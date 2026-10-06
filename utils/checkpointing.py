"""Convert and load Inkling checkpoint tensors."""

import json
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger
from safetensors import safe_open
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoConfig
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightConverter, WeightRenaming

from kernels.nvfp4_marlin import marlin_utils
from utils.flashpack_cache import (
    cache_directory,
    cache_ready,
    load_experts,
    load_pack,
    save_experts,
    save_pack,
)


def convert_checkpoint_tensors(
    checkpoint_tensors: dict[str, torch.Tensor], *, packed_experts: bool = False
) -> dict[str, torch.Tensor]:
    """Convert tensors; packed expert banks retain interleaved rows and separate scales."""
    mapping = get_checkpoint_conversion_mapping("inkling_mm_model")
    renamings = [item for item in mapping if isinstance(item, WeightRenaming)]
    converters = [item for item in mapping if isinstance(item, WeightConverter)]
    converted = {}
    for key, tensor in checkpoint_tensors.items():
        for renaming in renamings:
            key, _ = renaming.rename_source_key(key)

        outputs = {key: tensor}
        for converter in converters:
            renamed_key, source_pattern = converter.rename_source_key(key)
            if source_pattern is None:
                continue
            if packed_experts and ".mlp.experts." in key:
                tensors = {converter.target_patterns[0]: tensor}
            else:
                tensors = {source_pattern: tensor}
                for operation in converter.operations:
                    tensors = operation.convert(
                        tensors,
                        source_patterns=converter.source_patterns,
                        target_patterns=converter.target_patterns,
                    )
            prefix, _, suffix = renamed_key.partition(converter.target_patterns[0])
            outputs = {
                prefix + target + suffix: tensor for target, tensor in tensors.items()
            }
            break

        for target, tensor in outputs.items():
            if packed_experts and ".mlp.experts." in target:
                target = target.replace(".scale", "_scale")
            assert target not in converted, f"Duplicate converted key: {target}"
            converted[target] = tensor
    return converted


def convert_checkpoint_shapes(
    checkpoint_shapes: dict[str, tuple[int, ...]],
) -> dict[str, tuple[int, ...]]:
    """Apply the same conversion on meta tensors without allocating weight storage."""
    tensors = {
        key: torch.empty(shape, device="meta")
        for key, shape in checkpoint_shapes.items()
    }
    return {
        key: tuple(tensor.shape)
        for key, tensor in convert_checkpoint_tensors(tensors).items()
    }


def load_config(checkpoint_dir: str | Path):
    config = AutoConfig.from_pretrained(checkpoint_dir)
    text = json.loads((Path(checkpoint_dir) / "config.json").read_text())["text_config"]
    # Small's legacy config otherwise overwrites the routed-expert width.
    if "dense_intermediate_size" in text:
        config.text_config.moe_intermediate_size = text["intermediate_size"]
    return config


def _llm_tensors(checkpoint_dir: str | Path, device: str, *, experts: bool):
    """Yield ``(handle, key)`` for the routed-expert or the remaining LLM tensors."""
    for file in sorted(Path(checkpoint_dir).glob("*.safetensors")):
        if file.name == "mtp.safetensors":
            continue
        with safe_open(file, framework="pt", device=device) as handle:
            for key in sorted(handle.keys()):
                if (
                    key.startswith("model.llm.")
                    and (".mlp.experts." in key) == experts
                    and not key.endswith((".original_shape", ".input_amax"))
                ):
                    yield handle, key


def load_non_expert_state_dict(
    model: torch.nn.Module, checkpoint_dir: str | Path, device_mesh: DeviceMesh
) -> dict[str, torch.Tensor]:
    """
    Load a cached rank-local pack, or scatter/broadcast from rank 0 and cache.
    """
    rank = device_mesh.get_local_rank()
    group = device_mesh.get_group()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if device_mesh.device_type == "cuda"
        else torch.device(device_mesh.device_type)
    )
    cache = cache_directory(model, checkpoint_dir, device_mesh)
    if cache_ready(cache, device_mesh, device):
        logger.info("Loading rank {} non-experts from FlashPack cache {}", rank, cache)
        return load_pack(cache / f"rank-{rank}.flashpack", device)
    templates = model.state_dict()
    state = {}
    for handle, key in _llm_tensors(checkpoint_dir, str(device), experts=False):
        # All ranks read headers; only rank 0 reads the actual weight.
        names = convert_checkpoint_tensors(
            {key: torch.empty(handle.get_slice(key).get_shape(), device="meta")}
        )
        names = [name for name in names if name in templates]
        if not names:
            continue
        converted = (
            convert_checkpoint_tensors({key: handle.get_tensor(key)})
            if rank == 0
            else {}
        )
        for name in names:
            target = templates[name]
            module_name, _, parameter_name = name.rpartition(".")
            module = model.get_submodule(module_name)
            shard_dim = getattr(module, "_tp_shard_dims", {}).get(parameter_name)
            if rank == 0:
                full = converted[name].to(dtype=target.dtype).contiguous()
            if shard_dim is None:
                value = full if rank == 0 else torch.empty_like(target, device=device)
                dist.broadcast(value, src=0, group=group)
            else:
                value = torch.empty_like(target, device=device)
                shards = (
                    [
                        chunk.contiguous()
                        for chunk in full.chunk(device_mesh.size(), dim=shard_dim)
                    ]
                    if rank == 0
                    else None
                )
                dist.scatter(value, scatter_list=shards, src=0, group=group)
                del shards
            state[name] = value
            if rank == 0:
                del full
        del converted
    if cache is not None:
        save_pack(state, cache / f"rank-{rank}.flashpack")
    return state


@torch.no_grad()
def pack_experts_for_marlin(
    state: dict[str, dict[int, torch.Tensor]],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Repack NVFP4 expert banks from the checkpoint layout to the Marlin layout.

    Each tensor is rewritten inside its own allocation, which the Marlin layout
    fills exactly, so GPU and pinned-CPU placement are unchanged:

        weight  uint8 [N, K/2]  ->  int32 [K/16, 2N]
        scale   fp8 [N, K/16]   ->  fp8 [K/16, N]  (Marlin's byte encoding)
        scale2  float32 []      ->  ``dtype`` [2]  (exponent bias folded in, then padding)

    Banks of layers stored unquantized have no scales and are left alone.
    """
    for name in [name for name in state if name.endswith("_scale")]:
        weights, scales, scale2s = (
            state[name.removesuffix("_scale") + suffix]
            for suffix in ("", "_scale", "_scale2")
        )
        for expert in weights:
            packed = weights[expert].to(device)
            size_n, half_k = packed.shape
            codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1)
            target = weights[expert].view(-1).view(torch.int32)
            target = target.view(half_k // 8, 2 * size_n)
            target.copy_(marlin_utils.marlin_pack_weight(codes))
            weights[expert] = target

            scale = scales[expert].to(device).view(torch.float8_e4m3fn)
            target = scales[expert].view(half_k // 8, size_n)
            target.view(torch.uint8).copy_(
                marlin_utils.marlin_pack_scale(scale).view(torch.uint8)
            )
            scales[expert] = target

            scale2 = marlin_utils.marlin_pack_scale2(scale2s[expert].to(device), dtype)
            target = scale2s[expert].view(1).view(dtype)
            target.copy_(torch.cat((scale2, torch.zeros_like(scale2))))
            scale2s[expert] = target


def load_expert_state_dict(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
    device_mesh: DeviceMesh,
    gpu_experts_per_rank: int,
) -> dict[str, dict[int, torch.Tensor]]:
    """Load this rank's contiguous expert shard with per-expert CPU/GPU storage.

    The first ``gpu_experts_per_rank`` experts of each bank's shard go to this
    rank's GPU; the rest stay in pinned CPU memory. Every rank reads only its
    shard from disk.

    Returns ``{weight_name: {global_expert_id: tensor}}``. Mixed-device banks
    cannot be loaded directly with ``model.load_state_dict``. NVFP4 weights
    remain packed, with their ``_scale`` and ``_scale2`` tensors, and are
    returned in the Marlin layout (``pack_experts_for_marlin``). Layers stored
    unquantized have no scale banks. Either way, gate/up rows stay interleaved
    as in the checkpoint.

    Complete FlashPack caches bypass conversion and sharding. Cache misses use
    the original flow above, then save GPU packs and a shared CPU-expert pack.
    The cache holds the checkpoint layout; Marlin packing runs on every load.
    """
    rank = device_mesh.get_local_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    cache = cache_directory(
        model,
        checkpoint_dir,
        device_mesh,
        experts=True,
        gpu_experts=gpu_experts_per_rank,
    )
    templates = model.state_dict()
    dtype = next(v.dtype for k, v in templates.items() if ".mlp.experts." in k)
    if cache_ready(cache, device_mesh, device, experts=True):
        state = load_experts(cache, device_mesh, device)
        pack_experts_for_marlin(state, dtype, device)
        return state
    state = {}
    for handle, key in _llm_tensors(checkpoint_dir, "cpu", experts=True):
        tensor_slice = handle.get_slice(key)
        packed = tensor_slice.get_dtype() == "U8" or key.endswith((".scale", ".scale2"))
        num_experts = tensor_slice.get_shape()[0]
        shard_size = -(-num_experts // device_mesh.size())
        start = min(rank * shard_size, num_experts)
        converted = convert_checkpoint_tensors(
            {key: tensor_slice[start : start + shard_size]}, packed_experts=True
        )
        for name, bank in converted.items():
            if not packed:
                bank = bank.to(dtype=templates[name].dtype)
            state[name] = {
                start + local_id: weight.contiguous().to(device)
                if local_id < gpu_experts_per_rank
                else weight.contiguous().pin_memory()
                for local_id, weight in enumerate(bank)
            }
        del converted
    if cache is not None:
        save_experts(state, cache, device_mesh)
    pack_experts_for_marlin(state, dtype, device)
    return state
