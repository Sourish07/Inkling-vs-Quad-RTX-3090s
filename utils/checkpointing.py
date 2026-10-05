"""Convert and load Inkling checkpoint tensors."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
from transformers import AutoConfig
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightConverter, WeightRenaming


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
    Read dense and shared-expert weights on rank 0, then shard and distribute.

    Routed expert banks are excluded. Shared experts use the DTensor placements
    installed by ``apply_tp_plan``, including converted gate/up tensors.
    """
    rank = device_mesh.get_local_rank()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if device_mesh.device_type == "cuda"
        else torch.device(device_mesh.device_type)
    )
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
            distributed = isinstance(target, DTensor)
            placements = target.placements if distributed else [Replicate()]
            full = (
                converted[name].to(dtype=target.dtype).contiguous()
                if rank == 0
                else torch.empty(target.shape, dtype=target.dtype, device=device)
            )
            value = distribute_tensor(full, device_mesh, placements, src_data_rank=0)
            state[name] = value if distributed else value.to_local()
            del full, value
        del converted
    return state


def load_expert_state_dict(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
    device_mesh: DeviceMesh,
    gpu_experts_per_rank: int,
) -> dict[str, dict[int, torch.Tensor]]:
    """Load this rank's contiguous expert shard with per-expert CPU/GPU storage.

    The first ``gpu_experts_per_rank`` experts of each bank's shard go to this
    rank's GPU; the rest stay in pinned CPU memory. Every rank reads only its
    shard from disk. Shard boundaries match DTensor's ``Shard(0)`` placement.

    Returns ``{weight_name: {global_expert_id: tensor}}``. Mixed-device banks
    cannot be loaded directly with ``model.load_state_dict``. NVFP4 weights
    remain packed, with their ``_scale`` and ``_scale2`` tensors preserved.
    Layers stored unquantized have no scale banks. Either way, gate/up rows
    stay interleaved as in the checkpoint.
    """
    rank = device_mesh.get_local_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    templates = model.state_dict()
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
    return state
