"""Convert and load Inkling checkpoint tensors."""

from pathlib import Path

import torch
from safetensors import safe_open
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
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


def load_non_expert_state_dict(
    model: torch.nn.Module, checkpoint_dir: str | Path, device_mesh: DeviceMesh
) -> dict[str, torch.Tensor]:
    """
    Read dense and shared-expert weights on rank 0, then shard and distribute.

    Routed expert banks are excluded. Shared experts use the DTensor placements
    installed by ``apply_tp_plan``, including converted gate/up tensors.

    use `model.load_state_dict(state, strict=False, assign=True)`
    """
    if device_mesh.ndim != 1:
        raise ValueError("Pass the 1-D TP device mesh.")
    rank = device_mesh.get_local_rank()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if device_mesh.device_type == "cuda"
        else torch.device(device_mesh.device_type)
    )
    templates = model.state_dict()
    state = {}
    for file in sorted(Path(checkpoint_dir).glob("*.safetensors")):
        if file.name == "mtp.safetensors":
            continue
        with safe_open(file, framework="pt", device=str(device)) as handle:
            for key in sorted(handle.keys()):
                if (
                    not key.startswith("model.llm.")
                    or ".mlp.experts." in key
                    or key.endswith((".original_shape", ".input_amax"))
                ):
                    continue
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
                        else torch.empty(
                            target.shape, dtype=target.dtype, device=device
                        )
                    )
                    value = distribute_tensor(
                        full, device_mesh, placements, src_data_rank=0
                    )
                    state[name] = value if distributed else value.to_local()
                    del full, value
                del converted
    return state


def load_expert_state_dict(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
    device_mesh: DeviceMesh,
    gpu_expert_ids: set[int],
) -> dict[str, dict[int, torch.Tensor]]:
    """Load this rank's contiguous expert shard with per-expert CPU/GPU storage.

    ``gpu_expert_ids`` contains global IDs, applied to every layer. Other experts
    stay in pinned CPU memory. Requires a CUDA mesh. Every rank reads only its
    shard from disk. Shard boundaries match DTensor's ``Shard(0)`` placement.

    Returns ``{weight_name: {global_expert_id: tensor}}``. Mixed-device banks
    cannot be loaded directly with ``model.load_state_dict``.
    """
    if device_mesh.ndim != 1:
        raise ValueError("Pass the 1-D EP device mesh.")
    if any(expert_id < 0 for expert_id in gpu_expert_ids):
        raise ValueError("Expert IDs must be nonnegative.")
    if device_mesh.device_type != "cuda":
        raise ValueError("Expert loading requires a CUDA mesh.")
    rank = device_mesh.get_local_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    templates = model.state_dict()
    state = {}
    for file in sorted(Path(checkpoint_dir).glob("*.safetensors")):
        if file.name == "mtp.safetensors":
            continue
        with safe_open(file, framework="pt", device="cpu") as handle:
            for key in sorted(handle.keys()):
                if (
                    not key.startswith("model.llm.")
                    or ".mlp.experts." not in key
                    or key.endswith((".original_shape", ".input_amax"))
                ):
                    continue
                tensor_slice = handle.get_slice(key)
                shape = tensor_slice.get_shape()
                names = convert_checkpoint_tensors(
                    {key: torch.empty(shape, device="meta")}
                )
                names = [name for name in names if name in templates]
                if not names:
                    continue
                num_experts = shape[0]
                if any(expert_id >= num_experts for expert_id in gpu_expert_ids):
                    raise ValueError(
                        f"GPU expert IDs must be below {num_experts} for {key}."
                    )
                shard_size = (
                    num_experts + device_mesh.size() - 1
                ) // device_mesh.size()
                start = min(rank * shard_size, num_experts)
                end = min(start + shard_size, num_experts)
                if start == end:
                    state.update({name: {} for name in names})
                    continue
                converted = convert_checkpoint_tensors({key: tensor_slice[start:end]})
                for name in names:
                    target = templates[name]
                    experts = {}
                    for local_id, expert_id in enumerate(range(start, end)):
                        weight = (
                            converted[name][local_id]
                            .to(dtype=target.dtype)
                            .contiguous()
                        )
                        if expert_id in gpu_expert_ids:
                            weight = weight.to(device)
                        else:
                            weight = weight.pin_memory()
                        experts[expert_id] = weight
                    state[name] = experts
                del converted
    return state
