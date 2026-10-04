"""Convert Inkling checkpoint names and fused weights using Transformers' mapping."""

import torch
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
