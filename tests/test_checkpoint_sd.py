"""Compare MyInkling's text parameters with BF16 checkpoint headers."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from huggingface_hub import HfApi
from huggingface_hub.utils import SafetensorsFileMetadata
from transformers import AutoConfig

from my_inkling import MyInkling
from utils.checkpointing import convert_checkpoint_shapes


def test_checkpoint_state_dict() -> None:
    repo_id = "thinkingmachines/Inkling-Small"
    ignored_prefixes = (
        "model.audio_tower.",
        "model.vision_tower.",
        "model.mtp.",
        "mtp.",
    )
    api = HfApi()
    info = api.model_info(repo_id)
    files = sorted(
        file.rfilename
        for file in info.siblings
        if file.rfilename.endswith(".safetensors")
        and file.rfilename != "mtp.safetensors"
    )
    assert files, f"No safetensors files found in {repo_id}"

    def read_header(filename: str) -> SafetensorsFileMetadata:
        # HTTP range requests read only headers, without downloading tensor data.
        return api.parse_safetensors_file_metadata(repo_id, filename, revision=info.sha)

    checkpoint_shapes = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for metadata in pool.map(read_header, files):
            for key, tensor in metadata.tensors.items():
                assert key not in checkpoint_shapes, f"Duplicate checkpoint key: {key}"
                checkpoint_shapes[key] = tuple(tensor.shape)
    checkpoint_shapes = convert_checkpoint_shapes(checkpoint_shapes)
    checkpoint_shapes = {
        key: shape
        for key, shape in checkpoint_shapes.items()
        if not key.startswith(ignored_prefixes)
    }

    config = AutoConfig.from_pretrained(repo_id, revision=info.sha)
    # Small's legacy config loses the expert size during dense-size conversion.
    config.text_config.moe_intermediate_size = 2048
    with torch.device("meta"):
        model = MyInkling(config)
    state_dict = model.state_dict()
    assert all(tensor.is_meta for tensor in state_dict.values())
    model_shapes = {
        key: tuple(tensor.shape)
        for key, tensor in state_dict.items()
        if not key.startswith(ignored_prefixes)
    }
    mismatches = [
        f"{key}: model={model_shapes.get(key, '<missing>')}, "
        f"checkpoint={checkpoint_shapes.get(key, '<missing>')}"
        for key in sorted(model_shapes.keys() | checkpoint_shapes.keys())
        if model_shapes.get(key) != checkpoint_shapes.get(key)
    ]
    if mismatches:
        missing_from_model = len(checkpoint_shapes.keys() - model_shapes.keys())
        missing_from_checkpoint = len(model_shapes.keys() - checkpoint_shapes.keys())
        shape_mismatches = (
            len(mismatches) - missing_from_model - missing_from_checkpoint
        )
        # Keep details in captured stdout so pytest's summary cannot repeat them.
        print("\n".join(mismatches))
        pytest.fail(
            f"{len(mismatches)} keys remain unmatched: "
            f"{missing_from_model} missing from model, "
            f"{missing_from_checkpoint} missing from checkpoint, "
            f"{shape_mismatches} shape mismatches; see output above.",
            pytrace=False,
        )
