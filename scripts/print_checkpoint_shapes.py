"""Print keys and stored shapes from every safetensors file, without loading tensors."""

from pathlib import Path

import tyro
from huggingface_hub import HfApi, parse_local_safetensors_file_metadata
from huggingface_hub.utils import SafetensorsFileMetadata


def print_shapes(filename: str, metadata: SafetensorsFileMetadata) -> None:
    print(f"{filename}:")
    for key, tensor in sorted(metadata.tensors.items()):
        print(f"  {key}: {tuple(tensor.shape)}")


def main(
    checkpoint: str = "thinkingmachines/Inkling-Small-NVFP4",
    revision: str = "main",
) -> None:
    """Inspect all safetensors headers, including auxiliary files such as MTP weights.

    Args:
        checkpoint: Hugging Face model ID, local directory, or local safetensors file.
        revision: Hub branch, tag, or commit to inspect; ignored for local paths.
    """
    path = Path(checkpoint)
    if path.exists():
        files = sorted(path.rglob("*.safetensors")) if path.is_dir() else [path]
        if not files:
            raise FileNotFoundError(f"No safetensors files found in {path}")
        for file in files:
            print_shapes(str(file), parse_local_safetensors_file_metadata(file))
        return

    api = HfApi()
    info = api.model_info(checkpoint, revision=revision)
    filenames = sorted(
        file.rfilename
        for file in info.siblings
        if file.rfilename.endswith(".safetensors")
    )
    if not filenames:
        raise FileNotFoundError(f"No safetensors files found in {checkpoint}")
    for filename in filenames:
        metadata = api.parse_safetensors_file_metadata(
            checkpoint, filename, revision=info.sha
        )
        print_shapes(filename, metadata)


if __name__ == "__main__":
    tyro.cli(main)
