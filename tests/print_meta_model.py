"""Inspect Inkling's model state dictionary without downloading weights."""

import torch
import tyro
from transformers import AutoConfig, AutoModelForMultimodalLM


def main(checkpoint: str = "thinkingmachines/Inkling-Small-NVFP4") -> None:
    """Initialize the checkpoint's architecture on meta and print its state dictionary.

    Args:
        checkpoint: Hugging Face model ID or local configuration directory.
    """
    config = AutoConfig.from_pretrained(checkpoint)
    with torch.device("meta"):
        model = AutoModelForMultimodalLM.from_config(
            config, dtype=config.text_config.dtype
        )

    for name, tensor in model.state_dict().items():
        print(f"{name}: {tensor}")


if __name__ == "__main__":
    tyro.cli(main)
