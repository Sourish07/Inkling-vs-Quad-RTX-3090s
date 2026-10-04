"""Cached generation: our Inkling model, Accelerate offload, ModelOpt NVFP4."""

import json
import sys
import time
from pathlib import Path

import psutil
import torch
import tyro
from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from huggingface_hub import parse_local_safetensors_file_metadata, snapshot_download
from jaxtyping import Float as Fp
from jaxtyping import Int, Shaped, UInt8
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from safetensors import safe_open
from torch import Tensor as T
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoTokenizer

from my_inkling import MyInkling
from my_inkling.cache import MyInklingCache
from my_inkling.model import MyInklingMoE
from utils.checkpointing import convert_checkpoint_tensors

GIB = 1024**3
DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "U8": torch.uint8,
    "F8_E4M3": torch.float8_e4m3fn,
}


def load_config(path):
    config = AutoConfig.from_pretrained(path)
    text = json.loads((Path(path) / "config.json").read_text())["text_config"]
    # Small's legacy config otherwise overwrites the routed-expert width.
    if "dense_intermediate_size" in text:
        config.text_config.moe_intermediate_size = text["intermediate_size"]
    return config


class PackedExperts(nn.Module):
    """Accelerate moves the entire bank; ModelOpt decodes selected matrices locally."""

    # e: experts, t: tokens, k: selected experts/token, d: hidden width,
    # f: expert intermediate width, n: tokens routed to one expert.
    gate_up_proj: UInt8[T, "e 2*f d//2"] | Fp[T, "e 2*f d"]
    down_proj: UInt8[T, "e d f//2"] | Fp[T, "e d f"]
    gate_up_proj_scale: Fp[T, "e 2*f d//16"]
    down_proj_scale: Fp[T, "e d f//16"]
    gate_up_proj_scale2: Fp[T, " e"]
    down_proj_scale2: Fp[T, " e"]

    def __init__(self, tensors: dict[str, Shaped[T, "..."]]) -> None:
        super().__init__()
        for name, tensor in tensors.items():
            self.register_buffer(name, tensor)

    def matrix(
        self, projection: str, index: int, dtype: torch.dtype
    ) -> Fp[T, "rows cols"]:
        weight: Shaped[T, "rows stored_cols"] = getattr(self, projection)[index]
        if weight.dtype != torch.uint8:
            return weight
        shape = torch.Size((*weight.shape[:-1], weight.shape[-1] * 2))
        return NVFP4QTensor(shape, dtype, weight).dequantize(
            dtype=dtype,
            scale=getattr(self, projection + "_scale")[index],
            double_scale=getattr(self, projection + "_scale2")[index],
            block_sizes={-1: 16},
            fast=False,
        )

    def forward(
        self,
        hidden_states: Fp[T, "t d"],
        top_k_index: Int[T, "t k"],
        top_k_weights: Fp[T, "t k"],
    ) -> Fp[T, "t d"]:
        output: Fp[T, "t d"] = torch.zeros_like(hidden_states)
        for expert in torch.unique(top_k_index).tolist():
            if expert == self.gate_up_proj.shape[0]:
                continue
            tokens: Int[T, " n"]
            slots: Int[T, " n"]
            tokens, slots = torch.where(top_k_index == expert)
            weight = self.matrix("gate_up_proj", expert, hidden_states.dtype)
            projected: Fp[T, "n 2*f"] = F.linear(hidden_states[tokens], weight)
            del weight
            # Expert rows remain interleaved in the published checkpoint.
            activated: Fp[T, "n f"] = F.silu(projected[:, 0::2]) * projected[:, 1::2]
            weight = self.matrix("down_proj", expert, hidden_states.dtype)
            values: Fp[T, "n d"] = F.linear(activated, weight)
            del weight
            values = values * top_k_weights[tokens, slots, None]
            output.index_add_(0, tokens, values.to(hidden_states.dtype))
        return output


def checkpoint_headers(path):
    """Read only text weights; auxiliary quantization records are not model tensors."""
    tensors = {}
    files = sorted(
        file
        for file in Path(path).glob("*.safetensors")
        if file.name != "mtp.safetensors"
    )
    for file in files:
        for key, info in parse_local_safetensors_file_metadata(file).tensors.items():
            if key.startswith("model.llm.") and not key.endswith(
                (".original_shape", ".input_amax")
            ):
                tensors[key] = torch.empty(
                    info.shape, dtype=DTYPES[info.dtype], device="meta"
                )
    return files, tensors


def build_model(config, raw_tensors):
    tensors = convert_checkpoint_tensors(raw_tensors, packed_experts=True)
    with init_empty_weights(include_buffers=True):
        model = MyInkling(config).bfloat16()
        for i, layer in enumerate(model.model.language_model.layers):
            if isinstance(layer.mlp, MyInklingMoE):
                prefix = f"model.language_model.layers.{i}.mlp.experts."
                layer.mlp.add_module(
                    "experts",
                    PackedExperts(
                        {
                            key.removeprefix(prefix): value
                            for key, value in tensors.items()
                            if key.startswith(prefix)
                        }
                    ),
                )
        for name, parameter in model.named_parameters():
            if "conv1d.weight" in name:
                parameter.data = parameter.float()
        for layer in model.model.language_model.layers:
            if isinstance(layer.mlp, MyInklingMoE):
                layer.mlp.gate.register_buffer(
                    "e_score_correction_bias",
                    layer.mlp.gate.e_score_correction_bias.float(),
                )
    if set(model.state_dict()) != set(tensors):
        raise ValueError(
            f"Checkpoint/model keys differ: {set(model.state_dict()) ^ set(tensors)}"
        )
    return model.eval().requires_grad_(False)


def load_weights(model, files, raw_tensors, device_map, *, skip_checkpoint_loading=False):
    """Stream converted tensors into Accelerate's placements without a second checkpoint."""
    placements = sorted(device_map.items(), key=lambda item: len(item[0]), reverse=True)
    if skip_checkpoint_loading:
        print("Skipping checkpoint loading; allocating placeholder weights", file=sys.stderr)
        for name, tensor in model.state_dict().items():
            device = next(
                device
                for prefix, device in placements
                if not prefix or name == prefix or name.startswith(prefix + ".")
            )
            target = f"cuda:{device}" if isinstance(device, int) else device
            value = torch.zeros(tensor.shape, dtype=tensor.dtype, device=target)
            if name.endswith(("_scale", "_scale2", "global_scale")):
                value.fill_(1)
            set_module_tensor_to_device(model, name, device, value=value, clear_cache=False)
        return dispatch_model(model, device_map=device_map, offload_buffers=True)

    for file in files:
        print(f"Loading {file.name}", file=sys.stderr, flush=True)
        with safe_open(file, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 -- safe_open is not a dict
                if key not in raw_tensors:
                    continue
                converted = convert_checkpoint_tensors(
                    {key: handle.get_tensor(key)}, packed_experts=True
                )
                for name, value in converted.items():
                    device = next(
                        device
                        for prefix, device in placements
                        if not prefix or name == prefix or name.startswith(prefix + ".")
                    )
                    # CPU weights must own memory rather than retaining the file's mmap.
                    if device == "cpu":
                        value = value.clone()
                    set_module_tensor_to_device(
                        model, name, device, value=value, clear_cache=False
                    )
                    del value
                del converted
    if any(tensor.is_meta for tensor in model.state_dict().values()):
        raise ValueError("Some model tensors were not loaded")
    return dispatch_model(model, device_map=device_map, offload_buffers=True)


@torch.inference_mode()
def generate(model, input_ids, *, max_new_tokens, eos_token_ids):
    cache = MyInklingCache(model.config.text_config)
    for step in range(max_new_tokens):
        started = time.monotonic()
        logits = model(input_ids, cache=cache)[:, -1].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite logits")
        token = logits.argmax(dim=-1, keepdim=True)
        input_ids = token.to(input_ids.device)
        token_id = token.item()
        print(
            f"Token {step + 1}: id={token_id}, {time.monotonic() - started:.2f}s",
            file=sys.stderr,
            flush=True,
        )
        yield token_id
        if token_id in eos_token_ids:
            break


def main(
    checkpoint: str = "thinkingmachines/Inkling-Small-NVFP4",
    prompt: str = "What is 17 * 23?",
    max_new_tokens: int = 8,
    gpus: str = "0,1,2,3",
    gpu_gib: float = 20,
    cpu_gib: float = 100,
    max_sequence_length: int = 512,
    plan: bool = False,
    skip_checkpoint_loading: bool = False,
):
    """Generate greedily, prefilling once and decoding with cached state.

    Args:
        checkpoint: Local checkpoint directory or HF model ID (reuses the HF cache).
        prompt: User message; thinking effort is disabled.
        max_new_tokens: Maximum output tokens.
        gpus: Comma-separated GPU IDs; empty string runs on CPU.
        gpu_gib: Weight budget per GPU in GiB; leave room for decoding and activations.
        cpu_gib: CPU weight budget in GiB; capped by available RAM minus 8 GiB.
        max_sequence_length: Prompt plus generation limit for this sample.
        plan: Show Accelerate's placements without loading tensors.
        skip_checkpoint_loading: Allocate placeholder weights for the configured model
            instead of reading checkpoint tensors; config, headers, and tokenizer are still used.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    path = (
        Path(checkpoint)
        if Path(checkpoint).is_dir()
        else Path(
            snapshot_download(
                checkpoint,
                allow_patterns=[
                    "config.json",
                    "tokenizer*",
                    "special_tokens_map.json",
                    "*.jinja",
                    "model*.safetensors",
                ],
            )
        )
    )
    files, raw_tensors = checkpoint_headers(path)
    model = build_model(load_config(path), raw_tensors)
    memory: dict[int | str, int | str] = {
        int(gpu): int(gpu_gib * GIB) for gpu in gpus.split(",") if gpu.strip()
    }
    memory["cpu"] = min(
        int(cpu_gib * GIB), max(0, psutil.virtual_memory().available - 8 * GIB)
    )
    device_map = infer_auto_device_map(
        model,
        max_memory=memory,
        no_split_module_classes=["MyInklingDecoderLayer"],
        offload_buffers=True,
    )
    if "disk" in device_map.values():
        raise ValueError(
            "Weights exceed the GPU/CPU budgets; this sample does not use disk offloading"
        )
    for name, device in device_map.items():
        print(f"{name or '<model>'}: {device}", file=sys.stderr)
    if plan:
        return
    tokenizer = AutoTokenizer.from_pretrained(path)
    assert tokenizer is not None
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        reasoning_effort="none",
    )
    if not isinstance(input_ids, torch.Tensor):
        input_ids = input_ids["input_ids"]
    if input_ids.shape[-1] + max_new_tokens > max_sequence_length:
        raise ValueError("Prompt + generation exceeds max_sequence_length")
    input_device = next(
        (
            f"cuda:{device}" if isinstance(device, int) else device
            for device in device_map.values()
            if device != "cpu"
        ),
        "cpu",
    )
    input_ids = input_ids.to(input_device)
    started = time.monotonic()
    model = load_weights(
        model, files, raw_tensors, device_map,
        skip_checkpoint_loading=skip_checkpoint_loading,
    )
    print(
        f"Loaded in {time.monotonic() - started:.2f}s; prompt: {input_ids.shape[-1]} tokens",
        file=sys.stderr,
        flush=True,
    )
    eos = tokenizer.eos_token_id
    started = time.monotonic()
    tokens = list(
        generate(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=set(eos if isinstance(eos, list) else [eos]),
        )
    )
    elapsed = time.monotonic() - started
    print(
        f"Generated {len(tokens)} tokens in {elapsed:.2f}s ({len(tokens) / elapsed:.4f} tokens/s, excluding loading)",
        file=sys.stderr,
    )
    print(tokenizer.decode(tokens, skip_special_tokens=False))


if __name__ == "__main__":
    tyro.cli(main)
