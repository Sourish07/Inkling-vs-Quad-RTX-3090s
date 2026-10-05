"""Cached TP generation with Accelerate expert offload and ModelOpt NVFP4.

Run four GPUs with ``torchrun --standalone --nproc-per-node=4 -m scripts.run_naive``.
"""

import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import torch
import torch.distributed as dist
import tyro
from accelerate import (
    cpu_offload,
    infer_auto_device_map,
    init_empty_weights,
)
from accelerate.utils import set_module_tensor_to_device
from huggingface_hub import parse_local_safetensors_file_metadata, snapshot_download
from jaxtyping import Float as Fp
from jaxtyping import Int, Shaped, UInt8
from loguru import logger
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from safetensors import safe_open
from torch import Tensor as T
from torch import nn
from torch.distributed.tensor import DTensor
from torch.nn import functional as F
from transformers import AutoConfig, AutoTokenizer

from my_inkling import MyInkling
from my_inkling.cache import MyInklingCache
from my_inkling.model import MyInklingMoE
from my_inkling.tp_plan import apply_tp_plan
from utils.checkpointing import convert_checkpoint_tensors, load_non_expert_state_dict
from utils.dist import Timer, get_device_mesh, setup_ddp_local, setup_rank_aware_logger

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
    """One rank owns each bank; Accelerate offloads it and TP reduces its output."""

    # e: experts, t: tokens, k: selected experts/token, d: hidden width,
    # f: expert intermediate width, n: tokens routed to one expert.
    gate_up_proj: UInt8[T, "e 2*f d//2"] | Fp[T, "e 2*f d"]
    down_proj: UInt8[T, "e d f//2"] | Fp[T, "e d f"]
    gate_up_proj_scale: Fp[T, "e 2*f d//16"]
    down_proj_scale: Fp[T, "e d f//16"]
    gate_up_proj_scale2: Fp[T, " e"]
    down_proj_scale2: Fp[T, " e"]

    def __init__(self, tensors: dict[str, Shaped[T, "..."]], device_mesh=None) -> None:
        super().__init__()
        self.device_mesh = device_mesh
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
        for expert in torch.unique(top_k_index).tolist() if self._buffers else []:
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
        if self.device_mesh is not None:
            dist.all_reduce(output, group=self.device_mesh.get_group())
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
                layer.mlp.experts = PackedExperts(
                    {
                        key.removeprefix(prefix): value
                        for key, value in tensors.items()
                        if key.startswith(prefix)
                    }
                )
                layer.mlp.gate.e_score_correction_bias = (
                    layer.mlp.gate.e_score_correction_bias.float()
                )
        for name, parameter in model.named_parameters():
            if "conv1d.weight" in name:
                parameter.data = parameter.float()
    if set(model.state_dict()) != set(tensors):
        raise ValueError(
            f"Checkpoint/model keys differ: {set(model.state_dict()) ^ set(tensors)}"
        )
    return model.eval().requires_grad_(False)


def expert_placements(model, mesh, gpu_gib, cpu_gib):
    """Let Accelerate place whole banks after reserving resident TP weights."""
    banks = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, PackedExperts)
    }
    if not banks:
        return {}
    resident = sum(
        (tensor.to_local() if isinstance(tensor, DTensor) else tensor).numel()
        * tensor.element_size()
        for name, tensor in model.state_dict().items()
        if ".mlp.experts." not in name
    )
    memory = (
        {rank: max(0, int(gpu_gib * GIB) - resident) for rank in range(mesh.size())}
        if mesh.device_type == "cuda"
        else {}
    )
    memory["cpu"] = min(
        int(cpu_gib * GIB), max(0, psutil.virtual_memory().available - 8 * GIB)
    )
    # Rank 0 chooses once: available host RAM can differ slightly across processes.
    placements = [{}]
    if mesh.get_local_rank() == 0:
        devices = infer_auto_device_map(
            nn.ModuleList(banks.values()),
            max_memory=memory,
            no_split_module_classes=["PackedExperts"],
            offload_buffers=True,
            clean_result=False,
        )
        placements[0] = {name: devices[str(i)] for i, name in enumerate(banks)}
    dist.broadcast_object_list(placements, src=0, group=mesh.get_group())
    if "disk" in placements[0].values():
        raise ValueError(
            "Routed experts exceed the GPU/CPU budgets; disk offload is disabled"
        )
    return placements[0]


def load_weights(
    model,
    path,
    files,
    raw_tensors,
    device_map,
    mesh,
    *,
    skip_checkpoint_loading=False,
    cpu_load_workers=16,
):
    """Load TP weights through the shared loader; stream each bank only on its owner."""
    if cpu_load_workers < 1:
        raise ValueError("cpu_load_workers must be positive")
    rank = mesh.get_local_rank()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if mesh.device_type == "cuda"
        else torch.device("cpu")
    )
    owned = {
        name: storage
        for name, storage in device_map.items()
        if (0 if storage == "cpu" else storage) == rank
    }
    for name in device_map:
        if name not in owned:
            model.set_submodule(name, PackedExperts({}, mesh))
        else:
            model.get_submodule(name).device_mesh = mesh
    state = (
        {
            name: torch.zeros_like(tensor, device=device)
            for name, tensor in model.state_dict().items()
            if ".mlp.experts." not in name
        }
        if skip_checkpoint_loading
        else load_non_expert_state_dict(model, path, mesh)
    )
    if skip_checkpoint_loading:
        for name, tensor in state.items():
            if name.endswith("global_scale"):
                tensor.fill_(1)
    model.load_state_dict(state, strict=False, assign=True)
    del state
    # Native tensor copies release the GIL. Bound outstanding copies so completed
    # host tensors cannot accumulate outside the model. Only this thread mutates it.
    pending = deque()
    with ThreadPoolExecutor(
        max_workers=cpu_load_workers, initializer=torch.init_num_threads
    ) as pool:
        for file in files:
            with safe_open(file, framework="pt", device="cpu") as handle:
                for key in handle.keys():  # noqa: SIM118 -- safe_open is not a dict
                    if key not in raw_tensors or ".mlp.experts." not in key:
                        continue
                    name = next(
                        iter(
                            convert_checkpoint_tensors(
                                {key: raw_tensors[key]}, packed_experts=True
                            )
                        )
                    )
                    bank_name = name.rsplit(".", 1)[0]
                    if bank_name not in owned:
                        continue
                    value = (
                        torch.zeros_like(raw_tensors[key], device="cpu")
                        if skip_checkpoint_loading
                        else handle.get_tensor(key)
                    )
                    if skip_checkpoint_loading and name.endswith(("_scale", "_scale2")):
                        value.fill_(1)
                    if owned[bank_name] == "cpu" and cpu_load_workers > 1:
                        pending.append((name, pool.submit(value.clone)))
                        del value
                        if len(pending) < cpu_load_workers:
                            continue
                        name, future = pending.popleft()
                        value = future.result()
                        storage = "cpu"
                    else:
                        if owned[bank_name] == "cpu":
                            value = value.clone()  # Own host storage instead of retaining the checkpoint mmap.
                        storage = owned[bank_name]
                    set_module_tensor_to_device(
                        model, name, storage, value=value, clear_cache=False
                    )
                    del value
                # Finish copies while this file's mapping is still open.
                while pending:
                    name, future = pending.popleft()
                    value = future.result()
                    set_module_tensor_to_device(
                        model, name, "cpu", value=value, clear_cache=False
                    )
                    del value
    if any(tensor.is_meta for tensor in model.state_dict().values()):
        raise ValueError("Some model tensors were not loaded")
    if device.type == "cuda":
        for name, storage in owned.items():
            if storage == "cpu":
                cpu_offload(
                    model.get_submodule(name),
                    execution_device=device,
                    offload_buffers=True,
                )
    return model


@torch.no_grad()
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
        if dist.get_rank() == 0:
            logger.info(
                "Token {}: id={}, {:.2f}s",
                step + 1,
                token_id,
                time.monotonic() - started,
            )
        yield token_id
        if token_id in eos_token_ids:
            break


def main(
    checkpoint: str = "thinkingmachines/Inkling-Small-NVFP4",
    prompt: str = "What is 17 * 23?",
    max_new_tokens: int = 8,
    gpu_gib: float = 20,
    cpu_gib: float = 100,
    cpu_load_workers: int = 16,
    max_sequence_length: int = 512,
    plan: bool = False,
    skip_checkpoint_loading: bool = False,
):
    """Generate greedily, prefilling once and decoding with cached state.

    Args:
        checkpoint: Local checkpoint directory or HF model ID (reuses the HF cache).
        prompt: User message; thinking effort is disabled.
        max_new_tokens: Maximum output tokens.
        gpu_gib: Weight budget per GPU in GiB; leave room for decoding and activations.
        cpu_gib: CPU weight budget in GiB; capped by available RAM minus 8 GiB.
        cpu_load_workers: Concurrent CPU expert copies during loading; 1 uses serial copies.
        max_sequence_length: Prompt plus generation limit for this sample.
        plan: Show Accelerate's routed-expert placements without loading tensors.
        skip_checkpoint_loading: Allocate placeholder weights for the configured model
            instead of reading checkpoint tensors; config, headers, and tokenizer are still used.
    """
    device, _ = setup_ddp_local()
    mesh = get_device_mesh()
    setup_rank_aware_logger()
    is_main_process = dist.get_rank() == 0
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
    config = load_config(path)
    model = build_model(config, raw_tensors)
    apply_tp_plan(model, config, mesh)
    device_map = expert_placements(model, mesh, gpu_gib, cpu_gib)
    if is_main_process:
        for name, storage in device_map.items():
            logger.info("{}: {}", name, storage)
    if plan:
        dist.destroy_process_group()
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
    input_ids = input_ids.to(device)
    with Timer("Checkpoint loading"):
        model = load_weights(
            model,
            path,
            files,
            raw_tensors,
            device_map,
            mesh,
            skip_checkpoint_loading=skip_checkpoint_loading,
            cpu_load_workers=cpu_load_workers,
        )
    if is_main_process:
        logger.info("Prompt: {} tokens", input_ids.shape[-1])
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
    if is_main_process:
        logger.info(
            "Generated {} tokens in {:.2f}s ({:.4f} tokens/s, excluding loading)",
            len(tokens),
            elapsed,
            len(tokens) / elapsed,
        )
        print(tokenizer.decode(tokens, skip_special_tokens=False))
    dist.destroy_process_group()


if __name__ == "__main__":
    tyro.cli(main)
