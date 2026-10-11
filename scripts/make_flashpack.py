"""
Convert the Inkling checkpoint into the per-rank FlashPack files ``run.py`` loads.

Run on CPU with ``python -m scripts.make_flashpack --pack PACK``, writing
``PACK-rank-N.flashpack`` for every rank:

- ``non-experts``: the rank's TP shards of everything but the routed experts.
- ``experts``: expert parallel, a contiguous share of whole routed experts.
- ``experts-tp``: tensor parallel, the rank's slice of every routed expert.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import torch
import tyro
from flashpack import pack_to_file
from huggingface_hub import snapshot_download
from loguru import logger
from safetensors import safe_open

from my_inkling import MyInkling, apply_tp_plan
from utils import convert_checkpoint_tensors, load_config
from utils.flashpack_cache import FLASHPACK_DIR

hf_repo = "thinkingmachines/Inkling-Small-NVFP4"

# Intermediate-feature dimension of each stacked expert bank; `_scale2` is replicated.
EXPERT_TP_SHARD_DIMS = {
    "gate_up_proj": 1,
    "gate_up_proj_scale": 1,
    "down_proj": 2,
    "down_proj_scale": 2,
}


def _llm_tensors(checkpoint_dir: str | Path, *, experts: bool):
    """Yield ``(handle, key)`` for the routed-expert or the remaining LLM tensors."""
    for file in sorted(Path(checkpoint_dir).glob("*.safetensors")):
        if file.name == "mtp.safetensors":
            continue
        with safe_open(file, framework="pt", device="cpu") as handle:
            for key in sorted(handle.keys()):
                if (
                    key.startswith("model.llm.")
                    and (".mlp.experts." in key) == experts
                    and not key.endswith((".original_shape", ".input_amax"))
                ):
                    yield handle, key


def main(
    pack: Literal["non-experts", "experts", "experts-tp"], world_size: int = 4
) -> None:
    checkpoint_dir = snapshot_download(repo_id=hf_repo, local_files_only=True)
    config = load_config(checkpoint_dir)
    # The TP plan only needs the mesh for shard sizes while on the meta device.
    mesh = SimpleNamespace(size=lambda: world_size, get_local_rank=lambda: 0)
    with torch.device("meta"):
        model = MyInkling(config).bfloat16()
        model.restore_fp32()
        apply_tp_plan(model, config, mesh)
    templates = model.state_dict()
    experts = pack != "non-experts"

    # Experts go one rank at a time: a rank's fit in host memory, all of them do not.
    # Non-experts are small, so one read of the checkpoint serves every rank.
    ranks = range(world_size)
    for group in [[rank] for rank in ranks] if experts else [ranks]:
        states = {rank: {} for rank in group}
        for handle, key in _llm_tensors(checkpoint_dir, experts=experts):
            # Expert banks are only renamed, so they stay lazy until sliced below.
            lazy = handle.get_slice(key)
            tensor = lazy if experts else handle.get_tensor(key)
            converted = convert_checkpoint_tensors({key: tensor}, packed_experts=True)
            for name, tensor in converted.items():
                if not experts and name not in templates:
                    continue
                module_name, _, parameter_name = name.rpartition(".")
                if pack == "experts":
                    dim = 0  # Expert parallel splits the expert dimension.
                elif pack == "experts-tp":
                    dim = EXPERT_TP_SHARD_DIMS.get(parameter_name)
                else:
                    module = model.get_submodule(module_name)
                    dim = getattr(module, "_tp_shard_dims", {}).get(parameter_name)
                for rank, state in states.items():
                    index = slice(None)
                    if dim is not None:
                        shape = lazy.get_shape() if experts else tensor.shape
                        size = shape[dim] // world_size
                        index = (slice(None),) * dim + (
                            slice(rank * size, (rank + 1) * size),
                        )
                    shard = tensor[index]
                    if experts:
                        start = rank * len(shard) if pack == "experts" else 0
                        for local_id, weight in enumerate(shard):
                            state[f"{name}/{start + local_id}"] = weight
                    else:
                        state[name] = shard.to(templates[name].dtype).contiguous()
        for rank, state in states.items():
            logger.info("Writing rank {} {}", rank, pack)
            pack_to_file(
                state, str(FLASHPACK_DIR / f"{pack}-rank-{rank}.flashpack"), None
            )


if __name__ == "__main__":
    tyro.cli(main)
