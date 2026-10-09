"""Convert the Inkling checkpoint into the per-rank FlashPack files ``run.py`` loads.

Run once, on CPU, with ``python -m scripts.make_flashpack``. Each rank gets
``non-experts-rank-N.flashpack`` (its TP shards) and ``experts-rank-N.flashpack``
(its contiguous expert shard); the loader decides which experts go to the GPU.
"""

from pathlib import Path
from types import SimpleNamespace

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


def main(world_size: int = 4) -> None:
    checkpoint_dir = snapshot_download(repo_id=hf_repo, local_files_only=True)
    config = load_config(checkpoint_dir)
    # The TP plan only needs the mesh for shard sizes while on the meta device.
    mesh = SimpleNamespace(size=lambda: world_size, get_local_rank=lambda: 0)
    with torch.device("meta"):
        model = MyInkling(config).bfloat16()
        model.restore_fp32()
        apply_tp_plan(model, config, mesh)
    templates = model.state_dict()

    states = [{} for _ in range(world_size)]
    for handle, key in _llm_tensors(checkpoint_dir, experts=False):
        converted = convert_checkpoint_tensors({key: handle.get_tensor(key)})
        for name, full in converted.items():
            if name not in templates:
                continue
            full = full.to(dtype=templates[name].dtype)
            module_name, _, parameter_name = name.rpartition(".")
            module = model.get_submodule(module_name)
            shard_dim = getattr(module, "_tp_shard_dims", {}).get(parameter_name)
            shards = (
                [full] * world_size
                if shard_dim is None
                else full.chunk(world_size, dim=shard_dim)
            )
            for state, shard in zip(states, shards):
                state[name] = shard.contiguous()
    for rank, state in enumerate(states):
        logger.info("Writing rank {} non-experts", rank)
        pack_to_file(
            state, str(FLASHPACK_DIR / f"non-experts-rank-{rank}.flashpack"), None
        )
    del states

    # One rank at a time: a rank's experts fit in host memory, all of them do not.
    for rank in range(world_size):
        state = {}
        for handle, key in _llm_tensors(checkpoint_dir, experts=True):
            tensor_slice = handle.get_slice(key)
            packed = tensor_slice.get_dtype() == "U8" or key.endswith(
                (".scale", ".scale2")
            )
            shard_size = tensor_slice.get_shape()[0] // world_size
            start = rank * shard_size
            converted = convert_checkpoint_tensors(
                {key: tensor_slice[start : start + shard_size]}, packed_experts=True
            )
            for name, bank in converted.items():
                if not packed:
                    bank = bank.to(dtype=templates[name].dtype)
                for local_id, weight in enumerate(bank):
                    state[f"{name}/{start + local_id}"] = weight.contiguous()
        logger.info("Writing rank {} experts", rank)
        pack_to_file(state, str(FLASHPACK_DIR / f"experts-rank-{rank}.flashpack"), None)


if __name__ == "__main__":
    tyro.cli(main)
