"""Cache converted TP shards and mixed-device EP weights, without model changes."""

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
from flashpack import get_flashpack_file_metadata, pack_to_file
from flashpack.utils import string_to_dtype
from loguru import logger


def cache_directory(model, checkpoint_dir, mesh, *, experts=False, gpu_experts=None):
    """Invalidate on source changes, local layout, dtypes, or EP placement changes."""
    if os.environ.get("INKLING_FLASHPACK_CACHE") == "0":
        return None
    checkpoint_dir = Path(checkpoint_dir).resolve()
    sources = [
        checkpoint_dir / "config.json",
        *sorted(checkpoint_dir.glob("*.safetensors")),
    ]
    schema = []
    for name, tensor in model.state_dict().items():
        if (".mlp.experts." in name) != experts:
            continue
        module_name, _, parameter = name.rpartition(".")
        shard_dim = getattr(model.get_submodule(module_name), "_tp_shard_dims", {}).get(
            parameter
        )
        schema.append((name, list(tensor.shape), str(tensor.dtype), shard_dim))
    identity = {
        "version": 2,
        "source": str(checkpoint_dir),
        "files": [
            (p.name, p.stat().st_size, p.stat().st_mtime_ns)
            for p in sources
            if p.exists()
        ],
        "schema": schema,
        "world_size": mesh.size(),
        "gpu_experts": gpu_experts,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
        :20
    ]
    root = Path(
        os.environ.get("INKLING_FLASHPACK_CACHE", checkpoint_dir / ".flashpack")
    )
    return root / ("experts" if experts else "non-experts") / digest


def cache_ready(directory, mesh, device, *, experts=False):
    """Agree before TP fallback: partial caches must never split collective paths."""
    if directory is None:
        return False
    files = [directory / f"rank-{rank}.flashpack" for rank in range(mesh.size())]
    if experts:
        files += [
            directory / f"cpu-rank-{rank}.flashpack" for rank in range(mesh.size())
        ]
    ready = torch.tensor(int(all(p.is_file() for p in files)), device=device)
    dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=mesh.get_group())
    return bool(ready.item())


def save_pack(state, path):
    # FlashPack atomically replaces the destination. A sentinel supports empty
    # shards, including all-GPU/all-CPU layouts and ranks with no owned experts.
    path.parent.mkdir(parents=True, exist_ok=True)
    pack_to_file(
        state or {"__empty__": torch.zeros(1, dtype=torch.uint8)}, str(path), None
    )


def _layout(path):
    """Return a pack's macroblocks as (dtype, byte offset, bytes) and its records."""
    metadata = get_flashpack_file_metadata(str(path))
    if "macroblocks" in metadata:
        blocks = [
            (string_to_dtype(b["dtype"]), b["offset_bytes"], b["length_bytes"])
            for b in metadata["macroblocks"]
        ]
    else:  # Single-dtype packs, such as the empty sentinel, have one implicit block.
        dtype = string_to_dtype(metadata["target_dtype"])
        size = torch.empty((), dtype=dtype).element_size()
        blocks = [(dtype, 0, metadata["total_elems"] * size)]
    return blocks, [r for r in metadata["index"] if r["name"] != "__empty__"]


@contextmanager
def _direct_reader(path):
    """Yield fill(start, target): copy file bytes at start into a uint8 tensor.

    FlashPack's own readers are avoided. Faulting a pack in through mmap is
    several times slower than the disk and its page cache competes with the
    pinned experts for host memory, while the parallel CUDA reader's threads
    each open a context on cuda:0, costing rank 0 256 MiB per other rank.
    """
    # O_DIRECT needs block-aligned offsets, lengths, and memory; tensors are only
    # aligned to FlashPack's align_bytes, so bounce through an aligned buffer.
    # Large sequential reads keep the device queue full, unlike per-tensor reads.
    block_size = 4096
    bounce = torch.empty(
        64 << 20, dtype=torch.uint8, pin_memory=torch.cuda.is_available()
    )
    view = memoryview(bounce.numpy())
    direct = 0 if bounce.data_ptr() % block_size else os.O_DIRECT
    try:
        fd = os.open(path, os.O_RDONLY | direct)
    except OSError:  # Filesystems without O_DIRECT still take the same reads.
        fd = os.open(path, os.O_RDONLY)
    chunk_start = chunk_end = 0

    def fill(start, target):
        nonlocal chunk_start, chunk_end
        done = 0
        while done < target.numel():
            position = start + done
            if not chunk_start <= position < chunk_end:
                chunk_start = position - position % block_size
                read = os.preadv(fd, [view], chunk_start)
                if read <= position - chunk_start:
                    raise OSError(f"Short read at byte {position} of {path}")
                chunk_end = chunk_start + read
            skip = position - chunk_start
            count = min(target.numel() - done, chunk_end - position)
            target[done : done + count].copy_(bounce[skip : skip + count])
            done += count

    try:
        yield fill
    finally:
        os.close(fd)


def load_pack(path, device):
    """Read a pack onto device; tensors are views of one block per dtype."""
    blocks, records = _layout(path)
    storage = []
    with _direct_reader(path) as fill:
        for dtype, offset, length in blocks:
            block = torch.empty(length, dtype=torch.uint8, device=device)
            fill(offset, block)
            storage.append(block.view(dtype))
    return {
        r["name"]: storage[r.get("macroblock", 0)]
        .narrow(0, r["offset"], r["length"])
        .reshape(r["shape"])
        for r in records
    }


def read_pinned_pack(path):
    """Yield a pack's tensors, each in its own pinned allocation."""
    blocks, records = _layout(path)

    def span(record):
        dtype, offset, _ = blocks[record.get("macroblock", 0)]
        size = torch.empty((), dtype=dtype).element_size()
        return offset + record["offset"] * size, record["length"] * size, dtype

    with _direct_reader(path) as fill:
        for record in sorted(records, key=lambda r: span(r)[0]):
            start, length, dtype = span(record)
            value = torch.empty(length, dtype=torch.uint8, pin_memory=True)
            fill(start, value)
            yield record["name"], value.view(dtype).reshape(record["shape"])


def save_experts(state, directory, mesh):
    rank = mesh.get_local_rank()
    gpu, cpu = {}, {}
    for name, bank in state.items():
        for expert_id, value in bank.items():
            if value.is_cuda:
                gpu[f"{name}/{expert_id}"] = value
            else:
                cpu[f"{name}/{expert_id}"] = value
    # Each rank keeps its own CPU pack: merging them would rewrite every CPU
    # expert a second time. cache_ready requires every rank's pair of packs, so
    # an interrupted conversion cannot look like a complete cache to the next run.
    save_pack(cpu, directory / f"cpu-rank-{rank}.flashpack")
    save_pack(gpu, directory / f"rank-{rank}.flashpack")


def load_experts(directory, mesh, device):
    rank = mesh.get_local_rank()
    logger.info("Loading rank {} experts from FlashPack cache {}", rank, directory)
    state = {}
    for key, value in load_pack(directory / f"rank-{rank}.flashpack", device).items():
        name, expert_id = key.rsplit("/", 1)
        state.setdefault(name, {})[int(expert_id)] = value
    for key, value in read_pinned_pack(directory / f"cpu-rank-{rank}.flashpack"):
        name, expert_id = key.rsplit("/", 1)
        state.setdefault(name, {})[int(expert_id)] = value
    return state
