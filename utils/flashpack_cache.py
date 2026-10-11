"""
Load the per-rank FlashPack files written by scripts/make_flashpack.py.
"""

import os
from contextlib import contextmanager
from pathlib import Path

import torch
from flashpack import get_flashpack_file_metadata
from flashpack.utils import string_to_dtype
from loguru import logger

FLASHPACK_DIR = Path(__file__).resolve().parent.parent / ".flashpack"


def _layout(path):
    """
    Return a pack's macroblocks as (dtype, byte offset, bytes) and its records.
    """
    metadata = get_flashpack_file_metadata(str(path))
    blocks = [
        (string_to_dtype(b["dtype"]), b["offset_bytes"], b["length_bytes"])
        for b in metadata["macroblocks"]
    ]
    return blocks, metadata["index"]


@contextmanager
def _direct_reader(path):
    """
    Yield fill(start, target): copy file bytes at start into a uint8 tensor.

    FlashPack's own readers are avoided. Faulting a pack in through mmap is
    several times slower than the disk and its page cache competes with the
    pinned experts for host memory, while the parallel CUDA reader's threads
    each open a context on cuda:0, costing rank 0 256 MiB per other rank.
    """
    if os.environ.get("INKLING_SKIP_WEIGHTS") == "1":
        logger.info("Skipping weights loading from disk")
        # Headers still size and place every tensor; contents are zeros, not weights.
        yield lambda start, target: target.zero_()
        return
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


def load_non_expert_state_dict(rank, device):
    """
    Read this rank's non-experts onto device; tensors are views of one block per dtype.
    """
    path = FLASHPACK_DIR / f"non-experts-rank-{rank}.flashpack"
    blocks, records = _layout(path)
    storage = []
    with _direct_reader(path) as fill:
        for dtype, offset, length in blocks:
            block = torch.empty(length, dtype=torch.uint8, device=device)
            fill(offset, block)
            storage.append(block.view(dtype))
    return {
        r["name"]: storage[r["macroblock"]]
        .narrow(0, r["offset"], r["length"])
        .reshape(r["shape"])
        for r in records
    }


def load_expert_state_dict(rank, device, gpu_experts_per_rank, full_tp=False):
    """
    Read this rank's experts as {weight_name: {global_expert_id: tensor}}.

    full_tp reads the pack holding this rank's slice of every expert instead of
    its share of whole experts. The first gpu_experts_per_rank experts of the
    shard go to device; the rest each get their own pinned allocation. NVFP4 weights stay packed, with their
    _scale and _scale2 banks; gate/up rows stay interleaved as in the checkpoint.
    """
    prefix = "experts-tp" if full_tp else "experts"
    path = FLASHPACK_DIR / f"{prefix}-rank-{rank}.flashpack"
    blocks, records = _layout(path)

    def span(record):
        dtype, offset, _ = blocks[record["macroblock"]]
        size = torch.empty((), dtype=dtype).element_size()
        return offset + record["offset"] * size, record["length"] * size, dtype

    def expert_id(record):
        return int(record["name"].rsplit("/", 1)[1])

    first = min(map(expert_id, records))
    state = {}
    with _direct_reader(path) as fill:
        for record in sorted(records, key=lambda r: span(r)[0]):
            start, length, dtype = span(record)
            if expert_id(record) - first < gpu_experts_per_rank:
                value = torch.empty(length, dtype=torch.uint8, device=device)
            else:
                value = torch.empty(length, dtype=torch.uint8, pin_memory=True)
            fill(start, value)
            name = record["name"].rsplit("/", 1)[0]
            state.setdefault(name, {})[expert_id(record)] = value.view(dtype).reshape(
                record["shape"]
            )
    return state
