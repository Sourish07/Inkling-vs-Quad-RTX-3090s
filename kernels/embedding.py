"""
Vocab-shard embedding lookup in one launch; the table may live in pinned host memory.
"""

import torch
from jaxtyping import Float as Fp
from jaxtyping import Int
from torch import Tensor as T

from .cuda_jit import CudaModule

# One 16-byte read per thread for a 4096-wide BF16 row: one PCIe round trip.
_THREADS = 512
_EMBEDDING = CudaModule("embedding.cu")


def embed_rows(
    input_ids: Int[T, "bs s"], weight: Fp[T, "rows d"], start: int
) -> Fp[T, "bs s d"]:
    """
    Look up rows of a table on the GPU or in pinned host memory; the result is on
    the GPU.

    Token `start + i` is row i; tokens outside the table give zero rows.
    """
    rows, width = weight.shape
    row_bytes = width * weight.element_size()
    assert weight.device == input_ids.device or weight.is_pinned()
    assert weight.is_contiguous() and row_bytes % 16 == 0
    assert weight.data_ptr() % 16 == 0
    assert input_ids.is_cuda and input_ids.dtype == torch.int64
    assert input_ids.is_contiguous()
    out = torch.empty(
        (*input_ids.shape, width), dtype=weight.dtype, device=input_ids.device
    )
    _EMBEDDING["embed_rows"](
        (input_ids.numel(),),
        _THREADS,
        input_ids,
        weight,
        out,
        start,
        rows,
        row_bytes // 16,
    )
    return out
