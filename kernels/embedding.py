"""
Vocab-shard embedding lookup in one launch; the table may live in VRAM or pinned host memory.
"""

import cuda.bindings.driver as cuda_driver  # ty: ignore[unresolved-import]
import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from jaxtyping import Float as Fp
from jaxtyping import Int
from torch import Tensor as T

# One 16-byte read per thread for a 4096-wide BF16 row: one PCIe round trip.
_THREADS = 512
_VECTOR_BYTES = 16
_COMPILED: dict[tuple, object] = {}


@cute.kernel
def _embed_rows_kernel(
    ids: cute.Tensor,  # [tokens]
    table: cute.Tensor,  # [rows, d], device or host
    out: cute.Tensor,  # [tokens, d]
    start: cutlass.Int32,
):
    """
    Rows of one rank's shard of an embedding table. The table may live in pinned
    host memory, in which case the rows are read over PCIe.
    A token outside [start, start + rows) belongs to another rank: its row is zero.
    Grid: (tokens); every thread moves 16 bytes per step.
    """
    tidx, _, _ = cute.arch.thread_idx()
    token, _, _ = cute.arch.block_idx()
    threads, _, _ = cute.arch.block_dim()

    rows, width = table.shape
    values = _VECTOR_BYTES * 8 // table.element_type.width
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        table.element_type,
        num_bits_per_copy=_VECTOR_BYTES * 8,
    )
    # ((1, values), (rows or tokens, vectors)): the first mode is one 16-byte vector
    table_vectors = cute.zipped_divide(table, (1, values))
    out_vectors = cute.zipped_divide(out, (1, values))

    local = ids[token] - start
    mine = local >= 0 and local < rows
    vector = cute.make_rmem_tensor(((1, values),), table.element_type)
    for i in cutlass.range(tidx, width // values, threads):
        vector.fill(0)
        if mine:
            cute.copy(copy_atom, table_vectors[(None, (local, i))], vector)
        cute.copy(copy_atom, vector, out_vectors[(None, (token, i))])


@cute.jit
def _launch(
    ids: cute.Tensor,
    table: cute.Tensor,
    out: cute.Tensor,
    start: cutlass.Int32,
    stream: cuda_driver.CUstream,
):
    _embed_rows_kernel(ids, table, out, start).launch(
        grid=(ids.shape[0], 1, 1), block=(_THREADS, 1, 1), stream=stream
    )


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
    assert weight.is_contiguous() and row_bytes % _VECTOR_BYTES == 0
    assert weight.data_ptr() % _VECTOR_BYTES == 0
    assert input_ids.is_cuda and input_ids.dtype == torch.int64
    assert input_ids.is_contiguous()
    out = torch.empty(
        (*input_ids.shape, width), dtype=weight.dtype, device=input_ids.device
    )

    # The table's shape is a compile-time constant; the token count is not.
    ids = from_dlpack(input_ids.view(-1)).mark_layout_dynamic()
    table = from_dlpack(weight, assumed_align=_VECTOR_BYTES)
    rows_out = from_dlpack(out.view(-1, width), assumed_align=_VECTOR_BYTES)
    rows_out = rows_out.mark_compact_shape_dynamic(mode=0)
    # Launch on PyTorch's current stream; a CUDA graph capture records it like any
    # kernel.
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    key = (rows, width, weight.dtype, weight.is_cuda)
    if key not in _COMPILED:
        _COMPILED[key] = cute.compile(_launch, ids, table, rows_out, start, stream)
    _COMPILED[key](ids, table, rows_out, start, stream)
    return out
