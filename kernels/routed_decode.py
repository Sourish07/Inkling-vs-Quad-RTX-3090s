"""
Routed-expert projections for a few tokens: one matrix-vector product per route.
"""

import torch
from jaxtyping import Float as Fp
from jaxtyping import Int
from torch import Tensor as T

from .cuda_jit import CudaModule

_THREADS = 128
_MODULES: dict[tuple[int, bool], CudaModule] = {}


def _module(width: int, quantized: bool) -> CudaModule:
    key = (width, quantized)
    if key not in _MODULES:
        # A thread owns 32 columns and the threads of a block tile whole rows.
        assert width % 32 == 0 and _THREADS % (width // 32) == 0
        _MODULES[key] = CudaModule("routed_decode.cu", K=width, QUANT=int(quantized))
    return _MODULES[key]


def routed_decode(
    x: Fp[T, "t d"],
    indices: Int[T, "t k"],
    weights: Fp[T, "t k"],
    pointers: Int[T, " entries"],
    activated: Fp[T, "routes f"],
    output: Fp[T, "t d"],
    max_expert: int,
    quantized: bool,
    rows: int | None = None,
) -> None:
    """
    output[token] = sum over its routes of weight * down(silu(gate) * up).

    Routes whose pointer-table row is zero (experts of other ranks) are skipped.
    `activated` is scratch for the intermediate; `output` is overwritten.
    `rows` is the number of weight rows one thread block reads.
    """
    if rows is None:
        # Packed rows are short: more, smaller blocks keep more reads in flight.
        rows = 16 if quantized else 32
    tokens, hidden = x.shape
    top_k = indices.shape[1]
    intermediate = activated.shape[1]
    assert x.dtype == activated.dtype == weights.dtype == torch.bfloat16
    assert indices.dtype == pointers.dtype == torch.int64
    assert output.dtype == torch.float32 and output.shape == x.shape
    assert all(
        tensor.is_contiguous() and tensor.data_ptr() % 16 == 0
        for tensor in (x, indices, weights, activated, output)
    )
    _module(hidden, quantized)["gate_up"](
        (2 * intermediate // rows, tokens * top_k),
        _THREADS,
        x,
        indices,
        pointers,
        activated,
        top_k,
        max_expert,
        2 * intermediate,
        rows,
    )
    _module(intermediate, quantized)["down_reduce"](
        (hidden // rows, tokens),
        _THREADS,
        activated,
        indices,
        weights,
        pointers,
        output,
        top_k,
        max_expert,
        hidden,
        rows,
    )
