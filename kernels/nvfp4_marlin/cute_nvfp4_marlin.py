"""NVFP4 Marlin linear (W4A16) for Ampere, in CuTe DSL: the public entry point.

    out = nvfp4_linear(x, weight, scale, scale2, out=None)
    out, run = prepare(x, weight, scale, scale2)        # run() relaunches into out

    x       [M, K]       fp16 | bf16
    weight  [K/16, 2*N]  int32, Marlin-packed e2m1 codes
    scale   [K/16, N]    float8_e4m3fn, Marlin-processed block scales
    scale2  [1]          x's dtype
    out     [M, N]       x's dtype
    all contiguous, on one CUDA device; N % 128 == 0, K % 64 == 0, M >= 1.
    (marlin_utils has the layouts, the packing and an fp64 oracle.)

A fork of sglang's NVFP4 Marlin GEMM as standalone CuTe DSL kernels, tuned for
the routed MoE experts of Inkling-Small on an A100 (K = 4096 or 2048, N = 4096,
bf16, M = tokens routed to the expert, cold weights). Two kernels do the work
and this module only picks one per call:

    cute_nvfp4_decode   M <= 6, and M = 7, 8 on layers with N >= 7168
                        every CTA owns output columns outright: no cross-CTA
                        reduction, but every CTA reads all of x
    cute_nvfp4_batch    every other M >= 7
                        K-striped 64-column tiles, fp32 partials handed from
                        CTA to CTA; row chunks above 128 rows

The rule is measured, not assumed (README.md, "Dispatch", has the numbers).
On the expert shapes the two kernels tie at K = 4096 from M = 7 on while the
batch kernel is clearly ahead at K = 2048 for M = 7 and 8; at M = 6 they are
level over the two GEMMs of an expert, and below that the decode kernel wins.
On layers of 7168 columns or more the decode kernel keeps its lead up to its
last M. Both kernels accept M = 7 and 8, so the boundary is a matter of
speed only.
cute_nvfp4_marlin_port is the literal port of Marlin's architecture, kept as
a reference; nothing is dispatched to it.

What this module adds per call: ``size_m, size_k = x.shape`` and one branch
(for M = 7 and 8 also ``weight.nbytes``, to tell a wide layer). Both kernels
take what was read here instead of reading it again (``linear_sized``), so a
decode call through this module costs what a direct call of that module
costs, and a batch call a fraction of a microsecond more (README.md, "Call
path", has the measurements). All validation is the kernels' own: a call
outside the binding raises from the kernel it was sent to (TypeError /
ValueError), and nothing is launched for it.

One layer may be called with any M from call to call: each kernel keeps its
own state about a layer (the decode kernel a registry keyed on the three
tensor objects, the batch kernel per-shape launch blocks), and neither looks
at the other's.

    uv run --frozen -m nvfp4_marlin.cute_nvfp4_marlin                  # M = 1
    uv run --frozen -m nvfp4_marlin.cute_nvfp4_marlin --m 16
    uv run --frozen -m nvfp4_marlin.cute_nvfp4_marlin --shapes target

builds the two expert GEMMs at the given M, gates them against sglang's own
kernel and an fp64 oracle (harness.run_gate) and prints a cold-weight
benchmark against sglang and dense cuBLAS (harness.run_bench).
"""

import importlib
from dataclasses import dataclass
from typing import Literal

import tyro

from . import cute_nvfp4_batch, cute_nvfp4_decode

__all__ = ["kernel_for", "nvfp4_linear", "prepare"]

# The dispatch rule (README.md has the measurements behind it).
DECODE_MAX_M = 6  # M <= 6: the decode kernel ...
WIDE_MAX_M = 8  # ... which also keeps every M up to this one
WIDE_N = 7168  # on layers of at least this many columns

assert DECODE_MAX_M + 1 == cute_nvfp4_batch.MIN_M
assert WIDE_MAX_M == cute_nvfp4_decode.MAX_M

_SPLIT_M = DECODE_MAX_M + 1
_WIDE_COLS = 2 * WIDE_N  # weight is [K/16, 2 * N]
_decode_linear = cute_nvfp4_decode.nvfp4_linear
_decode_sized = cute_nvfp4_decode.linear_sized
_batch_sized = cute_nvfp4_batch.linear_sized


def kernel_for(size_m: int, size_n: int):
    """The kernel module that serves ``M = size_m`` on a layer of ``size_n`` columns."""
    if size_m <= DECODE_MAX_M or (size_m <= WIDE_MAX_M and size_n >= WIDE_N):
        return cute_nvfp4_decode
    return cute_nvfp4_batch


def _wide(weight) -> bool:
    try:
        return weight.shape[1] >= _WIDE_COLS
    except (AttributeError, IndexError, TypeError):
        return False  # not a weight: the batch kernel's checks say what is wrong


def nvfp4_linear(x, weight, scale, scale2, out=None):
    """NVFP4 linear: ``x [M, K] @ dequant(weight, scale, scale2) -> [M, N]``.

    ``out``, if given, is a preallocated contiguous [M, N] tensor of x's dtype
    on the same device that does not overlap x; it is written and returned as
    is. The kernel is launched on torch's current stream. Raises TypeError /
    ValueError for arguments outside the binding, before anything is launched.
    """
    try:
        size_m, size_k = x.shape
    except (ValueError, TypeError, AttributeError):
        # not a 2-D tensor: the decode kernel's checks raise the precise error
        return _decode_linear(x, weight, scale, scale2, out)
    if size_m < _SPLIT_M:
        return _decode_sized(x, weight, scale, scale2, out, size_m, size_k)
    if size_m <= WIDE_MAX_M:
        # M = 7, 8: the decode kernel on wide layers. weight is int32 [K/16, 2 N],
        # K * N / 2 bytes, so this is N >= WIDE_N without building weight.shape
        # (0.1 us cheaper); for a weight that does not fit x the kernel raises.
        try:
            wide = 2 * weight.nbytes >= WIDE_N * size_k
        except (AttributeError, TypeError):
            wide = False  # not a weight: the batch kernel's checks say what is wrong
        if wide:
            return _decode_sized(x, weight, scale, scale2, out, size_m, size_k)
    return _batch_sized(x, weight, scale, scale2, out, size_m)


def prepare(x, weight, scale, scale2):
    """One-time work for a call; returns ``(out, run)``: ``run()`` relaunches into ``out``.

    Dispatches like nvfp4_linear(). ``run()`` is bound to the memory the five
    tensors have now and to the M of this x; it is one C call (plus a look at
    the launch status).
    """
    try:
        size_m = x.shape[0]
    except (AttributeError, IndexError, TypeError):
        return cute_nvfp4_decode.prepare(x, weight, scale, scale2)  # raises
    if size_m < _SPLIT_M or (size_m <= WIDE_MAX_M and _wide(weight)):
        return cute_nvfp4_decode.prepare(x, weight, scale, scale2)
    return cute_nvfp4_batch.prepare(x, weight, scale, scale2)


# ---------------------------------------------------------------------------
# Entry point: gate and benchmark the target problem
# ---------------------------------------------------------------------------


@dataclass
class Args:
    m: int = 1
    """Rows of x: the tokens routed to the expert."""
    shapes: str = ""
    """Run these instead of the two expert GEMMs at --m: a harness shape set (decode, lowm, midm, target, prefill, ...) or 'M,K,N;M,K,N'."""
    dtype: Literal["bf16", "fp16"] = "bf16"
    """Activation dtype."""
    rounds: int = 7
    """Timing rounds per shape; the best round is reported."""
    gate: bool = True
    """Check the output against sglang's kernel and the fp64 oracle first."""
    bench: bool = True
    """Time run() and the full call against sglang and dense cuBLAS, cold weights."""


def main(args: Args) -> None:
    # The harness brings in sglang's own Marlin kernel (JIT-compiled from the
    # vendored source tree on first use: minutes, then cached).
    from nvfp4_marlin import harness

    # (under its real name, not as __main__: one module, one name in reports)
    impl = importlib.import_module("nvfp4_marlin.cute_nvfp4_marlin")
    if args.shapes:
        shapes = harness.parse_shapes(args.shapes, harness.BENCH_SHAPES)
    else:
        shapes = [(args.m, k, n) for k, n in harness.TARGET_KN]
    for m, k, n in shapes:
        name = kernel_for(m, n).__name__.rpartition(".")[2]
        print(f"[dispatch] M={m:<5} K={k:<5} N={n:<5} -> {name}")
    if args.gate:
        results = harness.run_gate(impl, shapes, [args.dtype])
        if not all(r.ok for r in results):
            raise SystemExit("gate FAILED")
    if args.bench:
        harness.run_bench(impl, shapes, [args.dtype], args.rounds, True, 0)


if __name__ == "__main__":
    main(tyro.cli(Args))
