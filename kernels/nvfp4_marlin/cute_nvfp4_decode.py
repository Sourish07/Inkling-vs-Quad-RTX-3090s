"""NVFP4 Marlin linear for 1 <= M <= 8 (decode): column ownership (CuTe DSL, sm80).

    out = nvfp4_linear(x, weight, scale, scale2, out=None)      # see marlin_utils

The DECODE kernel of the package. cute_nvfp4_marlin, the public entry, sends
M <= 6 here (and M = 7, 8 on layers with N >= 7168); from M = 7 on the
K-striped tiles of cute_nvfp4_batch are faster on the target shapes (README.md
has the crossover data). The module is complete for every M <= 8 on its own.

COLUMN OWNERSHIP. Every CTA owns a set of output columns
outright and walks all of K for them; there are no locks, no scratch buffers
and no cross-CTA reduction. One launch per run().

Geometry
  * Slice = the 8 output columns {64q + 8u + g : u = 0..7} = what the four
    lanes (g, t = 0..3) of a warp hold of a Marlin-packed k16 row: 64 B of
    weights (4 words per lane = the four n16 blocks) plus exactly the 8 scale
    bytes at q*64 + 8g. Slices are the ownership unit because they are whole
    32 B sectors of the packed stream: no CTA ever pulls a byte it does not
    use, and N = 4096 gives 512 of them, dealt 4 or 5 per CTA over 108 CTAs
    (one per SM) -- max/mean 1.055.
  * The MMA is Marlin's transposed mma.m16n8k16: A = 16 output columns x 16 k
    (dequantized weights, 4 registers per lane), B = 16 k x 8 "rows of x".
    A lane group of 4 (a "g-slot") is therefore one slice at one k16 row, and
    the 8 B columns say which rows of x at which k16 rows it is multiplied
    with. The K range of a slice is split over "classes" of k16 rows (class c
    = rows c, c + C, c + 2C, ...), one g-slot per (slice, class): at step j
    every CTA reads the band of rows [C j, C (j + 1)), i.e. the grid sweeps
    the packed matrix sequentially in K.
      DIAG (M = 1)      every B column carries x at a different k16 row, so
                        the 8 g-slots of a warp are 8 independent streams and
                        only the diagonal C[g'][g'], C[g' + 8][g'] is read.
      QUAD (M = 2..8)   B columns 0-3 / 4-7 carry rows 0..3 of x at the k16
                        rows of two classes; a second MMA on the same
                        dequantized A carries rows 4..7. The g-slots of a warp
                        split 4 + 4, or 5 + 3 | 2 + 5 (+ 1 idle) over a warp
                        pair when a CTA owns 5 slices, between its two classes.
    C need not divide K/16: the first K/16 % C classes run one more step, and
    warps holding none of them skip it.
  * 8 warps per CTA: an SM issues from four warp schedulers (warp w on
    scheduler w % 4) and two warps saturate one; with 10 equal warps the load
    is 3:3:2:2 and the CTA ends 0.8 us after its first warp does.
  * One step of a warp = for each of its lanes one int4 (16 B, cp.async .cg
    straight into a private smem ring, 5 steps deep), 8 scale bytes per
    g-slot, then 4 x [dequant 8 weights/lane, scale, MMA(s)]. Nothing in the
    loop is synchronous. A warp never reads weights or scales another one
    fetched, and it starts on its own first group.
  * x. Every CTA needs all of x, 108 SMs read the same L2 lines at the same
    time, and that is what makes M > 1 slower than M = 1: what it costs is the
    number of requests per 128 B line, not the bytes (measured at M=8,
    K=N=4096: all per-step x fetches removed 12.3 -> 11.2 us, half the bytes in
    the same lines -> 12.0, half the lines -> 11.1). Staging all of x up front
    is no way out either: 64 KB of hot x take an SM 2.5-3.2 us to pull.
      ring (DIAG, and QUAD wherever BAND was not measured to pay)   each warp
          fetches the k16 rows of x its own classes need, per step, through
          its private ring (cp.async .ca): half a line per row of x and warp,
          and a class that two warps share twice.
      BAND (QUAD with M >= 3, one CTA per SM and more than 15 steps; for
          M <= 4 with at most 12 classes, for M > 4 only in the 5-slice
          geometry -- N = 3584 .. 4224 -- and up to 60 steps: the table in
          _quad_plan)   x is fetched once per CTA and in whole lines,
          the columns of D steps at a time, into one of two smem buffers:
          warp w moves 512 B pieces of row w of x. Lanes then read what other
          warps fetched, so each group of D steps starts with a barrier --
          after the wait for the cp.async group that carried the group's x
          (it has landed for every fetcher) and before the next group's x is
          requested into the other buffer (nobody still reads what it held).
          The refill of that step goes out before the barrier, so the weight
          stream never waits for it. See _quad_kernel and _band_tail.
  * Dequant per word (bf16): sign/code nibbles -> bf16 bit pattern with
    ((q & nib) * spread) & mask (3 ops per packed register), block scales
    byte -> bf16 pattern with sglang's bit trick, native packed multiply
    (fma.rn.bf16x2 / mul.rn.f16x2) with the scale broadcast folded into the
    HFMA2 operand swizzle. The B values are bit-identical to sglang's.
  * End: every g-slot leaves its fp32 partial sums in smem, one barrier, then
    each owned output is summed over its classes in a fixed order
    (deterministic), converted fp32 -> dtype, multiplied by scale2 in that
    dtype (as sglang does, with the native cvt / packed multiply, so NaN and
    Inf propagate) and stored.
  GENERIC is the older, slower flavor (uneven per-lane streams, predicated
  loads) kept for geometries the planners decline (debug_knobs("flavor=generic")).

Supported domain: 1 <= M <= 8 (NotImplementedError above), any K % 64 == 0,
N % 128 == 0, fp16 and bf16, contiguous inputs on one CUDA device. The kernel
reads x and weight in 16 B and scale in 8 B units (fresh torch allocations and
row views are aligned); other contiguous views take a slow path through
aligned copies. scale2 is read as the aligned 4-byte word that contains it.

Host path. The launchers take raw pointers, so a launch is one C call on a
pre-marshalled argument block whose pointer / stream / status cells are the
only thing that differs between calls (_Plan); the status cell is read back
after every launch. prepare() gives run() a block of its own; run() launches
on the torch stream current at prepare(), on the memory the five tensors had
then (it keeps those allocations alive).
nvfp4_linear(x, weight, scale, scale2, out=None) checks x and out in full on
every call -- shape, dtype, device, contiguity, alignment, overlap; nothing
about them is remembered. weight, scale and scale2 are checked in full when a
triple of tensor objects is first seen and then registered (_LAYERS). A later
call trusts the registration only for the same three objects, at the same
addresses, on allocations that are still alive (weak references to the three
storages): other objects, tensors rebound to other memory, a new allocation
that landed on an old address are all validated from scratch. A call the fast
path declines is re-examined with nothing remembered before it raises, so a
valid call is never refused because of what the registry held. The one thing
not noticed is a REGISTERED tensor whose shape, strides or dtype are edited in
place on the same allocation at the same address (w.data = w.data[:r], t_()):
a call that still fits the registered layer is then computed as that layer --
on memory that was validated and is still owned, never out of bounds; reading
the two shapes back on every call would cost 0.45 us.
The call then takes a launch block of the layer (it already holds the layer's
three pointers), writes x, out and the current stream, and launches: ~3.5 us
of Python around the 3.6 us launch call, no planning and, with out=, no
allocation. Passing fresh tensor objects for the layer on every call
(param.data, W_all[e]) means full validation every time: ~15 us per call.
Compiled kernels are cached on disk (NVFP4_CUTE_CACHE, default
~/.cache/nvfp4_marlin_cute), keyed by this file's hash, the launcher, the
geometry / dtype, the GPU architecture and the toolchain versions. A cache
file is the relocatable object plus its sha256: a file that does not verify
(cut short, overwritten, one flipped bit) is rebuilt, never linked. A fresh
process pays ~3 ms per kernel instead of 0.5-1.1 s.

Development-time measurements (README.md has the numbers of the final files).
On the A100 (harness, bf16, cold weights, us per run(); K=N=4096 /
K=2048,N=4096; sglang 15.7 / 12.7): M=1 9.3 / 6.8, M=2 9.8 / 7.1,
M=4 9.9 / 7.1, M=8 11.4 / 8.8 (the same binary moves by 0.2-0.5 us between
sessions; from M=4 on a launch also depends on the ADDRESS of x by 0.5-1.2 us
-- M=8: 11.3 .. 12.4 / 8.4 .. 9.3 over 20 addresses -- and a harness row is
one draw). nvfp4_linear(out=) costs ~7.3 us of host time whatever x and out
are, so it reads as run() where the kernel takes longer and 7.3 (run() + 0.6
at most) on the three K=2048 points below that; the allocating call is
host-bound at ~10.4 us (torch.empty); sglang's own call is 30 us. At M=1
the same kernel without dequant and MMA takes 8.95 / 6.3 us, i.e. the
arithmetic costs 0.5 us on top
of the stream; of the 6.7 / 4.1 us after the 2.7 us launch floor, ~1.25 us
pass before the first step's data has landed and ~0.35 us go to the reduction
and the epilogue. The SMs do not finish together: 46 of the 108 enter the
kernel 0.15-0.27 us after the others (bidx b runs on SM 2b, or 2(b - 54) + 1),
and at M >= 4 a fixed set of 56 SMs runs 10-18 % slower than the other 52 as
soon as x is read through L2 -- the launch ends with the slowest group.

There are no environment tuning knobs (NVFP4_CUTE_CACHE only moves the cache).
DEVELOPER TOOLS, never called by the package: debug_knobs("warps=8,depth=5,grid=108,
flavor=generic,band=0,prof=1", compile_opts=...) is the development hook for
sweeps, profiling builds (per-warp phase stamps, see profile()) and SASS dumps.
"""

import collections
import ctypes
import dataclasses
import functools
import hashlib
import os
import re
import shutil
import struct
import tempfile
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path

import cuda.bindings.driver as cuda_driver  # ty: ignore[unresolved-import]
import cutlass
import torch
from cutlass import Float32, Int16, Int32, Int64, Uint32, Uint64, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_ptr

MAX_M = 8

ROW_W = 512  # 32 lanes x one int4 of packed weights
ROW_S = 64  # 8 g-slots x 8 scale bytes
ROW_X = 256  # 8 g-slots x 32 B of activations (GENERIC flavor only)
GROW_BYTES = ROW_W + ROW_S + ROW_X
NSTAMP = 10  # profiling stamps per warp: see profile()


def _i32(v):
    """Python int bit pattern -> signed 32-bit constant."""
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


_NEG0 = _i32(0x80008000)
_SIGN2 = _i32(0x80008000)
_CODE2 = 0x70007000


# ---------------------------------------------------------------------------
# Inline-PTX primitives the DSL has no spelling for
# ---------------------------------------------------------------------------


def _asm(res_type, operands, text, constraints):
    return llvm.inline_asm(
        res_type,
        operands,
        text,
        constraints,
        has_side_effects=False,
        is_align_stack=False,
    )


def _mul2(bf16, a, b):
    """Packed x2 multiply of two 16-bit float lanes per register (one HFMA2)."""
    if bf16:
        ops = [Int32(a).ir_value(), Int32(b).ir_value(), Int32(_NEG0).ir_value()]
        return Int32(
            _asm(Int32.mlir_type, ops, "fma.rn.bf16x2 $0, $1, $2, $3;", "=r,r,r,r")
        )
    ops = [Int32(a).ir_value(), Int32(b).ir_value()]
    return Int32(_asm(Int32.mlir_type, ops, "mul.rn.f16x2 $0, $1, $2;", "=r,r,r"))


def _dup16(h):
    """Both 16-bit lanes = ``h``; ptxas folds this into the HFMA2 operand swizzle (.H0_H0)."""
    return Int32(
        _asm(Int32.mlir_type, [Int16(h).ir_value()], "mov.b32 $0, {$1, $1};", "=r,h")
    )


def _mma(bf16, a, b, c):
    """One mma.m16n8k16 (fp32 accumulate) on raw registers; returns 4 Float32."""
    ty = "bf16" if bf16 else "f16"
    text = (
        f"mma.sync.aligned.m16n8k16.row.col.f32.{ty}.{ty}.f32 "
        "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};"
    )
    ops = [Int32(v).ir_value() for v in (*a, *b)] + [Float32(v).ir_value() for v in c]
    d = _asm(
        ir.Type.parse("!llvm.struct<(f32,f32,f32,f32)>"),
        ops,
        text,
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
    )
    return tuple(
        Float32(llvm.extractvalue(Float32.mlir_type, d, [i])) for i in range(4)
    )


def _clock():
    """SM cycle counter (profiling builds only)."""
    return Int64(
        llvm.inline_asm(
            Int64.mlir_type,
            [],
            "mov.u64 $0, %clock64;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
        )
    )


# ---------------------------------------------------------------------------
# Small helpers (plain Python: they run at trace time)
# ---------------------------------------------------------------------------


def _smem_ptr(dtype, addr, align):
    return cute.make_ptr(dtype, addr, cute.AddressSpace.smem, assumed_align=align)


def _gmem_ptr(dtype, addr, align):
    return cute.make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=align)


def _smem_scalar(dtype, addr):
    return cute.make_tensor(_smem_ptr(dtype, addr, 4), cute.make_layout((1,)))


def _gmem_scalar(dtype, addr, align):
    return cute.make_tensor(_gmem_ptr(dtype, addr, align), cute.make_layout((1,)))


def _lds(addr, dst, n, align):
    """smem -> registers, n int32 in one load."""
    cute.autovec_copy(
        cute.make_tensor(_smem_ptr(Int32, addr, align), cute.make_layout((n,))), dst
    )


def _fetch(dst, src, size, mode):
    cute.arch.cp_async_shared_global(
        _smem_ptr(Int32, dst, size), _gmem_ptr(Int32, src, size), size, mode
    )


def _sel(cond, a, b, ty=Int32):
    return ty(cutlass.select_(cond, a, b))


def _wide(a, b):
    """u32 x u32 -> 64-bit byte offset (one IMAD.WIDE.U32 instead of a 64x64 multiply)."""
    return Int64(Uint64(Uint32(a)) * Uint64(Uint32(b)))


def _mdiv(a, magic):
    """floor(a / d) for small non-negative a, with magic = floor(2^32 / d) + 1."""
    return Int32((Int64(a) * magic) >> 32)


def _tree_sum(vals):
    """Pairwise sum in a fixed order (trace-time helper)."""
    vals = list(vals)
    while len(vals) > 1:
        vals = [
            vals[i] + vals[i + 1] if i + 1 < len(vals) else vals[i]
            for i in range(0, len(vals), 2)
        ]
    return vals[0]


def _finish(bf16, tot, s2):
    """fp32 sum -> activation dtype (RNE), times scale2 in that dtype; returns the 16 stored bits.

    Native conversion and native packed multiply (what sglang's __float2bfloat16 / __hmul
    compile to): two instructions, and NaN / Inf propagate as IEEE says. ``s2`` holds
    scale2's bits in its low half.
    """
    cvt = "cvt.rn.bf16.f32 $0, $1;" if bf16 else "cvt.rn.f16.f32 $0, $1;"
    r1 = Int16(_asm(Int16.mlir_type, [Float32(tot).ir_value()], cvt, "=h,f"))
    return Int16(_mul2(bf16, Int32(r1) & 0xFFFF, s2))


def _dequant_word(bf16, j, q, sw):
    """Word j (8 weights of one n16 block) x block scale -> the four A registers of the MMA.

    Nibble p of each 16-bit half of ``q`` is (k = 2t,2t+1 | 2t+8,2t+9) x
    (column g | g + 8) for p = 0..3; scale bytes (j % 2) and (j % 2) + 2 of
    ``sw`` belong to the two column halves. MMA A registers are
    (p0, p2, p1, p3) = (g; 2t..), (g+8; 2t..), (g; 2t+8..), (g+8; 2t+8..).
    """
    if (
        bf16
    ):  # byte b -> ((b & 0x80) << 7) | ((b & 0x7F) << 4): sglang's dequant_fp8_scales
        if j % 2 == 0:
            p = ((sw << 7) & 0x40004000) | ((sw << 4) & 0x07F007F0)
        else:
            p = ((sw >> 1) & 0x40004000) | ((sw >> 4) & 0x07F007F0)
    else:  # byte b -> b << 7
        if j % 2 == 0:
            p = (sw & 0x00FF00FF) << 7
        else:
            p = (sw >> 1) & 0x7F807F80
    s_lo = _dup16(Int16(p))
    s_hi = _dup16(Int16(p >> 16))
    areg = []
    if bf16:
        # nibble [s e1 e0 m] -> sign at bit 15, code at bits 8..6: isolate, spread with one
        # multiply (two disjoint shifted copies), mask.
        q2 = q >> 8
        for src, msk, mul, sc in (
            (q, 0x000F000F, 0x1040, s_lo),
            (q2, 0x000F000F, 0x1040, s_hi),
            (q, 0x00F000F0, 0x0104, s_lo),
            (q2, 0x00F000F0, 0x0104, s_hi),
        ):
            areg.append(_mul2(True, ((src & msk) * mul) & _i32(0x81C081C0), sc))
    else:
        # fp16: sign at bit 15, code at bits 11..9 (the two copies would overlap: use shifts)
        for nib, sc in ((0, s_lo), (2, s_hi), (1, s_lo), (3, s_hi)):
            tq = q << (12 - 4 * nib) if nib < 3 else q
            areg.append(_mul2(False, (tq & _SIGN2) | ((tq & _CODE2) >> 3), sc))
    return areg


def _compute_step(bf16, wq, sq, pairs):
    """One step of a lane: 4 words -> 4 transposed MMAs per (x fragment, accumulators) pair."""
    for sl in range(4):
        areg = _dequant_word(bf16, sl, wq[sl], sq[sl // 2])
        for xr, acc in pairs:
            res = _mma(bf16, areg, (xr[0], xr[1]), [acc[4 * sl + e] for e in range(4)])
            for e in range(4):
                acc[4 * sl + e] = res[e]


# ---------------------------------------------------------------------------
# DIAG flavor: M = 1, eight independent (slice, k16-row class) streams per warp
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DiagCfg:
    bf16: bool
    warps: int
    depth: int  # cp.async ring depth in steps
    rch: int  # 16 B chunks per row of the partial-sum buffer (>= classes / 4)
    rbytes: int  # smem for the partial sums
    short: bool  # no more steps than the ring is deep: everything is fetched in the prologue
    tm: int  # full refill steps left over after the unrolled groups of `depth`
    ext: bool  # K/16 is not a multiple of the class count: the first classes run one more step
    prof: bool = False
    grouped: bool = False  # operands come from a device-side record table: see _diag_group_launch

    mpad = 1

    @property
    def threads(self):
        return 32 * self.warps

    # smem: three rings, each stage contiguous over the CTA (a thread's slot is tidx * 16 + const):
    #   weights  depth x threads x 16 B          scales  depth x threads / 4 x 8 B
    #   x        depth x warps x 256 B (the k16 rows of x the warp's streams read at that step)
    # then the partial sums and the scale2 word.
    @property
    def wstage(self):
        return self.threads * 16

    @property
    def sstage(self):
        return self.threads * 2

    @property
    def xstage(self):
        return self.warps * 256

    @property
    def s_off(self):
        return self.depth * self.wstage

    @property
    def x_off(self):
        return self.s_off + self.depth * self.sstage

    @property
    def red_off(self):
        return self.x_off + self.depth * self.xstage

    @property
    def smem_bytes(self):  # (+16: the scale2 word)
        return self.red_off + self.rbytes + 16


def _diag_consume(cfg, d, dst_w, dst_s, rd_x, wq, sq, xr):
    """Ring stage d -> registers: this lane's int4, its g-slot's 8 scale bytes, its k16 row of x."""
    _lds(dst_w + d * cfg.wstage, wq, 4, 16)
    _lds(dst_s + d * cfg.sstage, sq, 2, 8)
    xr[0] = _smem_scalar(Int32, rd_x + d * cfg.xstage)[0]
    xr[1] = _smem_scalar(Int32, rd_x + (d * cfg.xstage + 16))[0]


@cute.kernel
def _diag_kernel(
    pX: cute.Pointer,  # [1, K] activations
    pW: cute.Pointer,  # [K/16, 2N] int32 packed weights
    pS: cute.Pointer,  # [K/16, N] scale bytes
    pS2: cute.Pointer,  # [1] global scale
    pC: cute.Pointer,  # [1, N] output
    pD: cute.Pointer,  # profiling stamps (cfg.prof only)
    prob_n: Int32,
    qn: Int32,  # slices owned by a class-B CTA (class A: one more), rn = number of class-A CTAs
    rn: Int32,
    nux: Int32,  # slices of the busiest class
    mg: Int32,  # 16-bit division magic for 1 / nux
    nst: Int32,  # k16-row classes (= streams per slice)
    rps: Int32,  # k16 rows every class has; classes < rem have one more
    rem: Int32,
    ng: Int32,  # unrolled refill groups of `depth` steps
    nxl: Int32,  # lanes of a warp that fetch x (two per k16-row class the warp touches)
    cfg: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, gidx, _ = cute.arch.block_idx()
    ts = cute.make_rmem_tensor((NSTAMP,), Int64)
    if cutlass.const_expr(cfg.prof):
        ts[0] = _clock()
        for i in cutlass.range_constexpr(1, NSTAMP - 1):
            ts[i] = ts[0]
        ts[NSTAMP - 1] = cute.arch.globaltimer()
    D = cfg.depth

    x_addr = pX.toint()
    w_addr = pW.toint()
    s_addr = pS.toint()
    s2_addr = pS2.toint()
    c_addr = pC.toint()
    s2_lane = tidx == 0
    if cutlass.const_expr(cfg.grouped):
        # GROUPED: grid row gidx is one group of a device-side table, so a launch whose
        # arguments are frozen (a captured CUDA graph) follows operands chosen on the GPU.
        # pW -> 64 B records [weight, scale, scale2, ...] (pointers), pS / pS2 -> the record
        # words holding the group's row of x / of the output. A zero weight pointer is an
        # idle group: with no classes nothing is fetched, computed or stored.
        rec = Int64(gidx) * 64
        w_addr = _gmem_scalar(Int64, pW.toint() + rec, 8)[0]
        s_addr = _gmem_scalar(Int64, pW.toint() + rec + 8, 8)[0]
        s2_addr = _gmem_scalar(Int64, pW.toint() + rec + 16, 8)[0]
        x_row = _gmem_scalar(Int64, pS.toint() + rec, 8)[0]
        c_row = _gmem_scalar(Int64, pS2.toint() + rec, 8)[0]
        on = w_addr != 0
        x_addr = x_addr + x_row * Int64((rps * nst + rem) * 32)
        c_addr = c_addr + c_row * Int64(prob_n * 2)
        nst = _sel(on, nst, 0)
        s2_lane = s2_lane & on

    # ---- ownership, and nothing else, before the first fetch ---------------------------
    # Stream tau = tidx >> 2 (g-slot gp of warp w: tau = 8 w + gp) is slice tau % nux of this
    # CTA at k16-row class tau // nux: it walks the rows {class + nst * j}. So at step j the
    # whole CTA (and the whole grid) reads the band of rows [nst * j, nst * (j + 1)): the
    # packed matrix is swept sequentially in K, and a warp only ever needs the few rows of x
    # of its own classes. Class-B CTAs own one slice less: their streams of slice nux - 1 idle.
    tau = tidx >> 2
    t = tidx & 3
    pr = (tau * mg) >> 16  # k16-row class
    ui = tau - pr * nux  # slice within the CTA
    is_a = bidx < rn
    u0 = bidx * qn + _sel(is_a, bidx, rn)
    act = (ui < qn + _sel(is_a, 1, 0)) & (pr < nst)
    slc = u0 + ui  # weights at slc * 64 + t * 16, scales at slc * 8 of a k16 row

    smem = cutlass.utils.SmemAllocator()
    blob = smem.allocate_tensor(
        Int32, cute.make_layout(cfg.smem_bytes // 4), byte_alignment=16
    )
    sbase = blob.iterator.toint()
    n8 = Uint32(prob_n * 8)
    n1 = Uint32(prob_n)

    # ---- prologue: D steps in flight; no barrier, a warp only reads what it fetched ----
    # Step 0 goes out first (this lane's int4 / scale bytes / x chunk; step j is nst * j rows
    # further). All fetch conditions are flat, single-statement `if`s: they compile to
    # predicated LDGSTS, so a warp never diverges between a fetch and the wait that its
    # sibling lanes (which read the scales and x other lanes fetched) rely on.
    dst_w = sbase + tidx * 16
    wb = w_addr + _wide(pr, n8) + Int64(slc * 64 + t * 16)
    if act:
        _fetch(dst_w, wb, 16, "cg")
    dst_s = sbase + cfg.s_off + (tidx >> 2) * 8
    sb = s_addr + _wide(pr, n1) + Int64(slc * 8)
    if act & (t == 0):
        _fetch(dst_s, sb, 8, "ca")
    lane = tidx & 31
    pw0 = (((tidx >> 5) << 3) * mg) >> 16  # class of the warp's first stream
    xcl = pw0 + (lane >> 1)  # class whose row of x this lane fetches a half of
    dst_x = sbase + cfg.x_off + (tidx >> 5) * 256 + lane * 16
    xb = x_addr + Int64(pw0 * 32 + lane * 16)
    if (lane < nxl) & (xcl < nst):
        _fetch(dst_x, xb, 16, "ca")
    cute.arch.cp_async_commit_group()
    if cutlass.const_expr(cfg.prof):
        ts[1] = _clock()

    # steps (k16 rows) of this lane's weight / scale / x fetches: 0 = never
    nrw = _sel(act, rps + _sel(pr < rem, 1, 0), 0)
    nrs = _sel(t == 0, nrw, 0)
    nrx = _sel((lane < nxl) & (xcl < nst), rps + _sel(xcl < rem, 1, 0), 0)
    live = pw0 < nst  # warp-uniform: the warp has work
    ext_w = pw0 < rem  # warp-uniform: some stream of the warp runs the extra step
    rd_x = sbase + cfg.x_off + (tidx >> 5) * 256 + (pr - pw0) * 32 + t * 4
    red = sbase + cfg.red_off
    s2sm = sbase + (cfg.smem_bytes - 16)

    wq = cute.make_rmem_tensor((4,), Int32)
    sq = cute.make_rmem_tensor((2,), Int32)
    xr = cute.make_rmem_tensor((2,), Int32)
    # [loop counter, k16-row offset of the next fetch]
    st = cute.make_rmem_tensor((2,), Int32)
    for d in cutlass.range_constexpr(1, D):
        if cutlass.const_expr(d == D - 1):
            # The last prologue step is only requested once step 0 has landed: measured
            # -0.06 .. -0.08 us (the SMs that enter the kernel last get their first data
            # sooner; holding back more steps than this one costs more than it gains).
            cute.arch.cp_async_wait_group(d - 1)
        if d < nrw:
            _fetch(dst_w + d * cfg.wstage, wb + _wide(nst * d, n8), 16, "cg")
        if d < nrs:
            _fetch(dst_s + d * cfg.sstage, sb + _wide(nst * d, n1), 8, "ca")
        if d < nrx:
            _fetch(dst_x + d * cfg.xstage, xb + _wide(nst * d, 32), 16, "ca")
        if cutlass.const_expr(d == D - 1):
            # The word holding scale2 (read in the epilogue) rides in the newest group, so no
            # step waits on it; the reduction barrier is preceded by a wait for every group.
            if s2_lane:
                _fetch(s2sm, s2_addr - (s2_addr % 4), 4, "ca")
        cute.arch.cp_async_commit_group()
    if cutlass.const_expr(cfg.prof):
        ts[2] = _clock()
        ts[3] = ts[2]

    # ---- the rest of the setup hides behind the first-data latency ---------------------
    s2sh = Int32(s2_addr % 4) * 8
    n_out = (qn + _sel(is_a, 1, 0)) * 8
    if cutlass.const_expr(cfg.grouped):
        n_out = _sel(on, n_out, 0)
    # The partial-sum rows are 4 * rch floats wide and summed whole: their pad words (classes
    # nst .. 4 * rch - 1, which no stream ever stores) must read as zeros. Only the pads are
    # written here -- nothing orders this against the stores of a warp that is already done.
    row_w = 4 * cfg.rch
    nthr = cfg.threads
    st[0] = tidx
    while st[0] < nux * 8:
        st[1] = nst
        while st[1] < row_w:
            pad = red + (st[0] * row_w + st[1]) * 4
            _smem_scalar(Float32, pad)[0] = Float32(0.0)
            st[1] = st[1] + 1
        st[0] = st[0] + nthr
    acc = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(16):
        acc[i] = Float32(0.0)
    st[1] = nst * D
    if cutlass.const_expr(cfg.prof):
        cute.arch.cp_async_wait_group(D - 1)
        ts[4] = _clock()

    # ---- main loop: one step = 8 (slice, k16 row) items = 4 MMAs -----------------------
    if cutlass.const_expr(cfg.short):
        # At most D steps: all of them were fetched above. A lane whose stream is over reads a
        # ring stage nothing was fetched into: it must add exact zeros, so its scales and its x
        # are zeroed (stale bytes can decode to Inf / NaN).
        for d in cutlass.range_constexpr(D):
            if live & (d < rps + _sel(rem > 0, 1, 0)):
                cute.arch.cp_async_wait_group(D - 1 - d)
                _diag_consume(cfg, d, dst_w, dst_s, rd_x, wq, sq, xr)
                for i in cutlass.range_constexpr(2):
                    sq[i] = _sel(d < nrw, sq[i], 0)
                    xr[i] = _sel(d < nrw, xr[i], 0)
                _compute_step(cfg.bf16, wq, sq, ((xr, acc),))
        if cutlass.const_expr(cfg.prof):
            ts[5] = _clock()
            ts[6] = ts[5]
    else:
        if live:
            # ng groups of D steps: consume ring stage d, refill it with the step D ahead
            st[0] = 0
            while st[0] < ng:
                for d in cutlass.range_constexpr(D):
                    cute.arch.cp_async_wait_group(D - 1)
                    _diag_consume(cfg, d, dst_w, dst_s, rd_x, wq, sq, xr)
                    if act:
                        _fetch(dst_w + d * cfg.wstage, wb + _wide(st[1], n8), 16, "cg")
                    if nrs > 0:
                        _fetch(dst_s + d * cfg.sstage, sb + _wide(st[1], n1), 8, "ca")
                    if nrx > 0:
                        _fetch(dst_x + d * cfg.xstage, xb + _wide(st[1], 32), 16, "ca")
                    st[1] = st[1] + nst
                    cute.arch.cp_async_commit_group()
                    _compute_step(cfg.bf16, wq, sq, ((xr, acc),))
                st[0] = st[0] + 1
            if cutlass.const_expr(cfg.prof):
                ts[5] = _clock()
            # The remaining steps, straight line: tm more full refills, then (warps holding one
            # of the first `rem` classes only) the refill of the extra step, then D steps with
            # nothing left to fetch.
            for i in cutlass.range_constexpr(cfg.tm):
                cute.arch.cp_async_wait_group(D - 1)
                _diag_consume(cfg, i % D, dst_w, dst_s, rd_x, wq, sq, xr)
                if act:
                    _fetch(
                        dst_w + (i % D) * cfg.wstage, wb + _wide(st[1], n8), 16, "cg"
                    )
                if nrs > 0:
                    _fetch(dst_s + (i % D) * cfg.sstage, sb + _wide(st[1], n1), 8, "ca")
                if nrx > 0:
                    _fetch(
                        dst_x + (i % D) * cfg.xstage, xb + _wide(st[1], 32), 16, "ca"
                    )
                st[1] = st[1] + nst
                cute.arch.cp_async_commit_group()
                _compute_step(cfg.bf16, wq, sq, ((xr, acc),))
            if cutlass.const_expr(cfg.ext):
                if ext_w:
                    d0 = cfg.tm % D
                    cute.arch.cp_async_wait_group(D - 1)
                    _diag_consume(cfg, d0, dst_w, dst_s, rd_x, wq, sq, xr)
                    if nrw > rps:
                        _fetch(dst_w + d0 * cfg.wstage, wb + _wide(st[1], n8), 16, "cg")
                    if nrs > rps:
                        _fetch(dst_s + d0 * cfg.sstage, sb + _wide(st[1], n1), 8, "ca")
                    if nrx > rps:
                        _fetch(dst_x + d0 * cfg.xstage, xb + _wide(st[1], 32), 16, "ca")
                    cute.arch.cp_async_commit_group()
                    _compute_step(cfg.bf16, wq, sq, ((xr, acc),))
                    for i in cutlass.range_constexpr(D):
                        cute.arch.cp_async_wait_group(D - 1 - i)
                        _diag_consume(
                            cfg, (cfg.tm + 1 + i) % D, dst_w, dst_s, rd_x, wq, sq, xr
                        )
                        if cutlass.const_expr(i == D - 1):
                            # the extra step: streams without it re-read an old stage -> add zeros
                            xr[0] = _sel(nrw > rps, xr[0], 0)
                            xr[1] = _sel(nrw > rps, xr[1], 0)
                        _compute_step(cfg.bf16, wq, sq, ((xr, acc),))
                else:
                    for i in cutlass.range_constexpr(D):
                        cute.arch.cp_async_wait_group(D - 1 - i)
                        _diag_consume(
                            cfg, (cfg.tm + i) % D, dst_w, dst_s, rd_x, wq, sq, xr
                        )
                        _compute_step(cfg.bf16, wq, sq, ((xr, acc),))
            else:
                for i in cutlass.range_constexpr(D):
                    cute.arch.cp_async_wait_group(D - 1 - i)
                    _diag_consume(cfg, (cfg.tm + i) % D, dst_w, dst_s, rd_x, wq, sq, xr)
                    _compute_step(cfg.bf16, wq, sq, ((xr, acc),))
            if cutlass.const_expr(cfg.prof):
                ts[6] = _clock()

    # ---- intra-CTA reduction over the K-split streams ----------------------------------
    # red[row = (2 * block + half) * nux + slice][class], rows of 4 * rch floats (pads stay
    # zero). C[g'][g'] and C[g' + 8][g'] of g-slot g' sit in lane t = g' / 2, elements g' % 2
    # and 2 + g' % 2. The eight lanes of a warp that store here hit eight different smem
    # banks per instruction: the row stride is an odd number of 16 B chunks, and a warp's
    # streams are consecutive (slice, class) pairs. (Slice-major rows put the ~5 streams of
    # one class on the same bank: 5 passes per store instead of 1.)
    rstride = cfg.rch * 16
    if t == (tidx >> 3) & 3:  # noqa: SIM102 -- DSL regions: keep the nesting
        if act:
            odd = (tidx & 4) != 0
            rb = red + ui * rstride + pr * 4
            rjump = nux * rstride
            for jh in cutlass.range_constexpr(8):
                v = _sel(odd, acc[2 * jh + 1], acc[2 * jh], Float32)
                _smem_scalar(Float32, rb + jh * rjump)[0] = v
    # Everything fetched in the prologue must have landed on EVERY path before the epilogue
    # reads it (scale2's word): an idle warp never waits for anything above.
    cute.arch.cp_async_wait_group(0)
    if cutlass.const_expr(cfg.prof):
        ts[7] = _clock()
    cute.arch.barrier()

    # ---- epilogue: sum the classes of each owned output, scale, store -------------------
    s2f = (_smem_scalar(Int32, s2sm)[0] >> s2sh) & 0xFFFF
    fs = cute.make_rmem_tensor((4,), Float32)
    fv = cute.make_rmem_tensor((4 * cfg.rch,), Float32)
    cbase = c_addr
    st[0] = tidx
    while st[0] < n_out:
        o = st[0]
        jh = o & 7
        rb = red + (jh * nux + (o >> 3)) * rstride
        for qd in cutlass.range_constexpr(cfg.rch):
            cute.autovec_copy(
                cute.make_tensor(
                    _smem_ptr(Float32, rb + 16 * qd, 16), cute.make_layout((4,))
                ),
                cute.make_tensor(fv.iterator + 4 * qd, cute.make_layout((4,))),
            )
        for i in cutlass.range_constexpr(4):  # fixed order: deterministic
            fs[i] = _tree_sum([fv[4 * qd + i] for qd in range(cfg.rch)])
        tot = (fs[0] + fs[1]) + (fs[2] + fs[3])
        slc_o = u0 + (o >> 3)
        col = (slc_o >> 3) * 64 + ((jh & 6) << 3) + (slc_o & 7) + (jh & 1) * 8
        _gmem_scalar(Int16, cbase + _wide(col, 2), 2)[0] = _finish(cfg.bf16, tot, s2f)
        st[0] = o + cfg.threads
    if cutlass.const_expr(cfg.prof):
        ts[NSTAMP - 2] = _clock()
        if tidx & 31 == 0:
            dbase = pD.toint() + Int64((bidx * cfg.warps + (tidx >> 5)) * (NSTAMP * 8))
            for i in cutlass.range_constexpr(NSTAMP):
                _gmem_scalar(Int64, dbase + i * 8, 8)[0] = ts[i]


@cute.jit
def _diag_launch(
    pX: cute.Pointer,
    pW: cute.Pointer,
    pS: cute.Pointer,
    pS2: cute.Pointer,
    pC: cute.Pointer,
    pD: cute.Pointer,
    prob_m: Int32,
    prob_k: Int32,
    prob_n: Int32,
    grid: Int32,
    qn: Int32,
    rn: Int32,
    nux: Int32,
    mg: Int32,
    nst: Int32,
    rps: Int32,
    rem: Int32,
    ng: Int32,
    nxl: Int32,
    cfg: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
):
    _diag_kernel(
        pX, pW, pS, pS2, pC, pD, prob_n, qn, rn, nux, mg, nst, rps, rem, ng, nxl, cfg
    ).launch(grid=(grid, 1, 1), block=(cfg.threads, 1, 1), stream=stream)


@cute.jit
def _diag_group_launch(
    pX: cute.Pointer,  # [rows, K] activations
    pT: cute.Pointer,  # [groups] 64 B records: weight, scale, scale2 pointers
    pXR: cute.Pointer,  # the word of record 0 holding its row of x
    pCR: cute.Pointer,  # the word of record 0 holding its row of the output
    pC: cute.Pointer,  # [rows, N] output
    groups: Int32,
    prob_n: Int32,
    grid: Int32,
    qn: Int32,
    rn: Int32,
    nux: Int32,
    mg: Int32,
    nst: Int32,
    rps: Int32,
    rem: Int32,
    ng: Int32,
    nxl: Int32,
    cfg: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
):
    _diag_kernel(
        pX, pT, pXR, pCR, pC, pC, prob_n, qn, rn, nux, mg, nst, rps, rem, ng, nxl, cfg
    ).launch(grid=(grid, groups, 1), block=(cfg.threads, 1, 1), stream=stream)


# ---------------------------------------------------------------------------
# QUAD flavor: 2 <= M <= 8, two k16-row classes per warp
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _QuadCfg:
    bf16: bool
    nq: int  # quads of x rows per column block: 1 (M <= 4) or 2 (M <= 8) = MMAs per word
    warps: int
    depth: int  # cp.async ring depth in steps
    rch: int  # 16 B chunks per row of the partial-sum buffer (>= classes / 4)
    rbytes: int  # smem for the partial sums
    short: bool  # no more steps than the ring is deep: everything is fetched in the prologue
    tm: int  # full refill steps left over after the unrolled groups of `depth`
    ext: bool  # K/16 is not a multiple of the class count: the first classes run one more step
    prof: bool = False
    band: bool = (
        False  # x is fetched once per CTA (BAND, see _quad_kernel), not once per warp
    )

    @property
    def mpad(self):
        return 4 * self.nq

    @property
    def threads(self):
        return 32 * self.warps

    # smem: as DIAG, with an x ring of [quad][half of the k16 row][column block][m % 4][16 B]
    # per warp and stage: lane l fetches chunk l of it, and lane (g', t) reads its B fragment
    # of quad q at q * 256 + (0 | 128) + l * 4 -- 32 lanes on 32 different smem banks.
    @property
    def wstage(self):
        return self.threads * 16

    @property
    def sstage(self):
        return self.threads * 2

    @property
    def xwarp(self):
        return 256 * self.nq

    @property
    def xstage(self):
        return self.warps * self.xwarp

    @property
    def s_off(self):
        return self.depth * self.wstage

    @property
    def x_off(self):
        return self.s_off + self.depth * self.sstage

    # BAND: two buffers, each holding the columns of x that depth (+ 1) steps read: 4 * nq rows
    # of up to (depth + 1) * 4 * rch k16 rows (32 B each), + 256 B per row for the
    # bank-spreading pads and the reads of idle blocks.
    @property
    def xwpr(self):  # warps that fetch one row of x: a power of two
        return 1 << ((self.warps // self.mpad).bit_length() - 1)

    @property
    def xrounds(
        self,
    ):  # fetch instructions per group: each warp moves 512 B per instruction
        return -(-((self.depth + 1) * 8 * self.rch) // (32 * self.xwpr))

    @property
    def xbpitch(self):
        return (self.depth + 1) * 8 * self.rch * 16 + 256

    @property
    def xbuf(self):
        return self.mpad * self.xbpitch

    @property
    def red_off(self):
        return self.x_off + (2 * self.xbuf if self.band else self.depth * self.xstage)

    @property
    def smem_bytes(self):  # (+16: the scale2 word)
        return self.red_off + self.rbytes + 16


def _band_tail(depth: int, tm: int, ext: bool):
    """BAND: trace-time schedule of the steps after the unrolled refill groups.

    Returns (common, ext branch, other branch, absorb): lists of steps
    (kind, ring slot, pending, starts a group of x, fetches the next group's x).
    kind is "refill" (consume + refill), "extrefill" (the refill of the extra step) or
    "final" (consume only); `pending` is how many of the newest cp.async groups may still be
    in flight when the step starts. Groups of x start at the steps that are 1 mod depth --
    except at the very last step: a group of one step is not worth a barrier, so the group
    before it fetches one step more (`absorb`). The x of a group always rides in the
    cp.async group of a refill, and both kinds of warps start the same groups.
    """
    total = tm + (1 if ext else 0) + depth  # steps of a warp that runs the extra step
    absorb = (total - 1) % depth == 1  # i.e. tm + ext == 2

    def build(has_ext):
        seq = [("refill", i % depth) for i in range(tm)]
        if has_ext:
            seq.append(("extrefill", tm % depth))
        base = len(seq)
        seq += [("final", (base + k) % depth, depth - 1 - k) for k in range(depth)]
        steps = []
        for i, item in enumerate(seq):
            gstart = i % depth == 1 and not (absorb and i == total - 1)
            xfetch = (
                gstart and i + depth < total and not (absorb and i + depth == total - 1)
            )
            assert not xfetch or item[0] == "refill"
            steps.append(
                (
                    item[0],
                    item[1],
                    item[2] if len(item) > 2 else depth - 1,
                    gstart,
                    xfetch,
                )
            )
        return steps

    with_ext = build(ext)
    without = build(False)
    assert with_ext[:tm] == without[:tm]
    assert [x[3] for x in with_ext[tm:]][: len(without) - tm] == [
        x[3] for x in without[tm:]
    ]
    assert not any(x[3] for x in with_ext[len(without) :])
    return with_ext[:tm], with_ext[tm:], without[tm:], absorb


def _band_x(cfg, xp, xinc, xrs):
    """BAND: this lane's B fragments at the k16 row its block is at; xp walks the group's buffer."""
    for q in range(cfg.nq):  # quad q is four rows of x further
        xrs[q][0] = _smem_scalar(Int32, xp[0] + q * 4 * cfg.xbpitch)[0]
        xrs[q][1] = _smem_scalar(Int32, xp[0] + (q * 4 * cfg.xbpitch + 16))[0]
    xp[0] = xp[0] + xinc


def _quad_consume(cfg, d, dst_w, dst_s, rd_x, wq, sq, xrs):
    """Ring stage d -> registers: int4 of weights, 8 scale bytes, one B fragment per quad of x."""
    _lds(dst_w + d * cfg.wstage, wq, 4, 16)
    _lds(dst_s + d * cfg.sstage, sq, 2, 8)
    for q in range(cfg.nq):
        xrs[q][0] = _smem_scalar(Int32, rd_x + (d * cfg.xstage + q * 256))[0]
        xrs[q][1] = _smem_scalar(Int32, rd_x + (d * cfg.xstage + q * 256 + 128))[0]


@cute.kernel
def _quad_kernel(
    pX: cute.Pointer,  # [M, K] activations
    pW: cute.Pointer,  # [K/16, 2N] int32 packed weights
    pS: cute.Pointer,  # [K/16, N] scale bytes
    pS2: cute.Pointer,  # [1] global scale
    pC: cute.Pointer,  # [M, N] output
    pD: cute.Pointer,  # profiling stamps (cfg.prof only)
    prob_m: Int32,
    prob_k: Int32,
    prob_n: Int32,
    qn: Int32,  # slices owned by a class-B CTA (class A: one more), rn = number of class-A CTAs
    rn: Int32,
    sdiv: Int32,  # g-slots per k16-row class (>= slices of the busiest CTA class)
    mg: Int32,  # 16-bit division magic for 1 / sdiv
    pmul: Int32,  # 1: every 16th g-slot idles (5 slices: classes of 5 + 3 | 2 + 5 slots per warp pair)
    nst: Int32,  # k16-row classes
    rps: Int32,  # k16 rows every class has; classes < rem have one more
    rem: Int32,
    ng: Int32,  # unrolled refill groups of `depth` steps
    cfg: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    ts = cute.make_rmem_tensor((NSTAMP,), Int64)
    if cutlass.const_expr(cfg.prof):
        ts[0] = _clock()
        for i in cutlass.range_constexpr(1, NSTAMP - 1):
            ts[i] = ts[0]
        ts[NSTAMP - 1] = cute.arch.globaltimer()
    D = cfg.depth
    NQ = cfg.nq
    BAND = cfg.band
    TAIL = _band_tail(D, cfg.tm, cfg.ext)
    MP = cfg.mpad
    lmp = MP.bit_length() - 1

    # ---- ownership, and nothing else, before the first fetch ---------------------------
    # The MMA's eight B columns are two column blocks of four: block cb carries rows
    # m = 0..3 of x (and, as a second MMA, rows 4..7) at the k16 row of ONE class. So a warp
    # serves two classes, pw0 and pw0 + 1, and each of its g-slots (A rows g', g' + 8) is one
    # slice at one of the two. G-slot tau = tidx >> 2 of the CTA is slice te % sdiv at class
    # te // sdiv, with te = tau minus the idle slots before it: consecutive g-slots fill a
    # class slice by slice, which never puts more than two classes in a warp for sdiv = 4
    # (<= 4 slices: 4 + 4), 5 (with every 16th slot idle: 5 + 3 | 2 + 5 + idle) and >= 6.
    # Lane (g', t) therefore has two unrelated roles: its A registers are the weights of its
    # g-slot's (slice, class); its B registers are row g' & 3 of x at the class of block g' >> 2.
    tau = tidx >> 2
    t = tidx & 3
    te = tau - (tau >> 4) * pmul
    pr = (te * mg) >> 16  # k16-row class of this g-slot
    ui = te - pr * sdiv  # slice within the CTA
    is_a = bidx < rn
    u0 = bidx * qn + _sel(is_a, bidx, rn)
    act = (ui < qn + _sel(is_a, 1, 0)) & (pr < nst) & ((pmul == 0) | ((tau & 15) != 15))
    slc = u0 + ui  # weights at slc * 64 + t * 16, scales at slc * 8 of a k16 row

    smem = cutlass.utils.SmemAllocator()
    blob = smem.allocate_tensor(
        Int32, cute.make_layout(cfg.smem_bytes // 4), byte_alignment=16
    )
    sbase = blob.iterator.toint()
    n8 = Uint32(prob_n * 8)
    n1 = Uint32(prob_n)

    # ---- prologue: D steps in flight; no barrier, a warp only reads what it fetched ----
    dst_w = sbase + tidx * 16
    wb = pW.toint() + _wide(pr, n8) + Int64(slc * 64 + t * 16)
    if act:
        _fetch(dst_w, wb, 16, "cg")
    dst_s = sbase + cfg.s_off + (tidx >> 2) * 8
    sb = pS.toint() + _wide(pr, n1) + Int64(slc * 8)
    if act & (t == 0):
        _fetch(dst_s, sb, 8, "ca")
    # x: lane l < 16 * NQ fetches half (l >> 3) & 1 of row 4 * (l >> 4) + (l & 3) of x at the
    # class of block (l >> 2) & 1
    lane = tidx & 31
    tw = (tidx >> 5) << 3
    pw0 = ((tw - (tw >> 4) * pmul) * mg) >> 16  # class of the warp's first g-slot
    xm = ((lane >> 4) << 2) + (lane & 3)
    xcl = pw0 + ((lane >> 2) & 1)
    xok = (lane < 16 * NQ) & (xm < prob_m) & (xcl < nst)
    dst_x = sbase + cfg.x_off + (tidx >> 5) * cfg.xwarp + lane * 16
    xb = (
        pX.toint()
        + _wide(xm, Uint32(prob_k * 2))
        + Int64(xcl * 32 + ((lane >> 3) & 1) * 16)
    )
    # BAND. All CTAs read the same rows of x at the same time, and what that costs is the
    # number of requests per 128 B line, not the bytes. So x is not fetched by the warp that
    # needs it (two classes = half a line per row and step, and classes shared by two warps
    # twice) but once per CTA, in whole lines, for a group of D steps at a time: warp w
    # fetches 512 B pieces of row w / WPR of x into one of two buffers. Lanes read what other
    # warps fetched, so each group of D steps starts with a barrier, placed after the wait for
    # the cp.async group that carries the group's x (R1: it has landed for every fetcher) and
    # before the next group's x is requested into the other buffer (R2: nobody still reads
    # what that buffer held). Groups are steps 1 .. D, D + 1 .. 2 D, ...; step 0 is a group
    # of its own, so that the first data only waits for one step's worth of x. A group's x is
    # requested after the barrier of the group before it, in the cp.async group of that step's
    # refill -- which is requested BEFORE the barrier, so that waiting for the other warps
    # never delays the weight stream, only this warp's arithmetic.
    krow = prob_k * 2  # bytes per row of x
    xs = sbase + cfg.x_off
    xrow = (tidx >> 5) >> (cfg.xwpr.bit_length() - 1)
    xpos0 = (
        (tidx >> 5) & (cfg.xwpr - 1)
    ) * 32 + lane  # 16 B chunk of the group's columns
    xf_ok = xrow < prob_m
    xsrc = pX.toint() + _wide(xrow, Uint32(krow)) + Int64(xpos0 * 16)
    xdst = xs + xrow * cfg.xbpitch + ((xrow & 1) * 16 + (xrow & 2) * 32) + xpos0 * 16
    cpr = nst * (2 * D)  # 16 B chunks of a row of x per group
    # ... and what the last full group fetches on top (_band_tail)
    cpr1 = nst * 2 if TAIL[3] else 0
    if cutlass.const_expr(BAND):
        # step 0's columns only (buffer 0): the less of x the first data waits for, the better
        for r in cutlass.range_constexpr(-(-(8 * cfg.rch) // (32 * cfg.xwpr))):
            if xf_ok & (xpos0 + r * 32 * cfg.xwpr < nst * 2):
                _fetch(
                    xdst + r * 512 * cfg.xwpr,
                    xsrc + Int64(r * 512 * cfg.xwpr),
                    16,
                    "ca",
                )
    else:
        if xok:
            _fetch(dst_x, xb, 16, "ca")
    cute.arch.cp_async_commit_group()
    if cutlass.const_expr(cfg.prof):
        ts[1] = _clock()

    # steps (k16 rows) of this lane's weight / scale / x fetches, and of the class whose x its
    # B registers carry: 0 = never
    nrw = _sel(act, rps + _sel(pr < rem, 1, 0), 0)
    nrs = _sel(t == 0, nrw, 0)
    nrx = _sel(xok, rps + _sel(xcl < rem, 1, 0), 0)
    bcl = pw0 + (lane >> 4)
    nrb = _sel(bcl < nst, rps + _sel(bcl < rem, 1, 0), 0)
    live = pw0 < nst  # warp-uniform: the warp has work
    ext_w = pw0 < rem  # warp-uniform: a class of the warp runs the extra step
    rd_x = sbase + cfg.x_off + (tidx >> 5) * cfg.xwarp + lane * 4
    red = sbase + cfg.red_off
    s2sm = sbase + (cfg.smem_bytes - 16)

    wq = cute.make_rmem_tensor((4,), Int32)
    sq = cute.make_rmem_tensor((2,), Int32)
    # BAND: [address of this lane's x at the current step, buffer]
    xp = cute.make_rmem_tensor((2,), Int32)
    xr0 = cute.make_rmem_tensor((2,), Int32)
    xr1 = cute.make_rmem_tensor((2,), Int32)
    xrs = (xr0, xr1)[:NQ]
    # [loop counter, k16-row offset of the next fetch]
    st = cute.make_rmem_tensor((2,), Int32)
    for d in cutlass.range_constexpr(1, D):
        if d < nrw:
            _fetch(dst_w + d * cfg.wstage, wb + _wide(nst * d, n8), 16, "cg")
        if d < nrs:
            _fetch(dst_s + d * cfg.sstage, sb + _wide(nst * d, n1), 8, "ca")
        if cutlass.const_expr(not BAND):  # noqa: SIM102 -- DSL regions: keep the nesting
            if d < nrx:
                _fetch(dst_x + d * cfg.xstage, xb + _wide(nst * d, 32), 16, "ca")
        if cutlass.const_expr(BAND and d == 1):
            # BAND: the x of steps 1 .. D (buffer 1) rides with the weights of step 1
            for r in cutlass.range_constexpr(cfg.xrounds):
                if (
                    xf_ok
                    & (xpos0 + r * 32 * cfg.xwpr < cpr + _sel(ng == 0, cpr1, 0))
                    & (nst * 32 + (xpos0 + r * 32 * cfg.xwpr) * 16 < krow)
                ):
                    _fetch(
                        xdst + cfg.xbuf + r * 512 * cfg.xwpr,
                        xsrc + _wide(nst, 32) + Int64(r * 512 * cfg.xwpr),
                        16,
                        "ca",
                    )
        if cutlass.const_expr(d == D - 1):
            # The word holding scale2 (read in the epilogue) rides in the newest group, so no
            # step waits on it; the reduction barrier is preceded by a wait for every group.
            s2addr = pS2.toint()
            if tidx == 0:
                _fetch(s2sm, s2addr - (s2addr % 4), 4, "ca")
        cute.arch.cp_async_commit_group()
    if cutlass.const_expr(cfg.prof):
        ts[2] = _clock()
        ts[3] = ts[2]

    # ---- the rest of the setup hides behind the first-data latency ---------------------
    s2sh = Int32(pS2.toint() % 4) * 8
    n_out = (qn + _sel(is_a, 1, 0)) * (8 * MP)
    for i in cutlass.range_constexpr(4):
        wq[i] = 0
    # Rows M .. MP - 1 of x do not exist: their ring slots (never fetched into, read only by
    # this warp) must hold zeros.
    if cutlass.const_expr(not BAND):  # noqa: SIM102 -- DSL regions: keep the nesting
        if (lane < 16 * NQ) & (xm >= prob_m):
            for d in cutlass.range_constexpr(D):
                cute.autovec_copy(
                    wq,
                    cute.make_tensor(
                        _smem_ptr(Int32, dst_x + d * cfg.xstage, 16),
                        cute.make_layout((4,)),
                    ),
                )
    # BAND: lane (g', t) carries row g' & 3 (+ 4 per quad) of x at the class of block g' >> 2.
    # Rows >= M are never fetched: what is read there only reaches MMA columns nobody stores.
    # The row pads (0 / 16 / 64 / 80 B) put the 32 lanes of a warp on 32 different smem banks.
    xm0 = (lane >> 2) & 3
    xcons = (
        xs + xm0 * cfg.xbpitch + ((xm0 & 1) * 16 + (xm0 & 2) * 32) + bcl * 32 + t * 4
    )
    xinc = nst * 32
    xp[0] = xcons
    xp[1] = 0
    # The partial-sum rows are 4 * rch floats wide and summed whole: their pad words (classes
    # nst .. 4 * rch - 1, which no g-slot ever stores) must read as zeros. Only the pads are
    # written here -- nothing orders this against the stores of a warp that is already done.
    row_w = 4 * cfg.rch
    nthr = cfg.threads
    st[0] = tidx
    while st[0] < sdiv * (8 * MP):
        st[1] = nst
        while st[1] < row_w:
            pad = red + (st[0] * row_w + st[1]) * 4
            _smem_scalar(Float32, pad)[0] = Float32(0.0)
            st[1] = st[1] + 1
        st[0] = st[0] + nthr
    acc0 = cute.make_rmem_tensor((16,), Float32)
    acc1 = cute.make_rmem_tensor((16,), Float32)
    accs = (acc0, acc1)[:NQ]
    for i in cutlass.range_constexpr(16):
        for q in cutlass.range_constexpr(NQ):
            accs[q][i] = Float32(0.0)
    pairs = tuple(zip(xrs, accs))
    st[1] = nst * D
    if cutlass.const_expr(cfg.prof):
        cute.arch.cp_async_wait_group(D - 1)
        ts[4] = _clock()

    # ---- main loop: one step = 8 (slice, k16 row) items = 4 * NQ MMAs ------------------
    if cutlass.const_expr(BAND):
        # step 0's x, fetched by other warps, has landed for all of them
        cute.arch.cp_async_wait_group(D - 1)
        cute.arch.barrier()
        if cutlass.const_expr(cfg.short):
            # At most D steps, all fetched above. See the ring version below.
            for d in cutlass.range_constexpr(D):
                if cutlass.const_expr(d == 1):
                    cute.arch.cp_async_wait_group(D - 2)
                    cute.arch.barrier()
                    xp[1] = cfg.xbuf - xp[1]
                    xp[0] = xcons + xp[1]
                if live & (d < rps + _sel(rem > 0, 1, 0)):
                    cute.arch.cp_async_wait_group(D - 1 - d)
                    _lds(dst_w + d * cfg.wstage, wq, 4, 16)
                    _lds(dst_s + d * cfg.sstage, sq, 2, 8)
                    _band_x(cfg, xp, xinc, xrs)
                    for i in cutlass.range_constexpr(2):
                        sq[i] = _sel(d < nrw, sq[i], 0)
                        for q in cutlass.range_constexpr(NQ):
                            xrs[q][i] = _sel(d < nrb, xrs[q][i], 0)
                    _compute_step(cfg.bf16, wq, sq, pairs)
            if cutlass.const_expr(cfg.prof):
                ts[5] = _clock()
                ts[6] = ts[5]
        else:
            # Every warp walks the whole schedule (an idle one computes on garbage that is never
            # stored): all of them have to meet at the group barriers, and fetch x.
            st[0] = 0
            while st[0] < ng:
                for d in cutlass.range_constexpr(D):
                    cute.arch.cp_async_wait_group(D - 1)
                    _lds(dst_w + d * cfg.wstage, wq, 4, 16)
                    _lds(dst_s + d * cfg.sstage, sq, 2, 8)
                    if act:
                        _fetch(dst_w + d * cfg.wstage, wb + _wide(st[1], n8), 16, "cg")
                    if nrs > 0:
                        _fetch(dst_s + d * cfg.sstage, sb + _wide(st[1], n1), 8, "ca")
                    if cutlass.const_expr(d == 1):
                        cute.arch.barrier()
                        xp[1] = cfg.xbuf - xp[1]
                        xp[0] = xcons + xp[1]
                        # the next group's x (its first k16 row is st[1]) into the other buffer
                        for r in cutlass.range_constexpr(cfg.xrounds):
                            if (
                                xf_ok
                                & (
                                    xpos0 + r * 32 * cfg.xwpr
                                    < cpr + _sel(st[0] == ng - 1, cpr1, 0)
                                )
                                & (st[1] * 32 + (xpos0 + r * 32 * cfg.xwpr) * 16 < krow)
                            ):
                                _fetch(
                                    xdst + (cfg.xbuf - xp[1]) + r * 512 * cfg.xwpr,
                                    xsrc + _wide(st[1], 32) + Int64(r * 512 * cfg.xwpr),
                                    16,
                                    "ca",
                                )
                    st[1] = st[1] + nst
                    cute.arch.cp_async_commit_group()
                    _band_x(cfg, xp, xinc, xrs)
                    _compute_step(cfg.bf16, wq, sq, pairs)
                st[0] = st[0] + 1
            if cutlass.const_expr(cfg.prof):
                ts[5] = _clock()
            # The remaining steps, straight line (schedule: _band_tail): tm more refills, then
            # (warps holding one of the first `rem` classes only) the refill of the extra step,
            # then D steps with nothing left to fetch.
            for ix in cutlass.range_constexpr(len(TAIL[0])):
                cute.arch.cp_async_wait_group(TAIL[0][ix][2])
                _lds(dst_w + TAIL[0][ix][1] * cfg.wstage, wq, 4, 16)
                _lds(dst_s + TAIL[0][ix][1] * cfg.sstage, sq, 2, 8)
                if act:
                    _fetch(
                        dst_w + TAIL[0][ix][1] * cfg.wstage,
                        wb + _wide(st[1], n8),
                        16,
                        "cg",
                    )
                if nrs > 0:
                    _fetch(
                        dst_s + TAIL[0][ix][1] * cfg.sstage,
                        sb + _wide(st[1], n1),
                        8,
                        "ca",
                    )
                if cutlass.const_expr(TAIL[0][ix][3]):
                    cute.arch.barrier()
                    xp[1] = cfg.xbuf - xp[1]
                    xp[0] = xcons + xp[1]
                    if cutlass.const_expr(TAIL[0][ix][4]):
                        # the next group's x (its first k16 row is st[1]) into the other buffer
                        for r in cutlass.range_constexpr(cfg.xrounds):
                            if (
                                xf_ok
                                & (xpos0 + r * 32 * cfg.xwpr < cpr)
                                & (st[1] * 32 + (xpos0 + r * 32 * cfg.xwpr) * 16 < krow)
                            ):
                                _fetch(
                                    xdst + (cfg.xbuf - xp[1]) + r * 512 * cfg.xwpr,
                                    xsrc + _wide(st[1], 32) + Int64(r * 512 * cfg.xwpr),
                                    16,
                                    "ca",
                                )
                st[1] = st[1] + nst
                cute.arch.cp_async_commit_group()
                _band_x(cfg, xp, xinc, xrs)
                _compute_step(cfg.bf16, wq, sq, pairs)
            if cutlass.const_expr(cfg.ext):
                if ext_w:
                    for ix in cutlass.range_constexpr(len(TAIL[1])):
                        cute.arch.cp_async_wait_group(TAIL[1][ix][2])
                        _lds(dst_w + TAIL[1][ix][1] * cfg.wstage, wq, 4, 16)
                        _lds(dst_s + TAIL[1][ix][1] * cfg.sstage, sq, 2, 8)
                        if cutlass.const_expr(TAIL[1][ix][0] == "extrefill"):
                            if nrw > rps:
                                _fetch(
                                    dst_w + TAIL[1][ix][1] * cfg.wstage,
                                    wb + _wide(st[1], n8),
                                    16,
                                    "cg",
                                )
                            if nrs > rps:
                                _fetch(
                                    dst_s + TAIL[1][ix][1] * cfg.sstage,
                                    sb + _wide(st[1], n1),
                                    8,
                                    "ca",
                                )
                        if cutlass.const_expr(TAIL[1][ix][3]):
                            cute.arch.barrier()
                            xp[1] = cfg.xbuf - xp[1]
                            xp[0] = xcons + xp[1]
                            if cutlass.const_expr(TAIL[1][ix][4]):
                                # the next group's x (its first k16 row is st[1]) into the other buffer
                                for r in cutlass.range_constexpr(cfg.xrounds):
                                    if (
                                        xf_ok
                                        & (xpos0 + r * 32 * cfg.xwpr < cpr)
                                        & (
                                            st[1] * 32
                                            + (xpos0 + r * 32 * cfg.xwpr) * 16
                                            < krow
                                        )
                                    ):
                                        _fetch(
                                            xdst
                                            + (cfg.xbuf - xp[1])
                                            + r * 512 * cfg.xwpr,
                                            xsrc
                                            + _wide(st[1], 32)
                                            + Int64(r * 512 * cfg.xwpr),
                                            16,
                                            "ca",
                                        )
                        if cutlass.const_expr(TAIL[1][ix][0] == "extrefill"):
                            st[1] = st[1] + nst
                            cute.arch.cp_async_commit_group()
                        _band_x(cfg, xp, xinc, xrs)
                        if cutlass.const_expr(ix == len(TAIL[1]) - 1):
                            # The extra step: g-slots of a class without it re-read an old stage, and only
                            # ever use that class's block of x -> zero it.
                            for q in cutlass.range_constexpr(NQ):
                                xrs[q][0] = _sel(nrb > rps, xrs[q][0], 0)
                                xrs[q][1] = _sel(nrb > rps, xrs[q][1], 0)
                        _compute_step(cfg.bf16, wq, sq, pairs)
                else:
                    for ix in cutlass.range_constexpr(len(TAIL[2])):
                        cute.arch.cp_async_wait_group(TAIL[2][ix][2])
                        _lds(dst_w + TAIL[2][ix][1] * cfg.wstage, wq, 4, 16)
                        _lds(dst_s + TAIL[2][ix][1] * cfg.sstage, sq, 2, 8)
                        if cutlass.const_expr(TAIL[2][ix][3]):
                            cute.arch.barrier()
                            xp[1] = cfg.xbuf - xp[1]
                            xp[0] = xcons + xp[1]
                            if cutlass.const_expr(TAIL[2][ix][4]):
                                # the next group's x (its first k16 row is st[1]) into the other buffer
                                for r in cutlass.range_constexpr(cfg.xrounds):
                                    if (
                                        xf_ok
                                        & (xpos0 + r * 32 * cfg.xwpr < cpr)
                                        & (
                                            st[1] * 32
                                            + (xpos0 + r * 32 * cfg.xwpr) * 16
                                            < krow
                                        )
                                    ):
                                        _fetch(
                                            xdst
                                            + (cfg.xbuf - xp[1])
                                            + r * 512 * cfg.xwpr,
                                            xsrc
                                            + _wide(st[1], 32)
                                            + Int64(r * 512 * cfg.xwpr),
                                            16,
                                            "ca",
                                        )
                        _band_x(cfg, xp, xinc, xrs)
                        _compute_step(cfg.bf16, wq, sq, pairs)
            else:
                for ix in cutlass.range_constexpr(len(TAIL[2])):
                    cute.arch.cp_async_wait_group(TAIL[2][ix][2])
                    _lds(dst_w + TAIL[2][ix][1] * cfg.wstage, wq, 4, 16)
                    _lds(dst_s + TAIL[2][ix][1] * cfg.sstage, sq, 2, 8)
                    if cutlass.const_expr(TAIL[2][ix][3]):
                        cute.arch.barrier()
                        xp[1] = cfg.xbuf - xp[1]
                        xp[0] = xcons + xp[1]
                        if cutlass.const_expr(TAIL[2][ix][4]):
                            # the next group's x (its first k16 row is st[1]) into the other buffer
                            for r in cutlass.range_constexpr(cfg.xrounds):
                                if (
                                    xf_ok
                                    & (xpos0 + r * 32 * cfg.xwpr < cpr)
                                    & (
                                        st[1] * 32 + (xpos0 + r * 32 * cfg.xwpr) * 16
                                        < krow
                                    )
                                ):
                                    _fetch(
                                        xdst + (cfg.xbuf - xp[1]) + r * 512 * cfg.xwpr,
                                        xsrc
                                        + _wide(st[1], 32)
                                        + Int64(r * 512 * cfg.xwpr),
                                        16,
                                        "ca",
                                    )
                    _band_x(cfg, xp, xinc, xrs)
                    _compute_step(cfg.bf16, wq, sq, pairs)
            if cutlass.const_expr(cfg.prof):
                ts[6] = _clock()
    else:
        if cutlass.const_expr(cfg.short):
            # At most D steps: all of them were fetched above. A g-slot whose stream is over reads
            # a ring stage nothing was fetched into: it must add exact zeros, so its scales are
            # zeroed, and so is the x of a block whose class is over (stale bytes can be Inf / NaN).
            for d in cutlass.range_constexpr(D):
                if live & (d < rps + _sel(rem > 0, 1, 0)):
                    cute.arch.cp_async_wait_group(D - 1 - d)
                    _quad_consume(cfg, d, dst_w, dst_s, rd_x, wq, sq, xrs)
                    for i in cutlass.range_constexpr(2):
                        sq[i] = _sel(d < nrw, sq[i], 0)
                        for q in cutlass.range_constexpr(NQ):
                            xrs[q][i] = _sel(d < nrb, xrs[q][i], 0)
                    _compute_step(cfg.bf16, wq, sq, pairs)
            if cutlass.const_expr(cfg.prof):
                ts[5] = _clock()
                ts[6] = ts[5]
        else:
            if live:
                # ng groups of D steps: consume ring stage d, refill it with the step D ahead
                st[0] = 0
                while st[0] < ng:
                    for d in cutlass.range_constexpr(D):
                        cute.arch.cp_async_wait_group(D - 1)
                        _quad_consume(cfg, d, dst_w, dst_s, rd_x, wq, sq, xrs)
                        if act:
                            _fetch(
                                dst_w + d * cfg.wstage, wb + _wide(st[1], n8), 16, "cg"
                            )
                        if nrs > 0:
                            _fetch(
                                dst_s + d * cfg.sstage, sb + _wide(st[1], n1), 8, "ca"
                            )
                        if nrx > 0:
                            _fetch(
                                dst_x + d * cfg.xstage, xb + _wide(st[1], 32), 16, "ca"
                            )
                        st[1] = st[1] + nst
                        cute.arch.cp_async_commit_group()
                        _compute_step(cfg.bf16, wq, sq, pairs)
                    st[0] = st[0] + 1
                if cutlass.const_expr(cfg.prof):
                    ts[5] = _clock()
                # The remaining steps, straight line: tm more full refills, then (warps holding one
                # of the first `rem` classes only) the refill of the extra step, then D steps with
                # nothing left to fetch.
                for i in cutlass.range_constexpr(cfg.tm):
                    cute.arch.cp_async_wait_group(D - 1)
                    _quad_consume(cfg, i % D, dst_w, dst_s, rd_x, wq, sq, xrs)
                    if act:
                        _fetch(
                            dst_w + (i % D) * cfg.wstage,
                            wb + _wide(st[1], n8),
                            16,
                            "cg",
                        )
                    if nrs > 0:
                        _fetch(
                            dst_s + (i % D) * cfg.sstage, sb + _wide(st[1], n1), 8, "ca"
                        )
                    if nrx > 0:
                        _fetch(
                            dst_x + (i % D) * cfg.xstage,
                            xb + _wide(st[1], 32),
                            16,
                            "ca",
                        )
                    st[1] = st[1] + nst
                    cute.arch.cp_async_commit_group()
                    _compute_step(cfg.bf16, wq, sq, pairs)
                if cutlass.const_expr(cfg.ext):
                    if ext_w:
                        d0 = cfg.tm % D
                        cute.arch.cp_async_wait_group(D - 1)
                        _quad_consume(cfg, d0, dst_w, dst_s, rd_x, wq, sq, xrs)
                        if nrw > rps:
                            _fetch(
                                dst_w + d0 * cfg.wstage, wb + _wide(st[1], n8), 16, "cg"
                            )
                        if nrs > rps:
                            _fetch(
                                dst_s + d0 * cfg.sstage, sb + _wide(st[1], n1), 8, "ca"
                            )
                        if nrx > rps:
                            _fetch(
                                dst_x + d0 * cfg.xstage, xb + _wide(st[1], 32), 16, "ca"
                            )
                        cute.arch.cp_async_commit_group()
                        _compute_step(cfg.bf16, wq, sq, pairs)
                        for i in cutlass.range_constexpr(D):
                            cute.arch.cp_async_wait_group(D - 1 - i)
                            _quad_consume(
                                cfg,
                                (cfg.tm + 1 + i) % D,
                                dst_w,
                                dst_s,
                                rd_x,
                                wq,
                                sq,
                                xrs,
                            )
                            if cutlass.const_expr(i == D - 1):
                                # The extra step: g-slots of a class without it re-read an old
                                # stage, and only ever use that class's block of x -> zero it.
                                for q in cutlass.range_constexpr(NQ):
                                    xrs[q][0] = _sel(nrb > rps, xrs[q][0], 0)
                                    xrs[q][1] = _sel(nrb > rps, xrs[q][1], 0)
                            _compute_step(cfg.bf16, wq, sq, pairs)
                    else:
                        for i in cutlass.range_constexpr(D):
                            cute.arch.cp_async_wait_group(D - 1 - i)
                            _quad_consume(
                                cfg, (cfg.tm + i) % D, dst_w, dst_s, rd_x, wq, sq, xrs
                            )
                            _compute_step(cfg.bf16, wq, sq, pairs)
                else:
                    for i in cutlass.range_constexpr(D):
                        cute.arch.cp_async_wait_group(D - 1 - i)
                        _quad_consume(
                            cfg, (cfg.tm + i) % D, dst_w, dst_s, rd_x, wq, sq, xrs
                        )
                        _compute_step(cfg.bf16, wq, sq, pairs)
                if cutlass.const_expr(cfg.prof):
                    ts[6] = _clock()

    # ---- intra-CTA reduction over the classes -------------------------------------------
    # red[row = ((2 * block + half) * MP + m) * sdiv + slice][class], rows of 4 * rch floats
    # (pads stay zero). G-slot g' at block cb of its warp: C[g' (+ 8)][4 cb + m] sits in lane
    # t = 2 cb + m / 2, elements m % 2 (and 2 + m % 2); the second MMA holds rows m + 4.
    # (Slice is the innermost row index and rch is odd so that the 16 lanes of a warp that
    # store here spread over the smem banks: slice-major rows cost 0.5 us at M = 8.)
    rstride = cfg.rch * 16
    if (t >> 1) == pr - pw0:  # noqa: SIM102 -- DSL regions: keep the nesting
        if act:
            rb = red + ((((t & 1) * 2) * sdiv + ui) * rstride + pr * 4)
            rjump = sdiv * rstride  # one m further
            for q in cutlass.range_constexpr(NQ):
                for jb in cutlass.range_constexpr(4):
                    for e in cutlass.range_constexpr(4):
                        row = (2 * jb + (e >> 1)) * MP + 4 * q + (e & 1)
                        _smem_scalar(Float32, rb + row * rjump)[0] = accs[q][4 * jb + e]
    # Everything fetched in the prologue must have landed on EVERY path before the epilogue
    # reads it (scale2's word): an idle warp never waits for anything above.
    cute.arch.cp_async_wait_group(0)
    if cutlass.const_expr(cfg.prof):
        ts[7] = _clock()
    cute.arch.barrier()

    # ---- epilogue: sum the classes of each owned output, scale, store -------------------
    s2f = (_smem_scalar(Int32, s2sm)[0] >> s2sh) & 0xFFFF
    fs = cute.make_rmem_tensor((4,), Float32)
    fv = cute.make_rmem_tensor((4 * cfg.rch,), Float32)
    cbase = pC.toint()
    st[0] = tidx
    while st[0] < n_out:
        o = st[0]  # = (slice * 8 + 2 * block + half) * MP + m
        m = o & (MP - 1)
        jh = (o >> lmp) & 7
        if m < prob_m:
            rb = red + ((jh * MP + m) * sdiv + (o >> (3 + lmp))) * rstride
            for qd in cutlass.range_constexpr(cfg.rch):
                cute.autovec_copy(
                    cute.make_tensor(
                        _smem_ptr(Float32, rb + 16 * qd, 16), cute.make_layout((4,))
                    ),
                    cute.make_tensor(fv.iterator + 4 * qd, cute.make_layout((4,))),
                )
            for i in cutlass.range_constexpr(4):  # fixed order: deterministic
                fs[i] = _tree_sum([fv[4 * qd + i] for qd in range(cfg.rch)])
            tot = (fs[0] + fs[1]) + (fs[2] + fs[3])
            slc_o = u0 + (o >> (3 + lmp))
            col = (slc_o >> 3) * 64 + ((jh & 6) << 3) + (slc_o & 7) + (jh & 1) * 8
            _gmem_scalar(Int16, cbase + _wide(m * prob_n + col, 2), 2)[0] = _finish(
                cfg.bf16, tot, s2f
            )
        st[0] = o + nthr
    if cutlass.const_expr(cfg.prof):
        ts[NSTAMP - 2] = _clock()
        if tidx & 31 == 0:
            dbase = pD.toint() + Int64((bidx * cfg.warps + (tidx >> 5)) * (NSTAMP * 8))
            for i in cutlass.range_constexpr(NSTAMP):
                _gmem_scalar(Int64, dbase + i * 8, 8)[0] = ts[i]


@cute.jit
def _quad_launch(
    pX: cute.Pointer,
    pW: cute.Pointer,
    pS: cute.Pointer,
    pS2: cute.Pointer,
    pC: cute.Pointer,
    pD: cute.Pointer,
    prob_m: Int32,
    prob_k: Int32,
    prob_n: Int32,
    grid: Int32,
    qn: Int32,
    rn: Int32,
    sdiv: Int32,
    mg: Int32,
    pmul: Int32,
    nst: Int32,
    rps: Int32,
    rem: Int32,
    ng: Int32,
    cfg: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
):
    _quad_kernel(
        pX,
        pW,
        pS,
        pS2,
        pC,
        pD,
        prob_m,
        prob_k,
        prob_n,
        qn,
        rn,
        sdiv,
        mg,
        pmul,
        nst,
        rps,
        rem,
        ng,
        cfg,
    ).launch(grid=(grid, 1, 1), block=(cfg.threads, 1, 1), stream=stream)


# ---------------------------------------------------------------------------
# GENERIC flavor: uneven streams, x through the ring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _GenCfg:
    bf16: bool
    mpad: int
    warps: int
    depth: int
    prof: bool = False

    @property
    def ns(self):
        return 8 // self.mpad

    @property
    def streams(self):
        return self.warps * self.ns

    @property
    def threads(self):
        return 32 * self.warps

    @property
    def ring_bytes(self):
        return self.depth * GROW_BYTES

    @property
    def red_off(self):
        return self.warps * self.ring_bytes

    @property
    def smem_bytes(self):
        return self.red_off + self.streams * self.mpad * self.mpad * 32


@cute.kernel
def _gen_kernel(
    pX: cute.Pointer,
    pW: cute.Pointer,
    pS: cute.Pointer,
    pS2: cute.Pointer,
    pC: cute.Pointer,
    pD: cute.Pointer,
    prob_m: Int32,
    prob_k: Int32,
    prob_n: Int32,
    qn: Int32,
    rn: Int32,
    nu_a: Int32,  # per class: units owned, floor / remainder of streams per unit, loop count,
    tq_a: Int32,  # division magics for floor and floor + 1
    tr_a: Int32,
    st_a: Int32,
    mq_a: Int64,
    mq1_a: Int64,
    nu_b: Int32,
    tq_b: Int32,
    tr_b: Int32,
    st_b: Int32,
    mq_b: Int64,
    mq1_b: Int64,
    cfg: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    ts = cute.make_rmem_tensor((NSTAMP,), Int64)
    if cutlass.const_expr(cfg.prof):
        ts[0] = _clock()
        for i in cutlass.range_constexpr(1, NSTAMP - 1):
            ts[i] = ts[0]
        ts[NSTAMP - 1] = cute.arch.globaltimer()
    lane = tidx % 32
    warp = tidx // 32
    gp = lane // 4
    t = lane % 4
    mp = cfg.mpad
    D = cfg.depth
    sp = gp // mp
    si = gp % mp
    rows = prob_k // 16

    # ---- ownership: the first tr units get tq + 1 streams, the others tq ------
    is_a = bidx < rn
    nu = _sel(is_a, nu_a, nu_b)
    tq = _sel(is_a, tq_a, tq_b)
    tr = _sel(is_a, tr_a, tr_b)
    nsteps = _sel(is_a, st_a, st_b)
    mq = _sel(is_a, mq_a, mq_b, Int64)
    mq1 = _sel(is_a, mq1_a, mq1_b, Int64)
    u0 = bidx * qn + _sel(is_a, bidx, rn)

    tau = warp * cfg.ns + sp
    bq = tr * (tq + 1)
    big = tau < bq
    dd = _sel(big, tq + 1, tq)
    mg = _sel(big, mq1, mq, Int64)
    tau2 = tau - _sel(big, 0, bq)
    uu = _mdiv(tau2, mg)
    ui = _sel(big, 0, tr) + uu
    pidx = tau2 - uu * dd
    row0 = _mdiv(pidx * rows, mg)
    nrows = _mdiv((pidx + 1) * rows, mg) - row0
    slc = (u0 + ui) * mp + si

    nrows_s = _sel(t == 0, nrows, 0)
    nrows_xr = _sel(si < prob_m, nrows, 0)
    nrows_xf = _sel(t < 2, nrows_xr, 0)

    smem = cutlass.utils.SmemAllocator()
    blob = smem.allocate_tensor(
        Int32, cute.make_layout(cfg.smem_bytes // 4), byte_alignment=16
    )
    sbase = blob.iterator.toint()
    ring = sbase + warp * cfg.ring_bytes
    red = sbase + cfg.red_off

    ws32 = Uint32(prob_n * 8)
    ss32 = Uint32(prob_n)
    wptr0 = pW.toint() + _wide(row0, ws32) + Int64(slc * 64 + t * 16)
    sptr0 = pS.toint() + _wide(row0, ss32) + Int64(slc * 8)
    xptr0 = pX.toint() + _wide(si, Uint32(prob_k * 2)) + Int64(row0 * 32 + (t % 2) * 16)

    dst_w = ring + lane * 16
    dst_s = ring + ROW_W + gp * 8
    dst_x = ring + (ROW_W + ROW_S) + gp * 32 + (t % 2) * 16
    rd_x = ring + (ROW_W + ROW_S) + gp * 32 + t * 4

    s2f = Int32(_gmem_scalar(Int16, pS2.toint(), 2)[0]) & 0xFFFF

    acc = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(16):
        acc[i] = Float32(0.0)
    wq = cute.make_rmem_tensor((4,), Int32)
    sq = cute.make_rmem_tensor((2,), Int32)
    xr = cute.make_rmem_tensor((2,), Int32)
    st = cute.make_rmem_tensor((2,), Int32)

    # ---- prologue: put the first D steps in flight ---------------------------
    # (flat, single-statement conditions: they compile to predicated LDGSTS, so the warp never
    # diverges between a fetch and the wait its sibling lanes rely on)
    for d in cutlass.range_constexpr(D):
        if d < nrows:
            _fetch(dst_w + d * GROW_BYTES, wptr0 + _wide(d, ws32), 16, "cg")
        if d < nrows_s:
            _fetch(dst_s + d * GROW_BYTES, sptr0 + _wide(d, ss32), 8, "ca")
        if d < nrows_xf:
            _fetch(dst_x + d * GROW_BYTES, xptr0 + d * 32, 16, "ca")
        cute.arch.cp_async_commit_group()
    if cutlass.const_expr(cfg.prof):
        ts[1] = _clock()
        ts[2] = ts[1]
        ts[3] = ts[1]

    # ---- main loop -------------------------------------------------------------
    st[0] = 0
    while st[0] < nsteps:
        for d in cutlass.range_constexpr(D):
            j = st[0] + d
            if j < nsteps:
                cute.arch.cp_async_wait_group(D - 1)
                _lds(dst_w + d * GROW_BYTES, wq, 4, 16)
                # A finished stream (or a lane of a never-fetched ring row) must
                # contribute exact zeros: zero scales and zero activations.
                _lds(dst_s + d * GROW_BYTES, sq, 2, 8)
                xr[0] = _smem_scalar(Int32, rd_x + d * GROW_BYTES)[0]
                xr[1] = _smem_scalar(Int32, rd_x + (d * GROW_BYTES + 16))[0]
                for i in cutlass.range_constexpr(2):
                    sq[i] = _sel(j < nrows, sq[i], 0)
                    xr[i] = _sel(j < nrows_xr, xr[i], 0)
                jn = j + D
                if jn < nrows:
                    _fetch(dst_w + d * GROW_BYTES, wptr0 + _wide(jn, ws32), 16, "cg")
                if jn < nrows_s:
                    _fetch(dst_s + d * GROW_BYTES, sptr0 + _wide(jn, ss32), 8, "ca")
                if jn < nrows_xf:
                    _fetch(
                        dst_x + d * GROW_BYTES, xptr0 + _wide(jn, Uint32(32)), 16, "ca"
                    )
                cute.arch.cp_async_commit_group()
                _compute_step(cfg.bf16, wq, sq, ((xr, acc),))
        st[0] = st[0] + D
    if cutlass.const_expr(cfg.prof):
        ts[5] = _clock()
        ts[6] = ts[5]
        ts[7] = ts[5]

    # ---- intra-CTA reduction: red[stream][slice-in-unit][m][2 * block + half] ----
    live = nrows > 0
    if cutlass.const_expr(mp == 1):
        if t == gp // 2:
            odd = (gp % 2) == 1
            for jh in cutlass.range_constexpr(8):
                v = _sel(odd, acc[2 * jh + 1], acc[2 * jh], Float32)
                _smem_scalar(Float32, red + tau * 32 + jh * 4)[0] = _sel(
                    live, v, Float32(0.0), Float32
                )
    else:
        if t // (mp // 2) == sp:
            m_a = 2 * t - sp * mp
            rbase = red + ((tau * mp + si) * mp + m_a) * 32
            for e in cutlass.range_constexpr(4):
                for jb in cutlass.range_constexpr(4):
                    v = _sel(live, acc[4 * jb + e], Float32(0.0), Float32)
                    _smem_scalar(
                        Float32, rbase + ((e % 2) * 32 + (2 * jb + e // 2) * 4)
                    )[0] = v
    cute.arch.barrier()

    # ---- epilogue ---------------------------------------------------------------
    n_out = nu * (mp * mp * 8)
    fs = cute.make_rmem_tensor((1,), Float32)
    cbase = pC.toint()
    st[0] = tidx
    while st[0] < n_out:
        o = st[0]
        jh = o % 8
        m = (o // 8) % mp
        sl_o = o // (8 * mp)
        if m < prob_m:
            ui_o = sl_o // mp
            si_o = sl_o % mp
            big_o = ui_o < tr
            t0 = _sel(big_o, ui_o * (tq + 1), bq + (ui_o - tr) * tq)
            ns_o = _sel(big_o, tq + 1, tq)
            fs[0] = Float32(0.0)
            st[1] = 0
            while st[1] < ns_o:
                addr = red + (((t0 + st[1]) * mp + si_o) * mp + m) * 32 + jh * 4
                fs[0] = fs[0] + _smem_scalar(Float32, addr)[0]
                st[1] = st[1] + 1
            slc_o = (u0 + ui_o) * mp + si_o
            col = (slc_o // 8) * 64 + (jh // 2) * 16 + (slc_o % 8) + (jh % 2) * 8
            _gmem_scalar(Int16, cbase + _wide(m * prob_n + col, 2), 2)[0] = _finish(
                cfg.bf16, fs[0], s2f
            )
        st[0] = o + cfg.threads
    if cutlass.const_expr(cfg.prof):
        ts[NSTAMP - 2] = _clock()
        if lane == 0:
            dbase = pD.toint() + Int64((bidx * cfg.warps + warp) * (NSTAMP * 8))
            for i in cutlass.range_constexpr(NSTAMP):
                _gmem_scalar(Int64, dbase + i * 8, 8)[0] = ts[i]


@cute.jit
def _gen_launch(
    pX: cute.Pointer,
    pW: cute.Pointer,
    pS: cute.Pointer,
    pS2: cute.Pointer,
    pC: cute.Pointer,
    pD: cute.Pointer,
    prob_m: Int32,
    prob_k: Int32,
    prob_n: Int32,
    grid: Int32,
    qn: Int32,
    rn: Int32,
    nu_a: Int32,
    tq_a: Int32,
    tr_a: Int32,
    st_a: Int32,
    mq_a: Int64,
    mq1_a: Int64,
    nu_b: Int32,
    tq_b: Int32,
    tr_b: Int32,
    st_b: Int32,
    mq_b: Int64,
    mq1_b: Int64,
    cfg: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
):
    _gen_kernel(
        pX, pW, pS, pS2, pC, pD, prob_m, prob_k, prob_n, qn, rn,
        nu_a, tq_a, tr_a, st_a, mq_a, mq1_a, nu_b, tq_b, tr_b, st_b, mq_b, mq1_b, cfg,
    ).launch(grid=(grid, 1, 1), block=(cfg.threads, 1, 1), stream=stream)  # fmt: skip


# ---------------------------------------------------------------------------
# Host side: plan
# ---------------------------------------------------------------------------

_SMEM_MAX = 160 << 10


def _magic(d: int) -> Int64:
    return Int64((1 << 32) // d + 1)


def _magic16(d: int) -> int:
    """floor(a * magic >> 16) == a // d for 0 <= a, a * d < 2^16 (stream indices: a < 100)."""
    return (1 << 16) // d + 1


def _gen_class(nu: int, streams: int, rows: int):
    tq, tr = divmod(streams, nu)
    return [
        Int32(nu),
        Int32(tq),
        Int32(tr),
        Int32(-(-rows // tq)),
        _magic(tq),
        _magic(tq + 1),
    ]


def _pow2_at_least(need: int, lo: int) -> int:
    while lo < need:
        lo *= 2
    return lo


_RCH = (1, 3, 5, 7, 9, 13, 17, 25, 33)  # odd: see the DIAG reduction


def _tail_shape(rows: int, nst: int, depth: int):
    """Steps of a (K/16, classes) split: (rps, rem, short, groups, tm, ext) for the kernels."""
    rps, rem = divmod(rows, nst)
    steps = rps + (1 if rem else 0)
    short = steps <= depth  # everything is fetched in the prologue
    fills = steps - depth - (1 if rem else 0)  # full refills
    if short:
        return rps, rem, True, 0, 0, False
    return rps, rem, False, fills // depth, fills % depth, bool(rem)


_BAND_MIN_M = (
    3  # QUAD: from this M on, x may be fetched once per CTA (BAND): see _quad_plan
)
_BAND_MAX_STEPS = 60  # ... for M > 4 only up to this many steps
_WARPS = 8  # per CTA: two per warp scheduler
_DEPTH = 5  # cp.async ring depth in steps: 8 warps x 5 x 576 B = 23 KB in flight per SM


def _sched_load(live_warps, steps: int) -> int:
    """Warp-steps on the busiest of the SM's four warp schedulers (warp w runs on w % 4).

    That, not the warp-steps of the SM, is what bounds a compute-bound CTA: a scheduler
    issues for one warp at a time and is saturated by two (a lone warp only reaches about
    half its rate, hence the floor of 2). Measured (bf16, cold, us at K=4096 / K=2048, N=4096):
        M=1   10 warps x depth 4 (3:3:2:2)   9.95 / 7.25      86 CTAs x 12 warps      9.77 / 7.17
              8 x 4   9.60 / 6.79     8 x 5   9.37 / 6.73     12 x 4   9.44 / 6.98    16 x 3   9.84 / 7.18
        M=4   8 x 4  10.85 / 7.34     8 x 5  10.85 / 7.26     12 x 4  10.80 / 7.42    16 x 3  10.97 / 7.76
        M=8   8 x 4  12.41 / 9.53     8 x 5  12.48 / 9.42     12 x 4  12.23 / 9.87    16 x 3  13.38 / 10.14
    """
    return (
        max(max(sum(1 for w in live_warps if w % 4 == s) for s in range(4)), 2) * steps
    )


def _diag_plan(size_n: int, rows: int, sms: int, bf16: bool, knobs: tuple, prof: bool):
    """M = 1: grid, warps per CTA and k16-row classes per slice.

    An SM runs its warps on four schedulers (warp w on scheduler w % 4), each issuing for one
    warp at a time, so what bounds the kernel is the busiest scheduler: its warps x steps.
    Ten equal warps load them 3 : 3 : 2 : 2 -- measured 0.8 us between the first and the last
    warp to finish; 8 or 12 warps load them evenly. The class count need not divide K/16: the
    first K/16 % classes classes simply run one more step.
    """
    depth = int(knobs[1] or _DEPTH)
    units = size_n // 8  # slices
    best = None
    for warps in (int(knobs[0] or _WARPS),):
        grid = min(int(knobs[2] or sms), units)
        while -(-units // grid) > 8 * warps:  # every slice needs at least one stream
            grid += sms
        qn, rn = divmod(units, grid)
        nux = qn + 1 if rn else qn
        for nst in range(min(8 * warps // nux, rows), 0, -1):
            steps = -(-rows // nst)
            live = [w for w in range(warps) if (8 * w) // nux < nst]
            key = (_sched_load(live, steps), len(live) * steps, warps)
            if best is None or key < best[0]:
                best = (key, warps, grid, qn, rn, nux, nst)
    _, warps, grid, qn, rn, nux, nst = best
    rps, rem, short, groups, tm, ext = _tail_shape(rows, nst, depth)
    spans = [
        min((8 * w + 7) // nux, nst - 1) - (8 * w) // nux + 1
        for w in range(warps)
        if (8 * w) // nux < nst
    ]
    rch = next((c for c in _RCH if 4 * c >= nst), None)
    if rch is None:
        return None
    rbytes = _pow2_at_least(nux * 8 * rch * 16, 4 << 10)
    cfg = _DiagCfg(bf16, warps, depth, rch, rbytes, short, tm, ext, prof)
    if cfg.smem_bytes > _SMEM_MAX:
        return None
    return (
        cfg,
        grid,
        [qn, rn, nux, _magic16(nux), nst, rps, rem, groups, 2 * max(spans)],
    )


def _quad_slot(tau: int, sdiv: int, pmul: int):
    """Host mirror of the QUAD g-slot map: (class, slice, idle)."""
    te = tau - (tau >> 4) * pmul
    return te // sdiv, te % sdiv, bool(pmul) and tau & 15 == 15


def _quad_plan(
    size_m: int, size_n: int, rows: int, sms: int, bf16: bool, knobs: tuple, prof: bool
):
    """2 <= M <= 8: grid, warps per CTA, g-slot pattern and k16-row classes (see _quad_kernel)."""
    depth = int(knobs[1] or _DEPTH)
    nq = 1 if size_m <= 4 else 2
    units = size_n // 8  # slices
    best = None
    for warps in (int(knobs[0] or _WARPS),):
        grid = min(int(knobs[2] or sms), units)
        while -(-units // grid) > 8 * warps:
            grid += sms
        qn, rn = divmod(units, grid)
        nux = qn + 1 if rn else qn
        sdiv, pmul = (4, 0) if nux <= 4 else (5, 1) if nux == 5 else (nux, 0)
        slots = [_quad_slot(tau, sdiv, pmul) for tau in range(8 * warps)]
        have = {(c, sl) for c, sl, idle in slots if not idle}
        ncap = 0  # classes with a g-slot for every slice
        while all((ncap, sl) in have for sl in range(nux)):
            ncap += 1
        for nst in range(min(ncap, rows), 0, -1):
            steps = -(-rows // nst)
            live = [w for w in range(warps) if slots[8 * w][0] < nst]
            # every warp must hold at most two classes: its first g-slot's and the next one
            ok = all(
                c - slots[8 * (tau // 8)][0] in (0, 1)
                for tau, (c, sl, idle) in enumerate(slots)
                if not idle and sl < nux and c < nst
            )
            if not ok:
                continue
            key = (_sched_load(live, steps), len(live) * steps, warps)
            if best is None or key < best[0]:
                best = (key, warps, grid, qn, rn, sdiv, pmul, nst)
    if best is None:
        return None
    _, warps, grid, qn, rn, sdiv, pmul, nst = best
    rps, rem, short, groups, tm, ext = _tail_shape(rows, nst, depth)
    rch = next((c for c in _RCH if 4 * c >= nst), None)
    if rch is None:
        return None
    rbytes = _pow2_at_least(8 * sdiv * 4 * nq * rch * 16, 4 << 10)
    cfg = _QuadCfg(bf16, nq, warps, depth, rch, rbytes, short, tm, ext, prof)
    # BAND (x once per CTA, a barrier per group of `depth` steps) needs a warp per row of x
    # and a group's columns in a few fetches. Whether it pays was measured, not derived: band
    # minus ring in us (bf16, cold, mean over 6-8 addresses of x; the per-address spread is
    # 0.2-0.5), one CTA per SM:
    #                                            steps   M=3    M=4    M=5    M=6    M=8
    #   K=4096  N=2048 (3 slices, 16 classes)      16   +0.14  +0.28  +0.64  +0.64  +0.35
    #           N=3072 (4 slices, 16 classes)      16   -0.03  -0.19  +0.25  +0.21  -0.08
    #           N=3584..4224 (5 slices, 12)        22   -0.4   -0.5   -0.2   -0.3   -0.5
    #           N=5120 (6, 10 classes)             26   -0.08  -0.20  +0.48  +0.17  -0.45
    #           N=6144 (8, 8)                      32   -0.05  -0.30  +0.70  +0.43  +0.02
    #           N=8192 (10, 6)                     43   +0.04  -0.02  +0.81  +0.72  +0.48
    #           N=14336 (17, 3)                    86   -0.73  -0.81  +2.25  +1.99  +2.06
    #   N=4096  K=3072                             16   -0.10  -0.18  +0.12         -0.12
    #           K=5120 / 6144                   27/32   -0.09  -0.37  -0.4/0 -0.25  -0.3/-0.5
    #           K=8192 / 10240                  43/54   -0.78  -1.02  -0.6/-1.2     -0.9/-2.3
    #           K=12288 / 16384                 64/86   -0.78  -0.67  +0.1/+1.8     -0.2/+1.2
    #   fewer CTAs than SMs (N=128, 512)        16/64   +0.2 .. +0.9 (M=4), +0.3 .. +1.7 (M=8)
    # (K=2048, N=4096: 11 steps, three barriers for one group's worth of saving: -0.2 .. +0.2.)
    # So: with one row quad (M <= 4) whenever there are at most 12 classes; with two only in
    # the 5-slice geometry and up to 60 steps; never when the grid does not fill the GPU.
    band = dataclasses.replace(cfg, band=True)
    steps = rps + (1 if rem else 0)
    if knobs[5]:
        want = knobs[5] != "0"
    else:
        want = size_m >= _BAND_MIN_M and grid == sms and steps > 3 * depth
        want = want and (
            nst <= 12 if nq == 1 else pmul == 1 and steps <= _BAND_MAX_STEPS
        )
    if (
        want
        and warps >= cfg.mpad
        and depth >= 3
        and band.xrounds <= 8
        and band.smem_bytes <= _SMEM_MAX
    ):
        cfg = band
    if cfg.smem_bytes > _SMEM_MAX:
        return None
    return cfg, grid, [qn, rn, sdiv, _magic16(sdiv), pmul, nst, rps, rem, groups]


_DEBUG = ""  # geometry overrides / profiling builds for development: see debug_knobs()
_COMPILE_OPTS = (
    None  # extra cute.compile options for development (SASS dumps): see debug_knobs()
)


def _parse_knobs(spec: str) -> tuple:
    """(warps, depth, grid, flavor, prof, band) strings out of "name=value,..."; "" = default."""
    kv = dict(item.split("=", 1) for item in spec.split(",") if item)
    unknown = set(kv) - {"warps", "depth", "grid", "flavor", "prof", "band"}
    if unknown:
        raise ValueError(f"debug_knobs: unknown knob(s) {sorted(unknown)}")
    return tuple(
        kv.get(name, "")
        for name in ("warps", "depth", "grid", "flavor", "prof", "band")
    )


def _plan(size_m: int, size_k: int, size_n: int, bf16: bool, sms: int, spec=None):
    """(launcher, cfg, grid, runtime scalar args) for a problem."""
    return _plan_cached(
        size_m, size_k, size_n, bf16, sms, _DEBUG if spec is None else spec
    )


@functools.lru_cache(maxsize=1024)
def _plan_cached(
    size_m: int, size_k: int, size_n: int, bf16: bool, sms: int, spec: str
):
    knobs = _parse_knobs(spec)
    flavor = knobs[3]
    prof = bool(knobs[4]) and knobs[4] != "0"
    rows = size_k // 16
    got = None
    if flavor != "generic":
        if size_m == 1:
            got = _diag_plan(size_n, rows, sms, bf16, knobs, prof)
            launch = _diag_launch
        else:
            got = _quad_plan(size_m, size_n, rows, sms, bf16, knobs, prof)
            launch = _quad_launch
    if got is not None:
        cfg, grid, own = got
    else:
        # GENERIC: uneven streams, any geometry.
        mpad = 1 if size_m == 1 else 2 if size_m == 2 else 4 if size_m <= 4 else 8
        warps = int(knobs[0] or 12)
        streams = warps * (8 // mpad)
        units = (size_n // 8) // mpad
        grid = min(int(knobs[2] or sms), units)
        while -(-units // grid) > streams:  # every unit needs at least one stream
            grid += sms
        qn, rn = divmod(units, grid)
        nu_a = qn + 1 if rn else qn
        own = [qn, rn, *_gen_class(nu_a, streams, rows), *_gen_class(qn, streams, rows)]
        launch, cfg = _gen_launch, _GenCfg(bf16, mpad, warps, int(knobs[1] or 4), prof)
    if cfg.threads > 1024 or cfg.smem_bytes > (163 << 10):
        raise ValueError(
            f"cute_nvfp4_decode: invalid geometry ({cfg.threads} threads, {cfg.smem_bytes} B of shared memory per CTA)"
        )
    scalars = tuple(
        v if isinstance(v, Int64) else Int32(v)
        for v in (size_m, size_k, size_n, grid, *own)
    )
    return launch, cfg, grid, scalars


# ---------------------------------------------------------------------------
# Host side: launch ABI and compiled kernels
# ---------------------------------------------------------------------------

# x, weight, scale, scale2, out, profiling stamps: (element type, guaranteed alignment)
_PTR_TYPES = ((Int32, 16), (Int32, 16), (Int32, 8), (Int16, 2), (Int16, 2), (Int64, 8))
NPTR = len(_PTR_TYPES)
# A launch block owns eight 8-byte cells: the NPTR pointers, the CUstream, the launch status.
# The cells a call changes come first -- x, out, stream, status -- then weight, scale, scale2
# and the profiling stamps, which a layer writes once into the blocks it keeps.
_NCELL = NPTR + 2
_CELL_STATUS = 3
# launcher slots (6 pointers, stream, status) -> cell
_CELL_OF_SLOT = (0, 4, 5, 6, 1, 7, 2, _CELL_STATUS)
_PACK_CALL = struct.Struct("4Q").pack_into  # x, out, stream, status = 0
_PACK_LAYER = struct.Struct("4Q").pack_into  # at byte 32: weight, scale, scale2, stamps
_FAKE_STREAM = 0x5EED5EED00  # marshalled once per plan to locate the stream slot; never launched on

_COMPILED: dict = {}  # (cfg, compute capability, compile options) -> compiled launcher
_EXECUTORS: dict = {}  # (cfg, device, compile options) -> its executor on that device
# compiles, plan and layer registration (never taken by a launch)
_BUILD_LOCK = threading.RLock()


@functools.cache
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


@functools.cache
def _capability(index: int) -> tuple:
    return tuple(torch.cuda.get_device_capability(index))


# ---------------------------------------------------------------------------
# On-disk cache of compiled kernels.
#
# THIS BLOCK IS IDENTICAL IN cute_nvfp4_decode.py AND cute_nvfp4_batch.py, which
# share one cache directory: change both or neither.
#
#   directory  NVFP4_CUTE_CACHE, else ~/.cache/nvfp4_marlin_cute (read once, at
#              import; the only environment variable these modules look at).
#   file       <module>_<12 hex of the module source's sha256>_<16 hex of the
#              key>.o, the key being (launcher, specialization, GPU compute
#              capability, cutlass / torch / CUDA versions): an edit or an
#              upgrade can only miss, never load a stale kernel.
#   content    the relocatable object + _OBJ_SEAL + sha256(object). A file that
#              does not verify -- cut short, overwritten, one flipped bit -- is
#              rebuilt and replaced, never linked: one wrong byte in a kernel
#              image is silently wrong output otherwise. The JIT is handed the
#              very bytes that were verified.
#   writes     into a private mkdtemp directory, then one atomic rename.
#   pruning    on a miss a module removes ITS OWN objects (matched by name)
#              that no process has loaded for _CACHE_MAX_AGE -- every hit
#              refreshes the mtime, so objects in use never age -- and its own
#              export directories older than _TMP_MAX_AGE (their process was
#              killed). Files of the other module are never touched.
# ---------------------------------------------------------------------------
_CACHE_DIR = Path(
    os.environ.get("NVFP4_CUTE_CACHE") or Path.home() / ".cache" / "nvfp4_marlin_cute"
)
_SRC_STEM = re.sub(r"\W", "_", Path(__file__).stem)
_SRC_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_CACHE_TAG = f"{_SRC_STEM}_{_SRC_HASH[:12]}"
_OWN_OBJECT = re.compile(rf"{_SRC_STEM}_[0-9a-f]{{12}}_[0-9a-f]{{16}}\.o")
_OBJ_SEAL = b"\0nvfp4-cute-sha256\0"
_CACHE_MAX_AGE = 14 * 86400.0
_TMP_PREFIX = f".tmp_{_SRC_STEM}_"
_TMP_MAX_AGE = 3600.0


def _cache_name(launcher: str, spec, capability) -> str:
    """Name of the cache object (and of its entry point) of one specialization."""
    key = repr(
        (
            launcher,
            spec,
            tuple(capability),
            cutlass.__version__,
            torch.__version__,
            torch.version.cuda,
        )
    )
    return f"{_CACHE_TAG}_{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def _link(body: bytes, name: str):
    """JIT-link a relocatable object held in memory (load_module() takes a path)."""
    try:
        fd = os.memfd_create(name)
        path = f"/proc/self/fd/{fd}"
    except (AttributeError, OSError):  # no memfd here: a private temporary file
        fd, path = tempfile.mkstemp(suffix=".o")
    try:
        with open(fd, "wb", closefd=False) as f:
            f.write(body)
        return getattr(cute.runtime.load_module(path), name)
    finally:
        os.close(fd)
        if not path.startswith("/proc/"):
            os.unlink(path)


def _cache_load(name: str):
    """The launcher ``name`` from the cache; None if it is not there or does not verify."""
    obj = _CACHE_DIR / f"{name}.o"
    try:
        blob = obj.read_bytes()
    except OSError:
        return None
    cut = len(blob) - len(_OBJ_SEAL) - 32
    if (
        cut <= 0
        or blob[cut:-32] != _OBJ_SEAL
        or hashlib.sha256(blob[:cut]).digest() != blob[-32:]
    ):
        return None
    try:
        fn = _link(blob[:cut], name)
    except Exception:  # noqa: BLE001 -- e.g. an object of another toolchain build
        return None
    try:
        os.utime(obj)  # "in use": see the pruning rule
    except OSError:
        pass
    return fn


def _cache_prune() -> None:
    now = time.time()
    for old in _CACHE_DIR.iterdir():
        try:
            age = now - old.stat().st_mtime
            if old.name.startswith(_TMP_PREFIX):
                if age > _TMP_MAX_AGE:
                    shutil.rmtree(old, ignore_errors=True)
            elif age > _CACHE_MAX_AGE and _OWN_OBJECT.fullmatch(old.name):
                old.unlink(missing_ok=True)
        except OSError:
            pass


def _cache_store(fn, name: str) -> None:
    """Seal a freshly compiled launcher into the cache (best effort), then prune."""
    tmp = None
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # (mkdtemp: unique also across PID namespaces that share the directory)
        tmp = tempfile.mkdtemp(prefix=_TMP_PREFIX, dir=_CACHE_DIR)
        fn.export_to_c(tmp, name, function_prefix=name)
        body = Path(tmp, f"{name}.o").read_bytes()
        sealed = Path(tmp, "sealed")
        sealed.write_bytes(body + _OBJ_SEAL + hashlib.sha256(body).digest())
        os.replace(sealed, _CACHE_DIR / f"{name}.o")
        _cache_prune()
    except Exception:  # noqa: BLE001, S110 -- the cache is an optimization only
        pass
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------ end of the shared block ---------------------


def _compiled(launch, cfg, args, index: int):
    """The launcher specialized on ``cfg``: from this process, else the on-disk
    cache (key: this file's hash, the launcher, the geometry / dtype dataclass,
    the GPU architecture, the toolchain versions), else cute.compile."""
    cap = _capability(index)
    mem_key = (cfg, cap, _COMPILE_OPTS)
    fn = _COMPILED.get(mem_key)
    if fn is not None:
        return fn
    with _BUILD_LOCK:
        fn = _COMPILED.get(mem_key)
        if fn is not None:
            return fn
        # dump builds (--keep-sass ...) are neither stored nor loaded
        disk = _COMPILE_OPTS is None
        name = _cache_name(launch.__name__, cfg, cap)
        if disk:
            fn = _cache_load(name)
        if fn is None:
            kw = {} if disk else {"options": _COMPILE_OPTS}
            fn = cute.compile(launch, *args[:-1], cfg, args[-1], **kw)
            if disk:
                _cache_store(fn, name)
        _COMPILED[mem_key] = fn
        return fn


# ---------------------------------------------------------------------------
# Host side: plans (pre-marshalled launches) and the layer registry
# ---------------------------------------------------------------------------


def _launch_error(status: int):
    return RuntimeError(
        f"cute_nvfp4_decode: kernel launch failed (CUDA error {status})"
    )


class _Plan:
    """One problem (M, K, N, dtype, device): its compiled kernel and launch arguments, marshalled once.

    The JIT-ed launcher takes ``void**``: one slot per argument holding the address of that
    argument's storage (pointers first, the stream last), then the address the launch status is
    written to. A *block* is a private copy of that slot array whose pointer, stream and status
    slots point at eight cells of its own, so a launch is "write the cells, one C call, read the
    status cell" and blocks never share state: prepare() keeps one per run(), nvfp4_linear()
    takes one for the duration of a call from a pool (pop / append are atomic: no lock) -- from
    the second call of a layer object on the layer's own, whose blocks already hold the layer's
    three pointers.
    """

    __slots__ = (
        "_block_t",
        "_keep",
        "_slots",
        "_template",
        "capi",
        "cfg",
        "checked",
        "grid",
        "index",
        "nobytes",
        "oshape",
        "pool",
        "slow",
        "xbytes",
    )  # fmt: skip

    def __init__(self, ex, args, cfg, grid, compiled, index, size_m, size_k, size_n):
        exe_args, adapted = ex.generate_execution_args(*args)
        # a buffer the executor reuses: copy it
        packed = ex._get_invoke_packed_args(list(exe_args))
        n = len(exe_args)
        ok = (
            ex.cuda_result is not None
            and len(packed) > n
            and packed[n] == ctypes.addressof(ex.cuda_result)
        )
        ok = ok and ctypes.c_uint64.from_address(packed[n - 1]).value == _FAKE_STREAM
        ok = ok and all(
            ctypes.c_uint64.from_address(packed[i]).value == args[i]._pointer
            for i in range(NPTR)
        )
        if not ok:
            raise RuntimeError("cute_nvfp4_decode: unexpected CuTe DSL launch ABI")
        self.capi = ex.capi_func
        self.cfg, self.grid, self.index = cfg, grid, index
        # nvfp4_linear(): launches that need more than the cells
        self.slow = bool(cfg.prof)
        self.checked = False
        # idle blocks of no particular layer: see _launch_first
        self.pool = collections.deque()
        self.oshape = (size_m, size_n)
        # out - x must lie outside (nobytes, xbytes)
        self.xbytes, self.nobytes = 2 * size_m * size_k, -2 * size_m * size_n
        self._block_t = ctypes.c_void_p * len(packed)
        self._template = self._block_t(*packed)
        self._slots = (*range(NPTR), n - 1, n)
        self._keep = (ex, compiled, args, exe_args, adapted)

    def block(self, pw: int, ps: int, ps2: int, stamps: int = 0):
        """(cells, address of the argument block, the block): a launch's own pointers, stream and
        status, with the layer's three pointers filled in."""
        cells = (ctypes.c_uint64 * _NCELL)()
        blk = self._block_t.from_buffer_copy(self._template)
        base = ctypes.addressof(cells)
        for slot, cell in zip(self._slots, _CELL_OF_SLOT):
            blk[slot] = base + 8 * cell
        _PACK_LAYER(cells, 32, pw, ps, ps2, stamps)
        # (passing the address saves 0.2 us of ctypes conversion)
        return cells, ctypes.addressof(blk), blk

    def bind(
        self,
        px: int,
        pw: int,
        ps: int,
        ps2: int,
        po: int,
        stamps: int,
        stream: int,
        keep,
    ):
        """A relaunch closure with its own argument block. Every launch's status is read back:
        a launch the driver rejects raises instead of silently leaving stale output."""
        cells, addr, blk = self.block(pw, ps, ps2, stamps)
        _PACK_CALL(cells, 0, px, po, stream, 0)
        capi = self.capi

        def run():
            capi(addr)
            if cells[_CELL_STATUS]:
                raise _launch_error(cells[_CELL_STATUS])

        run._keep = (blk, self, keep)
        return run


def _raw_stream(index: int) -> int:
    """cudaStream_t of torch's current stream on the device (0.1 us; the public API is 2.6 us)."""
    return torch.cuda.current_stream(index).cuda_stream


_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", _raw_stream)

_SHAPES: dict = {}  # (K, N, dtype, device) -> plans indexed by M (None: not built yet)


def _build_plan(size_m: int, size_k: int, size_n: int, bf16: bool, index: int) -> _Plan:
    launch, cfg, grid, scalars = _plan_cached(
        size_m, size_k, size_n, bf16, _sm_count(index), _DEBUG
    )
    # Placeholder pointers with distinct addresses (checked in _Plan, never dereferenced); their
    # element type and alignment are what the kernel may assume of the real ones.
    ptrs = tuple(
        make_ptr(ty, 64 * (i + 1), cute.AddressSpace.gmem, assumed_align=align)
        for i, (ty, align) in enumerate(_PTR_TYPES)
    )
    args = (*ptrs, *scalars, cuda_driver.CUstream(_FAKE_STREAM))
    compiled = _compiled(launch, cfg, args, index)
    ex_key = (cfg, index, _COMPILE_OPTS)
    ex = _EXECUTORS.get(ex_key)
    if ex is None:
        ex = _EXECUTORS[ex_key] = compiled.to(index)
    return _Plan(ex, args, cfg, grid, compiled, index, size_m, size_k, size_n)


def _new_plan(lay, size_m: int) -> _Plan:
    with _BUILD_LOCK:
        plan = lay.plans[size_m]
        if plan is None:
            plan = lay.plans[size_m] = _build_plan(
                size_m, lay.k, lay.n, lay.dtype is torch.bfloat16, lay.index
            )
    return plan


class _Layer:
    """What was validated about one (weight, scale, scale2) triple, found again by tensor identity.

    wref / sref / s2ref are weak references to the three tensor objects, wmem / smem / s2mem to
    their storages. A registered layer is trusted for a call only if the caller passes the same
    three objects, each still at its validated address, and the three allocations validated at
    those addresses are still alive (a storage weak reference dies with the allocation). So a
    tensor rebound to other memory -- including a new, smaller allocation that the allocator
    put at the old address -- is validated again from scratch, and the kernel never reads an
    extent that was not validated for memory that is still owned.
    """

    __slots__ = (
        "device",
        "dtype",
        "index",
        "k",
        "n",
        "plans",
        "pools",
        "ps",
        "ps2",
        "pw",
        "s2mem",
        "s2ref",
        "slow",
        "smem",
        "sref",
        "wmem",
        "wref",
        "xmask",
    )  # fmt: skip


_LAYERS: dict = {}  # id(weight) -> _Layer; an entry is dropped when its weight tensor dies
_DEVICES: dict = {}  # device index -> torch.device
_SCALE_DTYPES = (torch.float8_e4m3fn, torch.uint8)
_Tensor = torch.Tensor
# _Layer.pools marker (never filled): one call so far at this M
_SEEN_ONCE = collections.deque()


def _forget(key: int, ref, layers=_LAYERS) -> None:
    """Weak-reference callback: the weight tensor of a registered layer died."""
    lay = layers.get(key)
    if lay is not None and lay.wref is ref:
        del layers[key]


def _layer(weight, scale, scale2) -> _Layer:
    """Validate the fixed operands of a layer in full and register them."""
    if not (
        (type(weight) is _Tensor or isinstance(weight, _Tensor))
        and (type(scale) is _Tensor or isinstance(scale, _Tensor))
        and (type(scale2) is _Tensor or isinstance(scale2, _Tensor))
    ):
        raise TypeError("weight, scale and scale2 must be tensors")
    if weight.dim() != 2 or scale.dim() != 2:
        raise ValueError("weight must be [K/16, 2N] and scale [K/16, N]")
    rows, n2 = weight.shape
    size_k, size_n = rows * 16, n2 // 2
    dtype = scale2.dtype
    if dtype is not torch.bfloat16 and dtype is not torch.float16:
        raise TypeError(f"scale2 (and x) must be float16 or bfloat16, got {dtype}")
    if (
        weight.dtype is not torch.int32
        or n2 % 2
        or size_n % 128
        or size_k % 64
        or size_k < 64
        or size_n < 128
    ):
        raise ValueError(
            f"weight must be int32 [K/16, 2N] with K % 64 == 0 and N % 128 == 0, got {weight.dtype} {tuple(weight.shape)}"
        )
    if scale.dtype not in _SCALE_DTYPES or scale.shape != (rows, size_n):
        raise ValueError(
            f"scale must be float8_e4m3fn [K/16, N], got {scale.dtype} {tuple(scale.shape)}"
        )
    if scale2.numel() != 1:
        raise ValueError(f"scale2 must hold one element, got {tuple(scale2.shape)}")
    if not (weight.is_contiguous() and scale.is_contiguous()):
        raise ValueError("weight and scale must be contiguous")
    index = weight.get_device()
    if (
        not (weight.is_cuda and scale.is_cuda and scale2.is_cuda)
        or scale.get_device() != index
        or scale2.get_device() != index
    ):
        raise ValueError("weight, scale and scale2 must be on the same CUDA device")
    lay = _Layer()
    lay.wmem = weakref.ref(weight.untyped_storage())
    lay.smem = weakref.ref(scale.untyped_storage())
    lay.s2mem = weakref.ref(scale2.untyped_storage())
    lay.pw, lay.ps, lay.ps2 = pw, ps, ps2 = (
        weight.data_ptr(),
        scale.data_ptr(),
        scale2.data_ptr(),
    )
    if ps2 % 2:
        raise ValueError("scale2 is not aligned to its element size")
    lay.k, lay.n, lay.dtype, lay.index = size_k, size_n, dtype, index
    lay.device = _DEVICES.get(index) or _DEVICES.setdefault(index, weight.device)
    # The kernel reads weight in 16 B and scale in 8 B units: other (contiguous) views are staged
    # through aligned copies on every call.
    lay.slow = bool(pw % 16 or ps % 8)
    # x's address & xmask != 0: the call takes the slow path
    lay.xmask = -1 if lay.slow else 15
    # per M: idle launch blocks that hold this layer's pointers
    lay.pools = [None] * (MAX_M + 1)
    shape_key = (size_k, size_n, dtype, index)
    lay.plans = _SHAPES.get(shape_key)
    if lay.plans is None:
        with _BUILD_LOCK:
            lay.plans = _SHAPES.setdefault(shape_key, [None] * (MAX_M + 1))
    key = id(weight)
    lay.wref = weakref.ref(weight, functools.partial(_forget, key))
    lay.sref, lay.s2ref = weakref.ref(scale), weakref.ref(scale2)
    _LAYERS[key] = lay
    return lay


def _validate(lay: _Layer, x, out) -> int:
    """Validate the per-call operands against a freshly validated layer; returns M. Raises on any bad call."""
    if not isinstance(x, _Tensor) or not (out is None or isinstance(out, _Tensor)):
        raise TypeError("x (and out) must be tensors")
    if x.dim() != 2:
        raise ValueError(f"x must be [M, K], got {tuple(x.shape)}")
    size_m, size_k = x.shape
    if x.dtype is not lay.dtype:
        raise TypeError(f"x must have scale2's dtype {lay.dtype}, got {x.dtype}")
    if size_k != lay.k:
        raise ValueError(
            f"x must be [M, {lay.k}] for this weight, got {tuple(x.shape)}"
        )
    if size_m < 1:
        raise ValueError("x must have at least one row")
    if size_m > MAX_M:
        raise NotImplementedError(
            f"cute_nvfp4_decode only handles M <= {MAX_M}, got M={size_m}"
        )
    if not x.is_cuda or x.get_device() != lay.index:
        raise ValueError(
            f"x must be on the weight's device {lay.device}, got {x.device}"
        )
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if out is not None:
        if out.shape != (size_m, lay.n) or out.dtype is not lay.dtype:
            raise ValueError(
                f"out must be {lay.dtype} [{size_m}, {lay.n}], got {out.dtype} {tuple(out.shape)}"
            )
        if not out.is_cuda or out.get_device() != lay.index:
            raise ValueError(
                f"out must be on the weight's device {lay.device}, got {out.device}"
            )
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        lo, hi = out.data_ptr(), out.data_ptr() + 2 * size_m * lay.n
        if lo % 2 or (lo < x.data_ptr() + 2 * size_m * size_k and x.data_ptr() < hi):
            raise ValueError("out must be 2-byte aligned and must not overlap x")
    return size_m


def _prepare_into(lay, x, weight, scale, scale2, out):
    size_m = x.shape[0]
    plan = lay.plans[size_m] or _new_plan(lay, size_m)
    # The kernel reads x and weight in 16 B and scale in 8 B units. A contiguous view at an odd
    # element offset is staged through an aligned copy that run() refreshes (slow, correct).
    stage = []
    xa, wa, sa = x, weight, scale
    if x.data_ptr() % 16:
        xa = torch.empty_like(x)
        stage.append((xa, x))
    if lay.pw % 16:
        wa = torch.empty_like(weight)
        stage.append((wa, weight))
    if lay.ps % 8:
        sa = torch.empty_like(scale)
        stage.append((sa, scale))
    dbg = None
    if plan.cfg.prof:
        dbg = torch.zeros(
            plan.grid * plan.cfg.warps * NSTAMP, dtype=torch.int64, device=x.device
        )
    stamps = dbg.data_ptr() if dbg is not None else 0
    # run() launches on raw addresses: it owns the allocations behind them (not just the tensor
    # objects, which could be rebound to other memory while run() is still in use).
    keep = (x, weight, scale, scale2, out, xa, wa, sa, dbg)
    keep += tuple(t.untyped_storage() for t in keep if t is not None)
    with torch.cuda.device(lay.index):
        kernel = plan.bind(
            xa.data_ptr(),
            wa.data_ptr(),
            sa.data_ptr(),
            lay.ps2,
            out.data_ptr(),
            stamps,
            _raw_stream(lay.index),
            keep,
        )
        if stage:

            def run():
                for dst, src in stage:
                    dst.copy_(src)
                kernel()

            run._keep = kernel._keep
        else:
            run = kernel
        run.dbg = dbg
        run.plan = plan
        if not plan.checked:
            run()  # the first launch of a kernel loads its module: not in the caller's first run()
            plan.checked = True
    return run


def prepare(x, weight, scale, scale2):
    """One-time work for a call; returns ``(out, run)``; ``run()`` relaunches into ``out``.

    ``run()`` is one C call on pre-marshalled arguments plus a look at the launch status; it
    launches on the torch stream that was current when prepare() was called, on the memory the
    five tensors had at that time (which it keeps alive).
    """
    lay = _layer(weight, scale, scale2)
    size_m = _validate(lay, x, None)
    out = torch.empty(size_m, lay.n, dtype=lay.dtype, device=lay.device)
    return out, _prepare_into(lay, x, weight, scale, scale2, out)


def _launch_slow(lay, plan, x, weight, scale, scale2, out):
    """Launch for validated operands that need more than five pointers: operands the kernel cannot
    read in place (staged through aligned copies), profiling builds, a device that is not the
    current one."""
    if plan.cfg.prof:
        _prepare_into(lay, x, weight, scale, scale2, out)()
        return out
    xa = x if x.data_ptr() % 16 == 0 else x.clone()
    wa = weight if lay.pw % 16 == 0 else weight.clone()
    sa = scale if lay.ps % 8 == 0 else scale.clone()
    with torch.cuda.device(lay.index):
        cells, addr, _blk = plan.block(wa.data_ptr(), sa.data_ptr(), lay.ps2)
        _PACK_CALL(cells, 0, xa.data_ptr(), out.data_ptr(), _raw_stream(lay.index), 0)
        plan.capi(addr)
        if cells[_CELL_STATUS]:
            raise _launch_error(cells[_CELL_STATUS])
    return out


def _call_checked(x, weight, scale, scale2, out):
    """nvfp4_linear() with no shortcut: the layer is validated and registered from scratch, then
    x and out against it. Every call the fast path declines ends here, so an invalid call gets
    its precise error and a valid one (first use of an M, operands the registry had other facts
    about, ...) is served whatever was cached."""
    lay = _layer(weight, scale, scale2)
    size_m = _validate(lay, x, out)
    plan = lay.plans[size_m] or _new_plan(lay, size_m)
    if out is None:
        out = torch.empty(size_m, lay.n, dtype=lay.dtype, device=lay.device)
    return _launch_slow(lay, plan, x, weight, scale, scale2, out)


def _launch_first(lay, plan, px: int, po: int) -> None:
    """The first call of a layer object at an M launches from the plan's shared blocks, writing
    all the cells: a block of its own (3 us to make) is only worth it for a layer that is seen
    again -- callers that pass fresh tensor objects on every call never are."""
    pool = plan.pool
    try:
        blk = pool.pop()
    except IndexError:
        blk = plan.block(0, 0, 0)
    cells = blk[0]
    _PACK_LAYER(cells, 32, lay.pw, lay.ps, lay.ps2, 0)
    _PACK_CALL(cells, 0, px, po, _raw_stream(lay.index), 0)
    plan.capi(blk[1])
    status = cells[_CELL_STATUS]
    pool.append(blk)
    if status:
        raise _launch_error(status)


# then the current device has to be checked per call
_MULTI_GPU = torch.cuda.device_count() > 1
_GUARD = _MULTI_GPU  # nvfp4_linear(): some launches may need more than the fast path (also: profiling builds)


def nvfp4_linear(x, weight, scale, scale2, out=None):
    """NVFP4 linear for M <= 8: ``x [M, K] @ dequant(weight, scale, scale2) -> [M, N]``.

    ``out``, if given, is a preallocated contiguous [M, N] tensor of x's dtype on the same
    device; it is written and returned as is. The kernel is launched on torch's current stream.

    Validation. x and out are checked in full on every call: shape, dtype, device, contiguity,
    alignment, and that they do not overlap. weight, scale and scale2 are checked in full when
    a triple of tensor objects is first seen; later calls check that they are the same three
    objects, at the same addresses, on allocations that are still alive (see _Layer) -- anything
    else is validated again from scratch. A call that fails any check is re-examined with
    nothing cached (_call_checked) before it raises.
    """
    try:
        size_m, size_k = x.shape
    except (
        ValueError,
        TypeError,
        AttributeError,
    ):  # not a 2-D tensor: the checks say what is wrong
        return _call_checked(x, weight, scale, scale2, out)
    return linear_sized(x, weight, scale, scale2, out, size_m, size_k)


def linear_sized(x, weight, scale, scale2, out, size_m: int, size_k: int):
    """nvfp4_linear() for a caller that has already unpacked ``size_m, size_k = x.shape``.

    The public entry (cute_nvfp4_marlin) dispatches on M, so it has read x.shape (0.2 us) by
    the time it gets here; this is the whole call path, with every check of nvfp4_linear().
    ``size_m`` and ``size_k`` must be exactly x.shape of a 2-D x; ``out`` may be None.
    """
    lay = _LAYERS.get(id(weight))
    if (
        lay is None
        or lay.wref() is not weight
        or lay.sref() is not scale
        or lay.s2ref() is not scale2
        or lay.wmem() is None
        or lay.smem() is None
        or lay.s2mem() is None
        or weight.data_ptr() != lay.pw
        or scale.data_ptr() != lay.ps
        or scale2.data_ptr() != lay.ps2
    ):
        lay = _layer(weight, scale, scale2)
    try:
        plan = lay.plans[size_m]
    except IndexError:  # M > 8
        return _call_checked(x, weight, scale, scale2, out)
    if (
        plan is None
        or size_k != lay.k
        or x.dtype is not lay.dtype
        or not x.is_cuda
        or not x.is_contiguous()
    ):
        return _call_checked(x, weight, scale, scale2, out)
    px = x.data_ptr()
    if out is None:
        out = torch.empty(size_m, lay.n, dtype=lay.dtype, device=lay.device)
        po = out.data_ptr()
    else:
        po = out.data_ptr()
        if (
            out.shape != plan.oshape
            or out.dtype is not lay.dtype
            or not out.is_cuda
            or not out.is_contiguous()
            or po & 1
            or plan.nobytes < po - px < plan.xbytes
        ):
            return _call_checked(x, weight, scale, scale2, out)
    if px & lay.xmask or _GUARD:
        if _MULTI_GPU and (
            x.get_device() != lay.index or out.get_device() != lay.index
        ):
            return _call_checked(x, weight, scale, scale2, out)
        if px & 15 or lay.slow or plan.slow or torch.cuda.current_device() != lay.index:
            return _launch_slow(lay, plan, x, weight, scale, scale2, out)
    pool = lay.pools[size_m]
    try:
        blk = pool.pop()
    except IndexError:  # no idle block: the layer's second call at this M, or another thread is launching
        if pool is _SEEN_ONCE:
            pool = lay.pools[size_m] = collections.deque()
        blk = plan.block(lay.pw, lay.ps, lay.ps2)
    except AttributeError:  # None: the layer's first call at this M
        lay.pools[size_m] = _SEEN_ONCE
        _launch_first(lay, plan, px, po)
        return out
    cells = blk[0]
    _PACK_CALL(cells, 0, px, po, _raw_stream(lay.index), 0)
    plan.capi(blk[1])
    status = cells[_CELL_STATUS]
    pool.append(blk)
    if status:
        raise _launch_error(status)
    return out


GROUP_RECORD = 8  # int64 words per record of a grouped launch


def prepare_grouped(
    x, records, x_word: int, out_word: int, out, groups: int, projection: int = 0
):
    """M = 1 for up to ``groups`` layers at once, the operands picked on the device.

    ``records`` is an int64 tensor of 64 B records, one per group. Words
    ``3 * projection + (0, 1, 2)`` hold the addresses of a layer's weight, scale and scale2
    (the nvfp4_linear binding, weight and scale 16 B aligned); words ``x_word`` and ``out_word``
    hold the row of ``x [rows, K]`` the group reads and the row of ``out [rows, N]`` it writes.
    A zero weight address is an idle group. Returns ``run()``, which launches on torch's
    current stream. Its arguments never change, so it can be captured in a CUDA graph and
    still follow whatever a kernel earlier in the graph wrote into ``records``.

    Nothing the records point to can be validated here: the caller guarantees the binding.
    """
    size_k, size_n = x.shape[1], out.shape[1]
    dtype, index = x.dtype, x.get_device()
    if dtype not in (torch.bfloat16, torch.float16) or out.dtype is not dtype:
        raise TypeError("cute_nvfp4_decode: x and out must both be fp16 or bf16")
    if records.dtype is not torch.int64 or records.numel() < groups * GROUP_RECORD:
        raise ValueError("cute_nvfp4_decode: records must hold 8 int64 per group")
    if size_k % 64 or size_n % 128:
        raise ValueError("cute_nvfp4_decode: needs K % 64 == 0 and N % 128 == 0")
    for name, tensor in (("x", x), ("out", out), ("records", records)):
        if not tensor.is_contiguous() or tensor.get_device() != index:
            raise ValueError(f"cute_nvfp4_decode: {name} must be contiguous on x's device")
        if tensor.data_ptr() % 16:
            raise ValueError(f"cute_nvfp4_decode: {name} must be 16 B aligned")
    launch, cfg, grid, scalars = _plan_cached(
        1, size_k, size_n, dtype is torch.bfloat16, _sm_count(index), ""
    )
    if launch is not _diag_launch:
        raise NotImplementedError("cute_nvfp4_decode: no grouped kernel for this layer")
    cfg = dataclasses.replace(cfg, grouped=True)
    base = records.data_ptr()
    addresses = (
        x.data_ptr(),
        base + 24 * projection,
        base + 8 * x_word,
        base + 8 * out_word,
        out.data_ptr(),
    )

    def pointers(values):
        return tuple(
            make_ptr(Int64, value, cute.AddressSpace.gmem, assumed_align=8)
            for value in values
        )

    # (prob_m and prob_k are not launcher arguments here)
    tail = (Int32(groups), *scalars[2:])
    fake = (*pointers(64 * (i + 1) for i in range(5)), *tail)
    compiled = _compiled(
        _diag_group_launch, cfg, (*fake, cuda_driver.CUstream(_FAKE_STREAM)), index
    )
    args = (*pointers(addresses), *tail)

    def run():
        compiled(*args, cuda_driver.CUstream(_raw_stream(index)))

    run._keep = (x, records, out)
    return run


def debug_knobs(spec: str = "", compile_opts: str | None = None) -> None:
    """Development hook (sweeps, profiling builds, SASS dumps); production never calls it.

    ``spec`` overrides the geometry of plans made from now on, e.g.
    ``"warps=12,depth=4,grid=86,flavor=generic,prof=1"``; ``compile_opts`` is passed to
    cute.compile (e.g. ``"--keep-sass --dump-dir=..."``) and bypasses the on-disk cache.
    Launches prepared earlier keep the kernel they were prepared with.
    """
    global _DEBUG, _COMPILE_OPTS, _GUARD
    prof = _parse_knobs(spec)[4]
    with _BUILD_LOCK:
        _DEBUG, _COMPILE_OPTS = spec, compile_opts
        _GUARD = _MULTI_GPU or (bool(prof) and prof != "0")
        _SHAPES.clear()
        _LAYERS.clear()


def profile(x, weight, scale, scale2, launches: int = 20):
    """Per-warp phase stamps of a profiling build (after ``debug_knobs("prof=1")``).

    Returns an int64 tensor [launches, grid, warps, NSTAMP]: SM cycle counter at kernel entry,
    step 0 requested, prologue requested, (unused), first data landed, refill loop done, last
    step done, at the reduction barrier, end; then %globaltimer (ns) at entry.
    """
    _out, run = prepare(x, weight, scale, scale2)
    if run.dbg is None:
        raise RuntimeError(
            'profile() needs a profiling build: call debug_knobs("prof=1") first'
        )
    acc = []
    for _ in range(launches):
        run()
        torch.cuda.synchronize()
        acc.append(run.dbg.clone())
    return torch.stack(acc).view(launches, -1, run.plan.cfg.warps, NSTAMP)
