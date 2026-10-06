"""NVFP4 Marlin linear for M >= 7 (batched decode and prefill): K-striped tiles.

    out = nvfp4_linear(x, weight, scale, scale2, out=None)      # see marlin_utils

The BATCH kernel of the package. cute_nvfp4_marlin, the public entry, sends
every M >= 9 here, and M = 7 and 8 unless the layer is wide (N >= 7168);
below that the column-owning cute_nvfp4_decode is faster (README.md has the
crossover data). The module is complete for every M >= 7 on its own.

A CuTe DSL tensor-core GEMM over Marlin-packed NVFP4 weights for batched
decode / prefill token counts. It keeps Marlin's fragment-ready weight stream
and its K-striping across CTAs, and changes what made the original slow here.

What bounds this regime on an A100 (measured during development): not the HBM
stream alone. An SM delivers ~2.5 "instruction units" per cycle with an
HMMA.16816 worth ~4.5 of them, so dequant and MMA add up; a lone warp is far
slower than that (~3 ns per instruction, ~20 cycles per predicate + branch);
cp.async is taken at ~11 ns per 512 B per SM and the issuing warp waits its
turn; memory answers an SM in request order, so anything asked for while cold
fetches are in flight comes back behind them; a partial handed from one SM to
another costs 2-3 us end to end (store, fence, flag, poll, fetch). The design
follows from those numbers:

  - Three kinds of warps per CTA, one CTA per SM, meeting at named barriers.
    Four MMA warps only read smem, dequantize and multiply. IO warps issue
    every cp.async of the pipeline: fetches no longer park the MMA warps (that
    time used to add to the compute), and their code is straight-line. One is
    enough up to 64 rows; a taller A tile is more than a lone warp can request
    in a unit's time (it issues one 512 B cp.async every ~14 ns, and a
    128-row unit takes 37 of them), so above that two warps share the A rows
    (M = 128: 28.4 -> 26.1 us at K = 4096). One X warp does everything that
    crosses CTAs.
  - Dequant, in registers, with sglang's exact bit patterns: the e2m1 nibble
    pair goes to sign | code3 in the activation dtype (bf16: one mask, one
    multiply by 2^12 + 2^6, one mask), the block-scale bytes are decoded two
    per register, and weight x scale is the native packed multiply
    (fma.rn.bf16x2 / mul.rn.f16x2) with the scale lane picked by an operand
    modifier. All B fragments of a k16 row are decoded before its MMAs, so
    the decodes interleave instead of chaining through two registers. B is
    bit-identical to sglang's for every code and scale byte, subnormal block
    scales included.
  - Geometry per M bucket. A warp owns 16*mb rows (mb = 1 .. 8) of 32
    columns, so one dequantized B fragment feeds mb MMAs and M = 128 is one
    problem (not two 64-row problems that each re-dequantize the layer); 2 x 2
    warps split a 64-column tile along N and a 64-deep stage along K. The
    step is one copy of the code with the pipeline stage as a loop-carried
    smem offset (unrolling it over the stages cost 50+ registers). Measured
    at 128 rows: the loop is at what an SM delivers for this instruction mix
    (taking the MMAs out saves 1.3 of 26 us, eight MMA warps are slower with
    or without a second dequant), and with two IO warps the A re-reads cost
    1.2 us.
  - M > 128: row chunks. The rows are cut into ceil(M / 128) chunks of equal
    height (the last one is moved up to end at M and leaves the rows it then
    shares with the chunk above to that one, so every chunk is a full tile
    and no A row needs a predicate), and "column tile of a row chunk" takes
    the place of "column tile" in the stripe schedule: one launch, one
    continuous pipeline per CTA across chunks, the same hand-over. A stripe
    is then longer than a column, so nobody waits for a partial that is
    computed last. These kernels are their own specialization; the ones for
    M <= 128 carry none of it.
  - One continuous cp.async pipeline per CTA: stages of [A tile | packed B
    rows | scale rows]; A through a swizzled tile + ldmatrix, B and scale
    bytes copied verbatim and read straight into registers. 3 to 8 stages,
    deepest at small M: the cold stream runs at about 3/4 of what HBM
    delivers, and its latency exceeds two units of lead when the units are
    short. With enough lead the 64-column tile beats a 128-column one at
    every M (twice the CTAs finish a column, half the partial bytes).
  - Stripes. The (k-tile, n-tile) units, column-major, are cut into one
    contiguous stripe per CTA by a host-built table (16 B per CTA, read once).
    A CTA walks its stripe from the end: the head of the later column first,
    the tail of the earlier column last. Stripes are not equal: a CTA that
    only computes a partial gets fewer units than one that finishes a column,
    so that its partial is on its way while the finisher still computes. A
    layer with few column tiles gets fewer CTAs than SMs (bounded fan-in).
  - Lock-free cross-CTA reduction. Every contributor to a column except the
    one holding its bottom publishes its fp32 partial to its own scratch slot:
    the threads that store it fence -- every one of them -- and after a
    barrier one of them release-stores the launch's generation number into
    the CTA's flag. (Mid-stripe that is the X warp, working from smem while
    the MMA warps go on; at the end of a stripe the MMA threads do it, four
    warps store faster than one.) On the CTA holding the bottom, the X warp
    acquire-polls the flags of its contributors (lower-index CTAs only) from
    the start and cp.asyncs each one's partial into smem as soon as its flag
    is up, usually before the MMA warps are done; they add them in a fixed
    order and write the output. Flags are compared against the generation:
    there is no state to restore between launches and a stale flag can never
    match.
  - Epilogue with every MMA thread working: each thread sums its share over
    the k-split warps (if any) and the fetched partials, converts two fp32
    sums at a time to dtype and applies scale2 in dtype (as sglang does; both
    packed, one instruction each), and the tile goes out through a row-major
    smem staging as coalesced 16 B stores. In an epilogue a warp runs alone
    on its scheduler: every instruction per thread there is ~3 ns of latency.
  - Host: one launch per call through the JIT-ed C entry point with an
    argument block this module owns (raw addresses and sizes). Every call
    checks all five tensors by value -- Python type, dtype, device, sizes and
    strides, in one C call against what the shape has seen pass the full
    checks -- and launches with the addresses it reads from them then, so
    nothing rests on a tensor object still being what it was at an earlier
    call. Launch state is per (device, K, N, dtype), not per layer; binding a
    call is one struct.pack_into of a block kept per shape, row split and
    thread. nvfp4_linear(out=) is about 2.9 us of host time above a prepared
    run() (4.4 us), which a kernel of 7.5 us or more hides. Scratch
    (partials + flags) is per (device, stream): launches on one stream never
    overlap, launches on different streams never share scratch. A launch goes
    to the stream that is current when it is made. Compiled kernels are
    cached on disk (~/.cache/nvfp4_marlin_cute, or NVFP4_CUTE_CACHE), keyed on
    this file's hash, the launcher, the geometry, the GPU architecture and the
    toolchain versions; a cache file carries the sha256 of its object and is
    rebuilt, never linked, if it does not verify (the scheme is shared with
    cute_nvfp4_decode: see the block above _executor).

Limits: M < 7 raises NotImplementedError (cute_nvfp4_decode covers those; the
specialization for M = 7 fetches rows 0 .. 3 of x without a predicate, the
one for M = 8 .. 16 rows 0 .. 7).
The tensors must live on the current CUDA device (the compiled module is
loaded into the current context). x, weight, scale and out must be contiguous
(ValueError otherwise); if one of them is not 16-byte aligned the call still
works, through an aligned copy of that tensor per call. An x that is 16 but
not 32 B aligned is read in place at a price (+1 us at M = 16 .. 32, +4 us at
M = 64) and, from 640 KB on, through an aligned copy (see X32_BYTES); torch's
own allocations and row slices of them are 32 B aligned. A launch goes to the
current stream and reads x when it runs: an x that is still being produced
on another stream must be synchronized by the caller, as for any kernel.
A prepared run() is not re-entrant, and it is bound to the memory its tensors
had at prepare(); nvfp4_linear() keeps a launch block per thread.
An X warp that polls a flag 2^20 times (0.3-0.5 s; it cannot happen while
launches that share scratch are serialized, which per-stream scratch
guarantees) gives up rather than hang the GPU: that output tile is written as
NaN and the launch's generation number is stored in the scratch's error word.
That launch returns normally. The host then learns of it without ever
synchronizing: every 256 launches of the process it copies the error words
to page-locked host memory (asynchronously, on a side stream) and looks at
what the previous copies brought back, so a LATER call into this module --
any layer, any thread, run() or nvfp4_linear(), at most 512 launches after
the timed-out one has run -- raises RuntimeError instead of launching, once
per timeout. ``poll_timeouts()`` reads the words synchronously. (Having the
kernel store to host memory itself would make it the very next call, but any
edit to the kernel reshuffles ptxas' schedule of the 16-row loop: measured
+0.6 / +0.35 us at M = 8 .. 16, K = 4096 / 2048.)

DSL pitfalls met on the way: a (V, M) two-mode register tensor crashes
retile / make_rmem_tensor with std::bad_variant_access (keep a unit third
mode); Uint8 -> Int32 `.to()` sign-extends; thread-private smem staging must
be lane-contiguous (a 64 B lane stride gave 16-way bank conflicts and a 4x
slower fetch); a runtime integer division in the loop is a subroutine call;
when a thread needs more than 255 registers ptxas silently falls back to 32
and spills everything (grep the SASS for SpillRefill); after any change to an
IO warp's loop, look at its run of LDGSTS for IMAD.WIDE (see _run_io);
cp.async groups that are not committed are not waited for by wait_group;
inside a dynamic if / while, never assign through an expression that mentions
the Constexpr geometry (``view(g.x)[0] = v``): the region then treats ``g`` as
written and its fields come back as runtime values (bind the view to a name
first).
"""

import bisect
import ctypes
import functools
import hashlib
import itertools
import os
import re
import shutil
import struct
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cuda.bindings.driver as cuda_driver  # ty: ignore[unresolved-import]
import cutlass
import torch
from cutlass import Float32, Int32, Int64, cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T

MIN_M = 7  # fewer rows are the decode kernel's business (cute_nvfp4_decode)
MAX_ROWS = 128  # rows of the tallest tile: a larger M is cut into row chunks
SPIN_LIMIT = 1 << 20  # flag polls before a finisher gives up (never hang the GPU)

# ---------------------------------------------------------------------------
# Inline-PTX primitives (the DSL has no packed 16-bit-float multiply)
# ---------------------------------------------------------------------------


def _asm_i32(asm: str, *args) -> Int32:
    r = llvm.inline_asm(
        T.i32(),
        [Int32(a).ir_value() for a in args],
        asm,
        "=r" + ",r" * len(args),
        has_side_effects=False,
        is_align_stack=False,
    )
    return Int32(r)


_NEG0 = 0x80008000 - (1 << 32)  # -0.0 in both lanes: fma(a, b, -0.0) == a * b


def _mul_lane(w, s, hi: bool, bf16: bool) -> Int32:
    """Both lanes of ``w`` times lane ``hi`` of ``s`` (packed fp16 / bf16).

    Written as a b16 re-pack inside the asm block, the lane broadcast becomes
    an operand modifier of the multiply (HFMA2 ..., R.H0_H0 / .H1_H1) instead
    of an instruction.
    """
    h = "hi" if hi else "lo"
    pre = "{ .reg .b16 lo, hi; .reg .b32 t; mov.b32 {lo, hi}, $2; mov.b32 t, {%s, %s}; " % (h, h)  # fmt: skip  # noqa: UP031 -- PTX text with literal braces
    if bf16:
        return _asm_i32(pre + "fma.rn.bf16x2 $0, $1, t, $3; }", w, s, _NEG0)
    return _asm_i32(pre + "mul.rn.f16x2 $0, $1, t; }", w, s)


def _cvt_pair(lo, hi, bf16: bool) -> Int32:
    """Two fp32 values -> one register of two packed fp16 / bf16 (round to
    nearest even): 1 instruction (F2FP.PACK_AB) instead of 2 conversions."""
    r = llvm.inline_asm(
        T.i32(),
        [Float32(hi).ir_value(), Float32(lo).ir_value()],
        "cvt.rn.%s.f32 $0, $1, $2;" % ("bf16x2" if bf16 else "f16x2"),
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Int32(r)


def _mul_pair(a, b, bf16: bool) -> Int32:
    """Lane-wise product of two packed fp16 / bf16 pairs, rounded once."""
    if bf16:
        return _asm_i32("fma.rn.bf16x2 $0, $1, $2, $3;", a, b, _NEG0)
    return _asm_i32("mul.rn.f16x2 $0, $1, $2;", a, b)


_SIGN = 0x80008000 - (1 << 32)  # both sign bits, as a signed Int32 constant
_F16_MAG = 0x0E000E00  # fp16: e2m1 [e1 e0 m] at bits 11..9
_F16_SCL = 0x7F807F80  # fp16: scale byte at bits 14..7
_BF16_W = 0x81C081C0 - (1 << 32)  # bf16: sign | [e1 e0 m] at bits 8..6

# ---------------------------------------------------------------------------
# Geometry: everything the kernel is specialized on. M, K, N and the stripe
# schedule are runtime values.
# ---------------------------------------------------------------------------

FB_MAX = 4  # cross-CTA partials fetched per cp.async batch
MAX_SMEM = 163 * 1024  # sm80 opt-in shared memory per CTA (166,912 B)

# Layout of the int32 flags tensor of a scratch (one per device and stream)
FLAG_CAP = 1024  # flags [0, FLAG_CAP): one per CTA
ERR_WORD = FLAG_CAP  # generation of the last launch whose flag poll timed out
DBG_OFF = 2 * FLAG_CAP  # profiling builds only: 3 x 16 timestamps per CTA (256 B apart)
FLAG_WORDS = DBG_OFF + 64 * FLAG_CAP


@dataclass(frozen=True)
class Geom:
    bf16: bool
    mb: int = 1  # m16 blocks per warp
    nb: int = 8  # n8 blocks per warp: 8 (a whole 64-column unit) or 4 (half of one)
    wm: int = 1  # warps along M (16 * mb rows each)
    wn: int = 2  # warps along N (8 * nb columns each)
    wk: int = 2  # warps along K (kpw k16 rows each per stage); power of two
    kpw: int = 2  # k16 rows per warp per stage (wk * kpw == 4: a stage is 64 of K)
    stages: int = 4  # cp.async pipeline depth
    m_min: int = (
        1  # fewest rows this specialization serves (rows below it need no predicate)
    )
    nio: int = 1  # IO warps: a unit's cp.async are split between them
    rc: int = 0  # 1: M is cut into row chunks (their own specialization: the others pay nothing)
    dbg: int = 0  # 1: record %globaltimer phase stamps (profiling builds only)

    @property
    def dtype(self):
        return cutlass.BFloat16 if self.bf16 else cutlass.Float16

    @property
    def tile_m(self) -> int:
        return 16 * self.mb * self.wm

    @property
    def tile_n(self) -> int:
        return 8 * self.nb * self.wn

    @property
    def kb(self) -> int:  # k16 rows per stage
        return self.wk * self.kpw

    @property
    def ks(self) -> int:  # K extent of one stage
        return 16 * self.kb

    @property
    def mma_threads(self) -> int:
        return 32 * self.wm * self.wn

    @property
    def threads(self) -> int:
        return self.mma_threads * self.wk

    @property
    def groups(self) -> int:  # 16 B accumulator groups (one MMA C fragment) per thread
        return self.nb * self.mb

    @property
    def block_threads(self) -> int:  # + the IO warps and the X warp
        return self.threads + 32 * self.nio + 32

    @property
    def io_bs(self) -> int:  # cp.async instructions of a unit's packed B and scale rows
        return self.kb * (self.tile_n // 64) + -(-(self.kb * self.tile_n // 16) // 32)

    def io_part(self, i: int) -> int:
        """The IO warp that fetches A row group ``i`` (four rows): warp 0 takes
        B, the scales and the first row groups, an even split of the instructions."""
        return (i + self.io_bs) * self.nio // (self.tile_m // 4 + self.io_bs)

    @property
    def rec_words(self) -> int:  # int32 words of a CTA's stripe record
        return 8 if self.rc else 4

    # ---- smem plan (bytes): pipeline = A | B | S, the epilogue scratch E
    # (k-warp reduction staging, output staging), the fetched cross-CTA partials
    # F, and one status word. ----
    @property
    def sa_stage(self) -> int:
        return self.tile_m * self.ks * 2

    @property
    def sb_stage(self) -> int:
        return self.kb * self.tile_n * 8

    @property
    def ss_stage(self) -> int:
        return self.kb * self.tile_n

    @property
    def stage_bytes(self) -> int:  # one pipeline stage: A | B | S of one unit
        return self.sa_stage + self.sb_stage + self.ss_stage

    @property
    def pipe_bytes(self) -> int:
        return self.stages * self.stage_bytes

    @property
    def part_bytes(self) -> int:  # one CTA's partial: every accumulator of one k-warp
        return self.tile_m * self.tile_n * 4

    @property
    def out_stride(
        self,
    ) -> int:  # output staging row (bytes), padded: conflict-free STS
        return self.tile_n * 2 + 16

    @property
    def cpt(self) -> int:  # 16 B accumulator groups each thread sums in the epilogue
        return self.groups // self.wk

    @property
    def dslot_bytes(self) -> int:  # one k-warp's dump: the groups other k-warps sum
        return self.part_bytes - self.part_bytes // self.wk

    @property
    def epi_bytes(self) -> int:
        return max(self.wk * self.dslot_bytes, self.tile_m * self.out_stride)

    @property
    def fb(self) -> int:  # cross-CTA partials staged per batch
        room = MAX_SMEM - 16 - self.pipe_bytes - self.epi_bytes
        return max(1, min(FB_MAX, room // self.part_bytes))

    @property
    def work_bytes(self) -> int:  # everything but the status word
        return self.pipe_bytes + self.epi_bytes + self.fb * self.part_bytes

    @property
    def smem_bytes(self) -> int:
        return self.work_bytes + 16

    @property
    def out_chunks(self) -> int:  # 16 B output chunks per thread (ceil)
        return -(-(self.tile_m * self.tile_n // 8) // self.threads)

    def check(self) -> None:
        assert self.mb >= 1 and self.wm >= 1 and self.wn >= 1 and self.kpw >= 1
        assert self.nb in (4, 8)
        assert self.tile_n % 64 == 0, "column tiles are whole 64-column units"
        assert self.wk >= 1 and self.wk & (self.wk - 1) == 0, (
            "wk must be a power of two"
        )
        assert self.ks == 64, "a stage is one K % 64 == 0 step of the binding"
        assert self.kpw % 2 == 0, "the raw-word double buffer flips once per k16 row"
        assert self.stage_bytes % 128 == 0
        assert self.stages >= 2
        assert self.nio in (1, 2)
        assert self.block_threads <= 1024
        assert self.groups % self.wk == 0
        assert self.smem_bytes <= MAX_SMEM, (
            f"geometry needs {self.smem_bytes} B of smem"
        )


def _make_sA_layout(g: Geom):
    """Canonically swizzled A tile of one stage: (tile_m, ks), 128 B rows."""
    atom_outer = cute.make_layout((8, g.ks), stride=(g.ks, 1))
    atom = cute.make_composed_layout(cute.make_swizzle(3, 3, 3), 0, atom_outer)
    return cute.tile_to_shape(atom, (g.tile_m, g.ks), (0, 1))


def _make_tiled_mma(g: Geom):
    """(wm x wn) warps; each owns 16*mb contiguous rows x 8*nb contiguous columns."""
    op = cute.nvgpu.warp.MmaF16BF16Op(g.dtype, Float32, (16, 8, 16))
    perm_m = cute.make_layout((16, g.wm, g.mb), stride=(1, 16 * g.mb, 16))
    perm_n = cute.make_layout((8, g.wn, g.nb), stride=(1, 8 * g.nb, 8))
    return cute.make_tiled_mma(
        op,
        atom_layout_mnk=(g.wm, g.wn, 1),
        permutation_mnk=(perm_m, perm_n, None),
    )


def _smem_view(dtype, addr, n, align):
    return cute.make_tensor(
        cute.make_ptr(dtype, addr, cute.AddressSpace.smem, assumed_align=align),
        cute.make_layout((n,)),
    )


def _gmem_view(dtype, addr, n, align):
    return cute.make_tensor(
        cute.make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=align),
        cute.make_layout((n,)),
    )


def _cp_async(nbytes, src_addr, dst_addr, cp_size=None):
    """One 16 B cp.async, L2 only (.cg). With ``cp_size`` = 0 the source is not
    read and the destination is zero-filled."""
    assert nbytes == 16
    cute.arch.cp_async_shared_global(
        cute.make_ptr(Int32, dst_addr, cute.AddressSpace.smem, assumed_align=16),
        cute.make_ptr(Int32, src_addr, cute.AddressSpace.gmem, assumed_align=16),
        16,
        "cg",
        cp_size=cp_size,
    )


def _stamp(g, ts, i):
    """Profiling builds: note %globaltimer_lo at phase ``i`` (flushed at kernel exit)."""
    if g.dbg:
        ts[i] = cute.arch.globaltimer_lo()


# Three kinds of warps share a CTA, and they meet at named barriers
# (bar.sync id, thread count):
#   MMA warps  [0, threads)        LDS / ldmatrix / dequant / MMA, epilogue math
#   IO warps   the next nio warps  every cp.async of the A | B | S pipeline
#   X warp     the last warp       the cross-CTA protocol: publishes this CTA's
#                                  partial, polls flags, fetches others' partials
BAR_STEP = 1  # MMA + IO: one per unit of the stripe
BAR_MMA = 2  # MMA warps only: phases of the epilogue
BAR_X = 3  # MMA + X: hand-over of a partial (either direction)
BAR_W = 4  # the X warp alone: orders its lanes


def _bar(g, which):
    n = {
        BAR_STEP: g.threads + 32 * g.nio,
        BAR_MMA: g.threads,
        BAR_X: g.threads + 32,
        BAR_W: 32,
    }
    cute.arch.barrier(barrier_id=which, number_of_threads=n[which])


# Scheduler state of the MMA warps (Int32 registers): unit, stage offset,
# column tile, k-tile and (row-chunked kernels) row chunk ...
ST_P, ST_SO, ST_CC, ST_CK, ST_CR = range(5)
# ... and an IO warp's running Int64 gmem pointers of the next unit to fetch
# (per lane), plus what a column change adds to B and S and (row-chunked
# kernels) what the next change of row chunk takes off A.
SP_A, SP_B, SP_S, SP_WB, SP_WS, SP_WA = range(6)

# ---------------------------------------------------------------------------
# Pipeline: the IO warp fetches, the MMA warps compute
# ---------------------------------------------------------------------------


@cute.jit
def _io_fetch(
    g: cutlass.Constexpr,
    part: cutlass.Constexpr,  # which IO warp this is
    so: Int32,  # smem byte offset of the stage to fill
    kt: Int32,  # k-tile of the unit
    k_tiles: Int32,
    size_k: Int32,
    size_n: Int32,
    mrow: Int32,
    a_dst0: Int32,
    a_dst1: Int32,
    sB_lane: Int32,
    sS_lane: Int32,
    sp: cute.Tensor,
):
    """IO warp ``part``: cp.async its share of the next unit into the smem
    stage at ``so`` -- A row groups, packed B and scale bytes -- then step the
    pointers to the unit before it (in the same row chunk).

    Straight-line code: every instruction moves 32 lanes x 16 B to
    lane-contiguous smem, every address is a per-lane running pointer plus an
    independent multiple of K or N (a lone warp is slow on dependent chains
    and on branches).
    """
    a = sp[SP_A]
    b = sp[SP_B]
    sc = sp[SP_S]
    a_d0 = a_dst0 + so
    a_d1 = a_dst1 + so
    b_d = sB_lane + so
    s_d = sS_lane + so
    # ---- A: lane = (row % 4, 16 B chunk c). Row r, chunk c of a stage sits at
    # 128 * r + 16 * (c ^ (r & 7)) (Sw<3,3,3>): a_dst0 / a_dst1 hold that for
    # rows lane / 8 of an even / odd group of four rows. Rows >= M are
    # zero-filled without reading the source.
    for i in cutlass.range_constexpr(g.tile_m // 4):
        if cutlass.const_expr(g.io_part(i) != part):
            continue
        src = a
        if cutlass.const_expr(i > 0):
            src = a + Int64(size_k) * Int64(8 * i)
        dst = (a_d1 if i % 2 else a_d0) + 512 * i
        if cutlass.const_expr(4 * i + 3 < g.m_min):
            _cp_async(16, src, dst)
        else:
            n = Int32(0)
            if 4 * i < mrow:
                n = Int32(16)
            _cp_async(16, src, dst, cp_size=n)
    # ---- B: kb k16 rows x (tile_n / 64) units of 512 B
    upr = g.tile_n // 64
    for row in cutlass.range_constexpr(g.kb if part == 0 else 0):
        src = b
        if cutlass.const_expr(row > 0):
            src = b + Int64(size_n) * Int64(8 * row)
        for u in cutlass.range_constexpr(upr):
            _cp_async(16, src + Int64(512 * u), b_d + 512 * (row * upr + u))
    # ---- scales: kb rows x tile_n bytes, 16 B per lane
    n_s = g.kb * g.tile_n // 16
    rows_per = 32 * 16 // g.tile_n  # k16 rows one instruction covers (when n_s >= 32)
    for v in cutlass.range_constexpr(-(-n_s // 32) if part == 0 else 0):
        src = sc
        if cutlass.const_expr(v > 0):
            src = sc + Int64(size_n) * Int64(v * rows_per)
        _cp_async(16, src, s_d + 512 * v)
    # ---- units are walked in descending column-major order
    if kt == 0:
        sp[SP_A] = a + Int64(k_tiles - 1) * Int64(2 * g.ks)
        sp[SP_B] = b + sp[SP_WB]
        sp[SP_S] = sc + sp[SP_WS]
    else:
        sp[SP_A] = a - Int64(2 * g.ks)
        sp[SP_B] = b - Int64(size_n) * Int64(8 * g.kb)
        sp[SP_S] = sc - Int64(size_n) * Int64(g.kb)


@cute.jit
def _io_chunk_up(size_m: Int32, size_k: Int32, size_n: Int32, sp: cute.Tensor):
    """An IO warp's pointers have just stepped from the first unit of a row
    chunk to "the column tile before it": make that the last column tile of the
    row chunk above."""
    sp[SP_A] = sp[SP_A] - sp[SP_WA]
    sp[SP_WA] = Int64(size_k) * Int64(2 * size_m)  # (only the last chunk is closer)
    sp[SP_B] = sp[SP_B] + Int64(size_n) * Int64(8)
    sp[SP_S] = sp[SP_S] + Int64(size_n)


@cute.jit
def _run_io(
    g: cutlass.Constexpr,
    part: cutlass.Constexpr,
    slen: Int32,
    fk: Int32,  # k-tile of the stripe's last unit (fetched first)
    left0: Int32,  # row-chunked kernels: units from that one to the first of its row chunk
    upc: Int32,  # ... and units per row chunk
    k_tiles: Int32,
    size_m: Int32,
    size_k: Int32,
    size_n: Int32,
    mrow: Int32,
    a_dst0: Int32,
    a_dst1: Int32,
    sB_lane: Int32,
    sS_lane: Int32,
    sp: cute.Tensor,
    ts: cute.Tensor,
):
    """An IO warp: one continuous cp.async pipeline over the whole stripe.

    It knows nothing about columns or epilogues: before the barrier of unit p
    it issues unit p + stages - 1 into the stage that the previous barrier
    freed and waits for unit p + 1 to land.

    An SM takes cp.async at ~11 ns per 512 B instruction and the issuing warp
    waits its turn. Issued by the MMA warps (all at the same point of a unit)
    that time added to the compute; here it overlaps with it.

    Row-chunked kernels walk the row chunks from the last one up. The change
    of chunk stays out of the per-unit code: the loop runs from one change to
    the next. A lone warp pays ~20 cycles per branch, and worse, with a real
    branch in the per-unit path ptxas stops keeping the A row step (8 * K) in
    a register and re-materializes it in every source address (a chain of
    IMAD.WIDE in place of IADD3 + IMAD.X: fewer instructions, +1.3 us at
    M = 128, +0.5 us at M = 16).
    """
    kt = fk
    left = left0
    for i in cutlass.range_constexpr(g.stages - 1):
        if i < slen:
            _io_fetch(
                g, part, Int32(i * g.stage_bytes), kt, k_tiles, size_k, size_n, mrow,
                a_dst0, a_dst1, sB_lane, sS_lane, sp,
            )  # fmt: skip
            kt = kt - 1
            if kt < 0:
                kt = k_tiles - 1
            if cutlass.const_expr(g.rc):
                left = left - 1
                if left == 0:
                    _io_chunk_up(size_m, size_k, size_n, sp)
                    left = upc
        cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(g.stages - 2)
    _stamp(g, ts, 1)
    _bar(g, BAR_STEP)
    so = Int32((g.stages - 1) * g.stage_bytes)  # the stage unit 0's barrier frees
    p = Int32(0)
    if cutlass.const_expr(g.rc):
        nf = slen - (g.stages - 1)  # units that are fetched from inside the loop
        while p < nf:
            stop = p + left
            if stop > nf:  # noqa: PLR1730 -- DSL regions: keep the shape of the if
                stop = nf
            left = left - (stop - p)
            while p < stop:
                _io_fetch(
                    g, part, so, kt, k_tiles, size_k, size_n, mrow,
                    a_dst0, a_dst1, sB_lane, sS_lane, sp,
                )  # fmt: skip
                kt = kt - 1
                if kt < 0:
                    kt = k_tiles - 1
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(g.stages - 2)
                _bar(g, BAR_STEP)
                so = so + g.stage_bytes
                if so == g.pipe_bytes:
                    so = Int32(0)
                p = p + 1
            if left == 0:
                _io_chunk_up(size_m, size_k, size_n, sp)
                left = upc
        while p < slen:  # the pipeline runs empty
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(g.stages - 2)
            _bar(g, BAR_STEP)
            p = p + 1
    else:
        while p < slen:
            if p + (g.stages - 1) < slen:
                _io_fetch(
                    g, part, so, kt, k_tiles, size_k, size_n, mrow,
                    a_dst0, a_dst1, sB_lane, sS_lane, sp,
                )  # fmt: skip
                kt = kt - 1
                if kt < 0:
                    kt = k_tiles - 1
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(g.stages - 2)
            _bar(g, BAR_STEP)
            so = so + g.stage_bytes
            if so == g.pipe_bytes:
                so = Int32(0)
            p = p + 1
    _stamp(g, ts, 4)


def _load_regs(g, b_addr, s_addr, buf, braw, sraw):
    """smem -> registers for one k16 row: the lane's raw packed words and scale bytes."""
    nw, nsw = g.nb // 2, g.nb // 4
    cute.autovec_copy(_smem_view(Int32, b_addr, nw, 4 * nw), braw[None, buf])
    cute.autovec_copy(_smem_view(Int32, s_addr, nsw, 4 * nsw), sraw[None, buf])


def _dequant_mma(g, buf, braw, sraw, tCrA, tCrB, tCrC, tiled_mma):
    """Dequantize one k16 row of the warp's columns in registers and run its
    nb * mb MMAs. The bit patterns are sglang's (dequant.h, kFE2M1f +
    dequant_fp8_scales), so B is the same number for every input.

    A packed word q of n16 block j holds, for both n8 halves ``blk``, the
    lane's two B registers; nibble i = 2 * blk + reg sits at bits 4i + 3..4i of
    both 16-bit halves (the (k, k + 1) pair).

      weight  bf16: sign -> bit 15, [e1 e0 m] -> bits 8..6  (e2m1 * 2^-126):
                    nibble * (2^12 + 2^6) puts both copies in place at once
              fp16: sign -> bit 15, [e1 e0 m] -> bits 11..9 (e2m1 * 2^-14)
      scale   bf16: byte bit 7 -> bit 14, bits 6..0 -> bits 10..4
              fp16: byte -> bits 14..7
    Scale bytes (0, 2) of a word are n8 blocks 4h, 4h + 1 and bytes (1, 3) are
    4h + 2, 4h + 3, so each decoded register holds two scales, one per lane.
    """
    sc = [None] * (g.nb // 2)
    for h in range(g.nb // 4):
        sq = sraw[h, buf]
        if g.bf16:
            sc[2 * h] = ((sq & 0x007F007F) << 4) | ((sq << 7) & 0x40004000)
            sc[2 * h + 1] = ((sq >> 4) & 0x07F007F0) | ((sq >> 1) & 0x40004000)
        else:
            sc[2 * h] = (sq << 7) & _F16_SCL
            sc[2 * h + 1] = (sq >> 1) & _F16_SCL
    # All B registers of the row first, then the MMAs: with every fragment
    # live at once ptxas gives each its own register and interleaves the
    # (independent) decodes instead of chaining them through two registers.
    frag = []
    for j in range(g.nb // 2):
        q = braw[j, buf]
        q8 = q >> 8
        for blk in range(2):
            nn = 2 * j + blk
            pat = [None, None]
            for r in range(2):
                if g.bf16:
                    src = q if blk == 0 else q8
                    if r == 0:
                        pat[r] = ((src & 0x000F000F) * 0x1040) & _BF16_W
                    else:
                        pat[r] = ((src & 0x00F000F0) * 0x0104) & _BF16_W
                else:
                    sh = 12 - 4 * (2 * blk + r)
                    v = (q << sh) if sh else q
                    pat[r] = (v & _SIGN) | ((v >> 3) & _F16_MAG)
            s = sc[nn // 2]
            frag.append(
                (
                    _mul_lane(pat[0], s, nn % 2 == 1, g.bf16),
                    _mul_lane(pat[1], s, nn % 2 == 1, g.bf16),
                )
            )
    for nn in range(g.nb):
        b_i32 = cute.recast_tensor(tCrB[nn], Int32)  # (2, 1) packed words
        b_i32[0] = frag[nn][0]
        b_i32[1] = frag[nn][1]
        cute.gemm(tiled_mma, tCrC[nn], tCrA, tCrB[nn], tCrC[nn])


def _stage_a(g, thr_s2r_A, sA_layout, addr):
    """This thread's ldmatrix view of the A tile of the stage at smem ``addr``:
    (CPY, CPY_M, kb)."""
    ptr = cute.make_ptr(g.dtype, addr, cute.AddressSpace.smem, assumed_align=16)
    return thr_s2r_A.partition_S(cute.make_tensor(ptr, sA_layout))


def _mma_step(g, so, so_n, smem_base, kblk0, sB_rd, sS_rd, thr_s2r_A, sA_layout, tCrA_v, tCrA, braw, sraw, tCrB, tCrC, s2r_A, tiled_mma):  # fmt: skip
    """MMA warps, one unit of the stripe (in the smem stage at ``so``): kpw k16
    rows each; the barrier before the last row hands over the next unit (at
    ``so_n``) and frees this stage for the IO warp."""
    for kk in range(g.kpw):
        buf = kk % 2
        if kk == g.kpw - 1:
            _bar(g, BAR_STEP)
            off, n_kk = so_n, 0
        else:
            off, n_kk = so, kk + 1
        # the raw words of the next k16 row are loaded ahead of this row's
        # dequant; A is single-buffered and reloaded behind this row's MMAs
        _load_regs(
            g,
            sB_rd + off + n_kk * (g.tile_n * 8),
            sS_rd + off + n_kk * g.tile_n,
            1 - buf,
            braw,
            sraw,
        )
        _dequant_mma(g, buf, braw, sraw, tCrA, tCrB, tCrC, tiled_mma)
        tCsA = _stage_a(g, thr_s2r_A, sA_layout, smem_base + off)
        cute.copy(s2r_A, tCsA[None, None, kblk0 + n_kk], tCrA_v)


# ---------------------------------------------------------------------------
# Cross-CTA reduction: the X warp
# ---------------------------------------------------------------------------


@cute.jit
def _run_x(
    g: cutlass.Constexpr,
    pub: Int32,  # this CTA computes a partial mid-stripe (the first of several slices)
    nc: Int32,  # contributors bidx-1 .. bidx-nc to the column it finishes (last slice)
    size_m: Int32,
    gen: Int32,
    pT: Int64,
    pF: Int64,
    smem_base: Int32,
    ts: cute.Tensor,
):
    """The X warp: everything that crosses CTAs, off the MMA warps' path.

    Publish (at most once per CTA, a partial computed mid-stripe): when the
    MMA threads have left the k-reduced sums of a slice that does not hold its
    column's bottom in the F region (which nothing else uses until this warp
    collects partials into it), store them to this CTA's scratch slot, fence
    -- every thread that stored does, so the stores are performed GPU-wide --
    and only then release-store the launch's generation into the CTA's flag.
    The MMA and IO warps have long moved on.

    Collect (on the CTA that holds a column's bottom): acquire-poll the flags
    of the contributors, fb at a time, and cp.async a contributor's partial
    into the F region as soon as its flag carries this launch's generation
    (the earlier ones are then in flight or landed when the last flag
    arrives); hand the batch to the MMA threads. The polling starts right
    away, so the partials are usually in smem before the MMA warps ask for
    them.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    f_lane = smem_base + g.pipe_bytes + g.epi_bytes + 16 * lane
    ok_i32 = _smem_view(Int32, smem_base + g.work_bytes, 1, 4)
    ok_i32[0] = Int32(0x3F800000)  # status word: 1.0, NaN after a poll timeout
    err_word = _gmem_view(Int32, pF + Int64(4 * ERR_WORD), 1, 4)
    v4 = cute.make_rmem_tensor((4,), Float32)
    if pub != 0:
        _bar(g, BAR_X)  # the sums are staged
        slot = pT + Int64(bidx) * Int64(g.part_bytes) + Int64(16 * lane)
        for mi in cutlass.range_constexpr(g.mb):
            if 16 * mi < size_m:  # m16 blocks >= M are never read
                for mw in cutlass.range_constexpr(g.wm * g.wn):
                    for nn in cutlass.range_constexpr(g.nb):
                        blk = mw * g.groups + nn * g.mb + mi
                        cute.autovec_copy(
                            _smem_view(Float32, f_lane + 512 * blk, 4, 16), v4
                        )
                        cute.autovec_copy(
                            v4, _gmem_view(Float32, slot + Int64(512 * blk), 4, 16)
                        )
        _stamp(g, ts, 6)
        cute.arch.fence_acq_rel_gpu()
        _stamp(g, ts, 7)
        _bar(g, BAR_W)  # every lane is past its fence
        if lane == 0:
            cute.arch.store(
                cute.make_ptr(
                    Int32, pF + Int64(4) * Int64(bidx), cute.AddressSpace.gmem, assumed_align=4
                ),
                gen,
                sem="release",
                scope="gpu",
            )  # fmt: skip
        _stamp(g, ts, 8)
    done = Int32(0)
    while done < nc:
        nb = nc - done
        if nb > g.fb:
            nb = Int32(g.fb)
        want = (Int32(1) << nb) - 1
        # bit f: the partial of contributor done + f is on its way to the F region
        got = Int32(0)
        spins = Int32(0)
        while got != want:
            # lane l polls the flag of contributor done + l until it has seen it
            here = Int32(0)
            if lane < nb:  # noqa: SIM102 -- DSL regions: keep the shape of the if
                if (got >> lane) & 1 == 0:
                    fp = cute.make_ptr(
                        Int32,
                        pF + Int64(4) * Int64(bidx - 1 - done - lane),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    )
                    if (
                        Int32(cute.arch.load(fp, Int32, sem="acquire", scope="gpu"))
                        == gen
                    ):
                        here = Int32(1)
            # (a warp-wide vote: every lane is past its acquire before any lane
            # loads a partial)
            new = Int32(cute.arch.vote_ballot_sync(here != 0)) & want
            spins = spins + 1
            if spins > SPIN_LIMIT:
                # give up (never hang the GPU): poison this tile's output and
                # leave a trace
                if lane == 0:
                    ok_i32[0] = Int32(0x7FC00000)
                    err_word[0] = gen
                new = (want ^ got) & want
            # fetch a contributor's partial as soon as its flag is up: the
            # others' are then already in flight when the last flag arrives
            for f in cutlass.range_constexpr(g.fb):
                if (new >> f) & 1 != 0:
                    slot = (
                        pT
                        + Int64(bidx - 1 - done - f) * Int64(g.part_bytes)
                        + Int64(16 * lane)
                    )
                    # blocks are [m/n warp][n8 block][m16 block]: skip m16 blocks >= M
                    for mi in cutlass.range_constexpr(g.mb):
                        if 16 * mi < size_m:
                            for mw in cutlass.range_constexpr(g.wm * g.wn):
                                for nn in cutlass.range_constexpr(g.nb):
                                    blk = mw * g.groups + nn * g.mb + mi
                                    _cp_async(
                                        16,
                                        slot + Int64(512 * blk),
                                        f_lane + f * g.part_bytes + 512 * blk,
                                    )
            got = got | new
        _stamp(g, ts, 9)
        cute.arch.cp_async_commit_group()
        _stamp(g, ts, 10)
        cute.arch.cp_async_wait_group(0)
        _stamp(g, ts, 11)
        _bar(g, BAR_X)  # the MMA threads add this batch
        done = done + g.fb
        if done < nc:
            _bar(g, BAR_X)  # ... and are done with the F region
    _stamp(g, ts, 15)


# ---------------------------------------------------------------------------
# Epilogue of the MMA warps: k-warp reduction, hand-over, output
# ---------------------------------------------------------------------------


def _grp_get(g, tCrC, grp, v4):
    nn, i = divmod(grp, g.mb)
    for e in range(4):
        v4[e] = tCrC[nn][4 * i + e]


@cute.jit
def _epilogue(
    g: cutlass.Constexpr,
    fin: Int32,  # this CTA holds the bottom of the column: it writes the output
    nc: Int32,  # lower-index CTAs that hold the rest of the column (fin only)
    last: Int32,  # the stripe ends here
    col: Int32,  # column tile
    row0: Int32,  # first row of x / out of the row chunk
    row_lo: Int32,  # tile rows below this one are the business of the chunk above
    size_m: Int32,
    size_n: Int32,
    smem_base: Int32,
    s2: Float32,
    gen: Int32,
    pC: Int64,
    pT: Int64,
    pF: Int64,
    tCrC: cutlass.Constexpr,
    ts: cute.Tensor,
):
    """MMA warps, end of a column slice.

    1. A thread owns cpt accumulator groups of its (m, n) warp position; each
       warp parks the groups it does not own in smem and every thread sums its
       groups over the k-warps.
    2a. Not the column's last contributor. Mid-stripe: the sums go to smem,
        the X warp publishes them and the MMA warps carry on. At the end of
        the stripe: the MMA threads publish them themselves (four warps store
        faster than one).
    2b. The last contributor: add the partials the X warp has brought into
        smem, convert to dtype x scale2 (two values per instruction) and write
        the tile through a row-major smem staging as coalesced stores.

    All sums run in a fixed order, so the result is deterministic.
    """
    tidx, _, _ = cute.arch.thread_idx()
    dtype = g.dtype
    lane = tidx % 32
    warp = tidx // 32
    mwarp = warp % (g.wm * g.wn)
    wm_i = warp % g.wm
    wn_i = (warp // g.wm) % g.wn
    wk_i = warp // (g.wm * g.wn)
    # this thread's 16 B slot inside a partial: [m/n warp][group][lane]
    toff = (mwarp * g.groups * 32 + lane) * 16
    row_w = 16 * g.mb * wm_i + lane // 4
    col_w = 8 * g.nb * wn_i + 2 * (lane % 4)
    e_base = smem_base + g.pipe_bytes
    f_base = e_base + g.epi_bytes
    ok_i32 = _smem_view(Int32, smem_base + g.work_bytes, 1, 4)
    v4 = cute.make_rmem_tensor((4,), Float32)
    acc = cute.make_rmem_tensor((4 * g.cpt,), Float32)

    # ---- 1: k-warp reduction. k-warp w sums groups [w * cpt, (w + 1) * cpt)
    # of its (m, n) warp position: its own copy is in registers, the other
    # k-warps park theirs in smem (a k-warp's slot skips the groups it sums) ----
    grp0 = wk_i * g.cpt
    d_off = (mwarp * (g.groups - g.cpt) * 32 + lane) * 16
    for w in cutlass.range_constexpr(g.wk):
        if wk_i == w:  # one warp-uniform branch per k-warp, not per group
            for j in cutlass.range_constexpr(g.cpt):
                _grp_get(g, tCrC, w * g.cpt + j, v4)
                for e in cutlass.range_constexpr(4):
                    acc[4 * j + e] = v4[e]
            d_wr = e_base + w * g.dslot_bytes + d_off
            for grp in cutlass.range_constexpr(g.groups):
                if cutlass.const_expr(g.wk > 1 and grp // g.cpt != w):
                    gi = grp if grp // g.cpt < w else grp - g.cpt  # compact index
                    _grp_get(g, tCrC, grp, v4)
                    cute.autovec_copy(v4, _smem_view(Float32, d_wr + 512 * gi, 4, 16))
    if cutlass.const_expr(g.wk > 1):
        _bar(g, BAR_MMA)
        # the other k-warps' copies, in k order (fixed, so deterministic)
        for k in cutlass.range_constexpr(g.wk):
            if wk_i != k:
                # groups of k-warp wk_i sit at compact index grp - cpt when wk_i > k
                gb = grp0
                if wk_i > k:
                    gb = grp0 - g.cpt
                rd = e_base + k * g.dslot_bytes + d_off + 512 * gb
                for j in cutlass.range_constexpr(g.cpt):
                    cute.autovec_copy(_smem_view(Float32, rd + 512 * j, 4, 16), v4)
                    for e in cutlass.range_constexpr(4):
                        acc[4 * j + e] = acc[4 * j + e] + v4[e]
    _stamp(g, ts, 5)

    if fin == 0:
        if last == 0:
            # ---- 2a, mid-stripe: hand the partial to the X warp, in the F
            # region (free until that warp collects other CTAs' partials) ----
            p_wr = f_base + toff + 512 * grp0
            for j in cutlass.range_constexpr(g.cpt):
                for e in cutlass.range_constexpr(4):
                    v4[e] = acc[4 * j + e]
                cute.autovec_copy(v4, _smem_view(Float32, p_wr + 512 * j, 4, 16))
            _bar(g, BAR_X)
            _stamp(g, ts, 6)
        else:
            # ---- 2a, end of the stripe (nothing left to compute, and the CTA
            # that finishes this column is waiting): every thread stores its
            # sums itself and fences -- its stores are then performed GPU-wide
            # -- and when all of them are past that, one thread release-stores
            # this launch's generation into the CTA's flag ----
            bidx, _, _ = cute.arch.block_idx()
            slot = pT + Int64(bidx) * Int64(g.part_bytes) + Int64(toff)
            for j in cutlass.range_constexpr(g.cpt):
                for e in cutlass.range_constexpr(4):
                    v4[e] = acc[4 * j + e]
                cute.autovec_copy(
                    v4, _gmem_view(Float32, slot + Int64(512) * Int64(grp0 + j), 4, 16)
                )
            _stamp(g, ts, 6)
            cute.arch.fence_acq_rel_gpu()
            _stamp(g, ts, 7)
            _bar(g, BAR_MMA)
            if tidx == 0:
                cute.arch.store(
                    cute.make_ptr(
                        Int32, pF + Int64(4) * Int64(bidx), cute.AddressSpace.gmem, assumed_align=4
                    ),
                    gen,
                    sem="release",
                    scope="gpu",
                )  # fmt: skip
            _stamp(g, ts, 8)
    else:
        # ---- 2b: finish the column (always the stripe's last slice if nc > 0):
        # add the partials of CTAs bidx-1 .. bidx-nc in that order, fb at a
        # time. m16 blocks >= M hold stale smem; those rows are never stored.
        done = Int32(0)
        while done < nc:
            _bar(g, BAR_X)  # the X warp has this batch in the F region
            _stamp(g, ts, 9)
            for f in cutlass.range_constexpr(g.fb):
                if done + f < nc:
                    for j in cutlass.range_constexpr(g.cpt):
                        cute.autovec_copy(
                            _smem_view(
                                Float32,
                                f_base + f * g.part_bytes + toff + 512 * (grp0 + j),
                                4,
                                16,
                            ),
                            v4,
                        )
                        for e in cutlass.range_constexpr(4):
                            acc[4 * j + e] = acc[4 * j + e] + v4[e]
            done = done + g.fb
            if done < nc:
                _bar(g, BAR_X)  # the F region is refilled
        _stamp(g, ts, 10)
        # ---- output: fp32 -> dtype, x scale2 in dtype (as sglang); the tile
        # is staged row-major in smem so the global stores are coalesced
        # 16 B. The status word is 1.0, or NaN after a flag-poll timeout.
        # Two values per register: packed convert, packed multiply.
        s2p = _cvt_pair(s2, s2, g.bf16)
        if nc > 0:
            s2p = cutlass.select_(
                ok_i32[0] == 0x3F800000,
                s2p,
                Int32(0x7FC07FC0 if g.bf16 else 0x7E007E00),
            )
        o2 = cute.make_rmem_tensor((2 * g.cpt,), Int32)
        for j in cutlass.range_constexpr(g.cpt):
            for hh in cutlass.range_constexpr(2):
                pair = _cvt_pair(acc[4 * j + 2 * hh], acc[4 * j + 2 * hh + 1], g.bf16)
                o2[2 * j + hh] = _mul_pair(pair, s2p, g.bf16)
        if cutlass.const_expr(g.wk > 1):
            _bar(g, BAR_MMA)  # everyone is done with the parked accumulators
        for j in cutlass.range_constexpr(g.cpt):
            grp = grp0 + j
            o_wr = (
                e_base
                + (row_w + 16 * (grp % g.mb)) * g.out_stride
                + (col_w + 8 * (grp // g.mb)) * 2
            )
            for hh in cutlass.range_constexpr(2):
                o_w = _smem_view(Int32, o_wr + 8 * hh * g.out_stride, 1, 4)
                o_w[0] = o2[2 * j + hh]
        _bar(g, BAR_MMA)
        _stamp(g, ts, 12)
        c_base = pC + Int64(col * g.tile_n) * 2
        if cutlass.const_expr(g.rc):
            c_base = c_base + Int64(row0) * Int64(size_n) * 2
        cpr = g.tile_n // 8  # 16 B chunks per output row
        o8 = cute.make_rmem_tensor((8,), dtype)
        for j in cutlass.range_constexpr(g.out_chunks):
            c = tidx + j * g.threads
            row = c // cpr
            keep = row < size_m
            if cutlass.const_expr(g.rc):  # row_lo <= row < size_m
                keep = (row - row_lo).to(cutlass.Uint32) < (size_m - row_lo).to(
                    cutlass.Uint32
                )
            if keep:
                cute.autovec_copy(
                    _smem_view(
                        dtype, e_base + row * g.out_stride + 16 * (c % cpr), 8, 16
                    ),
                    o8,
                )
                cute.autovec_copy(
                    o8,
                    _gmem_view(
                        dtype,
                        c_base
                        + (Int64(row) * Int64(size_n) + Int64(8 * (c % cpr))) * 2,
                        8,
                        16,
                    ),
                )
        if last == 0:
            _bar(g, BAR_MMA)  # the staging is reused by the next slice
        _stamp(g, ts, 13)


@cute.jit
def _run_mma(
    g: cutlass.Constexpr,
    slen: Int32,
    nc_tab: Int32,
    k_tiles: Int32,
    n_tiles: Int32,
    size_m: Int32,
    size_mt: Int32,
    size_n: Int32,
    s2: Float32,
    gen: Int32,
    pC: Int64,
    pT: Int64,
    pF: Int64,
    smem_base: Int32,
    kblk0: Int32,
    sB_rd: Int32,
    sS_rd: Int32,
    st: cute.Tensor,
    thr_s2r_A: cutlass.Constexpr,
    sA_layout: cutlass.Constexpr,
    tCrA_v: cute.Tensor,
    tCrA: cute.Tensor,
    braw: cute.Tensor,
    sraw: cute.Tensor,
    tCrB: cutlass.Constexpr,
    tCrC: cutlass.Constexpr,
    s2r_A: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
    ts: cute.Tensor,
):
    """The MMA warps' walk over the stripe [u0, u0 + slen) of (k-tile, n-tile)
    units, column-major, from the end: the head of the last column first (its
    partial is published early), the tail of the first column last (this CTA
    holds that column's bottom and finishes it)."""
    for nn in cutlass.range_constexpr(g.nb):
        tCrC[nn].fill(0.0)
    _bar(g, BAR_STEP)
    _stamp(g, ts, 1)
    _load_regs(g, sB_rd, sS_rd, 0, braw, sraw)
    tCsA = _stage_a(g, thr_s2r_A, sA_layout, smem_base)
    cute.copy(s2r_A, tCsA[None, None, kblk0], tCrA_v)

    while st[ST_P] < slen:
        # one slice: column st[CC], k-tiles st[CK] down to ra
        n_sl = st[ST_CK] + 1
        if n_sl > slen - st[ST_P]:  # noqa: PLR1730 -- DSL regions: keep the shape of the if
            n_sl = slen - st[ST_P]
        stop = st[ST_P] + n_sl
        while st[ST_P] < stop:
            # one unit; the stages are used round-robin
            so = st[ST_SO]
            so_n = so + g.stage_bytes
            if so_n == g.pipe_bytes:
                so_n = Int32(0)
            _mma_step(
                g, so, so_n, smem_base, kblk0, sB_rd, sS_rd, thr_s2r_A, sA_layout,
                tCrA_v, tCrA, braw, sraw, tCrB, tCrC, s2r_A, tiled_mma,
            )  # fmt: skip
            st[ST_SO] = so_n
            st[ST_P] = st[ST_P] + 1

        col = st[ST_CC]
        ra = st[ST_CK] + 1 - n_sl
        fin = Int32(0)
        if st[ST_CK] == k_tiles - 1:
            fin = Int32(1)
        # contributors above this CTA's part of the column: bidx-1 .. bidx-nc
        # (only the stripe's last slice can have any; the host counted them)
        nc = Int32(0)
        if ra > 0:
            nc = nc_tab
        last = Int32(0)
        if st[ST_P] == slen:
            last = Int32(1)
            _stamp(g, ts, 4)
        else:
            _stamp(g, ts, 2)
        # Row-chunked kernels: the chunk's size_m rows of x / out start at
        # row0; the last chunk is moved up to end at M and leaves the rows it
        # then shares with the chunk above to that one.
        row0 = Int32(0)
        row_lo = Int32(0)
        if cutlass.const_expr(g.rc):
            row0 = st[ST_CR] * size_m
            if row0 > size_mt - size_m:
                row_lo = row0 - (size_mt - size_m)
                row0 = size_mt - size_m
        _epilogue(
            g, fin, nc, last, col, row0, row_lo, size_m, size_n, smem_base, s2, gen,
            pC, pT, pF, tCrC, ts,
        )  # fmt: skip
        if last == 0:
            _stamp(g, ts, 3)
        for nn in cutlass.range_constexpr(g.nb):
            tCrC[nn].fill(0.0)
        st[ST_CC] = col - 1
        if cutlass.const_expr(g.rc):  # noqa: SIM102 -- DSL regions: keep the shape of the if
            if col == 0:  # on to the last column tile of the row chunk above
                st[ST_CC] = n_tiles - 1
                st[ST_CR] = st[ST_CR] - 1
        st[ST_CK] = k_tiles - 1
    _stamp(g, ts, 15)


@cute.jit
def _role_io(
    g: cutlass.Constexpr,
    part: cutlass.Constexpr,
    pA: Int64,
    pB: Int64,
    pS: Int64,
    size_m: Int32,
    size_mt: Int32,
    size_k: Int32,
    size_n: Int32,
    slen: Int32,
    fc: Int32,
    fr: Int32,
    fk: Int32,
    k_tiles: Int32,
    n_tiles: Int32,
    smem_base: Int32,
    ts: cute.Tensor,
):
    """IO warp ``part``: per-lane gmem pointers and smem targets of its chunks,
    then the pipeline. A: lane = (row % 4, chunk); B: lane = 16 B of a unit's
    512; scales: lane = (k16 row, 16 B chunk) -- fewer than 32 chunks are
    fetched twice."""
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % 32
    sB_addr = smem_base + g.sa_stage
    sS_addr = sB_addr + g.sb_stage
    a_row = lane // 8
    a_chk = lane % 8
    a_dst0 = smem_base + 128 * a_row + 16 * (a_chk ^ a_row)
    a_dst1 = smem_base + 128 * a_row + 16 * (a_chk ^ (a_row + 4))
    mrow = size_m - a_row
    sB_lane = sB_addr + 16 * lane
    s_chunks = min(32, g.kb * g.tile_n // 16)
    s_lane = lane % s_chunks
    scpr = g.tile_n // 16
    sS_lane = sS_addr + 16 * s_lane
    sp = cute.make_rmem_tensor((6 if g.rc else 5,), Int64)
    row_n = Int64(size_n) * Int64(fk * g.kb)  # scale bytes above the unit's k16 rows
    sp[SP_A] = (
        pA + Int64(size_k) * Int64(2 * a_row) + Int64(fk) * Int64(2 * g.ks) + Int64(16 * a_chk)
    )  # fmt: skip
    if cutlass.const_expr(g.rc):
        # the rows of the stripe's last row chunk (see _run_mma) and how far up
        # the chunk above it starts
        row0 = fr * size_m
        step = size_m
        if row0 > size_mt - size_m:
            step = size_mt - row0
            row0 = size_mt - size_m
        sp[SP_A] = sp[SP_A] + Int64(size_k) * Int64(2 * row0)
        sp[SP_WA] = Int64(size_k) * Int64(2 * step)
    sp[SP_B] = (
        pB + row_n * Int64(8) + Int64(fc) * Int64(g.tile_n * 8) + Int64(16 * lane)
    )
    sp[SP_S] = (
        pS
        + row_n
        + Int64(fc) * Int64(g.tile_n)
        + Int64(size_n) * Int64(s_lane // scpr)
        + Int64(16 * (s_lane % scpr))
    )
    wrap_n = Int64(size_n) * Int64((k_tiles - 1) * g.kb)  # up to the last k-tile ...
    sp[SP_WB] = wrap_n * Int64(8) - Int64(g.tile_n * 8)  # ... of the previous column
    sp[SP_WS] = wrap_n - Int64(g.tile_n)
    _run_io(
        g, part, slen, fk, fc * k_tiles + fk + 1, n_tiles * k_tiles, k_tiles, size_m, size_k,
        size_n, mrow, a_dst0, a_dst1, sB_lane, sS_lane, sp, ts,
    )  # fmt: skip


@cute.jit
def _role_mma(
    g: cutlass.Constexpr,
    pC: Int64,
    pS2: Int64,
    pT: Int64,
    pF: Int64,
    gen: Int32,
    size_m: Int32,
    size_mt: Int32,
    size_n: Int32,
    slen: Int32,
    nc_tab: Int32,
    fc: Int32,
    fr: Int32,
    fk: Int32,
    k_tiles: Int32,
    n_tiles: Int32,
    smem_base: Int32,
    tiled_mma: cute.TiledMma,
    sA_layout: cutlass.Constexpr,
    ts: cute.Tensor,
):
    """MMA warps: partitions, fragments and smem read addresses, then the walk."""
    tidx, _, _ = cute.arch.thread_idx()
    dtype = g.dtype
    lane = tidx % 32
    warp = tidx // 32
    wn_i = (warp // g.wm) % g.wn
    wk_i = warp // (g.wm * g.wn)
    sB_addr = smem_base + g.sa_stage
    sS_addr = sB_addr + g.sb_stage
    s2 = _gmem_view(dtype, pS2, 1, 2)[0].to(Float32)
    atom_s2r = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), dtype)
    s2r_A = cute.make_tiled_copy_A(atom_s2r, tiled_mma)
    thr_s2r_A = s2r_A.get_slice(tidx % g.mma_threads)
    psA = tiled_mma.partition_shape_A((g.tile_m, 16))  # (V, mb, 1)
    # (a 2-mode (V, mb) register tensor crashes retile: keep the unit K mode)
    tCrA_k = cute.make_rmem_tensor((psA[0], psA[1], 1), dtype)
    tCrA = tCrA_k[None, None, 0]
    tCrA_v = thr_s2r_A.retile(tCrA_k)[None, None, 0]
    # B fragments / accumulators, one tensor per n8 block of the warp's
    # columns: ((2,2), 1) and ((2,2), mb, 1)
    psB = tiled_mma.partition_shape_B((g.tile_n, 16))
    psC = tiled_mma.partition_shape_C((g.tile_m, g.tile_n))
    tCrB = [cute.make_rmem_tensor((psB[0], 1), dtype) for _ in range(g.nb)]
    tCrC = [cute.make_rmem_tensor((psC[0], psC[1], 1), Float32) for _ in range(g.nb)]
    braw = cute.make_rmem_tensor((g.nb // 2, 2), Int32)  # raw packed B, double-buffered
    sraw = cute.make_rmem_tensor((g.nb // 4, 2), Int32)  # raw scale bytes
    # this warp's slice of a staged k16 row: lane-major 16 B per 64-column
    # unit (4 B per n16 block) and 8 scale bytes per 4 lanes (4 per four n8
    # blocks); kblk0 is the warp's first k16 row of a stage
    unit = (wn_i * g.nb) // 8
    sub = wn_i % (8 // g.nb)
    kblk0 = wk_i * g.kpw
    sB_rd = sB_addr + 16 * (32 * unit + lane) + 2 * g.nb * sub + kblk0 * (g.tile_n * 8)
    sS_rd = sS_addr + 64 * unit + 8 * (lane // 4) + g.nb * sub + kblk0 * g.tile_n
    st = cute.make_rmem_tensor((5 if g.rc else 4,), Int32)
    st[ST_P] = 0
    st[ST_SO] = 0
    st[ST_CC] = fc
    st[ST_CK] = fk
    if cutlass.const_expr(g.rc):
        st[ST_CR] = fr
    _run_mma(
        g, slen, nc_tab, k_tiles, n_tiles, size_m, size_mt, size_n, s2, gen, pC, pT, pF,
        smem_base, kblk0,
        sB_rd, sS_rd, st, thr_s2r_A, sA_layout, tCrA_v, tCrA, braw, sraw,
        tCrB, tCrC, s2r_A, tiled_mma, ts,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


@cute.kernel
def _kernel(
    pA: Int64,  # x [M, K] fp16 | bf16
    pB: Int64,  # int32 [K/16, 2N], Marlin-packed
    pS: Int64,  # scale bytes [K/16, N]
    pC: Int64,  # out [M, N]
    pS2: Int64,  # scale2 [1]
    pT: Int64,  # fp32 scratch: one tile_m x tile_n partial per CTA
    pF: Int64,  # int32 flags, one per CTA (see FLAG_CAP)
    pR: Int64,  # int32 stripe records, 16 or 32 B per CTA (see _stripe_records)
    size_m: Int32,  # M; row-chunked kernels: the rows of a row chunk
    size_k: Int32,
    size_n: Int32,
    size_mt: Int32,  # M
    gen: Int32,  # this launch's generation (non-zero, unique per scratch)
    tiled_mma: cute.TiledMma,
    sA_layout: cute.ComposedLayout,
    g: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    ts = cute.make_rmem_tensor((16,), Int32)
    if cutlass.const_expr(g.dbg):
        for i in cutlass.range_constexpr(16):
            ts[i] = Int32(0)
    _stamp(g, ts, 0)
    # this CTA's stripe record: (first unit, units, contributors | publishes << 16,
    # column tile of the last unit) and, in row-chunked kernels, (k-tile and row
    # chunk of the last unit, 0, 0)
    rec = cute.make_rmem_tensor((g.rec_words,), Int32)
    cute.autovec_copy(
        _gmem_view(Int32, pR + Int64(4 * g.rec_words) * Int64(bidx), g.rec_words, 16),
        rec,
    )
    u0 = rec[0]
    slen = rec[1]
    nc_tab = rec[2] & 0xFFFF
    pub = rec[2] >> 16
    fc = rec[3]
    k_tiles = size_k // g.ks
    n_tiles = size_n // g.tile_n
    fk = u0 + slen - 1 - fc * k_tiles  # the stripe's last unit is fetched first
    fr = Int32(0)
    if cutlass.const_expr(g.rc):
        fk = rec[4]
        fr = rec[5]

    # ---- shared memory: stages of [A tile | packed B rows | scale rows],
    # then E | F | status ----
    smem = cutlass.utils.SmemAllocator()
    blob = smem.allocate_tensor(
        Int32, cute.make_layout(g.smem_bytes // 4), byte_alignment=16
    )
    smem_base = Int32(blob.iterator.toint())

    if tidx >= g.threads + 32 * g.nio:
        _run_x(g, pub, nc_tab, size_m, gen, pT, pF, smem_base, ts)
    elif tidx >= g.threads + 32:  # (no such thread unless there are two IO warps)
        _role_io(
            g, 1, pA, pB, pS, size_m, size_mt, size_k, size_n, slen, fc, fr, fk,
            k_tiles, n_tiles, smem_base, ts,
        )  # fmt: skip
    elif tidx >= g.threads:
        _role_io(
            g, 0, pA, pB, pS, size_m, size_mt, size_k, size_n, slen, fc, fr, fk,
            k_tiles, n_tiles, smem_base, ts,
        )  # fmt: skip
    else:
        _role_mma(
            g, pC, pS2, pT, pF, gen, size_m, size_mt, size_n, slen, nc_tab, fc, fr, fk,
            k_tiles, n_tiles, smem_base, tiled_mma, sA_layout, ts,
        )  # fmt: skip

    if cutlass.const_expr(g.dbg):
        ts[14] = slen
        for r in cutlass.range_constexpr(3):
            if tidx == (0, g.threads, g.threads + 32 * g.nio)[r]:
                dbg = _gmem_view(
                    Int32,
                    pF + Int64(4 * DBG_OFF + 64 * r) + Int64(256) * Int64(bidx),
                    16,
                    4,
                )
                for i in cutlass.range_constexpr(16):
                    dbg[i] = ts[i]


@cute.jit
def _launch(
    pA: Int64,
    pB: Int64,
    pS: Int64,
    pC: Int64,
    pS2: Int64,
    size_m: Int32,
    size_mt: Int32,
    gen: Int32,
    pT: Int64,
    pF: Int64,
    pR: Int64,
    size_k: Int32,
    size_n: Int32,
    grid: Int32,
    g: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
):
    _kernel(
        pA, pB, pS, pC, pS2, pT, pF, pR, size_m, size_k, size_n, size_mt, gen,
        _make_tiled_mma(g), _make_sA_layout(g), g,
    ).launch(grid=(grid, 1, 1), block=(g.block_threads, 1, 1), stream=stream)  # fmt: skip


# Runtime arguments of _launch, in order (the Constexpr geometry is not one).
# The first eight are what nvfp4_linear() rewrites on every call: one
# struct.pack_into of the argument block.
A_, B_, S_, C_, S2_, M_, MT_, GEN_, T_, F_, R_, K_, N_, GRID_, STREAM_ = range(15)
_NARG = 15
_SAMPLE = (*(Int64(0),) * 5, *(Int32(1),) * 3, *(Int64(0),) * 3, *(Int32(1),) * 3)
_pack_call = struct.Struct("8q").pack_into

# ---------------------------------------------------------------------------
# Host side: geometry, stripe schedule, compiled kernels
# ---------------------------------------------------------------------------

# DEVELOPER TOOL (tests / sweeps / profiling builds only): geometry field
# overrides per m16-block bucket -- {("geom", mb): {...}} or {"geom": {...}},
# e.g. {"dbg": 1} for a build that records phase stamps -- and schedule knobs
# ("grid", "e_mid", "d_pub"). Nothing in the package writes to it and no
# environment variable feeds it: it is empty unless a developer fills it in.
_TUNE: dict = {}


@functools.cache
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


# Stripe schedule, in units of work (see _stripe_starts): the lead of a stripe
# that ends with a partial, and the cost of a mid-stripe epilogue -- about one
# unit up to 80 rows, about three for a taller tile (its k-warp reduction and
# stores grow with the rows faster than a unit does: M = 128, 1 -> 3: 26.2 ->
# 25.8 and 17.3 -> 16.8 us), two when every stripe has several of them (row
# chunks: 1 / 2 / 3 at M = 1024, K = 2048: 86.6 / 84.6 / 86.0 us).
D_PUB = 6.0


def _e_mid(g: Geom) -> float:
    if g.rc:
        return 2.0
    return 3.0 if g.mb >= 6 else 1.0


# (most rows, m16 blocks, n8 blocks per warp, warps along N, warps along K,
# k16 rows per warp and stage, pipeline stages for short / long stripes, IO
# warps): one kernel specialization per row and stripe class. A warp owns
# 16 * mb rows x 32 columns and two k16 rows of a stage: 2 x 2 warps over a
# 64-column tile and a 64-deep stage in every bucket.
# Pipeline depth is what makes the 64-column tile right at small M too: the
# cold stream runs at about 3/4 of what HBM delivers, and with only two units
# of lead its latency stalls the short units of a small M again and again
# (M = 16, 4 stages: 13.9 / 9.4 us at K = 4096 / 2048; 8 stages: 11.8 / 8.4;
# a 128-column tile with four warps side by side: 12.5 / 8.9 at 4 stages,
# 12.0 / 8.9 at 6). Requesting more up front costs a little when a stripe is
# only a few units long. The tall buckets trade pipeline stages for room to
# stage two fetched partials, and their A tile is more than one warp can
# request in a unit's time (a lone warp issues a 512 B cp.async every ~14 ns:
# 37 of them per unit at 128 rows), so two warps share it.
# The first two buckets are the same 16-row tile and differ in Geom.m_min only
# (the rows of x that exist for every M of the bucket are fetched without a
# predicate, in whole groups of four): rows 0 .. 3 for M = 7, rows 0 .. 7 for
# M = 8 .. 16. Predicating rows 0 .. 3 as well would cost 0.5 us (measured at
# K = 4096 with every row predicated), which is why M = 7 does not simply
# lower the bound of the 16-row bucket to 1.
_BUCKETS = (
    (7, 1, 4, 2, 2, 2, 8, 8, 1),
    (16, 1, 4, 2, 2, 2, 8, 8, 1),
    (32, 2, 4, 2, 2, 2, 5, 6, 1),
    (48, 3, 4, 2, 2, 2, 6, 6, 1),
    (64, 4, 4, 2, 2, 2, 5, 5, 1),
    (80, 5, 4, 2, 2, 2, 5, 5, 2),
    (96, 6, 4, 2, 2, 2, 5, 5, 2),
    (112, 7, 4, 2, 2, 2, 4, 4, 2),
    (128, 8, 4, 2, 2, 2, 3, 3, 2),
)
assert _BUCKETS[-1][0] == MAX_ROWS
LONG_STRIPE = 12  # units per CTA from which a stripe counts as long


def split_rows(size_m: int) -> tuple[int, int]:
    """(row chunks, rows per chunk) for M rows: as few chunks as the tallest
    tile allows, all of the same height. The last one is moved up to end at
    M; the rows it then shares with the chunk above are written by that one."""
    chunks = -(-size_m // MAX_ROWS)
    return chunks, -(-size_m // chunks)


def select_geom(
    rows: int,
    bf16: bool,
    long_stripes: bool = True,
    chunked: bool = False,
    lone: bool = False,
) -> Geom:
    """Kernel geometry for ``rows`` rows (all of M, or the rows of a row chunk).
    ``lone``: no CTA ever collects more than one partial."""
    # Four MMA warps: with the fetches on warps of their own the SM is
    # throughput-bound with four already, and more only add per-warp overhead.
    lo = MIN_M
    for hi, mb, nb, wn, wk, kpw, stages, stages_long, nio in _BUCKETS:
        if rows <= hi:
            break
        lo = hi + 1
    if long_stripes:
        stages = stages_long
    if chunked and lone and mb == 8:
        # the room for a second fetched partial is worth more as a fourth stage
        # (M = 256: 45.0 -> 44.4 and 26.8 -> 26.5 us)
        stages = 4
    g = Geom(
        bf16, mb=mb, nb=nb, wm=1, wn=wn, wk=wk, kpw=kpw, stages=stages, m_min=lo, nio=nio,
        rc=int(chunked),
    )  # fmt: skip
    over = _TUNE.get(("geom", g.mb)) or _TUNE.get("geom")
    if over:
        g = replace(g, **over)
    return g


def _stripe_starts(
    k_tiles: int, n_tiles: int, grid: int, e_mid: float, d_pub: float
) -> list[int]:
    """First unit of every stripe (and the total), column-major: at most
    ``grid`` contiguous, non-empty stripes.

    The stripes are cut so that every CTA is *done* at about the same time
    rather than computes the same number of units. In units of work:

      - a stripe that crosses a column boundary pays a mid-stripe epilogue:
        ``e_mid`` per boundary;
      - a stripe that ends inside a column (it computes nothing but a partial)
        must end ``d_pub`` early: its partial has to be stored, fenced, flagged
        and fetched by the CTA that finishes the column before that one runs
        out of work.

    Greedy cut for a target length L, smallest L that needs <= grid stripes.
    """
    units = k_tiles * n_tiles
    if units < 8 * grid or (e_mid <= 0 and d_pub <= 0):
        # short stripes: plain equal split
        q, r = divmod(units, grid)
        return [q * b + min(b, r) for b in range(grid + 1)]

    def cut(length: float) -> list[int]:
        # x: where the cut would be with fractional units; the stripe
        # boundaries are x rounded, so that the lengths average out.
        starts, u, x = [0], 0, 0.0
        while u < units:
            col_end = (u // k_tiles + 1) * k_tiles
            if col_end - x > length:
                x += length - d_pub  # ends inside the column: a partial
                nxt = min(max(int(x + 0.5), u + 1), col_end - 1)
            else:
                budget = length - (col_end - x)  # holds the column's bottom ...
                x = float(col_end)
                while x < units and budget >= e_mid + 1:
                    take = min(budget - e_mid, float(k_tiles))  # ... and goes on
                    x += take
                    budget -= e_mid + take
                    if take < k_tiles:
                        break
                nxt = min(max(int(x + 0.5), col_end), units)
            if abs(x - nxt) > 0.5:
                x = float(nxt)
            u = nxt
            starts.append(u)
        return starts

    lo, hi = units / grid, units / grid + d_pub + e_mid + 2.0
    while len(cut(hi)) - 1 > grid:
        hi *= 1.5
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if len(cut(mid)) - 1 > grid:
            lo = mid
        else:
            hi = mid
    return cut(hi)


def _pick_grid(units: int, n_tiles: int, fb: int, sms: int) -> int:
    """Number of CTAs: one per SM unless the layer has so few column tiles that
    the CTAs finishing them would drown in partials.

    In units of work: a stripe is units / grid long; splitting columns across
    CTAs at all costs about the publish lead D_PUB once, and every further
    batch of fb partials a finisher has to collect about 3 more.
    """
    if "grid" in _TUNE:
        return min(units, sms, _TUNE["grid"])
    best, best_cost = 1, float("inf")
    for grid in range(1, min(units, sms) + 1):
        cost = units / grid
        fan_in = -(-grid // n_tiles) - 1  # contributors of the busiest finisher
        if fan_in > 0:
            cost += D_PUB + 3.0 * (-(-fan_in // fb) - 1)
        if cost <= best_cost:
            best, best_cost = grid, cost
    return best


def _stripe_records(
    starts: list[int], k_tiles: int, n_tiles: int
) -> list[tuple[int, ...]]:
    """Per CTA: (first unit, units, contributors | publishes << 16, column tile
    of the last unit, k-tile and row chunk of the last unit, 0, 0). A "column"
    of the schedule is one column tile of one row chunk, chunk-major,
    ``n_tiles`` per chunk; kernels that are not row-chunked read the first four
    words only.

    ``contributors`` is the number of lower-index CTAs that hold the top of the
    column whose bottom this CTA holds (0 if it holds no bottom, or all of the
    column): the CTAs it waits for in its last slice. ``publishes`` is set if
    the stripe ends inside a column it did not start in: the slice it computes
    first is a partial that the X warp publishes mid-stripe. (A stripe that
    lies inside one column is a partial too; its MMA warps publish it.)
    """
    grid = len(starts) - 1
    recs = []
    for b in range(grid):
        u0, u1 = starts[b], starts[b + 1]
        assert u1 > u0, "empty stripe"
        col0 = u0 // k_tiles * k_tiles
        # the CTA whose stripe holds the first unit of this CTA's first column
        first = bisect.bisect_right(starts, col0, 0, grid) - 1
        nc = b - first if u1 >= col0 + k_tiles else 0
        pub = 1 if u1 % k_tiles and (u1 - 1) // k_tiles != u0 // k_tiles else 0
        assert nc < 1 << 16
        col, kt = divmod(u1 - 1, k_tiles)
        chunk, col = divmod(col, n_tiles)
        recs.append((u0 & 0x7FFFFFFF, u1 - u0, nc | pub << 16, col, kt, chunk, 0, 0))
    return recs


# Compiled kernels: one per Geom and process, backed by the on-disk cache below.
_EXECUTORS: dict[Geom, tuple] = {}
_EXEC_LOCK = threading.Lock()


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


def _executor(g: Geom):
    """(executor, packed-argument template, compiled fn) of the kernel for ``g``."""
    entry = _EXECUTORS.get(g)
    if entry is not None:
        return entry
    with _EXEC_LOCK:
        entry = _EXECUTORS.get(g)
        if entry is not None:
            return entry
        g.check()
        stream = cuda_driver.CUstream(0)
        sample = (*_SAMPLE, stream)
        name = _cache_name(_launch.__name__, g, torch.cuda.get_device_capability())
        # (profiling builds are neither stored nor loaded)
        fn = None if g.dbg else _cache_load(name)
        if fn is None:
            fn = cute.compile(_launch, *sample[:-1], g, stream)
            if not g.dbg:
                _cache_store(fn, name)
        ex = fn.to(None)
        exe_args, _adapted = ex.generate_execution_args(*sample)
        packed = ex._get_invoke_packed_args(list(exe_args))
        if len(exe_args) != _NARG or ex.cuda_result is None:
            raise RuntimeError("cute_nvfp4_batch: unexpected CuTe DSL launch ABI")
        template = [packed[i] for i in range(len(packed))]
        entry = (ex, template, fn)
        _EXECUTORS[g] = entry
        return entry


# The entry point takes the address of the packed-argument block. Called
# through this prototype with a plain integer it costs 0.2 us less than the
# executor's own ctypes function called with the ctypes array.
_ENTRY = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


@dataclass(frozen=True, eq=False)
class _Plan:
    """Everything about a launch that depends only on (device, geometry, K, N,
    row chunks)."""

    g: Geom
    chunks: int
    grid: int
    table: torch.Tensor  # int32 [grid, 4 or 8] stripe records, on the device
    part_floats: int  # fp32 scratch the grid needs
    call: object  # the JIT-ed C entry point: call(address of an argument block)
    template: tuple
    keep: tuple


@functools.lru_cache(maxsize=1024)
def _plan(index: int, g: Geom, size_k: int, size_n: int, chunks: int) -> _Plan:
    assert size_k % g.ks == 0 and size_n % g.tile_n == 0 and (g.rc or chunks == 1)
    k_tiles, n_tiles = size_k // g.ks, size_n // g.tile_n
    # the schedule's columns: every column tile of every row chunk
    cols = n_tiles * chunks
    grid = _pick_grid(k_tiles * cols, cols, g.fb, min(_sm_count(index), FLAG_CAP))
    starts = _stripe_starts(
        k_tiles, cols, grid, _TUNE.get("e_mid", _e_mid(g)), _TUNE.get("d_pub", D_PUB)
    )
    grid = len(starts) - 1
    recs = [r[: g.rec_words] for r in _stripe_records(starts, k_tiles, n_tiles)]
    table = torch.tensor(recs, dtype=torch.int32, device=torch.device("cuda", index))
    ex, template, fn = _executor(g)
    return _Plan(
        g, chunks, grid, table, grid * g.tile_m * g.tile_n,
        _ENTRY(ctypes.cast(ex.capi_func, ctypes.c_void_p).value), tuple(template), (ex, fn),
    )  # fmt: skip


# Scratch (fp32 partial slots, int32 flags) per (device, stream). Launches on
# one stream run one after the other, so they can share it; launches on
# different streams may overlap and must not.
_SCRATCH: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
_SCRATCH_LOCK = threading.Lock()
_GEN = itertools.count(1)  # launch generations: unique across every scratch

# Flag-poll timeouts, made loud. A launch that gave up on a flag poll (_run_x)
# has returned normally, with one NaN tile and its generation in ERR_WORD of
# its scratch, on the device: the host cannot look there without synchronizing.
# So every _ERR_EVERY launches of the process (_err_sample) it issues a 4-byte
# asynchronous copy of every scratch's ERR_WORD into page-locked host memory --
# on a side stream, so that the copies neither wait for the kernels nor hold
# them up -- and looks at what the copies issued the time before brought back.
# A timeout therefore raises RuntimeError from a later call, at most
# 2 * _ERR_EVERY launches after the launch that timed out has run. Cost: one
# AND per launch, and ~4 us of host time once every _ERR_EVERY launches.
_ERR_EVERY = 256  # a power of two: the launch paths test ``gen & 255``
_ERR_CAP = 64  # scratches that are watched (torch hands out 32 raw streams per device)
_ERR_HOST = None  # (page-locked int32 tensor, ctypes view of it); False: unavailable
_ERR_SIDE: dict[int, tuple] = {}  # device index -> (torch stream, its CUstream)
# per watched scratch: [device index, flags tensor (kept alive: its address is
# read later), address of its ERR_WORD, last generation reported]
_ERR_SLOTS: list[list] = []


def _err_watch(index: int, flags: torch.Tensor) -> None:
    """A new scratch (called under _SCRATCH_LOCK): give its error word a host slot."""
    global _ERR_HOST
    if _ERR_HOST is None:
        try:
            pinned = torch.zeros(_ERR_CAP, dtype=torch.int32).pin_memory()
            if not pinned.is_pinned():
                raise RuntimeError("host memory was not page-locked")
            view = (ctypes.c_int32 * _ERR_CAP).from_address(pinned.data_ptr())
            _ERR_HOST = (pinned, view)
        except Exception:  # noqa: BLE001 -- no page-locked memory: poll_timeouts() only
            _ERR_HOST = False
    if _ERR_HOST and len(_ERR_SLOTS) < _ERR_CAP:
        _ERR_SLOTS.append([index, flags, flags.data_ptr() + 4 * ERR_WORD, 0])


def _err_sample(index: int) -> None:
    """Once every _ERR_EVERY launches, before a launch on device ``index`` (the
    current device): raise for a timeout that the previous round of copies has
    brought back, and issue the next round."""
    host = _ERR_HOST
    if not host:
        return
    side = _ERR_SIDE.get(index)
    if side is None:
        stream = torch.cuda.Stream(index)
        side = _ERR_SIDE[index] = (stream, cuda_driver.CUstream(stream.cuda_stream))
    base, words = host[0].data_ptr(), host[1]
    timed_out = 0
    for i, slot in enumerate(tuple(_ERR_SLOTS)):  # (other threads may append)
        gen = words[i]
        if gen and gen != slot[3]:
            slot[3] = timed_out = gen
        if slot[0] == index:
            cuda_driver.cuMemcpyDtoHAsync(base + 4 * i, slot[2], 4, side[1])
    if timed_out:
        raise RuntimeError(
            f"cute_nvfp4_batch: an earlier launch (generation {timed_out}) gave up "
            f"waiting for another CTA's partial after {SPIN_LIMIT} flag polls: one "
            "64-column tile of its output is NaN. This call was not launched. "
            "See poll_timeouts()."
        )


def _scratch(index: int, raw_stream: int, part_floats: int):
    key = (index, raw_stream)
    cur = _SCRATCH.get(key)
    if cur is None or cur[0].numel() < part_floats:
        with _SCRATCH_LOCK:
            cur = _SCRATCH.get(key)
            if cur is None or cur[0].numel() < part_floats:
                device = torch.device("cuda", index)
                cur = (
                    torch.empty(max(part_floats, 1 << 20), dtype=torch.float32, device=device),
                    torch.zeros(FLAG_WORDS, dtype=torch.int32, device=device),
                )  # fmt: skip
                # (bindings made earlier keep the tensors they were given alive)
                _SCRATCH[key] = cur
                _err_watch(index, cur[1])
    return cur


def poll_timeouts(device=None) -> dict[int, int]:
    """Diagnostics: {stream: generation of the last launch that gave up waiting
    for a cross-CTA partial} for every scratch of ``device`` (synchronizes). The
    words are never cleared; the RuntimeError of _err_sample comes once per value."""
    index = torch.device(device or "cuda").index or 0
    return {
        raw: int(flags[ERR_WORD])
        for (i, raw), (_, flags) in list(_SCRATCH.items())
        if i == index and int(flags[ERR_WORD])
    }


_raw_stream = torch._C._cuda_getCurrentRawStream
_get_ident = threading.get_ident


class _Bound:
    """One launch with its own argument block: ``call(arg)`` is the whole call.

    The block is the DSL's packed-argument layout: slot i points at the 8-byte
    cell holding runtime argument i, the slot after the last one at the launch
    result. ``mv`` is an int64 view of the cells (the cheapest way to store
    into them from Python).
    """

    __slots__ = (
        "arg",
        "call",
        "index",
        "keep",
        "mv",
        "own",
        "plan",
        "raw",
        "res",
        "vals",
    )

    def __init__(self, plan: _Plan, index: int):
        self.plan = plan
        self.index = index
        self.vals = (ctypes.c_int64 * _NARG)()
        self.mv = memoryview(self.vals).cast("B").cast("q")
        self.res = ctypes.c_int32(0)
        own = (ctypes.c_void_p * len(plan.template))(*plan.template)
        base = ctypes.addressof(self.vals)
        for i in range(_NARG):
            own[i] = base + 8 * i
        own[_NARG] = ctypes.addressof(self.res)
        self.own = own
        self.call = plan.call
        self.arg = ctypes.addressof(own)
        self.raw = None
        self.keep = None
        v = self.vals
        v[R_] = plan.table.data_ptr()
        v[GRID_] = plan.grid

    def set_stream(self, raw: int) -> None:
        part, flags = _scratch(self.index, raw, self.plan.part_floats)
        v = self.vals
        v[T_] = part.data_ptr()
        v[F_] = flags.data_ptr()
        v[STREAM_] = raw
        self.raw = raw
        self.keep = (part, flags)

    def set_call(self, pa, pb, ps, pc, ps2, size_m, size_k, size_n) -> None:
        v = self.vals
        v[A_] = pa
        v[B_] = pb
        v[S_] = ps
        v[C_] = pc
        v[S2_] = ps2
        v[M_] = -(-size_m // self.plan.chunks)  # the rows of a row chunk
        v[K_] = size_k
        v[N_] = size_n
        v[MT_] = size_m

    def launch(self) -> None:
        raw = _raw_stream(self.index)
        if raw != self.raw:
            self.set_stream(raw)
        gen = next(_GEN) & 0x7FFFFFFF or 1
        if not gen & 255:  # every _ERR_EVERY launches
            _err_sample(self.index)
        self.mv[GEN_] = gen
        self.call(self.arg)
        if self.res.value:
            raise RuntimeError(
                f"cute_nvfp4_batch: kernel launch failed (CUDA error {self.res.value})"
            )


# ---------------------------------------------------------------------------
# Call path. The integration calls nvfp4_linear() once per expert and token
# batch: fixed weight tensors, a new x (and M) every time. Nothing about a
# call is taken on trust from an earlier one: every call checks all five
# tensors by value -- type, dtype, device, sizes, strides -- and launches with
# the addresses it has just read from them, so the memory that was checked is
# the memory that is read. Tensor objects that were re-pointed in place since
# the last call (param.data = ..., resize_(), a module moved to the CPU and
# back) are either valid as they are now, and used as they are now, or raise.
#
# What keeps that cheap: launch state is per shape, not per layer (one _Shape
# per device, K, N and dtype -- a model has a handful -- with the weight
# address as a mere hint to it), and the checks against a shape are one C call
# (torch's TensorGuards: Python type, dtype, device, sizes and strides of
# several tensors at once, built from tensors that passed the same checks in
# Python). A shape keeps the guards of the last few kinds of tensors it has
# seen (another Python type, requires_grad, inference tensors, ...); a call
# that none of them knows, or a torch without TensorGuards, takes the checks in
# Python: 5 us slower, never different.
# ---------------------------------------------------------------------------

_SCALE_DTYPES = tuple(
    d for d in (getattr(torch, "float8_e4m3fn", None), torch.uint8, torch.int8) if d is not None
)  # fmt: skip
_ROW_BUCKET = tuple(
    next(i for i, b in enumerate(_BUCKETS) if rows <= b[0])
    for rows in range(MAX_ROWS + 1)
)
_NB = len(_BUCKETS)
# x at an address that is 16 mod 32 is legal (cp.async needs 16) but slow: the
# 128 B row piece of an A tile then starts and ends with half a 32 B sector,
# and an instruction that asks for half sectors costs the IO warps about twice
# the time (whatever the lane order: measured with permuted lanes). A tall
# tile has no IO time to spare: M = 128, K = 4096 takes 37 us instead of 26,
# M = 64 21.8 instead of 17.8. An x of this size or more is therefore read
# through an aligned copy, which costs one torch copy per call (about 17 us of
# host time, 3 us of GPU time); a smaller one is cheaper read where it is.
X32_BYTES = 640 << 10

try:  # one C call for the type, dtype, device, sizes and strides of several tensors
    from torch._C._dynamo.guards import TensorGuards as _TensorGuards
except Exception:  # noqa: BLE001 -- a torch without it: the checks run in Python
    _TensorGuards = None
_cur_device = torch._C._cuda_getDevice


def _never(*_args) -> bool:
    return False


class _Shape:
    """What every layer of one (device, K, N, dtype) shares: the sizes, the
    fast checks and the launch blocks (per thread: a block is not re-entrant)."""

    __slots__ = (
        "alt4",
        "alt5",
        "blocks",
        "chk4",
        "chk5",
        "device",
        "dtype",
        "index",
        "orow",
        "size_k",
        "size_n",
        "xrow",
    )  # fmt: skip

    def __init__(self, device, size_k: int, size_n: int, dtype):
        self.size_k, self.size_n, self.dtype = size_k, size_n, dtype
        self.device, self.index = device, device.index
        self.xrow, self.orow = 2 * size_k, 2 * size_n  # bytes of a row of x / out
        # thread id -> [block per row bucket ..., {(row chunks, bucket): block}]
        self.blocks: dict[int, list] = {}
        # the guard the fast path asks (calls without / with out=), and the
        # ones this shape has learned, most recently used first
        self.chk4 = self.chk5 = _never
        self.alt4: list = []
        self.alt5: list = []


_SHAPES: dict[tuple, _Shape] = {}
_HINT: dict[
    int, _Shape
] = {}  # weight address -> the shape seen there last (a hint only)
_HOST_LOCK = threading.Lock()
_MULTI_GPU = (
    False  # several CUDA devices: the current one is then checked on every call
)


def _shape(device, size_k: int, size_n: int, dtype) -> _Shape:
    key = (device.index, size_k, size_n, dtype)
    sh = _SHAPES.get(key)
    if sh is None:
        with _HOST_LOCK:
            sh = _SHAPES.get(key)
            if sh is None:
                global _MULTI_GPU
                _MULTI_GPU = torch.cuda.device_count() > 1
                sh = _SHAPES[key] = _Shape(device, size_k, size_n, dtype)
    return sh


_I32, _BF16, _FP16, _STRIDED, _Tensor = (
    torch.int32, torch.bfloat16, torch.float16, torch.strided, torch.Tensor,
)  # fmt: skip


def _checked(x, weight, scale, scale2, out):
    """Every check of the binding, by value, at its cheapest in Python: the
    shape record and M of a valid call, None for anything else (_explain then
    says what is wrong)."""
    try:
        ws = weight.shape
        xs = x.shape
        if len(ws) != 2 or len(xs) != 2:
            return None
        size_k, size_n, size_m = ws[0] * 16, ws[1] // 2, xs[0]
        dtype = scale2.dtype
        index = weight.get_device()
        if not (
            isinstance(weight, _Tensor)
            and weight.dtype is _I32
            and weight.layout is _STRIDED
            and weight.is_cuda
            and index == _cur_device()
            and weight.is_contiguous()
            and not (
                ws[1] % 2 or size_n % 128 or size_k % 64 or size_k < 64 or size_n < 128
            )
            and isinstance(scale, _Tensor)
            and scale.dtype in _SCALE_DTYPES
            and scale.shape == (size_k // 16, size_n)
            and scale.layout is _STRIDED
            and scale.is_cuda
            and scale.get_device() == index
            and scale.is_contiguous()
            and isinstance(scale2, _Tensor)
            and (dtype is _BF16 or dtype is _FP16)
            and scale2.numel() == 1
            and scale2.layout is _STRIDED
            and scale2.is_cuda
            and scale2.get_device() == index
            and isinstance(x, _Tensor)
            and x.dtype is dtype
            and xs[1] == size_k
            and x.layout is _STRIDED
            and x.is_cuda
            and x.get_device() == index
            and x.is_contiguous()
            and size_m >= MIN_M
        ):
            return None
        if out is not None:
            if not (
                isinstance(out, _Tensor)
                and out.dtype is dtype
                and out.shape == (size_m, size_n)
                and out.layout is _STRIDED
                and out.is_cuda
                and out.get_device() == index
                and out.is_contiguous()
            ):
                return None
            pa, pc = x.data_ptr(), out.data_ptr()
            if pc < pa + 2 * size_m * size_k and pa < pc + 2 * size_m * size_n:
                return None
        sh = _SHAPES.get((index, size_k, size_n, dtype))
        if sh is None:
            sh = _shape(weight.device, size_k, size_n, dtype)
        return sh, size_m
    except Exception:  # noqa: BLE001 -- not tensors, or tensors without sizes / strides
        return None


def _validate(x, weight, scale, scale2, out=None) -> tuple[_Shape, int]:
    """The shape record and M of a valid call; raises TypeError / ValueError
    for arguments outside the binding and NotImplementedError for M < 7."""
    return _checked(x, weight, scale, scale2, out) or _explain(
        x, weight, scale, scale2, out
    )


def _explain(x, weight, scale, scale2, out=None) -> tuple[_Shape, int]:
    """The checks of _checked once more, one by one, raising for the first
    that fails."""
    named = [("x", x), ("weight", weight), ("scale", scale), ("scale2", scale2)]
    if out is not None:
        named.append(("out", out))
    for name, t in named:
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
        if t.layout is not torch.strided:
            raise ValueError(f"{name} must be a dense tensor, got {t.layout}")
    if weight.dim() != 2 or weight.dtype != torch.int32:
        raise ValueError(
            f"weight must be int32 [K/16, 2N], got {weight.dtype} {tuple(weight.shape)}"
        )
    size_k, size_n = weight.shape[0] * 16, weight.shape[1] // 2
    if (
        weight.shape[1] % 2
        or size_n % 128
        or size_k % 64
        or size_k < 64
        or size_n < 128
    ):
        raise ValueError(
            f"need K % 64 == 0 and N % 128 == 0; weight {tuple(weight.shape)} gives K={size_k}, "
            f"N={weight.shape[1] / 2:g}"
        )
    if scale.dtype not in _SCALE_DTYPES or tuple(scale.shape) != (size_k // 16, size_n):
        raise ValueError(
            f"scale must be float8_e4m3fn [K/16, N] = [{size_k // 16}, {size_n}], "
            f"got {scale.dtype} {tuple(scale.shape)}"
        )
    dtype = scale2.dtype
    if dtype not in (torch.float16, torch.bfloat16) or scale2.numel() != 1:
        raise ValueError(
            f"scale2 must be one float16 / bfloat16 element, got {dtype} {tuple(scale2.shape)}"
        )
    device = weight.device
    if device.type != "cuda" or scale.device != device or scale2.device != device:
        raise ValueError(
            "weight, scale and scale2 must be on the same CUDA device, got "
            f"{weight.device}, {scale.device}, {scale2.device}"
        )
    if device.index != torch.cuda.current_device():
        raise ValueError(
            f"the tensors must be on the current CUDA device, got {device}"
        )
    if not (weight.is_contiguous() and scale.is_contiguous()):
        raise ValueError("weight and scale must be contiguous")

    if x.dtype != dtype:
        raise TypeError(f"x must be {dtype} like scale2, got {x.dtype}")
    if x.device != device:
        raise ValueError(f"x must be on {device} like the weights, got {x.device}")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.dim() != 2 or x.shape[1] != size_k:
        raise ValueError(f"x must be [M, {size_k}], got {tuple(x.shape)}")
    size_m = x.shape[0]
    if size_m < MIN_M:
        raise NotImplementedError(
            f"cute_nvfp4_batch handles M >= {MIN_M}, got M={size_m}"
        )
    if out is not None:
        if out.dtype != dtype:
            raise TypeError(f"out must be {dtype} like scale2, got {out.dtype}")
        if out.device != device:
            raise ValueError(
                f"out must be on {device} like the weights, got {out.device}"
            )
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if tuple(out.shape) != (size_m, size_n):
            raise ValueError(
                f"out must be [{size_m}, {size_n}], got {tuple(out.shape)}"
            )
        pa, pc = x.data_ptr(), out.data_ptr()
        if pc < pa + 2 * size_m * size_k and pa < pc + 2 * size_m * size_n:
            raise ValueError("out must not overlap x")
    return _shape(device, size_k, size_n, dtype), size_m


def _resolve(x, weight, scale, scale2, out, sh) -> _Shape:
    """The guard the fast path asked did not pass (``sh``: the shape it
    belongs to, if any). Ask the other guards of that shape; failing that,
    check the call in Python (raises) and teach the shape this kind of
    tensors."""
    args = (
        (weight, scale, scale2, x) if out is None else (weight, scale, scale2, x, out)
    )
    chk = None
    if sh is not None and not (_MULTI_GPU and _cur_device() != sh.index):
        try:
            for c in (sh.alt4 if out is None else sh.alt5)[1:]:
                if c(*args):
                    chk = c
                    break
        except Exception:  # noqa: BLE001, S110 -- the checks in Python will say
            pass
    if chk is None:
        sh, _ = _validate(x, weight, scale, scale2, out)
        if len(_HINT) >= 1 << 16:
            _HINT.clear()
        _HINT[weight.data_ptr()] = sh
        if _TensorGuards is None:
            return sh
    alts = sh.alt4 if out is None else sh.alt5
    try:
        if chk is None:  # (the hint was for another shape, or there was none)
            chk = next((c for c in alts if c(*args)), None)
        if chk is None:
            # Equal type, dtype, device, strides and sizes -- but for the rows
            # of x and out -- to tensors that have just passed _validate:
            # everything that checks is a function of these.
            sizes = [list(t.shape) for t in args]
            for s in sizes[3:]:
                s[0] = None
            chk = _TensorGuards(
                *args,
                dynamic_dims_sizes=sizes,
                dynamic_dims_strides=[list(t.stride()) for t in args],
            ).check
            if not chk(*args):
                return sh
    except Exception:  # noqa: BLE001 -- no guard for these: they stay on the checks in Python
        return sh
    alts = [chk] + [c for c in alts if c is not chk][:3]
    if out is None:
        sh.alt4, sh.chk4 = alts, chk
    else:
        sh.alt5, sh.chk5 = alts, chk
    return sh


def _plan_of(sh: _Shape, size_m: int) -> _Plan:
    bf16 = sh.dtype is torch.bfloat16
    chunks, rows = split_rows(size_m)
    sms = _sm_count(sh.index)
    g = select_geom(rows, bf16)
    cols = sh.size_n // g.tile_n * chunks
    g = select_geom(
        rows,
        bf16,
        long_stripes=sh.size_k // g.ks * cols >= LONG_STRIPE * sms,
        chunked=chunks > 1,
        lone=cols >= sms,  # every stripe then holds the end of a column
    )
    return _plan(sh.index, g, sh.size_k, sh.size_n, chunks)


def _staged(t: torch.Tensor, i: int) -> bool:
    """Argument ``i`` (x, weight, scale, scale2) cannot be read where it is
    (cp.async moves 16 B, scale2 is one 2 B load), or is better not (X32_BYTES)."""
    p = t.data_ptr()
    if i == 0:
        return bool(p & 15 or (p & 16 and t.nbytes >= X32_BYTES))
    return bool(p & (15 if i < 3 else 1))


def prepare(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, scale2: torch.Tensor):  # fmt: skip
    """Bind one call; returns ``(out, run)`` where ``run()`` relaunches into ``out``.

    The tensors are bound as they are now -- addresses and sizes -- and run()
    only relaunches: it sees values written into them later, not a tensor that
    is re-pointed or resized in place (bind again after that).
    """
    sh, size_m = _validate(x, weight, scale, scale2)
    out = torch.empty(size_m, sh.size_n, dtype=sh.dtype, device=sh.device)

    # Inputs the kernel does not read in place go through an aligned copy that
    # every run() refreshes.
    srcs = [x, weight, scale, scale2]
    copies = []
    for i, t in enumerate(srcs):
        if _staged(t, i):
            staged = torch.empty(t.shape, dtype=t.dtype, device=t.device)
            copies.append((staged, t))
            srcs[i] = staged

    bound = _Bound(_plan_of(sh, size_m), sh.index)
    bound.set_call(
        *(t.data_ptr() for t in srcs[:3]), out.data_ptr(), srcs[3].data_ptr(),
        size_m, sh.size_k, sh.size_n,
    )  # fmt: skip
    bound.set_stream(_raw_stream(sh.index))
    launch = bound.launch
    if copies:

        def run():
            for staged, t in copies:
                staged.copy_(t)
            launch()

    else:
        run = functools.partial(launch)
    # (aliases of the storages that were bound: they outlive a caller who
    # re-points one of the tensor objects)
    run.keep = (bound, out, copies, [t.detach() for t in (x, weight, scale, scale2)])
    return out, run


def _slow_call(sh: _Shape, x, weight, scale, scale2, out, size_m: int):
    """Everything nvfp4_linear() does not do in its straight line: a thread's
    first call for a shape and row split, and tensors that are staged through
    aligned temporaries (which only have to live until the launch is queued:
    later work on the stream is ordered behind it)."""
    chunks, rows = split_rows(size_m)
    bi = _ROW_BUCKET[rows]
    tid = _get_ident()
    tb = sh.blocks.get(tid)
    if tb is None:
        if len(sh.blocks) >= 256:  # thread ids of threads long gone
            sh.blocks.clear()
        tb = sh.blocks[tid] = [None] * _NB + [{}]
    if chunks == 1:
        bound = tb[bi]
        if bound is None:
            bound = tb[bi] = _Bound(_plan_of(sh, size_m), sh.index)
    else:
        big = tb[_NB]
        bound = big.get((chunks, bi))
        if bound is None:
            if len(big) >= 256:
                big.clear()
            bound = big[(chunks, bi)] = _Bound(_plan_of(sh, size_m), sh.index)
    srcs = [
        t.clone() if _staged(t, i) else t
        for i, t in enumerate((x, weight, scale, scale2))
    ]
    dst = out if not out.data_ptr() & 15 else torch.empty_like(out)
    bound.set_call(
        *(t.data_ptr() for t in srcs[:3]), dst.data_ptr(), srcs[3].data_ptr(),
        size_m, sh.size_k, sh.size_n,
    )  # fmt: skip
    bound.launch()
    if dst is not out:
        out.copy_(dst)
    return out


def nvfp4_linear(x, weight, scale, scale2, out=None):
    """NVFP4 Marlin linear: ``x [M, K] @ dequant(weight, scale, scale2) -> [M, N]``.

    ``out``, if given, is a contiguous [M, N] tensor of x's dtype on the same
    device that does not overlap x; it is written and returned as is. Raises
    TypeError / ValueError for arguments outside the binding and
    NotImplementedError for M < 7; never launches on them.
    """
    return linear_sized(x, weight, scale, scale2, out, -1)


def linear_sized(x, weight, scale, scale2, out, size_m: int):
    """nvfp4_linear() for a caller that has already read ``size_m = x.shape[0]``.

    The public entry (cute_nvfp4_marlin) dispatches on M, so it has x.shape in
    hand; this is the whole call path, with every check of nvfp4_linear().
    ``size_m`` must be x.shape[0] exactly, or negative if it is not known;
    ``out`` may be None.
    """
    # Type, dtype, device, sizes and strides of every tensor: one C call
    # against what the shape at this weight address has seen pass ...
    try:
        pb = weight.data_ptr()
        sh = _HINT[pb]
        if out is None:
            ok = sh.chk4(weight, scale, scale2, x)
        else:
            ok = sh.chk5(weight, scale, scale2, x, out)
    except Exception:  # noqa: BLE001 -- unknown weights, or not tensors at all
        sh = None
        ok = False
    # ... or, if that does not know them, another guard of the shape or all
    # the checks in Python (raises)
    if not ok or (_MULTI_GPU and _cur_device() != sh.index):
        sh = _resolve(x, weight, scale, scale2, out, sh)
        pb = weight.data_ptr()
    # What is left: the rows, out against them, and where everything is now.
    pa = x.data_ptr()
    ps = scale.data_ptr()
    ps2 = scale2.data_ptr()
    if size_m < 0:  # from the bytes of x: cheaper than x.shape
        xb = x.nbytes
        size_m = xb // sh.xrow
    else:
        xb = size_m * sh.xrow
    if size_m < MIN_M:
        raise NotImplementedError(
            f"cute_nvfp4_batch handles M >= {MIN_M}, got M={size_m}"
        )
    if out is None:
        # (sizes as separate arguments: 0.8 us cheaper than a tuple)
        out = torch.empty(size_m, sh.size_n, dtype=sh.dtype, device=sh.device)
        pc = out.data_ptr()
    else:
        pc = out.data_ptr()
        if out.nbytes != size_m * sh.orow:
            raise ValueError(
                f"out must be [{size_m}, {sh.size_n}], got {tuple(out.shape)}"
            )
        if pc < pa + xb and pa < pc + size_m * sh.orow:
            raise ValueError("out must not overlap x")
    bound = None
    tb = sh.blocks.get(_get_ident())
    if tb is not None:
        if size_m <= MAX_ROWS:
            rows = size_m
            bound = tb[_ROW_BUCKET[rows]]
        else:  # row chunks of equal height
            chunks = -(-size_m // MAX_ROWS)
            rows = -(-size_m // chunks)
            bound = tb[_NB].get((chunks, _ROW_BUCKET[rows]))
    if (
        bound is None
        or (pa | pb | ps | pc | ps2 << 3)
        & 15  # cp.async moves 16 B; scale2 is a 2 B load
        or (pa & 16 and xb >= X32_BYTES)
    ):
        return _slow_call(sh, x, weight, scale, scale2, out, size_m)
    # this thread's block for the shape and row split: eight cells of it
    # change from call to call
    raw = _raw_stream(sh.index)
    if raw != bound.raw:
        bound.set_stream(raw)
    gen = next(_GEN) & 0x7FFFFFFF or 1
    if not gen & 255:  # every _ERR_EVERY launches
        _err_sample(sh.index)
    _pack_call(bound.vals, 0, pa, pb, ps, pc, ps2, rows, size_m, gen)
    bound.call(bound.arg)
    if bound.res.value:
        raise RuntimeError(
            f"cute_nvfp4_batch: kernel launch failed (CUDA error {bound.res.value})"
        )
    return out
