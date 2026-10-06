# NVFP4 Marlin (W4A16) on Ampere — CuTe DSL fork of sglang's kernel

A fork of sglang's NVFP4 Marlin GEMM — weight-only 4-bit `e2m1` weights with
one fp8 block scale per 16 weights, fp16 / bf16 activations, for GPUs without
native FP4/FP8 — as standalone NVIDIA CuTe DSL (Python) kernels, made as fast
as this box's single **A100-SXM4-80GB** allows.

```python
out = nvfp4_linear(x, weight, scale, scale2, out=None)
```

Target workload: the routed MoE experts of Inkling-Small — fused gate/up
`x [M, 4096] -> [M, 4096]` and down `x [M, 2048] -> [M, 4096]`, bf16, `M` =
tokens routed to the expert (`M = 1` first: single-sequence decode), weights
always cold (a different expert every call).

On the 16 target points (`M` = 1 … 128 on both GEMMs) the kernels are a
geomean **1.82x** faster than sglang's own Marlin kernel on the same inputs
(1.76x at `M = 1`: 9.5 / 6.9 us against 15.7 / 12.8), and 1.35x at prefill
sizes (`M` = 256 … 1024). [Results](#results-a100-sxm4-80gb) has every row.

| file | what |
|---|---|
| `cute_nvfp4_marlin.py` | **public entry**: `nvfp4_linear`, `prepare`; dispatches on `M` (and on `N` at `M` = 7, 8); `main()` gates and benches the target problem |
| `cute_nvfp4_decode.py` | decode kernel, `1 <= M <= 8`: every CTA owns output columns, no cross-CTA reduction |
| `cute_nvfp4_batch.py` | batch kernel, `M >= 7`: K-striped tiles, fp32 partials handed from CTA to CTA |
| `cute_nvfp4_marlin_port.py` | the faithful port of Marlin's architecture, any `M`; reference only, not dispatched to |
| `marlin_utils.py` | pure-torch NVFP4 quantize / Marlin pack and an fp64 oracle |
| `sglang_ref.py` | sglang's real kernel, JIT-compiled from the submodule: ground truth and baseline |
| `harness.py` | correctness gate and cold-weight benchmark |

## Provenance

Everything is derived from the vendored sglang tree, `_third_party/sglang` at
`9cc7da2ab0b305f7aecd45f7b0854c7e20535640` (2026-09-19). Paths below are under
`python/sglang/`.

| here | sglang origin | relation |
|---|---|---|
| `cute_nvfp4_marlin_port.py` | `kernels/jit/csrc/gemm/marlin/{marlin_template.h, gptq_marlin.cuh, marlin.cuh, dequant.h}`, the `w_type == kFE2M1f` path | re-implemented in CuTe DSL, architecture kept |
| `cute_nvfp4_decode.py`, `cute_nvfp4_batch.py` | same files | keep the packed weight / scale layout, the dequant bit patterns and the fp32-accumulate, convert, `* scale2` epilogue; everything else is new |
| `marlin_utils.py` | `srt/layers/quantization/marlin_utils_fp4.py` (`prepare_nvfp4_layer_for_marlin`) and `kernels/jit/csrc/gemm/marlin/gptq_marlin_repack.cuh` | pure-torch port; `sglang_ref.check_pack_matches_sglang` asserts the packing is bit-identical to the CUDA repack |
| `sglang_ref.py` | `gptq_marlin.cuh`, `gptq_marlin_repack.cuh` through sglang's `load_jit`, called as `apply_fp4_marlin_linear` calls it | unmodified upstream, compiled with nvcc on first use |

The Marlin kernel itself descends from [IST-DASLab Marlin](https://github.com/IST-DASLab/marlin)
by way of vLLM's `gptq_marlin`; `../fp8_marlin/` has the W8A16 sibling and the
background on the offline layout.

## The binding

```
x       [M, K]       fp16 | bf16
weight  [K/16, 2*N]  int32            Marlin-packed e2m1 codes, 8 nibbles per word
scale   [K/16, N]    float8_e4m3fn    Marlin-processed block scales (one per 16 weights along K)
scale2  [1]          same dtype as x  global scale, exponent bias folded in
out     [M, N]       same dtype as x
all contiguous, on one CUDA device; N % 128 == 0, K % 64 == 0, M >= 1
```

`marlin_utils.py` documents the layouts and builds them
(`nvfp4_quantize` -> `marlin_prepare`); its `reference_linear` is an fp64
oracle built from the kernel's exact dequant bit patterns. The weight stream
is "fragment-ready": one `int4` per lane per k16 row lands in registers in
`mma.m16n8k16` operand order, so no kernel here needs `ldmatrix` or a swizzle
for the weights.

## Usage

Run everything from the repo root (see `AGENTS.md`).

```bash
# entry point: the two expert GEMMs at M (default 1), gated against sglang's
# kernel and the fp64 oracle, then benchmarked against sglang and dense cuBLAS
uv run --frozen -m nvfp4_marlin.cute_nvfp4_marlin
uv run --frozen -m nvfp4_marlin.cute_nvfp4_marlin --m 16
uv run --frozen -m nvfp4_marlin.cute_nvfp4_marlin --shapes target --no-gate

# correctness gate (--impl defaults to the public entry)
uv run --frozen -m nvfp4_marlin.harness --mode gate --gate full
uv run --frozen -m nvfp4_marlin.harness --mode gate --gate highm
uv run --frozen -m nvfp4_marlin.harness --impl nvfp4_marlin.cute_nvfp4_marlin_port --mode gate --gate full

# benchmark: prepared run() and, with --e2e, the full call three ways
uv run --frozen -m nvfp4_marlin.harness --mode bench --shapes target --e2e
uv run --frozen -m nvfp4_marlin.harness --mode bench --shapes prefill --e2e
uv run --frozen -m nvfp4_marlin.harness --mode bench --shapes "8,4096,8192;9,4096,8192"
```

```python
import torch

# (the package exports are lazy: marlin_utils alone needs no cutlass)
from nvfp4_marlin import marlin_utils as mu
from nvfp4_marlin import nvfp4_linear, prepare

# offline: quantize a layer's weight [N, K] and Marlin-pack it
w = torch.randn(4096, 2048, device="cuda")
b = mu.marlin_prepare(mu.nvfp4_quantize(w), torch.bfloat16)
x = torch.randn(3, 2048, device="cuda", dtype=torch.bfloat16)

# allocates y
y = nvfp4_linear(x, b.weight, b.scale, b.scale2)
# writes into y and returns it
nvfp4_linear(x, b.weight, b.scale, b.scale2, out=y)
# bind once, then only relaunch into `out`
out, run = prepare(x, b.weight, b.scale, b.scale2)
run()
```

The gate requires the error against the fp64 oracle to be no worse than
sglang's on the same inputs (2x on the max, 1.25x on the mean), for both
dtypes, realistic and stress weights, plus `run()` idempotency and
`nvfp4_linear() == prepare() + run()`. The benchmark rotates through 16
layers, each with its own `x`, so the weights are streamed from HBM on every
call rather than served from the A100's 40 MB L2. The first use of a kernel
specialization compiles it (0.6 – 1.6 s); compiled objects are cached in
`~/.cache/nvfp4_marlin_cute` (override: `NVFP4_CUTE_CACHE`, the only
environment variable the kernels read), after which a fresh process pays
3 – 16 ms per specialization. `sglang_ref` compiles sglang's kernel with nvcc
on its first use (minutes; cached under `~/.cache/sglang/jit`).

## Results (A100-SXM4-80GB)

bf16, cold weights, microseconds. `run()` is the prepared launch (the kernel
as the GPU sees it, launch included), next to sglang's kernel and dense
cuBLAS (`torch.matmul` on the pre-dequantized weight: 4x the weight bytes, a
reference point, not a competitor) timed the same way. The three call columns
are the full `nvfp4_linear()` call: into a preallocated `out=`, allocating its
output, and sglang's own call (`sglang_ref.nvfp4_linear`, which allocates its
output and reduction scratch per call as `apply_fp4_marlin_linear` does).
From `harness --mode bench --shapes target --e2e` and `--shapes prefill
--e2e` on the files as they are here (torch 2.14.0+cu132,
nvidia-cutlass-dsl 4.7.1); best of 7 interleaved rounds, median / best below
1.04 on every row.

| M | K | kernel | run() us | sglang us | speedup | dense us | call, `out=` us | call, allocating us | sglang call us |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | decode | 9.5 | 15.7 | 1.65x | 25.1 | 9.5 | 10.9 | 30.6 |
| 1 | 2048 | decode | 6.9 | 12.8 | 1.87x | 14.8 | 7.4 | 11.0 | 30.6 |
| 2 | 4096 | decode | 10.1 | 15.6 | 1.55x | 26.2 | 10.1 | 10.9 | 30.6 |
| 2 | 2048 | decode | 7.3 | 12.8 | 1.76x | 15.6 | 7.4 | 11.0 | 30.9 |
| 4 | 4096 | decode | 10.3 | 15.7 | 1.52x | 26.3 | 10.2 | 11.0 | 30.7 |
| 4 | 2048 | decode | 7.4 | 12.8 | 1.72x | 15.6 | 7.4 | 10.9 | 30.4 |
| 8 | 4096 | batch | 11.8 | 15.8 | 1.33x | 26.9 | 11.8 | 11.9 | 30.7 |
| 8 | 2048 | batch | 8.5 | 12.9 | 1.51x | 15.7 | 8.5 | 12.0 | 30.6 |
| 16 | 4096 | batch | 11.9 | 17.3 | 1.45x | 25.3 | 11.9 | 11.9 | 30.5 |
| 16 | 2048 | batch | 8.6 | 14.0 | 1.63x | 15.1 | 8.6 | 11.3 | 30.4 |
| 32 | 4096 | batch | 14.0 | 28.7 | 2.06x | 27.2 | 13.9 | 13.8 | 31.7 |
| 32 | 2048 | batch | 9.6 | 24.8 | 2.58x | 16.3 | 9.7 | 11.8 | 30.7 |
| 64 | 4096 | batch | 18.0 | 44.2 | 2.46x | 27.5 | 18.0 | 17.8 | 43.9 |
| 64 | 2048 | batch | 12.0 | 38.0 | 3.17x | 16.4 | 12.1 | 12.0 | 37.8 |
| 128 | 4096 | batch | 26.0 | 44.9 | 1.73x | 37.9 | 25.9 | 25.8 | 44.6 |
| 128 | 2048 | batch | 16.9 | 31.9 | 1.88x | 19.1 | 16.9 | 16.7 | 31.6 |
| 256 | 4096 | batch | 45.0 | 61.9 | 1.38x | 56.3 | 44.1 | 43.7 | 60.9 |
| 256 | 2048 | batch | 26.4 | 37.3 | 1.41x | 31.4 | 26.4 | 26.2 | 36.8 |
| 512 | 4096 | batch | 84.6 | 111.2 | 1.31x | 91.7 | 83.2 | 83.2 | 111.3 |
| 512 | 2048 | batch | 45.7 | 64.3 | 1.41x | 55.5 | 46.3 | 45.5 | 62.9 |
| 1024 | 4096 | batch | 164.1 | 208.5 | 1.27x | 206.8 | 159.9 | 159.8 | 206.2 |
| 1024 | 2048 | batch | 85.3 | 114.5 | 1.34x | 109.5 | 86.3 | 85.4 | 114.2 |

Geomean speedup over sglang: 1.76x at `M = 1`, 1.56x for `M` = 2 … 8, 2.06x
for `M` = 16 … 128, 1.82x over the 16 target points, 1.35x for prefill.

Reading the call columns: a call with `out=` costs about 7.2 – 7.6 us of
host time, so it reads as `run()` wherever the kernel takes longer than that
and is host-bound below (only `K = 2048`, `M <= 4`: 7.4 us). The allocating
call adds a `torch.empty` (3.5 us) and is host-bound at about 11 us through
the decode kernel and 11.3 – 12 us through the batch kernel wherever the
kernel is shorter than that. Prefer `out=`. Two runs of this benchmark a
quarter of an hour apart agreed within 0.2 us on every `run()` row up to
`M = 128`; the host-bound call columns moved by up to 0.4 us with the load
on the machine.

The gate on the same files: `--gate full` 192/192 (48 shapes x fp16 / bf16 x
realistic / stress), worst error ratio against sglang 1.003 on the max and
1.0003 on the mean; `--gate highm` 56/56 (`M` = 129 … 4097); the reference
port 192/192 on `--gate full`.

## Dispatch

```
cute_nvfp4_decode   M <= 6, and M = 7, 8 on layers with N >= 7168
cute_nvfp4_batch    every other M >= 7
```

Both kernels accept `M` = 7 and 8, so the boundary is a matter of speed only
(`cute_nvfp4_marlin.kernel_for(M, N)` names the kernel). The data behind the
rule: prepared `run()` of both kernel modules, bf16, cold weights, harness
timing, mean over three sets of `x` addresses, microseconds. `batch*` is the
batch kernel with its lower bound lifted through its developer hook — the
`M = 7` specialization for `M` = 4 … 6, every row of `x` predicated for `M` =
1 … 3 — which is not shipped; it passes the `lowm` gate (72/72 with every row
predicated for `M <= 7`).

| M | decode, K=4096 | batch, K=4096 | sglang | decode, K=2048 | batch, K=2048 | sglang | sum of the two GEMMs: decode / batch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.48 | 12.12\* | 15.63 | 6.82 | 8.37\* | 12.65 | 16.29 / 20.49\* |
| 2 | 10.01 | 12.10\* | 15.60 | 7.19 | 8.38\* | 12.62 | 17.20 / 20.48\* |
| 3 | 10.03 | 12.15\* | 15.69 | 7.27 | 8.39\* | 12.67 | 17.30 / 20.54\* |
| 4 | 10.23 | 11.68\* | 15.64 | 7.46 | 8.26\* | 12.64 | 17.69 / 19.94\* |
| 5 | 11.45 | 11.72\* | 15.72 | 8.27 | 8.30\* | 12.68 | 19.71 / 20.02\* |
| 6 | 11.61 | 11.73\* | 15.68 | 8.55 | 8.31\* | 12.67 | 20.17 / 20.04\* |
| 7 | 11.79 | 11.74 | 15.73 | 8.75 | 8.33 | 12.73 | 20.53 / 20.07 |
| 8 | 11.82 | 11.75 | 15.70 | 8.95 | 8.36 | 12.68 | 20.77 / 20.12 |
| 9 | – | 11.78 | 17.12 | – | 8.37 | 13.83 | – / 20.15 |
| 10 | – | 11.79 | 17.04 | – | 8.38 | 13.79 | – / 20.18 |
| 11 | – | 11.79 | 17.14 | – | 8.39 | 13.84 | – / 20.18 |
| 12 | – | 11.80 | 17.09 | – | 8.40 | 13.80 | – / 20.20 |
| 13 | – | 11.82 | 17.14 | – | 8.44 | 13.85 | – / 20.26 |
| 14 | – | 11.82 | 17.10 | – | 8.46 | 13.82 | – / 20.28 |
| 15 | – | 11.80 | 17.17 | – | 8.50 | 13.85 | – / 20.30 |
| 16 | – | 11.84 | 17.14 | – | 8.50 | 13.85 | – / 20.35 |

The decode kernel slows down with `M` (every CTA re-reads all of `x`); the
batch kernel is flat up to 16 rows. At `K = 4096` they meet at `M = 7`; at
`K = 2048` the batch kernel is ahead from `M = 6` on, by 0.4 us at `M = 7`
and 0.6 us at `M = 8`. Over the two GEMMs of an expert `M = 6` is level
(20.17 against 20.04), so the boundary is 6 | 7 and the batch kernel got one
more specialization for it (below). The decode kernel also depends on where
`x` lives from `M = 4` on: over the three address sets `M = 8` reads
11.68 – 12.00 / 8.80 – 9.07 us, the batch kernel 11.75 – 11.76 / 8.34 – 8.40.

Elsewhere, around the boundary (same method; the faster kernel in bold when
they differ by 0.1 us or more):

| K | N | M=6: decode / batch\* | M=7: decode / batch | M=8: decode / batch | M=9: batch | sglang at M=8 | dispatched at M=7, 8 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2048 | 2048 | 7.11 / **6.99** | 7.41 / **7.00** | 7.52 / **7.02** | 7.03 | 13.09 | batch |
| 4096 | 2048 | 9.36 / **8.79** | 9.70 / **8.81** | 10.03 / **8.82** | 8.82 | 16.13 | batch |
| 8192 | 2048 | 14.07 / **12.16** | 14.63 / **12.19** | 15.44 / **12.20** | 12.22 | 19.19 | batch |
| 4096 | 3072 | 10.24 / 10.18 | 10.63 / **10.21** | 10.91 / **10.24** | 10.25 | 14.29 | batch |
| 1024 | 4096 | 6.99 / 7.00 | 6.60 / 6.65 | 6.70 / 6.70 | 6.68 | 10.25 | batch |
| 8192 | 4096 | **18.20** / 18.39 | 18.43 / 18.43 | 18.71 / **18.44** | 18.46 | 21.02 | batch |
| 4096 | 5120 | **13.17** / 13.36 | 13.58 / **13.37** | 13.98 / **13.39** | 13.42 | 15.96 | batch |
| 2048 | 6144 | 10.22 / **9.79** | 10.50 / **9.78** | 10.85 / **9.83** | 9.86 | 11.92 | batch |
| 4096 | 6144 | **14.70** / 14.85 | 15.06 / **14.87** | 15.49 / **14.84** | 14.89 | 17.37 | batch |
| 8192 | 6144 | **24.32** / 25.13 | **24.67** / 25.30 | 25.48 / **25.21** | 25.05 | 25.88 | batch |
| 4096 | 7168 | **15.65** / 16.32 | **15.89** / 16.36 | **16.10** / 16.39 | 16.41 | 18.63 | decode |
| 1024 | 8192 | **8.12** / 8.30 | **8.18** / 8.32 | 8.29 / 8.34 | 8.34 | 11.47 | decode |
| 2048 | 8192 | 11.52 / 11.57 | 11.53 / 11.55 | 11.70 / **11.58** | 11.57 | 14.37 | decode |
| 4096 | 8192 | **17.08** / 18.14 | **17.28** / 18.15 | **17.46** / 18.18 | 18.17 | 20.20 | decode |
| 8192 | 8192 | **29.23** / 32.81 | **29.47** / 33.00 | **29.94** / 33.06 | 32.30 | 32.26 | decode |
| 4096 | 14336 | **26.86** / 29.35 | **27.03** / 29.49 | **27.23** / 29.66 | 29.24 | 29.21 | decode |

The crossover moves with `N`, not with `M` alone: up to `N = 6144` the batch
kernel wins or ties at `M` = 7, 8 with one exception (by up to 3.2 us on
narrow, deep layers), from `N = 7168` on the decode kernel does (by up to
3.5 us), which is the one shape term in the rule. The rule misses by 0.6 us
at (7, 8192, 6144), the exception, and by 0.1 us at (8, 2048, 8192). On the
widest layers the batch kernel is only at parity with sglang at `M = 8`
(33.1 against 32.3 us at `K = N = 8192`, 29.7 against 29.2 at
`N = 14336`), and the step from `M = 8` to `M = 9` is a cliff of 2.0 – 2.4 us
there (the decode kernel stops at 8 rows).

One caveat at `M` = 7, 8: the rule ranks the kernels, i.e. `run()` and the
`out=` call. An *allocating* call is host-bound at `K = 2048` whichever
kernel serves it, and the decode kernel's host path is the shorter one
(10.6 – 11.0 us against 12.0 – 12.1 through the public entry, where the
`out=` call is 8.5 against 8.8 – 9.0).

## How it works

### Measured facts the designs follow from

All from runs on this box with the final files (the first three with the
development probes: a spin kernel and a fetch-only kernel).

| fact | measured |
|---|---|
| host cost of one launch through the pre-packed C entry point | 3.4 – 3.7 us (trivial kernel, back to back); prepared `run()` 3.9 us (decode), 4.5 us (batch) |
| GPU-side fixed cost of a kernel, independent of CTA count, threads and smem | 2.6 – 2.9 us (intercept of a spin kernel's time against its work, 108 CTAs and more) |
| streaming one cold packed layer with no compute | 9.44 MB (K = N = 4096) in 9.3 – 9.6 us, 4.72 MB (K = 2048) in 5.6 – 6.5 us; cp.async -> smem, cp.async + LDS and plain LDG all land there. From L2 the 9.44 MB take 4.9 – 5.3 us |
| decode kernel at `M = 1` | 9.5 / 6.9 us: at the stream's floor |
| the batch architecture (tiles + hand-over) forced down to `M = 1` | 12.1 / 8.4 us, i.e. +2.6 / +1.6 us over column ownership |
| the Marlin port (cross-CTA reduction, k-split warps) at `M = 1` | 11.9 / 9.2 us |
| above the stream | `M = 16`: 11.9 us, `M = 128`: 26.0 us for the layer that streams in 9.3 – 9.6 us: bound by instructions (dequant, MMA, re-reads of `x`), not by HBM |
| bf16 `weight * scale` | native: `fma.rn.bf16x2` compiles to one `HFMA2.BF16_V2` per packed pair (257 of the 1920 instructions of the `M = 1`, `K = 2048` decode kernel, next to 64 `HMMA.16816.F32.BF16`; fp16 gets `HMUL2`). sglang's build uses the same instruction |

So at `M = 1`, `K = 2048` the kernel's 6.9 us are 2.6 – 2.9 us of fixed cost
plus about 4 us in which 4.72 MB have to arrive: the stream wants nearly all
of it, and anything serial at the end of the kernel — a partial stored by one
SM and fetched by another, a lock chain — is paid in full. With more rows the
stream stops being the bound and the kernel has to keep an SM's instruction
budget on dequant and MMA rather than on bookkeeping.

### Decode kernel (`cute_nvfp4_decode.py`, M <= 8): column ownership

Every CTA owns a set of output columns outright and walks all of `K` for
them: no locks, no scratch, no cross-CTA reduction, one launch. The
ownership unit is a *slice* — the 8 output columns that four lanes of a warp
hold of a Marlin-packed k16 row, i.e. 64 B of weights plus exactly 8 scale
bytes, whole 32 B sectors of the packed stream — so no CTA ever pulls a byte
it does not use; `N = 4096` is 512 slices, dealt 4 or 5 per CTA over one CTA
per SM. The MMA is Marlin's transposed `mma.m16n8k16` (weights as operand A,
`x` as operand B). The `K` range of a slice is split over *classes* of k16
rows, one group of four lanes per (slice, class), so the grid sweeps the
packed matrix sequentially in `K`. At `M = 1` (DIAG) each of the 8 B columns
carries `x` at a different k16 row: 8 independent streams per warp, only the
diagonal of the result is read. At `M` = 2 … 8 (QUAD) B columns 0-3 / 4-7
carry rows 0..3 of `x` for two classes and a second MMA on the same
dequantized A carries rows 4..7.

Eight warps per CTA (two per warp scheduler), each with a private 5-step
cp.async ring: per step one `int4` of weights per lane and 8 scale bytes per
group, then dequant (3 integer ops per packed register, one packed multiply)
and the MMAs. Nothing in the loop is synchronous and no warp reads what
another fetched. At the end every group leaves its fp32 partial sums in smem,
one barrier, and each output is summed over its classes in a fixed order,
converted, multiplied by `scale2` in the activation dtype and stored.

The price is `x`: every CTA needs all of it, 108 SMs ask L2 for the same
lines at the same time, and that — the number of requests per line, not the
bytes — is what makes `M > 1` slower than `M = 1` and the kernel sensitive to
the address of `x` from `M = 4` on. From `M = 3` the kernel can fetch `x`
once per CTA in whole lines, a few steps ahead, into one of two smem buffers
(BAND) instead of per warp through the ring; where that pays was decided by
measurement and is a table in `_quad_plan`.

### Batch kernel (`cute_nvfp4_batch.py`, M >= 7): K-striped tiles with a hand-over

Column ownership stops paying when `x` is large: with K-striping every CTA
reads only the `x` of its own k-tiles. The batch kernel keeps Marlin's
striping across CTAs and changes what surrounds it.

- **Three kinds of warps**, one CTA per SM, meeting at named barriers. Four
  MMA warps only read smem, dequantize and multiply; IO warps (one; two above
  64 rows) issue every cp.async of the pipeline in straight-line code; one X
  warp does everything that crosses CTAs.
- **Geometry per row bucket** (7, 8 … 16, … 32, … 128 rows): a warp owns
  `16 * mb` rows of 32 columns, 2 x 2 warps split a 64-column tile along `N`
  and a 64-deep stage along `K`, and one dequantized B fragment feeds `mb`
  MMAs — `M = 128` is one problem, not two 64-row problems that each
  dequantize the layer. One continuous cp.async pipeline of 3 – 8 stages
  `[A tile | packed B rows | scale rows]` per CTA; A through a swizzled tile
  and `ldmatrix`, B and scales copied verbatim and read straight into
  registers.
- **Stripes.** The (k-tile, column-tile) units, column-major, are cut into one
  contiguous stripe per CTA by a host-built table. Stripes are deliberately
  unequal: a CTA that only computes a partial gets fewer units than one that
  finishes a column, so that its partial is on its way while the finisher
  still computes.
- **Lock-free hand-over.** Every contributor to a column except the one
  holding its bottom stores its fp32 partial to its own scratch slot, fences,
  and release-stores the launch's generation number into its flag. On the
  CTA that finishes the column the X warp acquire-polls the flags of its
  contributors from the start and cp.asyncs each partial into smem as soon as
  its flag is up; the MMA threads add them in a fixed order and write the
  output. Flags are compared against the generation, so there is no state to
  reset between launches. Scratch is per (device, stream).
- **Row chunks above 128 rows**: `ceil(M / 128)` chunks of equal height in
  one launch, one continuous pipeline per CTA across chunks, the same
  hand-over.

What the hand-over costs is the +2.6 / +1.6 us of the table above at `M = 1`
(with the pipeline lead and epilogue that come with it), which is why it
only starts at `M = 7`. The `M = 7` bucket is the 16-row tile of the 8 … 16
bucket with one difference, `Geom.m_min`: rows of `x` that exist for every
`M` of a bucket are fetched without a predicate, in groups of four — rows
0..7 for 8 … 16, only rows 0..3 for `M = 7`. Predicating rows 0..3 as well
costs 0.4 – 0.5 us at `K = 4096` (the `batch*` rows for `M` <= 3 above), so
the bound was not simply lowered to 1.

### Reference port (`cute_nvfp4_marlin_port.py`, any M)

The literal port: Marlin's 4-stage cp.async pipeline carrying A, packed B
and scales together, double-buffered registers, fragment-ready B dequantized
in registers, threads split `n_warps x k_warps` with the k-split partial sums
reduced through smem in Marlin's tree order, the transposed MMA for
`M <= 8`, the striped CTA schedule with an fp32 cross-CTA reduction through
scratch, 64-row sub-problems above 64 rows, and Marlin's thread configs
(re-ranked by measurement). It departs from the `.cuh` where the A100 made
the original slow: the cross-CTA reduction polls a sentinel instead of
chaining a ticket lock, the k-group reduction is dealt to all k-groups, and
the fetch pipeline does not drain at a column boundary (its docstring has
the list). On the target points it reads 11.9 / 9.2 us at `M = 1`,
13.8 / 10.4 at `M = 16`, 24.2 / 17.8 at `M = 64` and 35.0 / 25.1 at
`M = 128` — 1.25x – 2.1x over sglang, behind the two specialists everywhere.
It kept its first-round host path: no on-disk compile cache (every process
compiles), Python validation on every call (`out=` call 13 us, allocating
17 us where the kernel is shorter).

## Call path, registries and their limits

The public entry reads `size_m, size_k = x.shape`, picks the kernel, and
hands the sizes on so that no kernel reads them twice; validation is each
kernel's own and happens once. Measured on a host-bound problem (`K = 64`,
`N = 128`), microseconds per `out=` call: the decode kernel's module called
directly 7.3, through the public entry 7.3 (+0.0); the batch kernel's 7.5 –
7.6 directly, +0.2 through the public entry and +0.4 – 0.5 at `M` = 7, 8,
where the wide-layer test reads `weight.nbytes` as well. On the expert shapes
every batch call is GPU-bound, so none of that shows in the results table.
(Both modules' own `nvfp4_linear` became thin wrappers around `linear_sized`
for this; against the modules as they were before the integration the public
entry measured +0.1 us on decode calls and +0.2 – 0.3 us on batch calls.)

A layer may be called with any `M` from call to call. The two kernels keep
separate state about it and neither looks at the other's:

**Decode kernel: a registry keyed on the three tensor objects.**
`weight`, `scale` and `scale2` are validated in full when a triple of tensor
*objects* is first seen. Later calls trust the registration only for the
same three objects, at the same addresses, on allocations that are still
alive (weak references to the storages); anything else is validated from
scratch. `x` and `out` are checked in full on every call.

- The fast path needs the **same Python objects** on every call. Passing
  fresh objects for the layer (`param.data`, `W_all[e]`) is correct but
  validates everything every time: 14.7 us per call instead of 7.2.
- A registered tensor **edited in place on the same allocation at the same
  address** (`w.data = w.data[:r]`, `t_()`, a dtype view) is not noticed by a
  call that still fits the registered layer: it is computed as the layer
  that was registered — on memory that was validated and is still owned,
  never out of bounds. A call that does not fit is re-validated. Re-pointing
  to *other* memory (`param.data = new`, `.cpu().cuda()`, `load_state_dict`
  into new storage) is noticed.

**Batch kernel: by value.** Every call checks all five tensors by value —
Python type, dtype, device, sizes, strides — in one C call
(`torch._C._dynamo.guards.TensorGuards`, a private torch API; without it the
same checks run in Python, slower and never different) and launches with the
addresses it has just read. Launch state is per (device, K, N, dtype), not
per layer, so it has no in-place-edit limit, and fresh layer objects cost
9.1 us per call instead of 7.6.

**`prepare()` / `run()`** (both kernels) bind the memory the five tensors
have at `prepare()` and the stream that is current then (decode) or at each
`run()` (batch); `run()` keeps those allocations alive and sees values
written into them later, not a tensor that is re-pointed or resized
afterwards. A prepared `run()` of the batch kernel is not re-entrant.

**The compile cache** is shared by the two kernels and they follow one
scheme (the block is byte-identical in both files): a file is
`<module>_<12 hex of the module's sha256>_<16 hex of the key>.o`, the key
being the launcher, the specialization, the GPU's compute capability and the
cutlass / torch / CUDA versions; its content is the relocatable object plus
its sha256, and a file that does not verify is rebuilt, never linked; writes
are atomic renames; on a miss a module prunes only *its own* objects that no
process has loaded for 14 days (every hit refreshes the mtime). Objects of
an edited source file are never picked up.

**A flag-poll timeout is reported late, not silently.** A finisher that has
polled a contributor's flag 2^20 times gives up rather than hang the GPU
(0.24 – 0.34 s in the white-box test; it cannot happen while launches that
share scratch are serialized, which per-stream scratch guarantees, and no
test other than editing a stripe record on the device has produced one).
That launch returns normally with one 64-column tile of its output NaN and
its generation number in the scratch's error word. The host cannot look
there without synchronizing, so every 256 launches it copies the error words
to page-locked host memory (asynchronously, on a side stream) and looks at
what the previous copies brought back: a later call into the batch kernel —
any layer, `run()` or `nvfp4_linear()`, at most 512 launches after the
timed-out one has run (508 – 509 in the test) — raises `RuntimeError`
instead of launching, once per timeout. `cute_nvfp4_batch.poll_timeouts()`
reads the words synchronously. Cost on the hot path: one AND per launch.

Developer tools that stay in the files and cannot change results unless a
developer calls them: `cute_nvfp4_decode.debug_knobs()` / `profile()`
(geometry sweeps, phase-stamp builds, SASS dumps), `cute_nvfp4_batch._TUNE`
(geometry and schedule overrides; `dbg` builds) and the port's `_OVERRIDE` /
`SPIN_LIMIT`. None is read from the environment.

## Where it deviates from sglang

- **Summation order.** A different tiling sums the fp32 partial products in
  a different order, so outputs are not bit-identical to sglang's; the gate
  holds the error against the fp64 oracle to sglang's own (worst ratio 1.003
  on the max, 1.0003 on the mean over the full gate). The dequantized
  weights, the fp32 accumulation, and the convert-then-multiply-by-`scale2`
  in the activation dtype are sglang's.
- **Deterministic.** Every reduction runs in a fixed order; relaunches are
  bit-identical (the gate checks it on every case).
- **NVFP4 only, one configuration.** No zero points, no act-order, no
  per-group index; fp32 reduction always (sglang's `use_fp32_reduce=True`,
  `use_atomic_add=False`, as `apply_fp4_marlin_linear` sets them).
- **No workspace arguments.** sglang's `workspace` (locks) and `c_tmp`
  (fp32 reduce scratch) are gone: the decode kernel needs neither, the batch
  kernel and the port keep their scratch in the module, per (device, stream).
- **No cross-CTA lock.** The ticket-lock chain is replaced by column
  ownership (decode), generation flags (batch) or a sentinel poll (port).
  The batch kernel's poll is bounded (see the timeout above); sglang's lock
  wait is not.
- **Rows.** sglang runs more than 64 rows as 64-row sub-problems; the batch
  kernel has tiles up to 128 rows and equal row chunks above.
- **Launch path.** Launches go through the JIT-ed C entry point on
  pre-packed arguments; inputs that are contiguous but not 16-byte aligned
  are served through aligned copies. A uint8 view is accepted for `scale`.

## Known limitations

- Tuned and verified on one A100 (sm80, 108 SMs) only. The planners take the
  SM count from the device, but nothing here has run on another GPU or with
  more than one device; the batch kernel requires its tensors on the
  *current* CUDA device.
- Decode kernel: `run()` depends on the address of `x` from `M = 4` on (the
  spreads under [Dispatch](#dispatch)); a benchmark row at those `M` is one
  draw.
- Batch kernel: an `x` at an address that is 16 mod 32 is read in place at a
  price, and through an aligned copy from 640 KB on. `K = N = 4096`, `out=`
  call, aligned -> 16 mod 32: 11.9 -> 12.0 us (`M = 16`), 13.8 -> 15.1 (32),
  18.1 -> 21.9 (64), 26.0 -> 29.7 (128), 44.7 -> 49.5 (256). torch's own
  allocations and row slices of them are 32 B aligned.
- Batch kernel: about 2 us per 16 rows at `K = 4096` (14.0 us at 32 rows,
  26.0 at 128), in steps at the bucket edges, and a cliff from `M = 128` to
  `M = 129` where row chunking starts (26.0 -> 31.2 us at `K = 4096`,
  16.9 -> 19.2 at `K = 2048`). Against dense cuBLAS the margin
  is thin where the GEMM is compute-bound: 16.9 against 19.1 us at
  (128, 2048, 4096), 1.1x – 1.3x at prefill sizes.
- Wide layers (`N >= 7168`): from `M = 9` on only the batch kernel applies,
  with a small margin over sglang (32.3 against 34.6 us at (9, 8192, 8192),
  29.2 against 31.8 at (9, 4096, 14336)); (8, 4096, 8192) -> (9, 4096, 8192)
  is 17.6 -> 18.2 us.
- The allocating call is host-bound up to `M = 32` at `K = 2048` (see
  Results), and at `M` = 7, 8 it is 1.1 – 1.4 us slower than the decode
  kernel's would be (see Dispatch). Use `out=`.
- The registries' limits above: same layer objects for the decode kernel's
  fast path; in-place edits of registered tensors; `prepare()` binds memory.
- The batch kernel's per-(device, stream) scratch (at least 4 MiB of
  partials and 0.26 MiB of flags) is never freed; a timeout is reported up
  to 512 launches late.
- The batch kernel's fast validation rests on a private torch API
  (`TensorGuards`); the fallback is correct and slower.
- The reference port has no compile cache and a slow host path, and it
  waits for partials without a bound, as Marlin's lock does.

## Open ideas

- **`M` = 5, 6 on narrow layers.** The `M = 7` specialization serves `M` =
  4 … 6 unchanged (`batch*` above): 0.2 us ahead of the decode kernel at
  (6, 2048, 4096) and 0.1 – 1.9 us ahead at `M = 6` on `N = 2048`, behind it
  at `K = 4096`, `N >= 4096`. A rule in `K * N` or in `N` for `M` = 5, 6 would
  collect that; over the two expert GEMMs it is a wash today.
- **Make the timeout raise on the very next call.** Having the give-up path
  store its generation to page-locked host memory works (a kernel can store
  there under unified addressing) and costs 0.05 us of host time per call,
  but that one extra load and store in a cold branch made ptxas reschedule
  the 16-row loop: +0.6 / +0.35 us at `M` = 8 … 16 (`K` = 4096 / 2048) in an
  A/B during integration, so it was not shipped. A formulation that leaves
  the hot loop's schedule alone would make the sampling unnecessary.
- **Host path.** The batch kernel's `out=` call is 0.3 – 0.4 us and its
  allocating call 0.5 us slower than the decode kernel's on the host, and
  `M` = 7, 8 pay 0.2 us for the wide-layer test. One compiled helper that
  returns the verdict on all five tensors and their addresses would take
  most of the 3 us of Python around a launch away for both kernels; it
  matters for kernels under 7.5 us (`K = 2048` at `M <= 4`, small layers).
- **Decode kernel.** The remaining gap at `M = 1`, `K = 2048` (6.9 us
  against a 5.6 – 6.5 us stream) is start-up: instructions ahead of the first
  cp.async. For `M` = 5 … 8, a shared `x` ring without the per-group barrier,
  and BAND groups longer than the ring depth on deep / wide layers.
- **Batch kernel.** A whole-sector A fetch for an `x` at 16 mod 32; a
  closed-form stripe schedule (the host plan for a new row-chunk count is
  Python, linear in the schedule's columns); a taller-and-wider tile with
  eight MMA warps for 96 … 128 rows, where the A fetch and the compute are
  each at capacity and only relieving both can show.
- **Integration level.** Overlapping consecutive experts on two streams
  fills the idle start, lead and tail of the short kernels; scratch is
  already per stream.
