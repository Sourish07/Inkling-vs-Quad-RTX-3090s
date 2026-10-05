# Single-expert NVFP4 linear for OffloadedExperts

The public API matches the per-expert projection loop in
`my_inkling/ep_plan.py`:

```python
import torch
from kernels import prepare_nvfp4, nvfp4_linear

# Once per projection/expert, before allocating the cache slots:
weight, scale, scale2 = prepare_nvfp4(
    checkpoint_weight, checkpoint_scale, checkpoint_scale2,
    dtype=torch.bfloat16,  # must match the activations
)

# After the expert's onload event has completed, with all tensors on CUDA:
projected = nvfp4_linear(hidden_states[token_idx], weight, scale, scale2)
```

`nvfp4_linear(x, weight, scale, scale2)` performs one bias-free projection and
returns `[tokens, output_dim]`. The caller gathers this expert's tokens, splits
interleaved gate/up rows, applies the activation, multiplies routing weights,
accumulates tokens, and performs the EP all-reduce. Those operations already
belong to `OffloadedExperts.forward`.

The binding infers dimensions, represents these rows as a single-expert launch,
chooses an 8- or 16-token block, allocates output/routing/scratch buffers, and
uses FP32 reduction on the current PyTorch CUDA stream. No expert IDs, pointer
table, top-k weights, scalar-type enum, output buffer, workspace, or kernel
flags appear in the public interface. Even though the caller uses EP, the
kernel's EP flag is false: every row of this launch targets this local expert.
Scratch is private to each call, so different CUDA streams can launch safely.

## Checkpoint preparation

`prepare_nvfp4(weight, scale, scale2, dtype=torch.bfloat16)` accepts the same
three ModelOpt tensors currently dequantized by `OffloadedExperts.matrix`:

| Tensor | Checkpoint input | Prepared output |
| --- | --- | --- |
| weight | uint8 `[N, K/2]` | int32 `[K/16, N*2]` |
| scale | E4M3 `[N, K/16]` | processed E4M3 `[K/16, N]` |
| scale2 | scalar | processed FP16/BF16 `[1]` |

Preparation works on CPU or CUDA, preserves output row order and the packed
weight/scale storage size, and preserves CPU pinning for asynchronous onload.
N must be a positive multiple of 128, and K a positive multiple of 64. There
is no padding or automatic dequantization fallback. Both FP16 and BF16
activations are supported; prepare separately if changing activation dtype.

`ep_plan.py` still uses its existing dequantization path. Integration requires
preparing its weight banks before cache allocation and replacing its
`F.linear(..., self.matrix(...))` calls with this operation, using the same
per-projection event waits. Prepared tensors must not be passed to the old
`matrix()` method, which expects checkpoint layout.

## Build and validation

Requires PyTorch, Ninja, a CUDA toolkit with `nvcc`, a C++20 compiler, and an
Ampere-or-newer GPU. No SGLang, `sgl_kernel`, or TVM FFI package is required.
Run through `uv run --frozen` so Ninja is on PATH. The extension is compiled
lazily and cached by activation dtype and GPU architecture.

```sh
MAX_JOBS=2 uv run --frozen -m pytest tests/test_nvfp4_marlin.py -q
```

Tests compare with ModelOpt dequantization plus PyTorch linear, cover FP16/BF16,
CPU/GPU preparation, pinned onload into reused slots, empty and padded token
batches, a non-default CUDA stream, and preservation of interleaved gate/up rows.

## Source provenance

Original CUDA headers were copied verbatim from:

- Source: `/home/sourish/inkling-sglang/_third_party/sglang`
- Commit: `e33584e32c7514f3b036a090724c9ed5cd1bea1f` (local `inkling-3090` branch)
- Original CUDA directory: `python/sglang/kernels/jit/csrc/gemm/marlin_moe/`

`marlin_moe/marlin_template.h` contains the unchanged CUDA GEMM implementation.
`marlin_moe/moe_wna16_marlin.cuh` is the original host launcher, retained for
reference. `marlin/` and `include/sgl_kernel/` contain its original helper headers.
`SHA256SUMS` records these verbatim copies and the Apache-2.0 license; verify
with `sha256sum -c SHA256SUMS` from this directory.

Local files are `moe_wna16_marlin.py` (lazy PyTorch loader), `nvfp4.py` (checkpoint
preparation), `csrc/bindings.cpp`, and `csrc/nvfp4_moe.cu` (four-tensor binding).
`csrc/nvfp4_dispatch.cuh` adapts the original host dispatcher to NVFP4 only.
`csrc/include/sgl_kernel/utils.cuh` supplies minimal PyTorch/CUDA host helpers,
replacing TVM-dependent utilities in the compilation include path.
