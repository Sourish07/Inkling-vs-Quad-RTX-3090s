# Running Inkling-Small-NVFP4 on Local Hardware!

Optimizing [Inkling Small](https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4) (total NVFP4 weights ~159 GiB) to run on 96 GiB of VRAM (across 4 RTX 3090s) & 128 GiB of host RAM.

The challenge: Non-expert weights are 15.38 GiB (9.7%) but the expert weights are 143.62 GiB (90.3%). This means the experts don't fit completely in host memory or sharded across GPUs. *We split experts across GPUs and host memory, and built a custom engine to stream experts from host memory as necessary based on routing results.*

List of optimizations (so far!):
- 1. Baseline, naive implementation (just with `accelerate`'s cpu offloading)
  - Speed: 10 s/tok
- 2. Caching for KV vectors & short convolutions
  - Speed: 8.6 s/tok
- 3. Tensor Parallelism (non-expert weights)
  - Speed: 7.45 s/tok
- 4. Expert Parallelism + CPU offloading (no expert caching)
  - Speed: 2.6 tok/s
- 5. CPU-expert caching
  - Speed: 3.28 tok/s
- 6. Removing DTensor interface
  - Speed: 4.39 tok/s
- 7. Grouped expert GEMMs with reusable routing and scratch buffers
  - Speed: 9.45 tok/s
- 8. Moving rest of hot path to GPU
  - Speed: 9.88 tok/s
- 9. Fusing Q/K/V/relative projections and simplifying attention preparation
  - Speed: 11.22 tok/s
- 10. Fusing normalization, short convolutions, and cache updates
  - Speed: 14.77 tok/s
- 11. Cudagraph replay for decode
  - Speed: 17.22 tok/s
- 12: Paired NVLink/PCIe tensor-parallel reductions
  - Speed: 18.14 tok/s
 
Please read [DEVLOG.md](DEVLOG.md) for more details about my journey.

Future features to come!
- Benchmark with TP
- Batched decode
- Custom GPU-pinned expert selection
- Vary num pinned experts by layers
- Port over marlin kernel from sglang
- Use separate stream for shared experts
- Paged KV cache
- Sequence packing + custom `flash_attn_varlen_func` & `flash_attn_with_kvcache` kernels (because additive bias isn't supported)
- Continuous batching + prefill/decode scheduling + HTTP interface
- Visual diagram of Inkling model
- Non-greedy sampling
- More efficient checkpoint loading
- More robust benchmarking
