# Running Inkling-Small-NVFP4 on Local Hardware!

Optimizing [Inkling Small](https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4) (total NVFP4 weights ~159 GiB) to run on 96 GiB of VRAM (across 4 RTX 3090s) & 128 GiB of host RAM.

The challenge: Non-expert weights are 15.38 GiB (9.7%) but the expert weights are 143.62 GiB (90.3%). This means the experts don't fit completely in host memory or sharded across GPUs. **I keep a subset of experts in VRAM, store the rest in pinned host memory, and stream experts to GPUs on demand.** CPU-resident experts are cached in VRAM and are evicted via LRU policy to avoid repeated transfers.

Result: ~0.10 tok/s → 18.14 tok/s (~181× improvement) on 4× RTX 3090s.

### Benchmarking

List of optimizations (so far!):

| Optimization | Speed |
|---|---:|
| Baseline — `accelerate` CPU offloading | 0.10 tok/s |
| KV + short-convolution caching | 0.12 tok/s |
| Tensor parallelism for non-expert weights | 0.13 tok/s |
| Expert parallelism + CPU offloading | 2.6 tok/s |
| CPU expert caching | 3.28 tok/s |
| Remove DTensor interface | 4.39 tok/s |
| Grouped expert GEMMs + reusable routing/scratch buffers | 9.45 tok/s |
| Move remaining hot path to GPU | 9.88 tok/s |
| Fuse Q/K/V/relative projections + simplify attention prep | 11.22 tok/s |
| Fuse normalization, short convolutions, and cache updates | 14.77 tok/s |
| CUDA Graph replay for decode | 17.22 tok/s |
| Paired NVLink/PCIe tensor-parallel reductions | **18.14 tok/s** |

Notes:
- All calculations are done with conc=1 for now. Warmup & prefill are excluded.
- MFU calculations (back of napkin math):
  - Peak BF16 flops for RTX 3090 (assuming stock boost clock speed): 71 TFLOP/s
  - Number of active parameters in forward pass: 12B
  - Number of FLOPs in decode forward pass: 2 * 12B = 24 GFLOPs = 0.024 TFLOPs
  - 18.14 tok/s = ~55 ms per token
  - 0.024 TFLOPs / 0.055 seconds = 0.436 TFLOP/s
  - MFU = 0.436 TFLOP/s / (4 * 71 TFLOP/s) = 0.00153 -> **0.153% MFU**
  - Not taking into account cost of dequantizing FP4 weights & applying scales

### Interesting things I learned

Please read [DEVLOG.md](DEVLOG.md) for more details about my journey.

### Hardware

GPU topology: 2 NVLink-connected RTX 3090 pairs; cross-pair traffic traverses PCIe.

- AORUS GeForce RTX 3090 XTREME 24G x4
- PNY NVIDIA NVLink Bridge x2
- AMD Ryzen Threadripper Pro 5955WX
- Pro WS WRX80E-SAGE SE WIFI
- VENGEANCE LPX 128GB (8 x 16GB) DDR4 DRAM 2666MHz C16 Memory Kit
- Lexar® PLAY 2280 SE PCIe 4.0 SSD

### Model Architecture

Inkling-Small is a MoE model with 42 layers (first two are dense) and 256 experts. Each token is routed to 6 experts and 2 shared experts. Attention is hybrid, between full attention (every 6th layer) and sliding window attention (window of 512 tokens). The model contains "short convolutions" for additional interaction with nearby tokens. Instead of RoPE, we use a learned relative position bias instead.

For the NVFP4 checkpoint, all routed experts except layer 2 are quantized to NVFP4.

### Future features to come!

- Add vision & audio towers
- Shard experts with TP instead of EP (and benchmark)
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
