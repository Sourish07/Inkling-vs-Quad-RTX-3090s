# Running Inkling-Small-NVFP4 on Local Hardware!

Optimizing [Inkling Small](https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4) (total NVFP4 weights ~159 GiB) to run on 96 GiB of VRAM (across 4 RTX 3090s) & 128 GiB of host RAM.

The challenge: Non-expert weights are 15.38 GiB (9.7%) but the expert weights are 143.62 GiB (90.3%). This means the experts don't fit completely in host memory or sharded across GPUs. **We split experts across GPUs and host memory, and built a custom engine to stream experts from host memory as necessary based on routing results.**

List of optimizations (so far!):

*the "s/tok" units are not a typo lol*

| Optimization | Speed |
|---|---:|
| Baseline — `accelerate` CPU offloading | 10 s/tok |
| KV + short-convolution caching | 8.6 s/tok |
| Tensor parallelism for non-expert weights | 7.45 s/tok |
| Expert parallelism + CPU offloading | 2.6 tok/s |
| CPU expert caching | 3.28 tok/s |
| Remove DTensor interface | 4.39 tok/s |
| Grouped expert GEMMs + reusable routing/scratch buffers | 9.45 tok/s |
| Move remaining hot path to GPU | 9.88 tok/s |
| Fuse Q/K/V/relative projections + simplify attention prep | 11.22 tok/s |
| Fuse normalization, short convolutions, and cache updates | 14.77 tok/s |
| CUDA Graph replay for decode | 17.22 tok/s |
| Paired NVLink/PCIe tensor-parallel reductions | **18.14 tok/s** |


### Interesting things I learned

Please read [DEVLOG.md](DEVLOG.md) for more details about my journey.

### Hardware

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
