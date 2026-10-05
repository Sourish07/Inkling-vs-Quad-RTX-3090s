# Running Inkling-Small-NVFP4 on Local Hardware!

Optimizing [Inkling Small](https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4) (total NVFP4 weights ~171 GB) to run on 96 GB of VRAM (across 4 RTX 3090s) & 128 GB of host RAM.

List of optimizations:
- 1. Baseline, naive implementation (because model doesn't fit entirely in GPUs or in host memory)
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
 
Please read [DEVLOG.md](DEVLOG.md) for more details about my journey.
