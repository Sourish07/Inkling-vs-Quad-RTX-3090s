# Custom Kernels for Inkling

I plan on porting these kernels over to CuTeDSL manually for learning purposes + increasing readability.

## Kernels still to port from `sourish/testing2`

This branch has the routed-expert matrix-vector kernel (`routed_decode.cu`). Everything below exists on `sourish/testing2` and not here yet. All of them are decode-only (up to 8 tokens per step); prefill keeps the PyTorch/Triton path. Speedups are the ones measured on `sourish/testing2` in the order they landed there (see `BENCHMARKS.md`), so they won't transfer exactly.

Shared headers the kernels below include:

- `bf16.cuh`: fixed-width types and BF16 <-> FP32 conversion (`from_bf16`, `to_bf16`, `round_bf16`).
- `matvec.cuh`: the row-reading code of every matrix-vector kernel (`load_inputs`, `bf16_row_dot`, `lane_sum` and the thread layout macros). `routed_decode.cu` currently carries its own copy of these.

### 1. `linear_decode.cu` (`linear`): BF16 linear layer as a matrix-vector product

- **Problem:** At batch size 1 every projection is a matrix times one vector, and cuBLAS GEMM is tuned for large batches. The router gate alone cost 18 us per layer, and the QKVR, attention output and dense-MLP projections go through the same path.
- **How:** One thread owns 32 consecutive input columns for the whole kernel, so its 32 inputs stay in registers. The threads that share a weight row read it as one contiguous run, multiply-accumulate in FP32, sum across the warp with shuffles and round to BF16 once (or keep FP32 for the ranks' partial sums). Grid is (row tiles, tokens). The weight is read exactly once per token at memory bandwidth.
- **Used for:** router logits (10 us for the 2 MiB gate matrix), then QKVR, output and dense-MLP projections (+3.8%). The LM head went back to cuBLAS (526 us vs 547 us).

### 2. `router.cu` (`route`): top-k routing in one launch

- **Problem:** After the logits, the router runs sigmoid, bias add, top-k, gather, cat, log-sigmoid, logsumexp, exp and scaling as separate PyTorch kernels: about 25 launches of 1-5 us each, per layer, per rank, per token.
- **How:** One thread block per token. Threads compute `sigmoid(logit) + bias` for every routed expert into shared memory, then each expert finds its rank by counting how many experts beat it (ties go to the lower ID); the ones with rank < top_k write themselves into slot `rank`. No sort needed for 256 experts. The same block then computes the softmax over the log-sigmoids of the chosen and shared experts and applies the route scale.
- **Details:** It rounds to BF16 everywhere the module produces a BF16 tensor (including inside logsumexp), so given the same logits the weights are bit-identical to `MyInklingTopkRouter`. Experts come out best first instead of in `topk`'s unspecified order.
- **Result:** +4.6% together with the linear kernel for the logits.

### 3. `attention_decode.cu` (`attention_chunks`, `attention_combine`): fused decode attention

- **Problem:** Between the QKVR projection and the output projection, one decode token runs about 35 small launches per layer: q/k RMS norms, K/V short convolutions, cache writes, relative-position bias, SDPA (31 us of it) and reshapes.
- **How:** Two launches.
  - `attention_chunks`: a block owns 32 cached keys of one key/value head of one sequence, plus the query heads that share that head. It normalizes its queries; if its chunk contains the new token's slot it also short-convolves the new key and value (advancing the FP32 histories), normalizes the key and writes both into the K/V cache. It then scores its keys (content dot product plus relative bias, with sliding-window wraparound and left-padding masks) and stores the chunk's max, softmax denominator and weighted value sum.
  - `attention_combine`: merges the chunks of each query head into the final softmax-weighted value (a log-sum-exp merge over chunk maxima).
- **Details:** Work is spread thin so no thread multiplies more than a few dozen elements. Handles full and sliding-window layers, the log-scaling `tau`, BF16 or FP32.
- **Result:** +19.0%. About 8-11 us per layer up to a few hundred cached keys.
- **Limitation:** The chunk merge is serial per head, so it slows down past a few thousand keys (28 us at 4000, 111 us at 16000).

### 4. Shared experts through `routed_decode.cu` (`SPLIT` variant, `shared_decode`)

- **Problem:** Shared experts run as three batched matmuls plus about 10 small kernels and their own all-reduce, so every MoE layer pays two all-reduces (routed and shared).
- **How:** `routed_decode.cu` gets a `SPLIT` compile-time flag for BF16 experts whose gate and up are two matrices instead of interleaved rows, an optional `gammas` argument so `gate_up` scales by the shared weight, and `accumulate` / `partial` flags on `down_reduce`. The shared experts then reuse the same two kernels, add into the same FP32 buffer as the routed experts on each rank, and the layer is reduced once.
- **Result:** +4.0%.

### 5. `reduce_conv.cu` (`reduce_conv`): all-reduce + short convolution + residual add

- **Problem:** Every branch (attention, MLP/MoE) ends with an all-reduce, a cast to BF16 and a short convolution added to the residual stream: three launches per branch, twice per layer.
- **How:** One kernel, one thread per channel. It does the paired reduction inline (each rank pushes its FP32 partial to its NVLink peer, sums the pair, pushes the pair sum over PCIe and sums again), then rounds to BF16, optionally scales, advances the channel's short-convolution history and adds the residual.
- **Details:** Data is its own ready flag: a sender never writes all-zero bits (+0.0 goes out as -0.0), and a receiver spins until its slot is non-zero and then zeroes it. Blocks alternate between two workspace phases so a peer that is one call ahead can't overwrite unread elements. Every rank adds in the same order, so all four get identical bits. Fixed to exactly four ranks in pairs (0, 1) and (2, 3).
- **Result:** +2.0%.

### 6. `expert_cache.cu` (`plan_copy`): plan and fill the expert cache in one launch

- **Problem:** The cache planner and the copy are two Triton launches (`expert_planner.py`), and Triton recompiles per route count. The host can't read the plan back inside a CUDA graph.
- **How:** One launch with several blocks. Block 0 plans: it counts routes per local expert, groups routes by expert, finds which routed experts are neither resident nor already in a slot, and hands each of those the least-recently-used slot this forward isn't using (ranked by age with a counting sort in shared memory). It then writes the pointer table the projection kernels read. The other blocks spin on a `planned` flag, then every block streams the missed experts from pinned host memory into their slots, 16 bytes per thread per step.
- **Details:** If a forward routes more missing experts than there are free slots, the leftovers keep their host address and the projection kernels read them straight from host memory. Also keeps counters (forwards, routes, copies, host reads).
- **Result:** +0.8% on its own; it was the prerequisite for expert TP on `sourish/testing2` and for the relay.

### 7. Relay in `expert_cache.cu` + `host_memory.py` (expert parallelism only)

- **Problem:** Under EP a cache miss moves a whole 13.5 MiB expert over one PCIe link while the link of the rank's NVLink peer sits idle.
- **How:** The two ranks of an NVLink pair split every miss. The owner copies the first half of each bank and the peer copies the second half, from the same host memory (`SharedHostMemory`: page-locked and mapped by both processes) into the same slot, with the peer's half arriving over NVLink. The ranks coordinate through an inbox each: after planning, a rank writes its (expert, slot) pairs and the forward number into the peer's inbox; each copying thread sets a flag when its share has landed.
- **Result:** +13.5% under EP (92.55 -> 105.08 tok/s). Not useful under expert TP, where a miss is already four slice copies over four links, so it is off in that layout.

### 8. `embedding.cu` (`embed_host`): embedding lookup from pinned host memory

- **Problem:** The embedding table sits in VRAM but decode reads one row of it per token. That VRAM is worth more as expert cache slots.
- **How:** The table stays in pinned host memory and the kernel reads the token's row over PCIe, 16 bytes per thread per step. A token outside this rank's shard gets a zero row, so the existing all-reduce still works.
- **Result:** Part of the `vram` change (with freeing the unfused Q/K/V/relative weights and sharing routing scratch across layers) that paid for two more cache slots per layer: +4.9%.

### Not a new kernel: thread-block sizing (`blocks`)

Smaller thread blocks in the matrix-vector kernels: 8 weight rows per block for NVFP4 and shared experts (this branch uses 16), 2 for the router gate. More, smaller blocks keep more memory reads in flight. +8.3% on `sourish/testing2`.
