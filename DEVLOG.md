# My journey

## 1. Porting over Inkling implementation

- Literally my first goal is to get just something running; `transformers` sample code won't work because the model is too large to fit across GPUs or even just within host memory.
- Used agents to create testing suite to ensure parity with transformers implementation
  - `from transformers import InklingForConditionalGeneration` to jump to reference quickly
- Architecture broadly makes sense. Using jaxtyping & einops makes tensors ops way more readable
- For now, omitting caching & vision/audio towers
- Relative positonal encodings is new to me
- What is the exact intuition behing these "short convolutions"?
- Also, organized repo so model definition lives in `my_inkling/` module
- utils/ will contain checkpointing & torch.distributed utilities
- `run_naive.py` runs the model on a single GPU at a time. The activations are moved to the GPU where the weights are resident, except CPU weights are moved to GPU 0 for computation. It uses `accelerate` for cpu offloading and Nvidia's `modelopt` to dequantize selected experts on the fly.
  - Running at ~10 s/tok! (lol)

## 2. Adding KV caching + short conv caching

- All the things that need to be cached:
  - Attention keys: `key_states`
  - Attention values: `value_states`
  - Key conv history: `conv_idx=0`
  - Value conv history: `conv_idx=1`
  - Attn-output conv history: `conv_idx=2`
  - MLP-output conv history: `conv_idx=3`
- Since ShortConvs were new to me, I had to figure out what exactly needed to be cached
```python
hidden_states = self.conv1d(hidden_states)[..., :seq_len]
```
  - Horribly wasteful for decode workloads... Only need to run the conv on full sequence for prefill
  - All we need to store is just the last `conv_kernel_size - 1` hidden states. We use a buffer of size `conv_kernel_size` though just to keep it simple
- Caching for sliding window attention was also new
  - Pretty intuitive; Just discard old cached tokens and add new ones
  - Need to add support for prefills that exceed the sliding window length
- `run_naive.py` is now running at ~8.6 s/tok! (only benchmarked 8 token generation haha)
- Probably can optimize away a couple redundant reallocations; i.e. `torch.roll` probably isn't ideal
- Need to add support for subsequent prefills after the first

## 3. Adding tensor parallelism

- `_VocabParallelEmbedding` performs a distributed embedding lookup
  - We cannot just use `RowwiseParallel` because we only want to split the embed lookup, and then ensure the all-reduce happens before the RMSNorm.  We need the lookup results combined before the RMSNorm.
  - We can't wrap `MyInklingNormedEmbedding` with `RowwiseParallel` because of the all-reduce requirement. We can't wrap `MyInklingNormedEmbedding.weight` either because `RowwiseParallel` only works on `nn.Module`s, not `Parameter`s.
  - `DTensor.from_local(..., Replicate())` declares replicated token IDs
  - This is needed because we call `redistribute` (the all-reduce) afterwards
- `_HeadParallelConv1d` performs a local convolution, with the split inputs & sharded weights
  - `self.weight.to_local()` is used here because everything is done locally with no collectives needed
- `_TPSharedExperts` is needed because shared experts use raw 3D parameters rather than `nn.Linear` modules
- `run_naive.py` now running at ~7.45 s/tok!

## 4. Adding expert parallelism + CPU loading

- Very naive version; Just created a drop-in replacement for `MyInklingExperts`
- Uses expert parallelism; No caching for now. Just loads & dequantizes each of the routed experts as needed
- state dictionary is loaded in two chunks, first the non-expert parameters, then the expert parameters
  - The expert parameters are split between GPU & CPU memory because all experts don't fit in host memory
- Checkpoint loading is still kinda slow; We need to optimize that
- Because we use TP for the non-expert layers, all expert layers receive the same input. No all2all necessary
- `run.py` is running at ~2.6 tok/s!

## 5. Adding expert caching for non-GPU experts

- First time implementing this feature and I didn't really have any good open source references
- Main new features:
  - Keep track of which experts are pinned to GPU vs not
  - Allocate slots for all "weight banks", i.e. (proj, scale, scale2) * (gate_up, down)
  - Uses separate copy stream to overlap data transfer with computation
  - Run prefetch for experts before entering the main expert loop in `forward`
  - Expert `gate` and `up` weights are interleaved

## 6. Removing DTensor

- This one was a little surprising... I just took some old implementation of my RowLinear & ColumnLinear classes and threw them in here; It's faster!
- `run.py` is now running at ~4.39 tok/s
- `_VocabParallelEmbedding` is present for the same previous reason (i.e. the RMSNorm needs its inputs all-reduced)
- `_TPSharedExperts` because the weights aren't `nn.Linear` modules, but rather just raw weight tensors
- `_HeadParallelConv1d` because `local_channels` is `total_channels // world_size`

## 7. Grouping expert GEMMs and reusing buffers

- Combines all expert GEMMs into a single batched GEMM
  - just a regular GEMM with the experts as a launch dimension
  - dequantization now happens in the kernel as well
  - a pointer table is used so the expert weights don't have to be contiguous
- `_route` just runs the "stream compaction algorithm" to quickly select which tokens to use for each expert
- The reused buffers are `self.host_metadata`, `self.metadata`, 
(the views into them), and then `self.counts`
  - also the ones created in `reserve()`
- Metadata stores pointer table & active expert list
  - Pointer table
    - one row for each expert ID
    - each row has `pointer_stride` entries; 6 for quantized layer, 2 for unquantized (bf16)
      - each entry is a raw GPU pointer to a weight tensor for that expert
  - Expert list
    - one entry for each local expert
    - first `group_size` entries hold the active expert list
      - entry `g` tells GEMM group `g` which expert it computes

## 8. GPU routing and expert cache decisions

- The top-k router was already on GPU. Expert ID downloads, Python LRU
  decisions, and repeated host pointer-table uploads remained in the hot path.
- Really bad sync points (each blocks host until GPU stream is empty):
  - `torch.unique(top_k_index).tolist()`
  - Using an `OrderedDict` for the LRU; each access is a sync
    - `_, slot = self.lru_slots.popitem(last=False)`
  - Updating metadata table from host and then copying to GPU
    - `self.metadata.copy_(self.host_metadata, non_blocking=True)`
- A GPU planner now compacts routing rows, protects active cache hits, assigns
  misses to least-recently-used slots, and writes the GEMM pointer table.
- For prefill, where there may be expert overflow, we directly pass in pointer to CUDA-mapped pinned CPU weights into cache slots
- Running at ~9.88 tok/s

## Some notes on pinned memory

- Memory that's allocated on host but cannot be "paged out" to disk
- CUDA driver then translates that physical host address to a GPU virtual address
  - Doesn't use VRAM until GPU accesses pointer
  - `.data_ptr()` will return the GPU virtual address, regardless of if it's actually in VRAM or host memory
  - When copying, there's no intermediate staging buffer (ex. because non-pinned memory may be on disk)

## 9. Fusing Q/K/V/relative projections and simplifying attention preparation

- Fusing the four projections (Q/K/V/relative) into a single kernel before attention
- Relative distance calculations only happen once
  - Used to happen in `MyInklingRelativeLogits.forward` and `MyInklingAttention.forward` when creating the causal/sliding mask
- Skipping log scaling when it's a no-op (tau == 1 until the sequence passes log_scaling_n_floor)
  - Separated out single token case too
- Running at ~11.22 tok/s

## 10. Fusing normalization, short convolutions, and cache updates

- Three Triton inference kernels: 
  - RMSNorm, which doesn't require the input to be contiguous
    - Rounds normalized activations to match PyTorch
  - Single-token depthwise convolution, which updates the conv cache as well
    - Also adds residual for the attn-output & mlp-output
  - Paired K/V cache writes, which for sliding window, writes the tokens twice
    - Uses a mirrored ring buffer; cost is double storage
    - Example (each row has a slide where tokens are in order):
      - [1, 2, 3, 1, 2, 3]
      - [4, 2, 3, 4, 2, 3]
      - [4, 5, 3, 4, 5, 3]
      - [4, 5, 6, 4, 5, 6]
    - Replaced `torch.roll`, which would create new allocation each time
- Running at ~14.77 tok/s!

## 11. Static buffers and complete decode CUDA graphs

- Cudagraphs requires static buffers; Graph replay uses the exact same memory addresses
- Added graph warmup
- KV cache allocations are all preallocated
  - Need to add paging later
- `position` is a GPU scaler that counts number of tokens seen so far
- Also put the `TextStreamer` on a background thread
- Running at 17.22 tok/s

## 12. Paired NVLink/PCIe tensor-parallel reductions

- Ported from SGLang
- GPUs 0 & 1 are connected via NVLink, same with GPUs 2 & 3
- The PCIe hop depends on driver-level p2p (which I have a patched driver for installed)
- Need to use PyTorch's symmetric memory for potential speed ups
- Running at 18.14 tok/s

## Adding batching support (Not an optimization)

- Reasoning about attention masks is complicated
- I just hardcoded some assumptions for now:
  - Prefill only runs once, i.e. no multi turn requests (yet)
  - Prefill length has to be less than or equal to 512 tokens
    - Otherwise, the sliding window attention mask logic needs extra logic
- Added some random asserts that need to be cleaned up
- Cross product of {prefill, decode} and {sliding window, full attn} results in four regimes that need to be handled separately
- Cleaned up `cache.py`
  - First, just allocated full kv cache for max seq len at init (will add paging later)
    - Fine for shorter sequences
  - Second, removed mirrored buffer
    - It was originally used as replacement for `torch.roll`
    - Simple way to always have a single slice return all the keys in order
      - Easy to calculate distance offsets from
    - Cuda graphs requires static memory addresses so we couldn't just "slice" the mirrored buffer anymore
      - Should've just removed the mirrored buffer then...
      - For decode in full attn layers, the size of the kv buffer is the final sequence length (i.e. `capacity`)
        - Kinda ineffecient, but we can optimize when we add paging
      - Simplified `decode_distance` (see fn's docstring)
- Updated kernels to support bs > 1
  - bs is now the first dim in the launch grid
  - requires adding offsets to input tensors
- Also fused the swiglu(gate) * up operation into the GEMM
  - The output tile for the first GEMM has half the number of columns now
  - `gate, up = tl.split(tl.reshape(acc, (BM, BN // 2, 2)))`
  - `acc` is `BM * BN` with the gate & up being interleaved along columns
    - As an example, take one row -> `[g0, u0, g1, u1, g2, u2, g3, u3]` | (shape: `(BN,)`)
    - Goal is -> `[[g0, u0], [g1, u1], [g2, u2], [g3, u3]]` | (shape: `(BN // 2, 2)`)
    - The 2 remains in the last dim because we want to keep the gate/up pairs together
    - If we did `.rehape((2, BN // 2))` we would get:
      - `[[g0, u0, g1, u1], [g2, u2, g3, u3]]` | (shape: `(2, BN // 2)`)
- Also, changed number of pinned GPU experts from 24 to 20 to accomodate bs=16. (bs=1 is now slower...)
  - Will add dynamic configuration soon
  - Running at 17.22 tok/s
