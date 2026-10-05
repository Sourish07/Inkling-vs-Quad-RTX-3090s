# My journey

## 1. Porting over Inkling implementation

- Literally my first goal is to get just something running; `transformers` sample code won't work because the model is too large to fit across GPUs or even just within host memory.
- Used agents to create testing suite to ensure parity with transformers implementation
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
