# My journey

## 1. Porting over Inkling implementation

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
- All we need to store is just the last `conv_kernel_size - 1` hidden states.
