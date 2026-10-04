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
