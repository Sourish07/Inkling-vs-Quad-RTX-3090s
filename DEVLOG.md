# My journey

## 1. Porting over Inkling implementation

- Used agents to create testing suite to ensure parity with transformers implementation
- Architecture broadly makes sense. Using jaxtyping & einops makes tensors ops way more readable
- For now, omitting caching & vision/audio towers
- Relative positonal encodings is new to me
- What is the exact intuition behing these "short convolutions"?
- Also, organized repo so model definition lives in `my_inkling/` module
- utils/ will contain checkpointing & torch.distributed utilities
- `run_naive.py` runs the model on a single GPU using other cards & host memory as staging room for additional weights. It uses accelerate for cpu offloading and Nvidia's ModelOpt to dequantize selected experts on the fly.
