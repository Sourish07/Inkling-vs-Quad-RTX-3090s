# My journey

## 1. Porting over Inkling implementation

- Used agents to create testing suite to ensure parity with transformers implementation
- Architecture broadly makes sense. Using jaxtyping & einops makes tensors ops way more readable
- For now, omitting caching & vision/audio towers
- Relative positonal encodings is new to me
- What is the exact intuition behing these "short convolutions"?