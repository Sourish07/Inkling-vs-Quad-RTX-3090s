"""One complete greedy decode graph for a prefilled batch of 1–16 requests."""

import torch
import torch.distributed as dist


class DecodeGraph:
    @torch.inference_mode()
    def __init__(self, model, cache, next_token, capacity, eos_token_ids=()):
        assert next_token.is_cuda and next_token.ndim == 2 and next_token.shape[1] == 1
        cache.validate_batch(next_token.shape[0])
        assert not model.training
        self.model = model
        self.cache = cache
        self.token = next_token.clone()
        self.eos_tokens = torch.tensor(
            eos_token_ids, dtype=torch.int64, device=next_token.device
        )
        self.finished = torch.isin(self.token[:, 0], self.eos_tokens)
        self.remaining = capacity - cache.layers[0].tokens_seen
        if isinstance(model, torch.nn.Module):
            for module in model.modules():
                prepare = getattr(module, "prepare_decode_workspace", None)
                if prepare is not None:
                    prepare(next_token.shape[0])
        cache.prepare_decode(capacity)
        buffers = [self.token, self.finished, *cache.decode_buffers()]
        saved = [buffer.clone() for buffer in buffers]

        def restore():
            for buffer, original in zip(buffers, saved):
                buffer.copy_(original)

        # JIT kernels, cuBLAS and NCCL must be initialized before capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._step()
                restore()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        # The graph uses a private allocator pool; return unused prefill/warmup
        # blocks to CUDA so they do not compete with capture allocations.
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self._step()
        restore()
        torch.cuda.synchronize()

    def _step(self):
        self.logits = self.model(self.token, cache=self.cache)
        torch.argmax(self.logits[:, -1], dim=-1, keepdim=True, out=self.token)
        if dist.is_initialized():
            dist.broadcast(self.token, src=0)
        if self.eos_tokens.numel():
            torch.where(
                self.finished[:, None], self.eos_tokens[0], self.token, out=self.token
            )
            self.finished.logical_or_(torch.isin(self.token[:, 0], self.eos_tokens))

    def replay(self):
        # A host bound prevents out-of-range writes without reading GPU state.
        assert self.remaining > 0, "decode graph capacity exhausted"
        self.remaining -= 1
        self.graph.replay()
        return self.token
