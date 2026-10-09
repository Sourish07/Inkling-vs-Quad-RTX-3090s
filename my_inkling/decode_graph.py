"""One complete greedy decode graph for a prefilled, batch-one request."""

import torch
import torch.distributed as dist


class DecodeGraph:
    @torch.inference_mode()
    def __init__(self, model, cache, next_token, capture=True):
        assert next_token.is_cuda
        assert not model.training
        self.model = model
        self.cache = cache
        self.token = next_token.clone()
        self.remaining = cache.capacity - cache.layers[0].tokens_seen
        cache.prepare_decode()
        buffers = [self.token, *cache.decode_buffers()]
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
        if dist.is_initialized():
            dist.barrier()
        # Without capture, replay() runs the same step eagerly.
        self.graph = None
        if capture:
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

    def replay(self):
        # A host bound prevents out-of-range writes without reading GPU state.
        assert self.remaining > 0, "decode graph capacity exhausted"
        self.remaining -= 1
        if self.graph is None:
            self._step()
        else:
            self.graph.replay()
        return self.token
