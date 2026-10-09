"""
Cache shape names.

bs               = batch size

d                = feature/channel width
hk               = key/value heads
c                = head width
s                = incoming sequence length
conv_kernel_size = short-convolution buffer length
capacity         = allocated full-attention buffer length
k_len            = returned sliding-window buffer length

K/V buffers are allocated once, at construction, with their final capacity.
Convolution histories are allocated on the first update and stay in FP32.
"""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float as Fp
from jaxtyping import Int
from torch import Tensor as T

from kernels.decode import update_kv

if TYPE_CHECKING:
    from my_inkling.model import InklingTextConfig


class ShortConvLayerCache:
    """
    Contains cache for short convolution layers
    """

    def __init__(
        self,
        config: "InklingTextConfig",
        is_kv_sconv: bool = True,
        is_swa: bool = True,
        batch_size: int = 16,
    ):
        self.is_kv_sconv = is_kv_sconv
        self.is_swa = is_swa
        self.batch_size = batch_size

        self.conv_kernel_size = config.conv_kernel_size

        if self.is_kv_sconv:
            if self.is_swa:
                dim = config.swa_num_key_value_heads * config.swa_head_dim
            else:
                dim = config.num_key_value_heads * config.head_dim
        else:
            dim = config.hidden_size

        self.dim = dim
        self.cache: Fp[T, "bs d conv_kernel_size"] | None = None

    def update_cache(self, tokens: Fp[T, "bs d s"]) -> Fp[T, "bs d conv_kernel_size"]:
        """
        Only ran during prefill
        """
        if self.cache is None:
            self.cache = tokens.new_zeros(
                (self.batch_size, self.dim, self.conv_kernel_size), dtype=torch.float32
            )

        roll_size = min(tokens.shape[2], self.conv_kernel_size)
        self.cache = torch.roll(self.cache, -roll_size, dims=-1)
        self.cache[..., -roll_size:].copy_(tokens[:, :, -roll_size:])

        return self.cache


class FullAttentionLayerCache:
    """
    Contains cache for full attention
    """

    def __init__(
        self, config: "InklingTextConfig", batch_size: int, capacity: int, dtype, device
    ):
        self.conv_caches = [
            ShortConvLayerCache(
                config, is_kv_sconv=True, is_swa=False, batch_size=batch_size
            ),
            ShortConvLayerCache(
                config, is_kv_sconv=True, is_swa=False, batch_size=batch_size
            ),
            ShortConvLayerCache(config, is_kv_sconv=False, batch_size=batch_size),
            ShortConvLayerCache(config, is_kv_sconv=False, batch_size=batch_size),
        ]

        shape = (batch_size, config.num_key_value_heads, capacity, config.head_dim)
        self.k_cache: Fp[T, "bs hk capacity c"] = torch.zeros(
            shape, dtype=dtype, device=device
        )
        self.v_cache: Fp[T, "bs hk capacity c"] = torch.zeros_like(self.k_cache)
        self.key_positions: T | None = None

    def update_cache(
        self,
        key_states: Fp[T, "bs hk s c"],
        value_states: Fp[T, "bs hk s c"],
        start: int,
    ) -> tuple[Fp[T, "bs hk k_len c"], Fp[T, "bs hk k_len c"]]:
        """
        Write the new keys at slot `start`; return every key written so far.
        """
        end = start + key_states.shape[2]
        # A sliding layer rolls over only in decode, where a mask selects its keys.
        assert end <= self.k_cache.shape[2], "K/V cache capacity exhausted"

        update_kv(key_states, value_states, self.k_cache, self.v_cache, start)

        return self.k_cache[:, :, :end], self.v_cache[:, :, :end]


class SlidingWindowAttentionLayerCache(FullAttentionLayerCache):
    """
    Contains cache for sliding window attention

    Prefill fills one window in order, as in full attention; decode writes a ring.
    """

    def __init__(
        self, config: "InklingTextConfig", batch_size: int, capacity: int, dtype, device
    ):
        """`capacity` is unused: the window size fixes the ring length."""
        self.conv_caches = [
            ShortConvLayerCache(
                config, is_kv_sconv=True, is_swa=True, batch_size=batch_size
            ),
            ShortConvLayerCache(
                config, is_kv_sconv=True, is_swa=True, batch_size=batch_size
            ),
            ShortConvLayerCache(config, is_kv_sconv=False, batch_size=batch_size),
            ShortConvLayerCache(config, is_kv_sconv=False, batch_size=batch_size),
        ]

        self.sliding_window_size = config.sliding_window_size

        shape = (
            batch_size,
            config.swa_num_key_value_heads,
            self.sliding_window_size,
            config.swa_head_dim,
        )
        self.k_cache: Fp[T, "bs hk capacity c"] = torch.zeros(
            shape, dtype=dtype, device=device
        )
        self.v_cache: Fp[T, "bs hk capacity c"] = torch.zeros_like(self.k_cache)
        self.key_positions: T | None = None


class MyInklingCache:
    def __init__(
        self,
        config: "InklingTextConfig",
        batch_size: int,
        capacity: int,
        pad_counts: T | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
    ):
        """`capacity` is the most tokens one request can hold: prompt plus output."""
        if pad_counts is None:
            pad_counts = torch.zeros(batch_size, dtype=torch.int64, device=device)
        self.pad_counts = pad_counts
        self.capacity = capacity

        layer_classes = {
            "hybrid": FullAttentionLayerCache,
            "hybrid_sliding": SlidingWindowAttentionLayerCache,
        }

        self.layers = [
            layer_classes[layer_type](config, batch_size, capacity, dtype, device)
            for layer_type in config.layer_types
        ]
        # Tokens stored by every layer. Prefill counts on the host in `tokens_seen`;
        # decode counts on the GPU in `position`, where a CUDA graph can advance it.
        self.tokens_seen = 0
        self.position: Int[T, ""] = torch.zeros((), dtype=torch.int64, device=device)
        self.decoding = False

    def advance(self, num_tokens: int) -> None:
        """Count the tokens of one forward, after every layer has stored them."""
        if self.decoding:
            self.position.add_(num_tokens)
        else:
            self.tokens_seen += num_tokens

    def prepare_decode(self) -> None:
        """
        Freeze a populated cache for decode.

        Sliding keys retain their physical ring order; attention masks use GPU
        distances instead of moving keys or changing tensor shapes each token.
        `position` becomes the live token counter; `tokens_seen` freezes.
        """
        assert not self.decoding
        assert 0 < self.tokens_seen < self.capacity

        for layer in self.layers:
            device = layer.k_cache.device
            layer.key_positions = torch.arange(layer.k_cache.shape[2], device=device)

        self.position.fill_(self.tokens_seen)
        self.decoding = True

    def decode_distance(self, layer_idx: int) -> T:
        """
        Returns the relative distance between the current position and
        each key position in the layer's kv cache.

        For full attention layers, since we've allocated the entire kv buffer already,
        some positions are going to be negative (i.e. if we're generating token 56, kv buffer is still 512 long)
        """
        layer = self.layers[layer_idx]
        assert self.decoding
        # self.position is the number of tokens seen so far
        # layer.key_positions is the arange of the entire sequence length
        distance = self.position - layer.key_positions
        if isinstance(layer, SlidingWindowAttentionLayerCache):
            distance = distance.remainder(layer.sliding_window_size)
        return distance.unsqueeze(0)

    def decode_buffers(self) -> list[T]:
        """Mutable request state to preserve across graph warmup and capture."""
        return [self.position] + [
            buffer
            for layer in self.layers
            for buffer in (
                layer.k_cache,
                layer.v_cache,
                *(conv.cache for conv in layer.conv_caches),
            )
        ]

    def update_attn_cache(
        self,
        key_states: Fp[T, "bs hk s c"],
        value_states: Fp[T, "bs hk s c"],
        layer_idx: int,
    ) -> tuple[Fp[T, "bs hk k_len c"], Fp[T, "bs hk k_len c"]]:
        layer = self.layers[layer_idx]
        if self.decoding:
            assert key_states.shape[2] == 1
            window = getattr(layer, "sliding_window_size", 0)
            update_kv(
                key_states,
                value_states,
                layer.k_cache,
                layer.v_cache,
                self.position,
                window,
            )
            return layer.k_cache, layer.v_cache
        return layer.update_cache(key_states, value_states, self.tokens_seen)

    def update_conv_cache(
        self,
        hidden_states: Fp[T, "bs d s"],
        layer_idx: int,
        conv_idx: int,
    ) -> Fp[T, "bs d conv_kernel_size"]:
        return self.layers[layer_idx].conv_caches[conv_idx].update_cache(hidden_states)
