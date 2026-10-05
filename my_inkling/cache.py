"""
Cache shape names (batch size is currently fixed to 1).

d                = feature/channel width
hk               = key/value heads
c                = head width
s                = incoming sequence length
conv_kernel_size = short-convolution buffer length
capacity         = allocated full-attention buffer length
k_len            = returned sliding-window buffer length

Buffers are allocated on the first update, using the input tensor's device.
K/V buffers also inherit the input dtype; convolution histories stay in FP32.
"""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float as Fp
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
    ):
        self.initialized = False

        self.is_kv_sconv = is_kv_sconv
        self.is_swa = is_swa

        self.conv_kernel_size = config.conv_kernel_size

        if self.is_kv_sconv:
            if self.is_swa:
                dim = config.swa_num_key_value_heads * config.swa_head_dim
            else:
                dim = config.num_key_value_heads * config.head_dim
        else:
            dim = config.hidden_size

        self.dim = dim
        self.cache: Fp[T, "1 d conv_kernel_size"] | None = None

    def update_cache(self, tokens: Fp[T, "1 d s"]) -> Fp[T, "1 d conv_kernel_size"]:
        """
        Only ran during prefill
        """
        if self.cache is None:
            self.cache = tokens.new_zeros(
                (1, self.dim, self.conv_kernel_size), dtype=torch.float32
            )

        roll_size = min(tokens.shape[2], self.conv_kernel_size)
        self.cache = torch.roll(self.cache, -roll_size, dims=-1)
        self.cache[..., -roll_size:].copy_(tokens[:, :, -roll_size:])

        self.initialized = True

        return self.cache


class FullAttentionLayerCache:
    """
    Contains cache for full attention
    """

    def __init__(self, config: "InklingTextConfig"):
        self.conv_caches = [
            ShortConvLayerCache(config, is_kv_sconv=True, is_swa=False),
            ShortConvLayerCache(config, is_kv_sconv=True, is_swa=False),
            ShortConvLayerCache(config, is_kv_sconv=False),
            ShortConvLayerCache(config, is_kv_sconv=False),
        ]
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.k_cache: Fp[T, "1 hk capacity c"] | None = None
        self.v_cache: Fp[T, "1 hk capacity c"] | None = None
        self.curr_size = 0
        self.tokens_seen = 0
        self.key_positions: T | None = None

    def allocate(
        self, reference: Fp[T, "1 hk reference_length c"], size: int = 256
    ) -> Fp[T, "1 hk size c"]:
        return reference.new_zeros((1, self.kv_heads, size, self.head_dim))

    def extend_cache(self) -> None:
        assert self.k_cache is not None and self.v_cache is not None
        self.k_cache = torch.cat([self.k_cache, self.allocate(self.k_cache)], dim=2)
        self.v_cache = torch.cat([self.v_cache, self.allocate(self.v_cache)], dim=2)

    def update_cache(
        self,
        key_states: Fp[T, "1 hk s c"],
        value_states: Fp[T, "1 hk s c"],
    ) -> tuple[Fp[T, "1 hk k_len c"], Fp[T, "1 hk k_len c"]]:
        if self.k_cache is None:
            self.k_cache = self.allocate(key_states)
            self.v_cache = self.allocate(value_states)

        seq_len = key_states.shape[2]

        old_size = self.curr_size
        new_size = old_size + seq_len

        while new_size > self.k_cache.shape[2]:
            self.extend_cache()

        update_kv(key_states, value_states, self.k_cache, self.v_cache, old_size)

        self.curr_size = new_size
        self.tokens_seen += seq_len
        return self.k_cache[:, :, :new_size], self.v_cache[:, :, :new_size]


class SlidingWindowAttentionLayerCache:
    """
    Contains cache for sliding window attention
    """

    def __init__(self, config: "InklingTextConfig"):
        self.conv_caches = [
            ShortConvLayerCache(config, is_kv_sconv=True, is_swa=True),
            ShortConvLayerCache(config, is_kv_sconv=True, is_swa=True),
            ShortConvLayerCache(config, is_kv_sconv=False),
            ShortConvLayerCache(config, is_kv_sconv=False),
        ]

        self.sliding_window_size = config.sliding_window_size

        self.k_cache: Fp[T, "1 hk capacity c"] | None = None
        self.v_cache: Fp[T, "1 hk capacity c"] | None = None
        self.curr_size = 0
        self.tokens_seen = 0
        self.key_positions: T | None = None

    def update_cache(
        self,
        key_states: Fp[T, "1 hk s c"],
        value_states: Fp[T, "1 hk s c"],
    ) -> tuple[Fp[T, "1 hk k_len c"], Fp[T, "1 hk k_len c"]]:
        roll_size = key_states.shape[2]
        # TODO: fix cases where prefill prompt > sliding window size
        assert roll_size <= self.sliding_window_size

        if self.k_cache is None:
            shape = (
                1,
                key_states.shape[1],
                2 * self.sliding_window_size,
                key_states.shape[3],
            )
            self.k_cache = key_states.new_zeros(shape)
            self.v_cache = value_states.new_zeros(shape)

        self.curr_size = min(self.curr_size + roll_size, self.sliding_window_size)

        # Mirror each position so rollover never needs to move existing tokens.
        update_kv(
            key_states,
            value_states,
            self.k_cache,
            self.v_cache,
            self.tokens_seen % self.sliding_window_size,
            self.sliding_window_size,
        )

        self.tokens_seen += roll_size

        end = (
            self.sliding_window_size
            + (self.tokens_seen - 1) % self.sliding_window_size
            + 1
        )
        start = end - self.curr_size
        return self.k_cache[:, :, start:end], self.v_cache[:, :, start:end]


class MyInklingCache:
    def __init__(self, config: "InklingTextConfig"):
        layer_classes = {
            "hybrid": FullAttentionLayerCache,
            "hybrid_sliding": SlidingWindowAttentionLayerCache,
        }
        self.layers = [
            layer_classes[layer_type](config) for layer_type in config.layer_types
        ]
        self.position = None

    def prepare_decode(self, capacity: int) -> None:
        """Freeze a populated batch-one cache for at most `capacity` total tokens.

        Sliding keys retain their physical ring order; attention masks use GPU
        distances instead of moving keys or changing tensor shapes each token.
        `position` becomes the live token counter; per-layer Python sizes freeze.
        """
        assert self.position is None
        seen = self.layers[0].tokens_seen
        assert 0 < seen < capacity
        for layer in self.layers:
            assert layer.tokens_seen == seen
            assert all(conv.initialized for conv in layer.conv_caches)
            assert layer.k_cache is not None and layer.v_cache is not None
            if isinstance(layer, FullAttentionLayerCache):
                shape = (1, layer.kv_heads, capacity, layer.head_dim)
                key = layer.k_cache.new_zeros(shape)
                value = layer.v_cache.new_zeros(shape)
                key[:, :, :seen].copy_(layer.k_cache[:, :, :seen])
                value[:, :, :seen].copy_(layer.v_cache[:, :, :seen])
            else:
                window = layer.sliding_window_size
                key = layer.k_cache[:, :, :window].clone()
                value = layer.v_cache[:, :, :window].clone()
            layer.k_cache, layer.v_cache = key, value
            layer.key_positions = torch.arange(key.shape[2], device=key.device)
        self.position = torch.tensor(seen, dtype=torch.int64, device=key.device)

    def decode_distance(self, layer_idx: int) -> T:
        layer = self.layers[layer_idx]
        assert self.position is not None and layer.key_positions is not None
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
        key_states: Fp[T, "1 hk s c"],
        value_states: Fp[T, "1 hk s c"],
        layer_idx: int,
    ) -> tuple[Fp[T, "1 hk k_len c"], Fp[T, "1 hk k_len c"]]:
        if self.position is not None:
            assert key_states.shape[0] == key_states.shape[2] == 1
            layer = self.layers[layer_idx]
            window = getattr(layer, "sliding_window_size", 0)
            update_kv(
                key_states,
                value_states,
                layer.k_cache,
                layer.v_cache,
                self.position,
                window,
                mirror=False,
            )
            return layer.k_cache, layer.v_cache
        return self.layers[layer_idx].update_cache(key_states, value_states)

    def update_conv_cache(
        self,
        hidden_states: Fp[T, "1 d s"],
        layer_idx: int,
        conv_idx: int,
    ) -> Fp[T, "1 d conv_kernel_size"]:
        return self.layers[layer_idx].conv_caches[conv_idx].update_cache(hidden_states)

    def has_previous_state(self, layer_idx: int, conv_idx: int) -> bool:
        conv_cache = self.layers[layer_idx].conv_caches[conv_idx]
        return conv_cache.initialized
