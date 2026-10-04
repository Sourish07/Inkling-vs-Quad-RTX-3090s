"""Cache shape names (batch size is currently fixed to 1).

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
from torch import nn

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
        device: torch.device | str = "cpu",
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

    def __init__(
        self,
        config: "InklingTextConfig",
        device: torch.device | str = "cpu",
    ):
        self.conv_caches = [
            ShortConvLayerCache(config, is_kv_sconv=True, is_swa=False, device=device),
            ShortConvLayerCache(config, is_kv_sconv=True, is_swa=False, device=device),
            ShortConvLayerCache(config, is_kv_sconv=False, device=device),
            ShortConvLayerCache(config, is_kv_sconv=False, device=device),
        ]
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.k_cache: Fp[T, "1 hk capacity c"] | None = None
        self.v_cache: Fp[T, "1 hk capacity c"] | None = None
        self.curr_size = 0
        self.tokens_seen = 0

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

        self.k_cache[:, :, old_size:new_size, :] = key_states
        self.v_cache[:, :, old_size:new_size, :] = value_states

        self.curr_size = new_size
        self.tokens_seen += seq_len
        return self.k_cache[:, :, :new_size], self.v_cache[:, :, :new_size]


class SlidingWindowAttentionLayerCache:
    """
    Contains cache for sliding window attention
    """

    def __init__(
        self,
        config: "InklingTextConfig",
        device: torch.device | str = "cpu",
    ):
        self.conv_caches = [
            ShortConvLayerCache(config, is_kv_sconv=True, is_swa=True, device=device),
            ShortConvLayerCache(config, is_kv_sconv=True, is_swa=True, device=device),
            ShortConvLayerCache(config, is_kv_sconv=False, device=device),
            ShortConvLayerCache(config, is_kv_sconv=False, device=device),
        ]

        self.sliding_window_size = config.sliding_window_size

        self.k_cache: Fp[T, "1 hk capacity c"] | None = None
        self.v_cache: Fp[T, "1 hk capacity c"] | None = None
        self.curr_size = 0
        self.tokens_seen = 0

    def update_cache(
        self,
        key_states: Fp[T, "1 hk s c"],
        value_states: Fp[T, "1 hk s c"],
    ) -> tuple[Fp[T, "1 hk k_len c"], Fp[T, "1 hk k_len c"]]:
        roll_size = key_states.shape[2]
        # TODO: fix cases where prefill prompt > sliding window size
        assert roll_size <= self.sliding_window_size

        if self.k_cache is None:
            shape = (1, key_states.shape[1], self.sliding_window_size, key_states.shape[3])
            self.k_cache = key_states.new_zeros(shape)
            self.v_cache = value_states.new_zeros(shape)

        self.curr_size = min(self.curr_size + roll_size, self.sliding_window_size)

        self.k_cache = torch.roll(self.k_cache, -roll_size, dims=2)
        self.v_cache = torch.roll(self.v_cache, -roll_size, dims=2)

        self.k_cache[:, :, -roll_size:] = key_states
        self.v_cache[:, :, -roll_size:] = value_states

        self.tokens_seen += roll_size

        return self.k_cache[:, :, -self.curr_size :], self.v_cache[
            :, :, -self.curr_size :
        ]


class MyInklingCache:
    def __init__(
        self,
        config: "InklingTextConfig",
        device: torch.device | str = "cpu",
    ):
        layer_classes = {
            "hybrid": FullAttentionLayerCache,
            "hybrid_sliding": SlidingWindowAttentionLayerCache,
        }
        self.layers = [
            layer_classes[layer_type](config, device=device)
            for layer_type in config.layer_types
        ]

    def update_attn_cache(
        self,
        key_states: Fp[T, "1 hk s c"],
        value_states: Fp[T, "1 hk s c"],
        layer_idx: int,
    ) -> tuple[Fp[T, "1 hk k_len c"], Fp[T, "1 hk k_len c"]]:
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
