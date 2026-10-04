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

        # bs=1 for now
        self.cache = torch.zeros(
            1, dim, self.conv_kernel_size, dtype=torch.float32, device=device
        )

    def update_cache(self, token: Fp[T, "1 d"]) -> Fp[T, "1 d conv_kernel_size"]:
        self.cache = torch.roll(self.cache, -1, dims=-1)
        self.cache[..., -1].copy_(token)

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

        self.k_cache = torch.zeros((1, config.hidden_size, 256))
        self.v_cache = torch.zeros((1, config.hidden_size, 256))
        self.curr_size = 0

    def extend_cache(self):
        self.k_cache = torch.cat(
            [self.k_cache, torch.zeros((1, self.k_cache.shape[1], 256))], dim=2
        )
        self.v_cache = torch.cat(
            [self.v_cache, torch.zeros((1, self.v_cache.shape[1], 256))], dim=2
        )

    def update_cache(self, key_states: torch.Tensor, value_states: torch.Tensor):
        self.curr_size += key_states.shape[2]
        if self.curr_size >= self.k_cache.shape[2]:
            self.extend_cache()

        self.k_cache[:, :, -key_states.shape[2] :] = key_states
        self.v_cache[:, :, -value_states.shape[2] :] = value_states

        return self.k_cache, self.v_cache


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

        self.k_cache = torch.zeros((1, config.hidden_size, self.sliding_window_size))
        self.v_cache = torch.zeros((1, config.hidden_size, self.sliding_window_size))
        self.curr_size = 0

    def update_cache(self, key_states: torch.Tensor, value_states: torch.Tensor):
        roll_size = key_states.shape[2]
        self.curr_size = min(self.curr_size + roll_size, self.sliding_window_size)
        assert roll_size < self.sliding_window_size

        torch.roll(self.k_cache, roll_size, dims=2)
        torch.roll(self.v_cache, roll_size, dims=2)

        self.k_cache[:, :, -roll_size:] = key_states
        self.v_cache[:, :, -roll_size:] = value_states

        return self.k_cache[:, :, : self.curr_size], self.v_cache[
            :, :, : self.curr_size
        ]


class MyInklingCache:
    def __init__(self, config: "InklingTextConfig", device: torch.device | str = "cpu"):
        layer_classes = {
            "hybrid": FullAttentionLayerCache,
            "hybrid_sliding": SlidingWindowAttentionLayerCache,
        }
        self.layers = [
            layer_classes[layer_type](config, device=device)
            for layer_type in config.layer_types
        ]

    def update_attn_cache(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ):
        self.layers[layer_idx].update_cache(key_states, value_states)

    def update_conv_cache(
        self, hidden_states: torch.Tensor, layer_idx: int, conv_idx: int
    ):
        self.layers[layer_idx].conv_caches[conv_idx].update_cache(hidden_states)
        return hidden_states

    def has_previous_state(self, layer_idx, conv_idx) -> bool:
        conv_cache = self.layers[layer_idx].conv_caches[conv_idx]
        return conv_cache.initialized
