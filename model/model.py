from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Bool, Float
from torch import nn
from transformers import AutoProcessor, InklingForConditionalGeneration


@dataclass
class InklingTextConfig:
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    pad_token_id: int
    rms_norm_eps: float = 1e-6
    mlp_layer_types: list[str] = None


@dataclass
class InklingConfig:
    text_config: InklingTextConfig
    hidden_size: int
    vocab_size: int


class MyInklingRelativeLogits(nn.Module):
    def __init__(self, d_rel: int, rel_extent: int):
        super().__init__()
        self.d_rel = d_rel
        self.rel_extent = rel_extent


class MyInklingRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps


class MyInklingMLP(nn.Module):
    def __init__(self, config: InklingTextConfig):
        super().__init__()


class MyInklingMoE(nn.Module):
    def __init__(self, config: InklingTextConfig):
        super().__init__()


class MyInklingShortConv(nn.Module):
    """
    TODO: What exactly is happening here? What is a 1d conv? Isn't that just a linear?
    """

    def __init__(
        self, hidden_size: int, conv_kernel_size: int, layer_idx: int, conv_idx: int
    ):
        super().__init__()
        self.dim = hidden_size
        self.kernel_size = conv_kernel_size
        self.layer_idx = layer_idx
        self.conv_idx = conv_idx

        self.conv1d = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=conv_kernel_size,
            groups=hidden_size,
            padding=conv_kernel_size - 1,
            bias=False,
        )


class MyInklingAttention(nn.Module):
    def __init__(self, config: InklingTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sliding = config.layer_types[self.layer_idx] == "hybrid_sliding"
        if self.is_sliding:
            self.head_dim = config.swa_head_dim
            self.num_heads = config.swa_num_attention_heads
            self.num_key_value_heads = config.swa_num_key_value_heads
            self.sliding_window = config.sliding_window_size
            self.rel_extent = config.sliding_window_size
        else:
            self.head_dim = config.head_dim
            self.num_heads = config.num_attention_heads
            self.num_key_value_heads = config.num_key_value_heads
            self.sliding_window = None
            self.rel_extent = config.rel_extent

        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = 1.0 / self.head_dim
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )

        self.r_proj = nn.Linear(
            config.hidden_size, self.num_heads * config.d_rel, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size)

        self.k_sconv = MyInklingShortConv(
            self.num_key_value_heads * self.head_dim,
            config.sconv_kernel_size,
            layer_idx,
            conv_idx=0,
        )
        self.v_sconv = MyInklingShortConv(
            self.num_key_value_heads * self.head_dim,
            config.sconv_kernel_size,
            layer_idx,
            conv_idx=1,
        )

        self.q_norm = MyInklingRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MyInklingRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rel_logits_proj = MyInklingRelativeLogits(config.d_rel, self.rel_extent)


class MyInklingNormedEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, padding_idx: int, eps: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.eps = eps


class MyInklingDecoderLayer(nn.Module):
    def __init__(self, config: InklingTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = MyInklingAttention(config, layer_idx)

        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = MyInklingMLP(config)
        else:
            self.mlp = MyInklingMoE(config)

        self.input_layernorm = MyInklingRMSNorm(config)


class MyInklingTextTower(nn.Module):
    def __init__(self, config: InklingTextConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = MyInklingNormedEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, config.rms_norm_eps
        )
        self.layers = nn.ModuleList(
            [
                MyInklingDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = MyInklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class MyInklingVisionTower(nn.Module):
    def __init__(self, config: InklingConfig):
        super().__init__()
        self.config = config


class MyInklingModel(nn.Module):
    def __init__(self, config: InklingConfig):
        super().__init__()
        self.language_model = MyInklingTextTower(config.text_config)
        self.audio_tower = None
        self.vision_tower = MyInklingVisionTower(config)


class MyInkling(nn.Module):
    def __init__(self, config: InklingConfig):
        super().__init__()
        self.model = MyInklingModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )

        self.mtp = None  # TODO: add MTP support
