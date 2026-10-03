from dataclasses import dataclass

import torch
from einops import rearrange, repeat
from jaxtyping import Bool, Float, Int
from jaxtyping import Float as Fp
from torch import Tensor as T
from torch import nn
from torch.nn import functional as F
from transformers import AutoProcessor, InklingForConditionalGeneration

ACT2FN = {"silu": nn.functional.silu}


@dataclass
class InklingTextConfig:
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    pad_token_id: int
    rms_norm_eps: float = 1e-6
    mlp_layer_types: list[str] | None = None
    layer_types: list[str] | None = None

    intermediate_size: int = 24576
    hidden_act: str = "silu"
    moe_intermediate_size: int = 3072
    n_routed_experts: int = 256
    num_experts_per_tok: int = 6
    n_shared_experts: int = 2
    route_scale: float = 8.0

    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    swa_num_attention_heads: int = 64
    swa_num_key_value_heads: int = 16
    swa_head_dim: int = 128
    sliding_window_size: int = 512
    d_rel: int = 16
    rel_extent: int = 1024
    attention_dropout: float = 0.0
    conv_kernel_size: int = 4

    def __post_init__(self) -> None:
        if self.layer_types is None:
            self.layer_types = [
                "hybrid" if (i + 1) % 6 == 0 else "hybrid_sliding"
                for i in range(self.num_hidden_layers)
            ]
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["sparse"] * self.num_hidden_layers

    @property
    def sconv_kernel_size(self) -> int:
        # Transformers aliases this checkpoint name to conv_kernel_size.
        return self.conv_kernel_size


@dataclass
class InklingConfig:
    text_config: InklingTextConfig
    hidden_size: int
    vocab_size: int


class MyInklingRelativeLogits(nn.Module):
    """
    TODO: Double check these comments

    No RoPE in this model; All positional embeddings are relative.

    Parameters:
        d_rel: Number of learned distance patterns, i.e. one may prioritize closer tokens, etc.
        rel_extent: How many backwards tokens are covered
    """

    def __init__(self, d_rel: int, rel_extent: int):
        super().__init__()
        self.d_rel = d_rel
        self.rel_extent = rel_extent

        self.proj = nn.Parameter(torch.empty(d_rel, rel_extent))

    def forward(
        self,
        relative_states: Float[T, "bs q_len num_heads d_rel"],
        query_positions: Int[T, " q_len"],
        key_positions: Int[T, " k_len"],
    ) -> Float[T, "bs num_heads q_len k_len"]:
        rel_logits = rearrange(
            relative_states @ self.proj,
            "bs q_len num_heads rel_extent -> bs num_heads q_len rel_extent",
        )

        distance: Int[T, "q_len k_len"] = rearrange(
            query_positions, "q_len -> q_len 1"
        ) - rearrange(key_positions, "k_len -> 1 k_len")

        gather_index = repeat(
            distance.clamp(0, self.rel_extent - 1),
            "q_len k_len -> bs num_heads q_len k_len",
            bs=rel_logits.shape[0],
            num_heads=rel_logits.shape[1],
        )

        position_bias: Float[T, "bs num_heads q_len k_len"] = rel_logits.gather(
            -1, gather_index
        )

        return position_bias.masked_fill(
            (distance < 0) | (distance >= self.rel_extent), 0.0
        )


class MyInklingRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.variance_epsilon = eps

        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self, hidden_states: Float[T, "... hidden_size"]
    ) -> Float[T, "... hidden_size"]:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance: Float[T, "... 1"] = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MyInklingMLP(nn.Module):
    def __init__(self, config: InklingTextConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]
        self.global_scale = nn.Parameter(torch.ones(1))

    def forward(
        self, hidden_states: Float[T, "bs seq_len hidden_size"]
    ) -> Float[T, "bs seq_len hidden_size"]:
        gate = self.act_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up) * self.global_scale


class MyInklingSharedExperts(nn.Module):
    def __init__(self, config: InklingTextConfig):
        super().__init__()

        self.n_shared_experts = config.n_shared_experts
        intermediate_dim = config.moe_intermediate_size

        self.gate_proj = nn.Parameter(
            torch.empty(config.n_shared_experts, intermediate_dim, config.hidden_size)
        )
        self.up_proj = nn.Parameter(
            torch.empty(config.n_shared_experts, intermediate_dim, config.hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(config.n_shared_experts, config.hidden_size, intermediate_dim)
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: Float[T, "bs s d"],
        gammas: Float[T, "bs*s e"],
    ) -> Float[T, "bs s d"]:
        """
        t = bs * s
        s = seq_len
        d = hidden_size
        f = moe_intermediate_size
        e = n_shared_experts
        """
        input_shape = hidden_states.shape

        hidden_states: Float[T, "e t d"] = repeat(
            hidden_states,
            "bs s d -> e (bs s) d",
            e=self.n_shared_experts,
        )

        gammas = rearrange(gammas, "t e -> e t 1")

        gate_proj, up_proj = (
            rearrange(layer, "e f d -> e d f")
            for layer in [self.gate_proj, self.up_proj]
        )
        gate = torch.bmm(hidden_states, gate_proj)
        up = torch.bmm(hidden_states, up_proj)
        activated: Float[T, "e t f"] = self.act_fn(gate) * up * gammas

        down_proj = rearrange(self.down_proj, "e d f -> e f d")
        down: Float[T, "e t d"] = torch.bmm(activated, down_proj)

        out: Float[T, "t d"] = down.float().sum(dim=0).to(hidden_states.dtype)
        return out.view(input_shape)


class MyInklingExperts(nn.Module):
    def __init__(self, config: InklingTextConfig):
        super().__init__()

        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: Fp[T, "t d"],
        top_k_index: Int[T, "t k"],
        top_k_weights: Fp[T, "t k"],
    ) -> Fp[T, "t d"]:
        """t = flattened tokens, d = hidden_size, k = experts per token."""
        final_hidden_states = torch.zeros_like(hidden_states)

        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1)
        expert_mask: Int[T, "e k t"] = rearrange(expert_mask, "t k e -> e k t")

        _expert_hit: Int[T, " e"] = expert_mask.sum(dim=(-1, -2))
        expert_hit: Int[T, " e"] = torch.greater(_expert_hit, 0).nonzero().squeeze()

        for expert_idx in expert_hit:
            if expert_idx == self.num_experts:
                continue

            # expert_mask[expert_idx]: Bool[Tensor, "k t"]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            # tok_k_pos and token_idx: Int[T, "num_tok_routed_to_expert"]

            # current_state: Float[T, "num_tok_routed_to_expert d"]
            current_state = hidden_states[token_idx]

            gate, up = F.linear(
                current_state,
                self.gate_up_proj[expert_idx],
            ).chunk(2, dim=-1)

            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(
                current_hidden_states,
                self.down_proj[expert_idx],
            )

            current_hidden_states *= top_k_weights[token_idx, top_k_pos, None]

            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states


class MyInklingTopkRouter(nn.Module):
    def __init__(self, config: InklingTextConfig):
        super().__init__()

        self.num_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.n_total_experts = self.num_experts + self.n_shared_experts
        self.hidden_dim = config.hidden_size
        self.route_scale = config.route_scale
        self.top_k = config.num_experts_per_tok

        self.weight = nn.Parameter(
            torch.empty(self.n_total_experts, config.hidden_size)
        )
        self.global_scale = nn.Parameter(torch.ones(1))
        self.e_score_correction_bias = nn.Buffer(torch.zeros(self.num_experts))

    def forward(
        self, hidden_states: Fp[T, "bs s d"]
    ) -> tuple[
        Fp[T, "bs*s e_r"],  # routed logits
        Fp[T, "bs*s k"],  # routed weights
        Int[T, "bs*s k"],  # routed expert indices
        Fp[T, "bs*s e_s"],  # shared expert weights
    ]:
        """
        t: total tokens
        e: total experts
        e_r: routed experts
        e_s: shared experts
        """
        flat = rearrange(hidden_states, "bs s d -> (bs s) d")
        router_logits: Fp[T, "t total_experts"] = F.linear(flat, self.weight)

        scores = router_logits.sigmoid()
        routed_scores = scores[..., : -self.n_shared_experts]
        scores_for_choice: Fp[T, "t e_r"] = routed_scores + self.e_score_correction_bias
        topk_indices: Int[T, "t k"] = torch.topk(
            scores_for_choice, self.top_k, dim=-1, sorted=False
        )[1]

        routed_logits: Fp[T, "t e_r"] = router_logits[..., : -self.n_shared_experts]
        shared_logits: Fp[T, "t e_s"] = router_logits[..., -self.n_shared_experts :]
        topk_logits: Fp[T, "t k+e_s"] = torch.cat(
            [routed_logits.gather(-1, topk_indices), shared_logits], dim=-1
        )
        topk_log_probs = F.logsigmoid(topk_logits)
        topk_weights = torch.exp(
            topk_log_probs - torch.logsumexp(topk_log_probs, dim=-1, keepdim=True)
        )

        topk_weights = topk_weights * self.route_scale * self.global_scale

        shared_gammas = topk_weights[..., -self.n_shared_experts :].contiguous()
        topk_weights = topk_weights[..., : self.top_k].contiguous()

        return routed_logits, topk_weights, topk_indices, shared_gammas


class MyInklingMoE(nn.Module):
    def __init__(self, config: InklingTextConfig):
        super().__init__()

        self.gate = MyInklingTopkRouter(config)
        self.experts = MyInklingExperts(config)
        self.shared_experts = MyInklingSharedExperts(config)


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
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

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


class MyInklingNormedEmbedding(nn.Embedding):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int, norm_eps: float
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_norm = MyInklingRMSNorm(embedding_dim, eps=norm_eps)


class MyInklingDecoderLayer(nn.Module):
    def __init__(self, config: InklingTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = MyInklingAttention(config, layer_idx)

        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = MyInklingMoE(config)
        else:
            self.mlp = MyInklingMLP(config)

        self.input_layernorm = MyInklingRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = MyInklingRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.layer_type = config.layer_types[layer_idx]
        self.attn_sconv = MyInklingShortConv(
            config.hidden_size, config.conv_kernel_size, layer_idx=layer_idx, conv_idx=2
        )
        self.mlp_sconv = MyInklingShortConv(
            config.hidden_size, config.conv_kernel_size, layer_idx=layer_idx, conv_idx=3
        )


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
