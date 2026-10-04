"""Tensor parallelism for Inkling's text tower (expert weights stay replicated)."""

import torch
from jaxtyping import Float as Fp
from jaxtyping import Int
from torch import Tensor as T
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.nn import functional as F

from .model import (
    InklingConfig,
    MyInkling,
    MyInklingAttention,
    MyInklingMLP,
    MyInklingNormedEmbedding,
)


class _VocabParallelEmbedding(MyInklingNormedEmbedding):
    """
    Reduce vocabulary-sharded lookups *before* Inkling's embedding RMSNorm.

    Equivalent to Row-wise Parallel, i.e. the inputs are split and the outputs are all-reduced
    """

    def __init__(self, embedding: MyInklingNormedEmbedding, mesh: DeviceMesh):
        super().__init__(
            embedding.num_embeddings,
            embedding.embedding_dim,
            embedding.padding_idx,
            embedding.embed_norm.variance_epsilon,
        )
        # embedding.weight originally is [vocab_size, hidden_size]
        self.weight = nn.Parameter(
            distribute_tensor(embedding.weight, mesh, [Shard(0)], src_data_rank=None),
            requires_grad=embedding.weight.requires_grad,
        )
        self.embed_norm = embedding.embed_norm
        self.device_mesh = mesh

    def forward(self, input_ids: Int[T, "bs s"]) -> Fp[T, "bs s d"]:
        # Explicitly declaring that `input_ids` is replicated across the mesh by `Replicate()`
        input_ids = DTensor.from_local(
            input_ids, self.device_mesh, [Replicate()], run_check=False
        )
        # DTensor handles the splitting of the inputs since self.weight is sharded along `vocab_size` dim
        embeddings: DTensor = F.embedding(input_ids, self.weight, self.padding_idx)

        # Redistribute to `Replicate()` placement to match `embed_norm` input
        # All-reduce is performed implicitly during `redistribute`
        embeddings = embeddings.redistribute(placements=[Replicate()]).to_local()
        return self.embed_norm(embeddings)


class _HeadParallelConv1d(nn.Conv1d):
    """Keep global DTensor weights, but convolve only this rank's K/V channels."""

    weight: DTensor  # Overriding type for ty

    def __init__(self, conv: nn.Conv1d, mesh: DeviceMesh):
        local_channels = conv.in_channels // mesh.size()
        super().__init__(
            local_channels,
            local_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=local_channels,
            bias=False,
            padding_mode=conv.padding_mode,
            device="meta",
            dtype=conv.weight.dtype,
        )
        self.weight = nn.Parameter(
            distribute_tensor(conv.weight, mesh, [Shard(0)], src_data_rank=None),
            requires_grad=conv.weight.requires_grad,
        )

    def forward(self, input: Fp[T, "bs d//ws s_in"]) -> Fp[T, "bs d//ws s_out"]:
        """Convolve local K/V channels; output length depends on the padding."""
        return self._conv_forward(input, self.weight.to_local(), self.bias)


def apply_tp_plan(
    model: MyInkling, config: InklingConfig, device_mesh: DeviceMesh
) -> MyInkling:
    """
    Apply TP to a meta model and update config's Q/KV counts to local heads.

    Construct caches normally with ``MyInklingCache(config.text_config)`` after
    applying TP. Hidden size, vocabulary size, per-head dimensions, and expert
    settings retain their existing values.
    """
    tp_size = device_mesh.size()
    tower = model.model.language_model

    if tp_size == 1:
        return model

    with torch.device("meta"):
        tower.embed_tokens = _VocabParallelEmbedding(tower.embed_tokens, device_mesh)

    for layer in tower.layers:
        attention: MyInklingAttention = layer.self_attn
        parallelize_module(
            attention,
            device_mesh,
            {
                "q_proj": ColwiseParallel(),
                "k_proj": ColwiseParallel(),
                "v_proj": ColwiseParallel(),
                "r_proj": ColwiseParallel(),
                "o_proj": RowwiseParallel(),
            },
            src_data_rank=None,
        )
        attention.num_heads //= tp_size
        attention.num_key_value_heads //= tp_size

        for short_conv in (attention.k_sconv, attention.v_sconv):
            short_conv.conv1d = _HeadParallelConv1d(short_conv.conv1d, device_mesh)
            short_conv.dim //= tp_size

        if isinstance(layer.mlp, MyInklingMLP):
            parallelize_module(
                layer.mlp,
                device_mesh,
                {
                    "gate_proj": ColwiseParallel(),
                    "up_proj": ColwiseParallel(),
                    "down_proj": RowwiseParallel(),
                },
                src_data_rank=None,
            )

    parallelize_module(
        model,
        device_mesh,
        {"lm_head": ColwiseParallel(output_layouts=Replicate())},
        src_data_rank=None,
    )
    text_config = config.text_config
    text_config.num_attention_heads //= tp_size
    text_config.num_key_value_heads //= tp_size
    text_config.swa_num_attention_heads //= tp_size
    text_config.swa_num_key_value_heads //= tp_size
    return model
