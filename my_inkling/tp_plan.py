"""
Local tensor parallelism; all sharded dimensions must divide evenly.
"""

import torch
import torch.distributed as dist
from jaxtyping import Float as Fp
from jaxtyping import Int
from torch import Tensor as T
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import functional as F

from kernels.paired_all_reduce import all_reduce

from .model import (
    InklingConfig,
    MyInkling,
    MyInklingAttention,
    MyInklingMLP,
    MyInklingMoE,
    MyInklingNormedEmbedding,
    MyInklingSharedExperts,
)


def _shard_parameter(
    module: nn.Module, name: str, weight: T, dim: int, mesh: DeviceMesh
):
    """
    Copy the local shard and record its dimension for checkpoint loading.
    """
    size = weight.shape[dim] // mesh.size()
    local = (
        weight.detach()
        .narrow(dim, mesh.get_local_rank() * size, size)
        .clone()
        .contiguous()
    )
    module.register_parameter(
        name, nn.Parameter(local, requires_grad=weight.requires_grad)
    )
    if not hasattr(module, "_tp_shard_dims"):
        module._tp_shard_dims = {}
    # Used to determine which dimension to shard along when loading checkpoints.
    module._tp_shard_dims[name] = dim


class RowLinear(nn.Linear):
    """
    Project an already-sharded input and sum the outputs.
    """

    def __init__(self, linear: nn.Linear, device_mesh: DeviceMesh):
        super().__init__(
            linear.in_features // device_mesh.size(),
            linear.out_features,
            bias=False,
            device="meta",
            dtype=linear.weight.dtype,
        )
        self.device_mesh = device_mesh
        _shard_parameter(self, "weight", linear.weight, 1, device_mesh)

    def forward(self, x: T) -> T:
        output = super().forward(x)
        return all_reduce(output, self.device_mesh.get_group())


class ColumnLinear(nn.Linear):
    """
    Project replicated inputs; optionally gather the output shards.
    """

    def __init__(
        self, linear: nn.Linear, device_mesh: DeviceMesh, sync_after_forward=False
    ):
        self.tp_size = device_mesh.size()
        super().__init__(
            linear.in_features,
            linear.out_features // self.tp_size,
            bias=False,
            device="meta",
            dtype=linear.weight.dtype,
        )
        self.device_mesh = device_mesh
        self.sync_after_forward = sync_after_forward
        _shard_parameter(self, "weight", linear.weight, 0, device_mesh)

    def forward(self, x: T) -> T:
        output = super().forward(x)
        if self.sync_after_forward:
            outputs = [torch.empty_like(output) for _ in range(self.tp_size)]
            dist.all_gather(outputs, output, group=self.device_mesh.get_group())
            output = torch.cat(outputs, dim=-1)
        return output


class _VocabParallelEmbedding(MyInklingNormedEmbedding):
    """
    Sum local vocabulary lookups before embedding RMSNorm.
    """

    def __init__(self, embedding: MyInklingNormedEmbedding, device_mesh: DeviceMesh):
        nn.Embedding.__init__(
            self,
            embedding.num_embeddings // device_mesh.size(),
            embedding.embedding_dim,
            device="meta",
            dtype=embedding.weight.dtype,
        )
        _shard_parameter(self, "weight", embedding.weight, 0, device_mesh)
        self.vocab_start = device_mesh.get_local_rank() * self.num_embeddings
        self.embed_norm = embedding.embed_norm
        self.device_mesh = device_mesh

    def forward(self, input_ids: Int[T, "bs s"]) -> Fp[T, "bs s d"]:
        local_ids = input_ids - self.vocab_start
        outside = (local_ids < 0) | (local_ids >= self.num_embeddings)
        embeddings = F.embedding(local_ids.masked_fill(outside, 0), self.weight)
        embeddings.masked_fill_(outside.unsqueeze(-1), 0)

        embeddings = all_reduce(embeddings, self.device_mesh.get_group())

        return self.embed_norm(embeddings)


class _TPSharedExperts(MyInklingSharedExperts):
    """
    Shard intermediate features and sum partial shared-expert outputs.
    Necessary because shared experts are all stored in one tensor and used in torch.bmm
    """

    def __init__(self, shared: MyInklingSharedExperts, device_mesh: DeviceMesh):
        nn.Module.__init__(self)
        self.n_shared_experts = shared.n_shared_experts
        self.act_fn = shared.act_fn
        self.device_mesh = device_mesh
        for name, dim in (("gate_proj", 1), ("up_proj", 1), ("down_proj", 2)):
            _shard_parameter(self, name, getattr(shared, name), dim, device_mesh)

    def forward(
        self, hidden_states: Fp[T, "bs s d"], gammas: Fp[T, "bs*s e_s"]
    ) -> Fp[T, "bs s d"]:
        output = super().forward(hidden_states, gammas)
        return all_reduce(output, self.device_mesh.get_group())


class _HeadParallelConv1d(nn.Conv1d):
    """
    Convolve this rank's K/V channels.
    """

    def __init__(self, conv: nn.Conv1d, device_mesh: DeviceMesh):
        channels = conv.in_channels // device_mesh.size()
        super().__init__(
            channels,
            channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=channels,
            bias=False,
            padding_mode=conv.padding_mode,
            device="meta",
            dtype=conv.weight.dtype,
        )
        _shard_parameter(self, "weight", conv.weight, 0, device_mesh)


def apply_tp_plan(
    model: MyInkling, config: InklingConfig, device_mesh: DeviceMesh
) -> MyInkling:
    """
    Install local shards before checkpoint loading; update head counts for caches.
    """
    tp_size = device_mesh.size()
    if tp_size == 1:
        return model
    tower = model.model.language_model
    tower.embed_tokens = _VocabParallelEmbedding(tower.embed_tokens, device_mesh)
    for layer in tower.layers:
        attention: MyInklingAttention = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "r_proj"):
            setattr(
                attention, name, ColumnLinear(getattr(attention, name), device_mesh)
            )
        attention.o_proj = RowLinear(attention.o_proj, device_mesh)
        attention.num_heads //= tp_size
        attention.num_key_value_heads //= tp_size

        for short_conv in (attention.k_sconv, attention.v_sconv):
            short_conv.conv1d = _HeadParallelConv1d(short_conv.conv1d, device_mesh)
            short_conv.dim //= tp_size

        if isinstance(layer.mlp, MyInklingMLP):
            for name in ("gate_proj", "up_proj"):
                setattr(
                    layer.mlp, name, ColumnLinear(getattr(layer.mlp, name), device_mesh)
                )
            layer.mlp.down_proj = RowLinear(layer.mlp.down_proj, device_mesh)
        elif isinstance(layer.mlp, MyInklingMoE):
            layer.mlp.shared_experts = _TPSharedExperts(
                layer.mlp.shared_experts, device_mesh
            )

    model.lm_head = ColumnLinear(model.lm_head, device_mesh, sync_after_forward=True)

    for name in (
        "num_attention_heads",
        "num_key_value_heads",
        "swa_num_attention_heads",
        "swa_num_key_value_heads",
    ):
        setattr(config.text_config, name, getattr(config.text_config, name) // tp_size)
    return model
