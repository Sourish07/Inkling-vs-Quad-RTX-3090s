import torch
from jaxtyping import Float as Fp
from jaxtyping import Int
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from torch import Tensor as T
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import functional as F

from .model import (
    MyInkling,
    MyInklingExperts,
)


class OffloadedExperts(nn.Module):
    """
    Drop-in replacement for `MyInklingExperts` that stores experts on CPU.
    - Compute local experts, copying CPU weights to GPU on demand.
    - Expects weights already sharded with EP and replicated inputs/routing.

    `weights` maps `gate_up_proj` and `down_proj` to `{expert_id: tensor}`. NVFP4
    banks stay packed and add `<projection>_scale` and `<projection>_scale2`.
    """

    def __init__(
        self,
        experts_module: MyInklingExperts,
        device_mesh: DeviceMesh,
        weights: dict[str, dict[int, T]],
    ):
        super().__init__()
        self.act_fn = experts_module.act_fn
        self.device_mesh = device_mesh
        self.weights = weights
        self.packed = "gate_up_proj_scale" in weights

    def matrix(self, projection: str, expert_id: int, like: T) -> T:
        weight = self.weights[projection][expert_id].to(like.device)
        if not self.packed:
            return weight
        scale, scale2 = (
            self.weights[projection + suffix][expert_id].to(like.device)
            for suffix in ("_scale", "_scale2")
        )
        shape = torch.Size((*weight.shape[:-1], weight.shape[-1] * 2))
        return NVFP4QTensor(shape, like.dtype, weight).dequantize(
            dtype=like.dtype,
            scale=scale,
            double_scale=scale2,
            block_sizes={-1: 16},
            fast=False,  # fast=True not supported on RTX 3090
        )

    def forward(
        self,
        hidden_states: Fp[T, "t d"],
        top_k_index: Int[T, "t k"],
        top_k_weights: Fp[T, "t k"],
    ) -> Fp[T, "t d"]:
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)

        for expert_id in torch.unique(top_k_index).tolist():
            if expert_id not in self.weights["down_proj"]:
                continue
            token_idx, top_k_pos = torch.where(top_k_index == expert_id)
            projected = F.linear(
                hidden_states[token_idx],
                self.matrix("gate_up_proj", expert_id, hidden_states),
            )
            # Packed checkpoints retain interleaved gate/up rows.
            gate, up = (
                (projected[:, 0::2], projected[:, 1::2])
                if self.packed
                else projected.chunk(2, dim=-1)
            )
            current_hidden_states = F.linear(
                self.act_fn(gate) * up,
                self.matrix("down_proj", expert_id, hidden_states),
            )
            current_hidden_states *= top_k_weights[token_idx, top_k_pos, None]

            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        torch.distributed.all_reduce(
            final_hidden_states, group=self.device_mesh.get_group()
        )
        return final_hidden_states.to(hidden_states.dtype)


def apply_ep_plan(
    model: MyInkling,
    device_mesh: DeviceMesh,
    expert_state_dict: dict[str, dict[int, torch.Tensor]],
) -> MyInkling:
    for name, module in list(model.named_modules()):
        if isinstance(module, MyInklingExperts):
            weights = {
                key.removeprefix(f"{name}."): bank
                for key, bank in expert_state_dict.items()
                if key.startswith(f"{name}.")
            }
            model.set_submodule(name, OffloadedExperts(module, device_mesh, weights))
    return model
