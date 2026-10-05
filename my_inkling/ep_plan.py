import torch
from jaxtyping import Float as Fp
from jaxtyping import Int
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
    Compute local experts, copying CPU weights to GPU on demand.

    Expects weights already sharded with EP and replicated inputs/routing.
    f: expert intermediate dimension; d: hidden dimension.
    """

    def __init__(
        self,
        experts_module: MyInklingExperts,
        device_mesh: DeviceMesh,
        gate_up: dict[int, Fp[T, "2*f d"]],
        down: dict[int, Fp[T, "d f"]],
    ):
        super().__init__()

        self.num_experts = experts_module.num_experts
        self.hidden_dim = experts_module.hidden_dim
        self.intermediate_dim = experts_module.intermediate_dim
        self.act_fn = experts_module.act_fn

        self.device_mesh = device_mesh
        self.weights = {
            expert_id: (gate_up_weight, down[expert_id])
            for expert_id, gate_up_weight in gate_up.items()
        }

    def forward(
        self,
        hidden_states: Fp[T, "t d"],
        top_k_index: Int[T, "t k"],
        top_k_weights: Fp[T, "t k"],
    ) -> Fp[T, "t d"]:
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
        for expert_id in torch.unique(top_k_index).tolist():
            if expert_id not in self.weights:
                continue
            gate_up, down = (
                weight.to(hidden_states.device) for weight in self.weights[expert_id]
            )
            token_idx, top_k_pos = torch.where(top_k_index == expert_id)
            current_state = hidden_states[token_idx]

            gate, up = F.linear(current_state, gate_up).chunk(2, dim=-1)

            current_hidden_states = F.linear(self.act_fn(gate) * up, down)
            current_hidden_states *= top_k_weights[token_idx, top_k_pos, None]

            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

            del gate_up, down

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
            gate_up = expert_state_dict[f"{name}.gate_up_proj"]
            down = expert_state_dict[f"{name}.down_proj"]

            model.set_submodule(
                name,
                OffloadedExperts(
                    module, device_mesh=device_mesh, gate_up=gate_up, down=down
                ),
            )
    return model
