import torch
from jaxtyping import Float as Fp
from jaxtyping import Int
from torch import Tensor as T
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from kernels import ExpertCache, GroupedExperts
from kernels.paired_all_reduce import all_reduce

from .model import (
    MyInkling,
    MyInklingExperts,
)


class OffloadedExperts(nn.Module):
    """
    Drop-in replacement for `MyInklingExperts` with a GPU-managed LRU expert cache in VRAM
    - Keep resident GPU experts and cache CPU experts in `num_slots` slots.
    - Expects weights already sharded with EP with replicated inputs.
    - Assumes mixed CPU/GPU storage; cache slots are shared across the batch.
    - Overflow experts read mapped pinned host weights directly, including in decode.
    - NVFP4 checkpoint: layer 2 is BF16; all other expert layers are quantized.

    Example weights dict format:
    weights = {
        "gate_up_proj":        {0: packed_tensor, 1: packed_tensor},
        "gate_up_proj_scale":  {0: scale_tensor,  1: scale_tensor},
        "gate_up_proj_scale2": {0: scale2_tensor, 1: scale2_tensor},
        "down_proj":           {0: packed_tensor, 1: packed_tensor},
        "down_proj_scale":     {0: scale_tensor,  1: scale_tensor},
        "down_proj_scale2":    {0: scale2_tensor, 1: scale2_tensor},
    }
    """

    def __init__(
        self,
        experts_module: MyInklingExperts,
        device_mesh: DeviceMesh,
        weights: dict[str, dict[int, T]],
        num_slots: int,
        layer_idx: int,
    ):
        super().__init__()

        self.act_fn = experts_module.act_fn
        self.device_mesh = device_mesh
        self.weights = weights
        self.quantized = layer_idx != 2

        suffixes = ("", "_scale", "_scale2") if self.quantized else ("",)
        self.projection_names = [
            [projection + suffix for suffix in suffixes]
            for projection in ("gate_up_proj", "down_proj")
        ]
        self.weight_names = [name for names in self.projection_names for name in names]

        self.local_expert_ids = set(weights["down_proj"])
        device = torch.device("cuda", torch.cuda.current_device())

        self.grouped = GroupedExperts(
            self.local_expert_ids,
            experts_module.hidden_dim,
            experts_module.intermediate_dim,
            experts_module.gate_up_proj.dtype,
            device,
            self.quantized,
            self.act_fn,
        )

        self.cache = ExpertCache(weights, self.weight_names, num_slots, self.grouped)

    def prepare_decode_workspace(self, batch_size):
        """Release prefill scratch before allocating graph state and its snapshot."""
        self.grouped.reserve(batch_size * self.choices_per_token, shrink=True)

    def forward(
        self,
        hidden_states: Fp[T, "t d"],
        top_k_index: Int[T, "t k"],
        top_k_weights: Fp[T, "t k"],
    ) -> Fp[T, "t d"]:
        self.choices_per_token = top_k_index.shape[1]
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
        # Bound persistent per-layer scratch and the planner's expert-by-route
        # scan during batched prefill. Decode (at most 16 tokens) uses one chunk.
        for start in range(0, hidden_states.shape[0], 64):
            end = start + 64
            indices = top_k_index[start:end]
            self.cache.prepare(indices)
            self.grouped.forward(
                hidden_states[start:end],
                indices,
                top_k_weights[start:end],
                final_hidden_states[start:end],
            )
        final_hidden_states = all_reduce(
            final_hidden_states, self.device_mesh.get_group()
        )
        return final_hidden_states.to(hidden_states.dtype)


def apply_ep_plan(
    model: MyInkling,
    device_mesh: DeviceMesh,
    expert_state_dict: dict[str, dict[int, torch.Tensor]],
    num_slots: int,
) -> MyInkling:
    for name, module in list(model.named_modules()):
        if isinstance(module, MyInklingExperts):
            # Isolate just the 6 weight banks (proj, scale, scale2) * (gate_up, down) for this layer
            # the "per-expert" dictionaries (ex. {0: packed_tensor, 1: packed_tensor}) are
            # prebuilt from checkpoint loader
            weights = {
                key.removeprefix(f"{name}."): bank
                for key, bank in expert_state_dict.items()
                if key.startswith(f"{name}.")
            }
            model.set_submodule(
                name,
                OffloadedExperts(
                    module,
                    device_mesh,
                    weights,
                    num_slots,
                    layer_idx=int(name.split(".")[-3]),
                ),
            )
    return model
