from collections import OrderedDict

import torch
from jaxtyping import Float as Fp
from jaxtyping import Int
from torch import Tensor as T
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from kernels.grouped_experts import GroupedExperts

from .model import (
    MyInkling,
    MyInklingExperts,
)


class OffloadedExperts(nn.Module):
    """
    Drop-in replacement for `MyInklingExperts` with an LRU expert cache in VRAM
    - Keep resident GPU experts and cache CPU experts in `num_slots` slots.
    - Expects weights already sharded with EP with replicated inputs.
    - Assumes mixed CPU/GPU storage and enough cache slots for all routed CPU experts.
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
        self.weight_names = [
            projection + suffix
            for projection in ("gate_up_proj", "down_proj")
            for suffix in suffixes
        ]

        # stores which experts are GPU pinned or in CPU memory
        self.cpu_expert_ids = {
            e for e, w in weights["down_proj"].items() if w.device.type == "cpu"
        }
        self.gpu_expert_ids = {
            e for e, w in weights["down_proj"].items() if w.device.type == "cuda"
        }
        # all experts that belong to this rank
        self.local_expert_ids = self.cpu_expert_ids | self.gpu_expert_ids

        self.num_slots = num_slots
        device = torch.device("cuda", torch.cuda.current_device())

        # Iterates over the 6 weight banks (proj, scale, scale2) * (gate_up, down)
        # Allocates self.num_slots for each bank
        for name in self.weight_names:
            weight = next(iter(weights[name].values()))
            self.register_buffer(
                "cache_" + name,
                torch.empty(
                    (self.num_slots, *weight.shape), dtype=weight.dtype, device=device
                ),
                persistent=False,
            )

        self.copy_stream = torch.cuda.Stream(device)

        # expert index -> slot, ordered from least to most recently used
        # we pop and then reinsert each time an expert is hit to update "lru"
        # max number of keys in lru_slots is num_slots
        self.lru_slots: OrderedDict[int, int] = OrderedDict()

        self.grouped = GroupedExperts(
            self.local_expert_ids,
            experts_module.hidden_dim,
            experts_module.intermediate_dim,
            experts_module.gate_up_proj.dtype,
            device,
            self.quantized,
            self.act_fn,
        )

    def _load(self, expert_id: int) -> int:
        """
        Called on experts that aren't in cache
        """
        if len(self.lru_slots) < self.num_slots:
            slot = len(self.lru_slots)
        else:
            _, slot = self.lru_slots.popitem(last=False)  # LRU eviction
        self.lru_slots[expert_id] = slot  # assigning new expert to slot

        # Initiate copy of 6 weights (2 if unquantized) on copy stream
        with torch.cuda.stream(self.copy_stream):
            for name in self.weight_names:
                getattr(self, "cache_" + name)[slot].copy_(
                    self.weights[name][expert_id], non_blocking=True
                )
        return slot

    def prefetch(self, expert_ids: list[int]) -> None:
        """
        Prefetch only recieves expert ids that aren't already pinned in GPU VRAM.
        """
        missing = []  # Stores which experts aren't in cache
        for expert_id in expert_ids:
            if expert_id in self.lru_slots:
                self.lru_slots.move_to_end(expert_id)
            else:
                missing.append(expert_id)

        # Wait for the previous forward before overwriting cache slots.
        self.copy_stream.wait_stream(torch.cuda.current_stream())
        for expert_id in missing:
            self._load(expert_id)

    def tensors(self, expert_id: int) -> list[T]:
        """
        Checkpoint-layout tensors from a resident expert or its cache slot.
        Assumes all experts are resident or already in cache.
        """
        if expert_id in self.gpu_expert_ids:
            return [self.weights[name][expert_id] for name in self.weight_names]
        slot = self.lru_slots[expert_id]
        return [getattr(self, "cache_" + name)[slot] for name in self.weight_names]

    def forward(
        self,
        hidden_states: Fp[T, "t d"],
        top_k_index: Int[T, "t k"],
        top_k_weights: Fp[T, "t k"],
    ) -> Fp[T, "t d"]:
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
        expert_ids = [
            e for e in torch.unique(top_k_index).tolist() if e in self.local_expert_ids
        ]
        if expert_ids:
            self.prefetch([e for e in expert_ids if e in self.cpu_expert_ids])
            torch.cuda.current_stream().wait_stream(self.copy_stream)
            self.grouped.set_weights({e: self.tensors(e) for e in expert_ids})
            self.grouped.forward(
                hidden_states, top_k_index, top_k_weights, final_hidden_states
            )
        torch.distributed.all_reduce(
            final_hidden_states, group=self.device_mesh.get_group()
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
