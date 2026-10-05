from collections import OrderedDict

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
    Drop-in replacement for `MyInklingExperts` with an LRU expert cache in VRAM
    - Keep resident GPU experts and cache CPU experts in `num_slots` slots.
    - Expects weights already sharded with EP with replicated inputs.
    - `num_slots` must fit all CPU experts requested by a forward pass.

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
    ):
        super().__init__()

        self.act_fn = experts_module.act_fn
        self.device_mesh = device_mesh
        self.weights = weights

        # stores which experts are GPU pinned or in CPU memory
        self.cpu_expert_ids = {
            e for e, w in weights["down_proj"].items() if w.device.type == "cpu"
        }
        self.gpu_expert_ids = {
            e for e, w in weights["down_proj"].items() if w.device.type == "cuda"
        }
        # all experts that belong to this rank
        self.local_expert_ids = self.cpu_expert_ids | self.gpu_expert_ids

        self.num_slots = min(num_slots, len(self.cpu_expert_ids))
        device = torch.device("cuda", torch.cuda.current_device())

        # Iterates over the 6 weight banks (proj, scale, scale2) * (gate_up, down)
        # Allocates self.num_slots for each bank
        for name, bank in weights.items():
            weight = next(iter(bank.values()))
            self.register_buffer(
                "cache_" + name,
                torch.empty(
                    (self.num_slots, *weight.shape), dtype=weight.dtype, device=device
                ),
                persistent=False,
            )

        self.copy_stream = torch.cuda.Stream(device)

        # pending = {slot_id: {gate_up_proj: Event, down_proj: Event}}
        self.pending: dict[int, dict[str, torch.cuda.Event]] = {}

        # expert index -> slot, ordered from least to most recently used
        # we pop and then reinsert each time an expert is hit to update "lru"
        # max number of keys in lru_slots is num_slots
        self.lru_slots: OrderedDict[int, int] = OrderedDict()

    def _load(self, expert_id: int) -> int:
        """
        Called on experts that aren't in cache
        """
        if len(self.lru_slots) < self.num_slots:
            slot = len(self.lru_slots)
        else:
            _, slot = self.lru_slots.popitem(last=False) # LRU eviction
        self.lru_slots[expert_id] = slot # assigning new expert to slot

        # Initiate copy of 6 weights on copy stream
        copied = {}
        with torch.cuda.stream(self.copy_stream):
            for projection in ("gate_up_proj", "down_proj"):
                for suffix in ("", "_scale", "_scale2"):
                    name = projection + suffix
                    getattr(self, "cache_" + name)[slot].copy_(
                        self.weights[name][expert_id], non_blocking=True
                    )
                copied[projection] = torch.cuda.Event()
                copied[projection].record()
        self.pending[slot] = copied
        return slot

    def prefetch(self, expert_ids: list[int]) -> None:
        """
        Prefetch only recieves expert ids that aren't already pinned in GPU VRAM.
        """
        missing = [] # Stores which experts aren't in cache
        for expert_id in expert_ids:
            if expert_id in self.lru_slots:
                self.lru_slots.move_to_end(expert_id)
            else:
                missing.append(expert_id)

        # Since copy stream & compute stream operate independently, we don't want to enqueue copies
        # that may override slots a GEMM in compute stream requires
        # Technically not necessary due to other sync points in model
        self.copy_stream.wait_stream(torch.cuda.current_stream())
        for expert_id in missing:
            self._load(expert_id)

    def matrix(self, projection: str, expert_id: int, like: T) -> T:
        """
        Returns the weight matrix for the given projection and expert ID, converted to the
        same device and dtype as `like`.
        """
        slot = self.lru_slots.get(expert_id)
        if slot is not None and slot in self.pending:
            # self.pending[slot] = {gate_up_proj: Event, down_proj: Event}
            torch.cuda.current_stream().wait_event(self.pending[slot].pop(projection))

            # Remove the slot when both projections are done
            if not self.pending[slot]:
                del self.pending[slot]

        # slot is None means it's a GPU pinned expert
        weight = (
            self.weights[projection][expert_id].to(like.device)
            if slot is None
            else getattr(self, "cache_" + projection)[slot]
        )
        scale, scale2 = (
            self.weights[projection + suffix][expert_id].to(like.device)
            if slot is None
            else getattr(self, "cache_" + projection + suffix)[slot]
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

        expert_ids = [
            e
            for e in torch.unique(top_k_index).tolist()
            if e in self.local_expert_ids
        ]

        # prefetch only non-gpu pinned experts
        self.prefetch([e for e in expert_ids if e in self.cpu_expert_ids])
        for expert_id in expert_ids:
            token_idx, top_k_pos = torch.where(top_k_index == expert_id)
            projected = F.linear(
                hidden_states[token_idx],
                self.matrix("gate_up_proj", expert_id, hidden_states),
            )

            # Packed checkpoints retain interleaved gate/up rows.
            gate, up = projected[:, 0::2], projected[:, 1::2]

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
                name, OffloadedExperts(module, device_mesh, weights, num_slots)
            )
    return model
