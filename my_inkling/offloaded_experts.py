import torch
import triton
from jaxtyping import Float as Fp
from jaxtyping import Int
from torch import Tensor as T
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from kernels import expert_copy, expert_gemm, expert_plan, expert_reduce
from kernels.paired_all_reduce_triton import all_reduce

from .model import MyInkling, MyInklingExperts


class OffloadedExperts(nn.Module):
    """
    Drop-in replacement for `MyInklingExperts` with a GPU-managed LRU expert cache in VRAM
    - Keep resident GPU experts and cache CPU experts in `num_slots` slots.
    - Expects weights already sharded (whole experts for EP, slices of every expert
      for TP) with replicated inputs.
    - Assumes mixed CPU/GPU storage and enough cache slots for one token's CPU experts.
    - Overflow prefill experts read mapped pinned host weights directly.
    - NVFP4 checkpoint: layer 2 is BF16; all other expert layers are quantized.
    - Calls run sequentially on the module's compute stream; internal buffers are
      overwritten on the next call.

    "routes" are defined as one (token, expert) pair!
    - (i.e. we're routing that token to that expert)

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
        """
        Allocate the cache slots, the address tables the planner selects from and the
        routing tables it rewrites, all on the GPU once.

        Every address a pointer-table entry can hold (resident tensor, pinned host
        tensor or cache slot) is recorded here, so a forward does no allocation, no
        host upload and no host work to find weights.
        """
        super().__init__()

        self.device_mesh = device_mesh
        self.weights = weights  # Keep mapped host allocations alive.
        self.hidden_dim = experts_module.hidden_dim
        self.intermediate_dim = experts_module.intermediate_dim
        self.dtype = experts_module.gate_up_proj.dtype
        self.quantized = layer_idx != 2
        self.device = torch.device("cuda", torch.cuda.current_device())

        # Bank order per expert; `expert_gemm` relies on it to find each projection.
        suffixes = ("", "_scale", "_scale2") if self.quantized else ("",)
        names = [
            projection + suffix
            for projection in ("gate_up_proj", "down_proj")
            for suffix in suffixes
        ]
        self.stride = len(names)

        ids = sorted(weights[names[0]])
        self.start = ids[0]
        self.local = len(ids)
        self.max_expert = ids[-1]
        assert ids == list(range(self.start, self.start + self.local))
        assert num_slots > 0
        self.num_slots = num_slots
        # GPU kernels read host weights by address, so they must be pinned.
        assert all(
            w.is_contiguous() and (w.is_cuda or w.is_pinned())
            for name in names
            for w in weights[name].values()
        )

        # Cache slots and the addresses of every place an expert's weights can be.
        first = [weights[name][ids[0]] for name in names]
        self.banks = [
            torch.empty((num_slots, *w.shape), dtype=w.dtype, device=self.device)
            for w in first
        ]
        self.sources = torch.tensor(
            [[weights[name][e].data_ptr() for name in names] for e in ids],
            dtype=torch.int64,
            device=self.device,
        )
        self.destinations = torch.tensor(
            [[bank[s].data_ptr() for bank in self.banks] for s in range(num_slots)],
            dtype=torch.int64,
            device=self.device,
        )
        self.resident = torch.tensor(
            [weights[names[0]][e].is_cuda for e in ids],
            dtype=torch.int32,
            device=self.device,
        )
        byte_sizes = [w.numel() * w.element_size() for w in first]
        assert all(size % 4 == 0 for size in byte_sizes)
        self.sizes = tuple(size // 4 for size in byte_sizes)

        # LRU state, rewritten by the planner.
        int32 = {"dtype": torch.int32, "device": self.device}
        self.slot_experts = torch.full((num_slots,), -1, **int32)
        self.ages = torch.zeros(num_slots, **int32)
        self.clock = torch.zeros((), **int32)
        self.copy_experts = torch.empty(num_slots, **int32)

        # Routing tables, rewritten by the planner and read by the projections:
        # weight addresses by global expert ID (zero: not active on this rank), then
        # the compacted active experts with their route counts and route rows.
        self.pointers = torch.zeros(
            (self.max_expert + 1) * self.stride, dtype=torch.int64, device=self.device
        )
        self.experts = torch.empty(self.local, **int32)
        self.counts = torch.empty(self.local, **int32)
        self.capacity = 0

    def _reserve(self, routes: int) -> None:
        """
        Grow the routing rows and projection scratch to hold `routes` routes.

        Prefill has many more routes than decode. Growing to a power of two and
        never shrinking lets later forwards reuse the same buffers.
        """
        if routes <= self.capacity:
            return

        self.capacity = triton.next_power_of_2(routes)
        self.rows = torch.empty(
            (self.local, self.capacity), dtype=torch.int32, device=self.device
        )
        options = {"dtype": self.dtype, "device": self.device}
        self.activated = torch.empty((self.capacity, self.intermediate_dim), **options)
        self.down = torch.empty((self.capacity, self.hidden_dim), **options)

    def _plan(self, indices: Int[T, "t k"]) -> None:
        """
        Plan this forward's routing and cache state, then load the missing experts.
        """
        routes = indices.numel()
        expert_plan[(1,)](
            indices,
            self.sources,
            self.resident,
            self.destinations,
            self.slot_experts,
            self.ages,
            self.clock,
            self.copy_experts,
            self.pointers,
            self.experts,
            self.counts,
            self.rows,
            self.start,
            self.local,
            self.num_slots,
            self.stride,
            routes,
            self.capacity,
            triton.next_power_of_2(self.local),
            triton.next_power_of_2(self.num_slots),
            triton.next_power_of_2(routes),
            num_warps=4,
        )
        expert_copy[(128, self.num_slots)](
            self.sources,
            self.destinations,
            self.copy_experts,
            self.sizes,
            self.stride,
            1024,
            128,
        )

    def _project(
        self,
        x: Fp[T, "t d"],
        indices: Int[T, "t k"],
        weights: Fp[T, "t k"],
        output: Fp[T, "t d"],
    ) -> None:
        """
        Run gate/up, activation, down and the weighted sum for all active experts.

        Reads the tables `_plan` just filled, so the number of launches is fixed per
        layer and does not depend on how many experts are active.
        """
        routes = indices.numel()  # num_tokens * top_k
        top_k = indices.shape[1]
        block_m = 16
        block_n = 64
        block_k = 64
        # The host does not know how many experts are active, so launch the upper
        # bound; the planner compacts active groups and unused ones have count zero.
        # a "group" is one active local expert along with all tokens routed to it
        group_size = min(routes, self.local)

        for projection, inputs, target, n, k in (
            (0, x, self.activated, 2 * self.intermediate_dim, self.hidden_dim),
            (1, self.activated, self.down, self.hidden_dim, self.intermediate_dim),
        ):
            grid = (triton.cdiv(routes, block_m), triton.cdiv(n, block_n), group_size)
            expert_gemm[grid](
                inputs,
                self.pointers,
                self.experts,
                self.rows,
                self.counts,
                target,
                self.capacity,
                n,
                k,
                top_k,
                projection,
                self.quantized,
                block_m,
                block_n,
                block_k,
            )

        # For EP, we're summing across just this rank's experts
        # For TP, we're summing across all experts, albiet just a quarter of their features
        expert_reduce[(x.shape[0], triton.cdiv(self.hidden_dim, 256))](
            self.down,
            indices,
            weights,
            self.pointers,
            output,
            self.hidden_dim,
            top_k,
            self.stride,
            self.max_expert,
            256,
        )

    def forward(
        self,
        hidden_states: Fp[T, "t d"],
        top_k_index: Int[T, "t k"],
        top_k_weights: Fp[T, "t k"],
    ) -> Fp[T, "t d"]:
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
        self._reserve(top_k_index.numel())
        self._plan(top_k_index)
        self._project(hidden_states, top_k_index, top_k_weights, final_hidden_states)

        """
        All-reduce required for both EP & TP!
        - For EP, a token's routed experts are distributed, so we need to sum the outputs
            - We're summing across experts
            - If a token wasn't routed to a rank, its column will remain zero from the torch.zeros_like()
        - For TP, a token's routed experts are sliced, so we need RowLinear's all-reduce
            - We're summing across experts' features
        (It's almost "coincidental" that EP & TP both require an all-reduce here)
        """
        final_hidden_states = all_reduce(
            final_hidden_states, self.device_mesh.get_group()
        )

        return final_hidden_states.to(hidden_states.dtype)


def apply_offload_plan(
    model: MyInkling,
    device_mesh: DeviceMesh,
    expert_state_dict: dict[str, dict[int, torch.Tensor]],
    num_slots: int,
) -> MyInkling:
    """
    Replace every routed-expert module by `OffloadedExperts` over this rank's weights.
    """
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
