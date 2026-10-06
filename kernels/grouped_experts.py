"""
Grouped expert projections over BF16 or Marlin-packed NVFP4 weights.
"""

import torch
import triton
import triton.language as tl

from .nvfp4_marlin import nvfp4_linear
from .nvfp4_marlin.cute_nvfp4_decode import prepare_grouped

# The larger tiles of the Marlin batch kernel need more shared memory than sm86 has.
MARLIN_MAX_ROWS = 16


@triton.jit
def _gemm(
    X,
    Pointers,
    Experts,
    Rows,
    Counts,
    Y,
    CAPACITY: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TOP_K: tl.constexpr,
    PROJECTION: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """
    Compute one projection for one tile of one active expert's routed rows.

    Each program finds its BF16 weights through the pointer table. This lets one
    launch cover all active experts, wherever their weights are, with no launch
    per expert. NVFP4 layers use the Marlin kernels instead.
    """
    group = tl.program_id(1)
    tile = tl.program_id(0)
    row = tile // tl.cdiv(N, BN) * BM + tl.arange(0, BM)
    count = tl.load(Counts + group)
    if tile // tl.cdiv(N, BN) * BM < count:
        expert = tl.load(Experts + group)
        dtype = X.dtype.element_ty
        weight = tl.load(Pointers + expert * 2 + PROJECTION).to(tl.pointer_type(dtype))
        route = tl.load(Rows + group * CAPACITY + row, row < count, 0)
        input_row = route // TOP_K if PROJECTION == 0 else route
        cols = tile % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for start in range(tl.cdiv(K, BK)):
            k = start * BK + ks
            a = tl.load(
                X + input_row[:, None] * K + k[None, :],
                (row[:, None] < count) & (k[None, :] < K),
                0,
            )
            b = tl.load(
                weight + cols[None, :] * K + k[:, None],
                (cols[None, :] < N) & (k[:, None] < K),
                0,
            )
            acc += tl.dot(a, b)
        tl.store(
            Y + route[:, None] * N + cols[None, :],
            acc.to(dtype),
            (row[:, None] < count) & (cols[None, :] < N),
        )


@triton.jit
def _reduce(
    Down,
    Indices,
    Weights,
    Pointers,
    Output,
    WIDTH: tl.constexpr,
    TOP_K: tl.constexpr,
    POINTER_STRIDE: tl.constexpr,
    MAX_EXPERT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Add each token's routed expert outputs, scaled by the router weights, to Output.

    The GEMMs write one row per route, so the rows must be combined per token.
    Routes whose pointer row is zero belong to other ranks and are skipped, which
    keeps their stale scratch rows out of the sum.
    """
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for choice in range(TOP_K):
        route = token * TOP_K + choice
        expert = tl.load(Indices + route)
        active = (
            tl.load(Pointers + expert * POINTER_STRIDE, expert <= MAX_EXPERT, 0) != 0
        )
        value = tl.load(Down + route * WIDTH + cols, active & (cols < WIDTH), 0).to(
            tl.float32
        )
        weight = tl.load(Weights + route).to(tl.float32)
        acc += (value * weight).to(Down.dtype.element_ty).to(tl.float32)
    output = Output + token * WIDTH + cols
    tl.store(output, tl.load(output, cols < WIDTH, 0) + acc, cols < WIDTH)


class GroupedExperts:
    """
    Reusable pointer table, routing workspace and projection scratch for one layer.
    - Pointer table: self.pointers
    - Routing workspace: self.experts, self.counts, self.rows
    - Projection scratch: self.gate_up, self.activated, self.down

    Calls run sequentially on the module's compute stream. The output is owned
    by the caller; internal buffers are overwritten on the next call.
    """

    def __init__(
        self, expert_ids, hidden_dim, intermediate_dim, dtype, device, quantized, act_fn
    ):
        """
        Allocate the pointer table, group list and counts on the GPU once.

        The cache planner rewrites them on the GPU each forward, so the hot path
        needs no allocation and no host upload.
        """
        self.max_expert = max(expert_ids)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dtype = dtype
        self.device = device
        self.quantized = quantized
        self.act_fn = act_fn
        self.pointer_stride = 6 if quantized else 2
        self.pointers = torch.zeros(
            (self.max_expert + 1) * self.pointer_stride,
            dtype=torch.int64,
            device=device,
        )
        self.experts = torch.empty(len(expert_ids), dtype=torch.int32, device=device)
        self.counts = torch.empty(len(expert_ids), dtype=torch.int32, device=device)
        # One 64-byte record per group for the grouped Marlin kernel (see _plan).
        self.records = torch.zeros(
            (len(expert_ids), 8), dtype=torch.int64, device=device
        )
        self.capacity = 0
        self.group_size = 0  # Set by ExpertCache.prepare before each forward.
        self.lookup = None  # Set by ExpertCache: expert -> weight tensors.

    def reserve(self, routes):
        """
        Grow the routing rows and projection scratch to hold `routes` routes.

        Prefill has many more routes than decode. Growing to a power of two and
        never shrinking lets later forwards reuse the same buffers.
        """
        if routes <= self.capacity:
            return
        self.capacity = triton.next_power_of_2(routes)
        self.rows = torch.empty(
            (len(self.experts), self.capacity), dtype=torch.int32, device=self.device
        )
        options = {"dtype": self.dtype, "device": self.device}
        self.gate_up = torch.empty(
            (self.capacity, 2 * self.intermediate_dim), **options
        )
        self.activated = torch.empty((self.capacity, self.intermediate_dim), **options)
        self.down = torch.empty((self.capacity, self.hidden_dim), **options)

    def forward(self, x, indices, weights, output):
        """
        Run gate/up, activation, down and the weighted sum for all active experts.

        `ExpertCache.prepare` must run first: it fills the pointers, groups, counts
        and rows that these kernels read. The number of launches is then fixed per
        layer and does not depend on how many experts are active.
        """
        routes = indices.numel()
        self.reserve(routes)
        top_k = indices.shape[1]
        if not self.quantized:
            self._project_bf16(x, routes, top_k)
        elif x.shape[0] == 1:
            self._project_marlin_token(x, routes)
        else:
            self._project_marlin_rows(x, top_k)
        _reduce[(x.shape[0], triton.cdiv(self.hidden_dim, 256))](
            self.down,
            indices,
            weights,
            self.pointers,
            output,
            self.hidden_dim,
            top_k,
            self.pointer_stride,
            self.max_expert,
            256,
        )

    def _activate(self, projected, out=None):
        return torch.mul(self.act_fn(projected[:, 0::2]), projected[:, 1::2], out=out)

    def _project_marlin_token(self, x, routes):
        """
        One token: every active expert has one row, so each projection is one launch.

        The kernel finds each group's weights and rows in the planner's records.
        Its arguments are fixed, which keeps the launch valid in the decode graph.
        """
        if x.data_ptr() % 16 or not x.is_contiguous():
            x = x.contiguous().clone()
        groups = self.group_size
        # Record words 6 and 7 hold the group's route and that route's token.
        prepare_grouped(x, self.records, 7, 6, self.gate_up, groups, projection=0)()
        self._activate(self.gate_up[:routes], out=self.activated[:routes])
        prepare_grouped(
            self.activated, self.records, 6, 6, self.down, groups, projection=1
        )()

    def _project_marlin_rows(self, x, top_k):
        """
        Many tokens (prefill): one host-dispatched Marlin call per active expert.

        Reading the routing back synchronizes with the GPU, which only prefill
        can afford. Rows are chunked to the largest tile that fits sm86.
        """
        tensors = self.lookup()
        experts = self.experts[: self.group_size].tolist()
        counts = self.counts[: self.group_size].tolist()
        for group, (expert, count) in enumerate(zip(experts, counts)):
            if count == 0:
                continue
            weight, scale, scale2, down_weight, down_scale, down_scale2 = tensors(
                expert
            )
            routes = self.rows[group, :count].long()
            for chunk in routes.split(MARLIN_MAX_ROWS):
                projected = nvfp4_linear(x[chunk // top_k], weight, scale, scale2[:1])
                self.down[chunk] = nvfp4_linear(
                    self._activate(projected), down_weight, down_scale, down_scale2[:1]
                )

    def _project_bf16(self, x, routes, top_k):
        for projection, inputs, target, n, k in (
            (0, x, self.gate_up, 2 * self.intermediate_dim, self.hidden_dim),
            (1, self.activated, self.down, self.hidden_dim, self.intermediate_dim),
        ):
            # The GPU planner compacts active groups; unused groups have count zero.
            _gemm[(triton.cdiv(routes, 16) * triton.cdiv(n, 64), self.group_size)](
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
                16,
                64,
                64,
            )
            # TODO: I can probably fuse
            if projection == 0:
                self._activate(self.gate_up[:routes], out=self.activated[:routes])
