"""
Grouped expert projections over checkpoint-layout BF16 or NVFP4 weights.
"""

import torch
import triton
import triton.language as tl


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
    QUANTIZED: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """
    Compute one projection for one tile of one active expert's routed rows.

    Each program finds its weights through the pointer table and dequantizes
    NVFP4 inline. This lets one launch cover all active experts, wherever their
    weights are, with no launch per expert and no dequantized copy of the weights.
    """
    group = tl.program_id(1)
    tile = tl.program_id(0)
    row = tile // tl.cdiv(N, BN) * BM + tl.arange(0, BM)
    count = tl.load(Counts + group)
    if tile // tl.cdiv(N, BN) * BM < count:
        expert = tl.load(Experts + group)
        pointer_stride = 6 if QUANTIZED else 2
        projection_stride = 3 if QUANTIZED else 1
        pointers = Pointers + expert * pointer_stride + PROJECTION * projection_stride
        dtype = X.dtype.element_ty
        weight_dtype = tl.uint8 if QUANTIZED else dtype
        weight = tl.load(pointers).to(tl.pointer_type(weight_dtype))
        if QUANTIZED:
            scale = tl.load(pointers + 1).to(tl.pointer_type(tl.uint8))
            scale2 = tl.load(tl.load(pointers + 2).to(tl.pointer_type(tl.float32)))
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
            if QUANTIZED:
                packed = tl.load(
                    weight + cols[None, :] * (K // 2) + k[:, None] // 2,
                    (cols[None, :] < N) & (k[:, None] < K),
                    0,
                )
                code = (packed >> ((k[:, None] % 2) * 4)) & 15
                magnitude = code & 7
                value = tl.where(
                    magnitude == 0,
                    0.0,
                    tl.where(
                        magnitude == 1,
                        0.5,
                        (2 + magnitude % 2)
                        * tl.exp2((magnitude // 2).to(tl.float32) - 2),
                    ),
                )
                value = tl.where((code & 8) != 0, -value, value)
                bits = tl.load(
                    scale + cols[None, :] * (K // 16) + k[:, None] // 16,
                    (cols[None, :] < N) & (k[:, None] < K),
                    0,
                )
                # Decode E4M3 in FP32: Ampere has no native FP8 conversion.
                # NVFP4 scales are E4M3
                exponent = (bits >> 3) & 15
                mantissa = bits & 7
                block_scale = tl.where(
                    exponent == 0,
                    mantissa * (2.0**-9),
                    (1 + mantissa * 0.125) * tl.exp2(exponent.to(tl.float32) - 7),
                )
                block_scale = tl.where((bits & 128) != 0, -block_scale, block_scale)
                b = (value * (block_scale * scale2)).to(dtype)
            else:
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
        self.capacity = 0
        self.group_size = 0  # Set by ExpertCache.prepare before each forward.

    def reserve(self, routes, *, shrink=False):
        """
        Grow the routing rows and projection scratch to hold `routes` routes.

        Forwards grow to a power of two and reuse their buffers. At the transition
        to captured decode, shrink=True releases excess prefill scratch once.
        """
        capacity = triton.next_power_of_2(routes)
        if capacity == self.capacity or (routes <= self.capacity and not shrink):
            return
        self.capacity = capacity
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
                self.quantized,
                16,
                64,
                64,
            )
            # TODO: I can probably fuse
            if projection == 0:
                projected = self.gate_up[:routes]
                torch.mul(
                    self.act_fn(projected[:, 0::2]),
                    projected[:, 1::2],
                    out=self.activated[:routes],
                )
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
