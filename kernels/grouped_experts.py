"""
Grouped expert projections over checkpoint-layout BF16 or NVFP4 weights.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _route(
    Indices,
    Experts,
    Rows,
    Counts,
    ROUTES: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0)
    expert = tl.load(Experts + group)
    routes = tl.arange(0, BLOCK)
    indices = tl.load(Indices + routes, routes < ROUTES, -1)
    matches = (routes < ROUTES) & (indices == expert)
    positions = tl.cumsum(matches.to(tl.int32)) - 1
    tl.store(Rows + group * CAPACITY + positions, routes, matches)
    tl.store(Counts + group, tl.sum(matches.to(tl.int32)))


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

    Calls run sequentially on the module's compute stream. The output is owned
    by the caller; internal buffers are overwritten on the next call.
    """

    def __init__(
        self, expert_ids, hidden_dim, intermediate_dim, dtype, device, quantized, act_fn
    ):
        self.max_expert = max(expert_ids)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dtype = dtype
        self.device = device
        self.quantized = quantized
        self.act_fn = act_fn
        # Copy active expert IDs and their pointer table together.
        self.pointer_stride = 6 if quantized else 2
        pointer_size = (self.max_expert + 1) * self.pointer_stride
        self.host_metadata = torch.zeros(
            pointer_size + len(expert_ids), dtype=torch.int64, pin_memory=True
        )
        self.metadata = torch.empty_like(self.host_metadata, device=device)
        self.pointer_array = (
            self.host_metadata[:pointer_size].numpy().reshape(-1, self.pointer_stride)
        )
        self.expert_array = self.host_metadata[pointer_size:].numpy()
        self.pointers = self.metadata[:pointer_size]
        self.experts = self.metadata[pointer_size:]
        self.metadata_copied = torch.cuda.Event()
        self.counts = torch.empty(len(expert_ids), dtype=torch.int32, device=device)
        self.capacity = 0

    def set_weights(self, weights):
        # The previous async copy must finish before we rewrite pinned memory.
        self.metadata_copied.synchronize()
        self.pointer_array[:] = 0
        self.group_size = len(weights)
        self.expert_array[: self.group_size] = list(weights)
        for expert, tensors in weights.items():
            self.pointer_array[expert] = [w.data_ptr() for w in tensors]
        self.metadata.copy_(self.host_metadata, non_blocking=True)
        self.metadata_copied.record()

    def reserve(self, routes):
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
        routes = indices.numel()
        self.reserve(routes)
        top_k = indices.shape[1]
        _route[(self.group_size,)](
            indices,
            self.experts,
            self.rows,
            self.counts,
            routes,
            self.capacity,
            triton.next_power_of_2(routes),
        )
        for projection, inputs, target, n, k in (
            (0, x, self.gate_up, 2 * self.intermediate_dim, self.hidden_dim),
            (1, self.activated, self.down, self.hidden_dim, self.intermediate_dim),
        ):
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
