"""
Grouped expert projections over checkpoint-layout BF16 or NVFP4 weights.
"""

import triton
import triton.language as tl


@triton.jit
def expert_gemm(
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
    row_tile_number = tl.program_id(0)
    column_tile_number = tl.program_id(1)
    group = tl.program_id(2)
    row = row_tile_number * BM + tl.arange(0, BM)
    count = tl.load(Counts + group)

    if row_tile_number * BM < count:
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
        cols = column_tile_number * BN + tl.arange(0, BN)
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

        if PROJECTION == 0:
            # acc: (BM, BN)
            # output matrix: (self.capacity, self.intermediate_dim)
            gate, up = tl.split(tl.reshape(acc, (BM, BN // 2, 2)))  # See DEVLOG.md
            activated = gate * tl.sigmoid(gate)  # Swiglu

            # Output tile: (BM, BN // 2)
            _cols = column_tile_number * (BN // 2) + tl.arange(0, BN // 2)
            _N = N // 2
            output = (activated * up).to(dtype)
        else:
            # Output tile: (BM, BN)
            _cols = cols
            _N = N
            output = acc.to(dtype)

        tl.store(
            Y + route[:, None] * _N + _cols[None, :],
            output,
            (row[:, None] < count) & (_cols[None, :] < _N),
        )


@triton.jit
def expert_reduce(
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
