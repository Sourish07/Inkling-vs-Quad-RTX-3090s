"""
Small inference kernels; convolution history is FP32.
"""

import torch
import triton
import triton.language as tl
from jaxtyping import Float as Fp
from jaxtyping import Int
from torch import Tensor as T


@triton.jit
def _rms_norm(
    X,
    W,
    Y,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    RMS-normalize one row of X over its last dimension and scale it by W.

    Each program finds its row through SHAPE and STRIDES, so transposed heads and
    slices of the fused projection are read in place with no contiguous copy.
    The reduction runs in FP32.
    """
    row = tl.program_id(0)
    remaining = row
    offset = 0
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        offset += (remaining % SHAPE[axis]) * STRIDES[axis]
        remaining //= SHAPE[axis]
    d = tl.arange(0, BLOCK)
    x = tl.load(X + offset + d * STRIDES[-1], d < D, 0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    # PyTorch rounds normalized activations before multiplying by the weight.
    normalized = (x * scale).to(X.dtype.element_ty).to(tl.float32)
    weight = tl.load(W + d, d < D, 0).to(tl.float32)
    tl.store(Y + row * D + d, normalized * weight, d < D)


def rms_norm(
    x: Fp[T, "*batch d"], weight: Fp[T, " d"], eps: float
) -> Fp[T, "*batch d"]:
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    width = x.shape[-1]
    _rms_norm[(x.numel() // width,)](
        x,
        weight,
        output,
        tuple(x.shape[:-1]),
        tuple(x.stride()),
        width,
        eps,
        triton.next_power_of_2(width),
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _short_conv(
    X,
    W,
    History,
    Residual,
    Y,
    D: tl.constexpr,
    K: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Convolve one new token with its cached history and add the skip connection.

    The FP32 History is shifted and the token appended in place, so decode needs
    no separate cache update. ADD_RESIDUAL also adds the decoder residual in the
    same launch.
    """
    d = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    current = tl.load(X + d, d < D, 0).to(tl.float32)
    total = tl.full((BLOCK,), 0, tl.float32)
    # A channel belongs to one lane: load each old value before overwriting it.
    for j in tl.static_range(K - 1):
        previous = tl.load(History + d * K + j + 1, d < D, 0)
        weight = tl.load(W + d * K + j, d < D, 0).to(tl.float32)
        total += previous * weight
        tl.store(History + d * K + j, previous, d < D)
    weight = tl.load(W + d * K + K - 1, d < D, 0).to(tl.float32)
    total += current * weight
    tl.store(History + d * K + K - 1, current, d < D)
    output = (total + current).to(Y.dtype.element_ty)
    if ADD_RESIDUAL:
        residual = tl.load(Residual + d, d < D, 0).to(tl.float32)
        output = output.to(tl.float32) + residual
    tl.store(Y + d, output, d < D)


def short_conv(
    x: Fp[T, "1 1 d"],
    weight: Fp[T, "d 1 conv_kernel_size"],
    history: Fp[T, "1 d conv_kernel_size"],
    residual: Fp[T, "1 1 d"] | None = None,
) -> Fp[T, "1 1 d"]:
    output = torch.empty_like(x)
    width = x.shape[-1]
    _short_conv[(triton.cdiv(width, 256),)](
        x,
        weight,
        history,
        residual if residual is not None else x,
        output,
        width,
        weight.shape[-1],
        residual is not None,
        256,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _update_kv(
    K,
    V,
    KCache,
    VCache,
    HEADS: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    K_STRIDES: tl.constexpr,
    V_STRIDES: tl.constexpr,
    CAPACITY: tl.constexpr,
    START,
    DEVICE_START: tl.constexpr,
    WINDOW: tl.constexpr,
    MIRROR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Copy the new key and value tokens into both caches, starting at START.

    K and V are read through their strides, so the transposed head views need no
    contiguous copy. A nonzero WINDOW selects the sliding cache: positions wrap
    and each token is written twice, WINDOW apart.
    """
    K += tl.program_id(0) * K_STRIDES[0]
    V += tl.program_id(0) * V_STRIDES[0]
    KCache += tl.program_id(0) * HEADS * CAPACITY * D
    VCache += tl.program_id(0) * HEADS * CAPACITY * D

    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    channel = i % D
    token = i // D % S
    head = i // (D * S)
    key = tl.load(
        K + head * K_STRIDES[1] + token * K_STRIDES[2] + channel * K_STRIDES[3],
        i < HEADS * S * D,
        0,
    )
    value = tl.load(
        V + head * V_STRIDES[1] + token * V_STRIDES[2] + channel * V_STRIDES[3],
        i < HEADS * S * D,
        0,
    )
    start = tl.load(START) if DEVICE_START else START
    position = start + token
    if WINDOW:
        position %= WINDOW
    destination = (head * CAPACITY + position) * D + channel
    tl.store(KCache + destination, key, i < HEADS * S * D)
    tl.store(VCache + destination, value, i < HEADS * S * D)
    if WINDOW and MIRROR:
        # Mirrored ring: the newest window is always a contiguous chronological view.
        tl.store(KCache + destination + WINDOW * D, key, i < HEADS * S * D)
        tl.store(VCache + destination + WINDOW * D, value, i < HEADS * S * D)


def update_kv(
    key: Fp[T, "bs hk s c"],
    value: Fp[T, "bs hk s c"],
    k_cache: Fp[T, "bs hk capacity c"],
    v_cache: Fp[T, "bs hk capacity c"],
    start: int | Int[T, ""],  # Start will remain the same for all sequences for now
    window: int = 0,
    mirror: bool = True,
) -> None:
    batch_size = key.shape[0]
    block_size = 256
    grid = (
        batch_size,
        triton.cdiv(key[0].numel(), block_size),
    )

    _update_kv[grid](
        key,
        value,
        k_cache,
        v_cache,
        key.shape[1],
        key.shape[2],
        key.shape[3],
        tuple(key.stride()),
        tuple(value.stride()),
        k_cache.shape[2],
        start,
        isinstance(start, torch.Tensor),
        window,
        mirror,
        block_size,
    )
