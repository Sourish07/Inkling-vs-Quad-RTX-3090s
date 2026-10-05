"""One-time preparation of a ModelOpt expert's packed checkpoint tensors."""

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _load_cpu_prepare():
    from torch.utils.cpp_extension import load

    return load(
        name="inkling_nvfp4_prepare_cpu",
        sources=[str(Path(__file__).resolve().parent / "csrc/nvfp4_prepare.cpp")],
        extra_cflags=["-O3", "-std=c++20"],
        with_cuda=False,
    )


def prepare_nvfp4(
    weight: torch.Tensor,
    scale: torch.Tensor,
    scale2: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert [N, K/2] checkpoint weight, [N, K/16] scale and scalar scale2.

    Returns Marlin weight [K/16, N*2], scale [K/16, N], and processed scale2.
    Run once before creating the GPU cache, on CPU or CUDA. Preserve the output
    row order, including interleaved gate/up rows. dtype must match activations.
    Pinned CPU inputs produce pinned CPU outputs for asynchronous onload.
    """
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Activations must be FP16 or BF16")
    if weight.ndim != 2 or weight.dtype != torch.uint8:
        raise ValueError("Checkpoint weight must be uint8 [N, K / 2]")
    n, packed_k = weight.shape
    k = packed_k * 2
    if n == 0 or k == 0 or n % 128 or k % 64:
        raise ValueError("Marlin requires positive N divisible by 128 and K divisible by 64")
    if scale.shape != (n, k // 16) or scale.dtype != torch.float8_e4m3fn:
        raise ValueError("Checkpoint scale must be E4M3 [N, K / 16]")
    if scale2.numel() != 1:
        raise ValueError("Checkpoint scale2 must be a scalar")
    if scale.device != weight.device or scale2.device != weight.device:
        raise ValueError("Checkpoint tensors must share a device")

    if weight.device.type == "cpu":
        return tuple(_load_cpu_prepare().prepare(
            weight, scale, scale2, dtype == torch.bfloat16
        ))

    # Unpack E2M1 codes, transpose to [K,N], and arrange tensor-core fragments.
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).reshape(n, k).T
    permutation = []
    for i in range(32):
        fragment = [
            16 * row + i // 4 + 8 * block
            for block in (0, 1)
            for row in (2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1)
        ]
        for j in range(4):
            permutation.extend(p + 256 * j for p in fragment)
    permutation = torch.tensor(permutation, device=weight.device).reshape(-1, 8)
    permutation = permutation[:, [0, 2, 4, 6, 1, 3, 5, 7]].flatten()
    tiles = codes.reshape(k // 16, 16, n // 16, 16).permute(0, 2, 1, 3)
    tiles = tiles.reshape(-1, 1024)[:, permutation].reshape(k // 16, n * 2, 8).long()
    shifts = torch.arange(8, device=weight.device) * 4
    packed = (tiles << shifts).sum(-1).to(torch.int32).contiguous()

    # FP8 block scales need both a layout permutation and an exponent adjustment.
    scale_perm = [i + 8 * j for i in range(8) for j in range(8)]
    s = scale.float().T.contiguous().reshape(-1, 64)[:, scale_perm]
    s = s.reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(k // 16, n)
    bits = (s.half() * 128).view(torch.int16) << 1
    processed_scale = bits.view(torch.float8_e4m3fn)[:, 1::2].contiguous()
    processed_scale2 = scale2.to(dtype).reshape(1) * (2.0 ** (7 if dtype == torch.float16 else 119))
    result = (packed, processed_scale, processed_scale2)
    if weight.device.type == "cpu" and weight.is_pinned():
        result = tuple(t.pin_memory() for t in result)
    return result
