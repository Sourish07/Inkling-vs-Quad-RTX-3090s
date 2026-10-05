from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=None)
def _load_ext(dtype: torch.dtype, capability: tuple[int, int]):
    from torch.utils.cpp_extension import load

    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Marlin requires FP16 or BF16 activations")
    if capability[0] < 8:
        raise ValueError("Marlin requires CUDA compute capability 8.0 or newer")
    root = Path(__file__).resolve().parent
    arch = f"{capability[0]}{capability[1]}"
    return load(
        name=f"inkling_nvfp4_linear_{str(dtype).split('.')[-1]}_sm{arch}",
        sources=[str(root / "csrc/bindings.cpp"), str(root / "csrc/nvfp4_moe.cu")],
        extra_include_paths=[str(root / "csrc/include"), str(root / "include")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            "-O3", "-std=c++20", "-lineinfo", "--expt-relaxed-constexpr",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
            f"-DINKLING_BF16={int(dtype == torch.bfloat16)}",
        ],
    )


def nvfp4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    scale2: torch.Tensor,
) -> torch.Tensor:
    """One expert's bias-free linear projection, returning [tokens, output_dim].

    x is contiguous CUDA FP16/BF16 [tokens, input_dim]. weight, scale and scale2
    are the three tensors returned by prepare_nvfp4, on x's device. Routing,
    gate/up splitting and routing-weight multiplication belong to the caller.
    """
    if not x.is_cuda:
        raise ValueError("Marlin requires CUDA tensors")
    return _load_ext(x.dtype, torch.cuda.get_device_capability(x.device)).linear(
        x, weight, scale, scale2
    )
