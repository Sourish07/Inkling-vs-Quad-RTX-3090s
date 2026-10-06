"""NVFP4 Marlin (W4A16) for Ampere: a CuTe DSL fork of sglang's kernel.

    from kernels.nvfp4_marlin import nvfp4_linear, prepare

    out = nvfp4_linear(x, weight, scale, scale2, out=None)
    out, run = prepare(x, weight, scale, scale2)

See README.md. cute_nvfp4_marlin is the public entry (it dispatches on M to
cute_nvfp4_decode and cute_nvfp4_batch), marlin_utils has the tensor layouts
and an fp64 oracle, sglang_ref the upstream kernel used as ground truth, and
harness the correctness gate and the benchmark.
"""

import importlib

__all__ = ["nvfp4_linear", "prepare"]


def __getattr__(name: str):
    # Lazy: the kernels import cutlass and touch the GPU; marlin_utils (pure
    # torch) must stay importable without either.
    if name in __all__:
        entry = importlib.import_module(f"{__name__}.cute_nvfp4_marlin")
        value = globals()[name] = getattr(entry, name)
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
