"""
Raw CUDA kernels: compiled by NVRTC, launched on PyTorch's current stream.

Triton cannot keep per-thread state in registers across a loop, index shared
memory or pick its own thread layout; the kernels that need those are written
in CUDA C++ instead. NVRTC and the driver bindings ship with PyTorch.
"""

from pathlib import Path

import numpy as np
import torch
from cuda.bindings import driver, nvrtc

_SOURCES = Path(__file__).resolve().parent


def _check(result):
    error, *values = result
    if int(error) != 0:
        raise RuntimeError(f"CUDA call failed: {error!r}")
    return values[0] if len(values) == 1 else values


def _compile(source: str, name: str, defines: dict[str, int]) -> bytes:
    major, minor = torch.cuda.get_device_capability()
    options = [f"--gpu-architecture=compute_{major}{minor}".encode(), b"-std=c++17"]
    options += [f"-D{key}={value}".encode() for key, value in defines.items()]
    program = _check(
        nvrtc.nvrtcCreateProgram(source.encode(), name.encode(), 0, [], [])
    )
    (error,) = nvrtc.nvrtcCompileProgram(program, len(options), options)
    if int(error) != 0:
        size = _check(nvrtc.nvrtcGetProgramLogSize(program))
        log = b" " * size
        _check(nvrtc.nvrtcGetProgramLog(program, log))
        raise RuntimeError(f"NVRTC failed on {name}:\n{log.decode(errors='replace')}")
    ptx = b" " * _check(nvrtc.nvrtcGetPTXSize(program))
    _check(nvrtc.nvrtcGetPTX(program, ptx))
    _check(nvrtc.nvrtcDestroyProgram(program))
    return ptx


class CudaModule:
    """
    One .cu file of this directory, compiled for a set of -D constants.

    The module is built on first use, in the calling process's CUDA context;
    each rank is its own process, so one build per module and rank.
    """

    def __init__(self, filename: str, **defines: int):
        self.filename = filename
        self.defines = defines
        self._module = None
        self._functions = {}

    def __getitem__(self, name: str) -> "CudaKernel":
        if name not in self._functions:
            if self._module is None:
                torch.cuda.init()
                source = (_SOURCES / self.filename).read_text()
                ptx = _compile(source, self.filename, self.defines)
                self._module = _check(driver.cuModuleLoadData(ptx))
            function = _check(driver.cuModuleGetFunction(self._module, name.encode()))
            self._functions[name] = CudaKernel(function)
        return self._functions[name]


class CudaKernel:
    def __init__(self, function):
        self.function = function

    def __call__(self, grid, block: int, *args, shared: int = 0) -> None:
        """
        Launch on the current stream; a CUDA graph capture records it like any kernel.

        Tensors are passed as device pointers, Python ints as int32 and floats as
        float32. Arguments are copied at launch, so they may be temporaries.
        """
        values = [
            np.array(a.data_ptr(), dtype=np.uint64)
            if isinstance(a, torch.Tensor)
            else np.array(a, dtype=np.float32 if isinstance(a, float) else np.int32)
            for a in args
        ]
        addresses = np.array([v.ctypes.data for v in values], dtype=np.uint64)
        grid = (*grid, 1, 1)[:3]
        _check(
            driver.cuLaunchKernel(
                self.function,
                *grid,
                block,
                1,
                1,
                shared,
                torch.cuda.current_stream().cuda_stream,
                addresses.ctypes.data,
                0,
            )
        )
