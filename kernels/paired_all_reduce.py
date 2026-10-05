"""Four TP ranks in NVLink pairs (0, 1), (2, 3), with one PCIe hop each."""

import ctypes
import fcntl
import hashlib
import os
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist

_COMMUNICATORS = {}
_MAX_BYTES = 512 * 1024


def _library():
    source = Path(__file__).with_suffix(".cu")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache = Path.home() / ".cache" / "inkling-kernels"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"paired-{digest}.so"
    with (cache / "paired.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            subprocess.run(
                [
                    str(Path(os.getenv("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"),
                    "-O3",
                    "-std=c++17",
                    "-shared",
                    "-Xcompiler",
                    "-fPIC",
                    "-arch=sm_86",
                    str(source),
                    "-o",
                    str(target),
                ],
                check=True,
            )
    library = ctypes.CDLL(str(target))
    pointer = ctypes.c_void_p
    library.paired_allocate.argtypes = [ctypes.POINTER(pointer), ctypes.c_int]
    library.paired_get_handle.argtypes = [pointer, pointer]
    library.paired_open_handle.argtypes = [pointer, ctypes.POINTER(pointer)]
    library.paired_launch.argtypes = [
        pointer,
        pointer,
        ctypes.POINTER(pointer),
        pointer,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        pointer,
    ]
    library.paired_error.argtypes = [ctypes.c_int]
    library.paired_error.restype = ctypes.c_char_p
    return library


class PairedAllReduce:
    def __init__(self, group):
        self.library = _library()
        self.rank = dist.get_rank(group)
        self.workspace = ctypes.c_void_p()
        self._check(
            self.library.paired_allocate(ctypes.byref(self.workspace), 8 * _MAX_BYTES)
        )
        self.counter = torch.zeros(16, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        handle = ctypes.create_string_buffer(64)
        self._check(self.library.paired_get_handle(self.workspace, handle))
        handles = [None] * 4
        dist.all_gather_object(handles, handle.raw, group=group)
        pointers = []
        for rank, raw in enumerate(handles):
            if rank == self.rank:
                pointer = self.workspace
            else:
                pointer = ctypes.c_void_p()
                self._check(self.library.paired_open_handle(raw, ctypes.byref(pointer)))
            pointers.append(pointer)
        self.pointers = (ctypes.c_void_p * 4)(*pointers)
        dist.barrier(group=group)

    def _check(self, status):
        if status:
            raise RuntimeError(self.library.paired_error(status).decode())

    def all_reduce(self, x):
        output = torch.empty_like(x)
        self._check(
            self.library.paired_launch(
                x.data_ptr(),
                output.data_ptr(),
                self.pointers,
                self.counter.data_ptr(),
                x.numel() * x.element_size(),
                _MAX_BYTES,
                self.rank,
                x.dtype == torch.bfloat16,
                torch.cuda.current_stream().cuda_stream,
            )
        )
        return output


def all_reduce(x, group):
    size = x.numel() * x.element_size()
    if (
        x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float32)
        and x.is_contiguous()
        and size % 16 == 0
        and size <= _MAX_BYTES
        and dist.get_world_size(group) == 4
        and os.getenv("INKLING_PAIRED_ALLREDUCE", "1") == "1"
    ):
        if group not in _COMMUNICATORS:
            _COMMUNICATORS[group] = PairedAllReduce(group)
        return _COMMUNICATORS[group].all_reduce(x)
    dist.all_reduce(x, group=group)
    return x
