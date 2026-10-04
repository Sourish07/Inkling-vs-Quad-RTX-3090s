import os
import random
import socket
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.distributed import get_rank, get_world_size
from torch.distributed.device_mesh import init_device_mesh
from torch.profiler import ProfilerActivity, profile

if socket.gethostname() == "sk-ml":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,3,1,2")
_DEVICE_MESH = {}


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_ddp_local():
    os.environ["TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING"] = "false"
    if os.environ.get("WORLD_SIZE") is None:
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = os.environ["RANK"]
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        master_port_str = os.environ.get("MASTER_PORT")
        if master_port_str is None:
            os.environ["MASTER_PORT"] = str(random.randint(20000, 60000))

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=local_rank)
    return device, local_rank


def setup_rank_aware_logger():
    logger.remove()

    RANK_COLORS = [
        "<green>",
        "<blue>",
        "<red>",
        "<yellow>",
        "<magenta>",
        "<cyan>",
        "<white>",
        "<black>",
    ]

    rank = get_rank()
    rank_color = RANK_COLORS[rank % len(RANK_COLORS)]

    # Loguru supports color tags inside the format string
    fmt = f"{rank_color}<rank {rank}> | {{time:HH:mm:ss}} | {{level}} | {{message}}</>"

    # INFO and below -> stdout
    logger.add(
        sys.stdout,
        format=fmt,
        level="INFO",
        filter=lambda r: r["level"].no < 30,
        colorize=True,  # Setting to true forces colors even when not in a TTY
    )

    logger.add(sys.stderr, format=fmt, level="WARNING", colorize=True)


class Timer:
    """Simple context manager to time code blocks and log the duration.

    Usage:
        with Timer("data_loading"):
            load_data()
    """

    def __init__(self, name: str):
        self.name = name
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start is not None:
            duration_seconds = time.perf_counter() - self._start
            logger.info(f"{self.name} took {duration_seconds}s")
        # Propagate exceptions, don't suppress
        return False


class Profiler:
    def __init__(self, enable: bool = False, export_all_ranks: bool = False):
        self.profiler = None

        self.rank = get_rank() if torch.distributed.is_initialized() else 0
        profiling_enabled = enable or os.getenv("ENABLE_PROFILING", "0") == "1"
        should_profile = profiling_enabled and (export_all_ranks or self.rank == 0)

        if should_profile:
            self.profiler = profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA,
                ],
                with_stack=True,
                acc_events=True,
            )

    def start(self):
        if self.profiler is not None:
            logger.info("Starting profiler")
            self.profiler.start()

    def stop(self):
        if self.profiler is not None:
            logger.info("Stopping profiler")
            self.profiler.stop()

    def step(self):
        if self.profiler is not None:
            logger.info("Stepping profiler")
            self.profiler.step()

    def export_chrome_trace(self, path: str):
        if self.profiler is not None:
            # Insert rank into the filename if exporting from multiple ranks
            if self.rank > 0:
                ext = ".json.gz"
                base = path.split(ext)[0]
                path = f"{base}_rank{self.rank}{ext}"
            logger.info("Exporting profiler trace")
            self.profiler.export_chrome_trace(path)
            logger.info(f"Profiler trace exported to {path}")

    def get_callback_on_step_end(self) -> Callable[..., Any]:
        def step_profiler_callback(*args, **kwargs) -> dict:
            self.step()
            return {}

        return step_profiler_callback


def get_device_mesh(num_dimensions: int = 1):
    if num_dimensions not in _DEVICE_MESH:
        grid = (2, 2) if num_dimensions == 2 else (get_world_size(),)
        mesh_dim_names = ("pcie", "nvlink") if num_dimensions == 2 else None

        _DEVICE_MESH[num_dimensions] = init_device_mesh(
            "cuda", grid, mesh_dim_names=mesh_dim_names
        )
    return _DEVICE_MESH[num_dimensions]
