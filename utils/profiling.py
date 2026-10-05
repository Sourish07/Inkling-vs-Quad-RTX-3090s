import os
import time
from collections.abc import Callable
from typing import Any

import torch
from loguru import logger
from torch.distributed import get_rank
from torch.profiler import ProfilerActivity, profile


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
