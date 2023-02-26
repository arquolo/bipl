__all__ = ['get_gpu_capability', 'get_gpu_memory_info']

import os
from contextlib import contextmanager
from typing import NamedTuple

import pynvml

from .. import si_bin


@contextmanager
def _nvml():
    pynvml.nvmlInit()
    try:
        yield
    finally:
        pynvml.nvmlShutdown()


def _get_device_handles() -> list:
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices is not None:
        indices = [int(dev) for dev in devices.split(',')]
    else:
        *indices, = range(int(pynvml.nvmlDeviceGetCount()))

    return [pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices]


class _GpuState(NamedTuple):
    free: list[int]
    total: list[int]


def get_gpu_capability() -> list[tuple[int, int]]:
    """Gives CUDA capability for each GPU"""
    with _nvml():
        handles = _get_device_handles()
        return [pynvml.nvmlDeviceGetCudaComputeCapability(h) for h in handles]


def get_gpu_memory_info() -> _GpuState:
    """Gives size of free and total VRAM memory for each GPU"""
    with _nvml():
        handles = _get_device_handles()
        *infos, = map(pynvml.nvmlDeviceGetMemoryInfo, handles)

    return _GpuState([si_bin(i.free) for i in infos],
                     [si_bin(i.total) for i in infos])
