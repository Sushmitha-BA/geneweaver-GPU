"""Numba engine: CUDA kernel on NVIDIA GPUs, parallel-CPU fallback otherwise."""
import numpy as np
from numba import njit, prange
from .config import GUIDE_LENGTH

try:
    from numba import cuda
    GPU_AVAILABLE = cuda.is_available()
except Exception:                       # no driver / no GPU
    cuda, GPU_AVAILABLE = None, False

L = GUIDE_LENGTH


@njit(parallel=True, cache=True)
def _scan_numba_cpu(chunk, guide, max_mm, out):
    n = chunk.shape[0] - L - 2
    for i in prange(n):
        out[i] = -1
        if chunk[i + L + 1] != 2 or chunk[i + L + 2] != 2:
            continue
        mm = 0
        for j in range(L):
            if chunk[i + j] != guide[j]:
                mm += 1
        if mm <= max_mm:
            out[i] = mm


if GPU_AVAILABLE:
    @cuda.jit
    def _scan_kernel(chunk, guide, max_mm, out):
        i = cuda.grid(1)               # one GPU thread per genome position
        if i >= out.shape[0]:
            return
        out[i] = -1
        if chunk[i + L + 1] != 2 or chunk[i + L + 2] != 2:
            return
        mm = 0
        for j in range(L):
            if chunk[i + j] != guide[j]:
                mm += 1
        if mm <= max_mm:
            out[i] = mm
