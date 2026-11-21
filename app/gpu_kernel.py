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


def active_backend():
    return "cuda-gpu" if GPU_AVAILABLE else "numba-cpu"


def scan_chunk(chunk, guide, max_mm, backend="auto"):
    """Return [(index, mismatches)] - same result as cpu_baseline.scan_cpu."""
    n = chunk.shape[0] - L - 2
    if n <= 0:
        return []
    out = np.full(n, -1, dtype=np.int8)
    if GPU_AVAILABLE and backend in ("auto", "gpu"):
        d_chunk = cuda.to_device(chunk)              # host RAM -> GPU VRAM
        d_guide = cuda.to_device(guide)
        d_out = cuda.to_device(out)
        threads = 256
        blocks = (n + threads - 1) // threads
        _scan_kernel[blocks, threads](d_chunk, d_guide, max_mm, d_out)
        out = d_out.copy_to_host()                   # GPU VRAM -> host RAM
    else:
        _scan_numba_cpu(chunk, guide, max_mm, out)
    idx = np.nonzero(out >= 0)[0]
    return [(int(i), int(out[i])) for i in idx]


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
