
from concurrent.futures import ThreadPoolExecutor


def scan_parallel(arrays, guide_enc, max_mm, workers=None):
    """
    Scan multiple genome chunks in parallel.

    Each chunk is processed using the existing GPU kernel,
    which automatically falls back to the CPU when CUDA
    is unavailable.
    """
    from .gpu_kernel import scan_chunk

    if not arrays:
        return []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(scan_chunk, arr, guide_enc, max_mm)
            for arr in arrays
        ]

        return [future.result() for future in futures]