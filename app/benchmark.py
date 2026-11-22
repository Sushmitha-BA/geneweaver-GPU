"""CPU baseline vs Numba/CUDA benchmark. Run: python -m app.benchmark"""

import json
import tempfile
import time
from pathlib import Path

from .config import DATA_DIR
from .sequence_utils import make_mock_genome, encode
from .parser import iter_chunks
from .cpu_baseline import scan_cpu
from .gpu_kernel import scan_chunk, active_backend

GUIDE = "GACGTTACCGGATTACAGTC"


def run(length=500_000, max_mm=4):
    fa = Path(tempfile.gettempdir()) / "gw_bench.fa"

    make_mock_genome(fa, length, GUIDE)
    guide = encode(GUIDE)

    chunks = [c for _, _, c in iter_chunks(fa)]

    # Warm-up: compile the Numba function if needed.
    scan_chunk(chunks[0], guide, max_mm)

    t = time.perf_counter()
    cpu_hits = sum(
        len(scan_cpu(c, guide, max_mm)) for c in chunks
    )
    cpu_s = time.perf_counter() - t

    t = time.perf_counter()
    fast_hits = sum(
        len(scan_chunk(c, guide, max_mm)) for c in chunks
    )
    fast_s = time.perf_counter() - t

    result = {
        "bases": length,
        "backend": active_backend(),
        "cpu_seconds": round(cpu_s, 4),
        "fast_seconds": round(fast_s, 4),
        "speedup": round(cpu_s / max(fast_s, 1e-9), 1),
        "hits_match": cpu_hits == fast_hits,
        "hits": fast_hits
    }

    (DATA_DIR / "benchmark.json").write_text(
        json.dumps(result, indent=2)
    )

    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
