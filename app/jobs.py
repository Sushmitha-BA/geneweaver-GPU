
"""Runs a scan in a background thread and stores progress + hits."""
import threading, time
from . import database as db
from .config import SITE_LEN, GUIDE_LENGTH
from .parser import iter_chunks, genome_stats
from .sequence_utils import encode, decode, rc_encoded
from .gpu_kernel import scan_chunk, active_backend

try:                                     # plugged in on Day 17
    from .scoring import score_hit
except ImportError:
    def score_hit(guide, site):
        return 0.0, "Unscored", []

try:                                   # plugged in on Day 19
    from .scheduler import scan_parallel
except ImportError:
    scan_parallel = None


def _run_batch(batch, guide_enc, max_mm):
    """Scan both strands of every chunk in the batch (Dask if available)."""
    tasks = [(rec, off, len(ch), strand, arr)
             for rec, off, ch in batch
             for strand, arr in (("+", ch), ("-", rc_encoded(ch)))]
    arrays = [t[4] for t in tasks]
    if scan_parallel:
        results = scan_parallel(arrays, guide_enc, max_mm)
    else:
        results = [scan_chunk(a, guide_enc, max_mm) for a in arrays]
    for (rec, off, clen, strand, arr), hits in zip(tasks, results):
        for i, mm in hits:
            yield rec, off, clen, strand, arr, i, mm


def _scan_all(path, guide_enc, max_mm, on_progress, batch_size=16):
    done, batch = 0, []
    for item in iter_chunks(path):
        batch.append(item)
        if len(batch) == batch_size:
            yield from _run_batch(batch, guide_enc, max_mm)
            done += sum(len(b[2]) for b in batch)
            on_progress(done)
            batch = []
    if batch:
        yield from _run_batch(batch, guide_enc, max_mm)
        done += sum(len(b[2]) for b in batch)
        on_progress(done)
