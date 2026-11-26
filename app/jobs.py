
def run_job(job_id, path, guide, max_mm):
    """Run a genome scanning job and save the results."""
    import time

    t0 = time.time()

    try:
        db.update_job(
            job_id,
            status="running",
            backend=active_backend()
        )

        _, total = genome_stats(path)
        guide_enc = encode(guide)

        def on_progress(done):
            progress = 100 if total == 0 else min(done / total, 1) * 100
            db.update_job(
                job_id,
                progress=round(progress, 1)
            )

        rows = []

        for rec, off, clen, strand, arr, i, mm in _scan_all(
            path,
            guide_enc,
            max_mm,
            on_progress
        ):
            site = decode(arr[i:i + SITE_LEN])

            if strand == "+":
                pos = off + i
            else:
                pos = off + clen - i - SITE_LEN

            score, severity, mismatches = score_hit(guide, site)

            rows.append((
                rec,
                pos,
                strand,
                mm,
                site,
                score,
                severity,
                ",".join(map(str, mismatches))
            ))

        db.add_hits(job_id, rows)

        db.update_job(
            job_id,
            status="done",
            progress=100,
            seconds=round(time.time() - t0, 3)
        )

    except Exception as exc:
        db.update_job(
            job_id,
            status="error",
            error=str(exc)
        )


def start_job(job_id, path, guide, max_mm):
    """Start a genome scanning job in a background thread."""
    import threading

    thread = threading.Thread(
        target=run_job,
        args=(job_id, path, guide, max_mm),
        daemon=True
    )

    thread.start()
    return threadpython -m 
