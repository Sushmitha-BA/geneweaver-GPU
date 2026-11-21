
"""GeneWeaver API for genome alignment and analysis."""

import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import database
from .config import UPLOAD_DIR, MAX_UPLOAD_MB
from .sequence_utils import encode, decode, rc_encoded
from .parser import iter_chunks, genome_stats
from .gpu_kernel import scan_chunk, active_backend


app = FastAPI(
    title="GeneWeaver API",
    description="CRISPR Genome Alignment Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

database.init_db()
database.fail_stale_jobs()


@app.get("/")
def home():
    return {"message": "Welcome to GeneWeaver API"}


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "project": "GeneWeaver",
        "backend": active_backend()
    }


@app.post("/api/analyze")
async def analyze_genome(
    guide: str = Form(...),
    genome: UploadFile = File(...),
    max_mm: int = Form(4)
):
    guide = guide.strip().upper()

    if len(guide) != 20 or any(base not in "ACGT" for base in guide):
        raise HTTPException(
            status_code=400,
            detail="Guide RNA must contain exactly 20 A, C, G or T bases."
        )

    if not genome.filename or Path(genome.filename).suffix.lower() not in (".fa", ".fasta"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a .fa or .fasta file."
        )

    if not 0 <= max_mm <= 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum mismatches must be between 0 and 20."
        )

    content = await genome.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds the upload limit.")

    filename = f"{uuid.uuid4().hex}_{Path(genome.filename).name}"
    file_path = UPLOAD_DIR / filename
    file_path.write_bytes(content)

    job_id = database.create_job(filename, guide, max_mm)

    try:
        database.update_job(
            job_id,
            status="running",
            backend=active_backend(),
            progress=0
        )

        record_count, total_bases = genome_stats(file_path)

        if record_count == 0:
            raise ValueError("No FASTA records were found in the uploaded file.")

        guide_array = encode(guide)
        rows = []
        seen = set()

        for record, offset, chunk in iter_chunks(file_path):
            # Scan the forward strand.
            for index, mismatches in scan_chunk(chunk, guide_array, max_mm):
                position = offset + index
                key = (record, position, "+")

                if key in seen:
                    continue
                seen.add(key)

                site = decode(chunk[index:index + 23])
                mismatch_positions = [
                    str(i + 1)
                    for i in range(20)
                    if site[i] != guide[i]
                ]

                score = round((20 - mismatches) / 20 * 100, 2)
                severity = (
                    "high" if mismatches <= 1
                    else "medium" if mismatches <= 3
                    else "low"
                )

                rows.append((
                    record, position, "+", mismatches, site,
                    score, severity, ",".join(mismatch_positions)
                ))

            # Scan the reverse strand using the reverse-complemented chunk.
            reverse_chunk = rc_encoded(chunk)

            for index, mismatches in scan_chunk(
                reverse_chunk, guide_array, max_mm
            ):
                position = offset + len(chunk) - index - 23
                key = (record, position, "-")

                if key in seen:
                    continue
                seen.add(key)

                site = decode(reverse_chunk[index:index + 23])
                mismatch_positions = [
                    str(i + 1)
                    for i in range(20)
                    if site[i] != guide[i]
                ]

                score = round((20 - mismatches) / 20 * 100, 2)
                severity = (
                    "high" if mismatches <= 1
                    else "medium" if mismatches <= 3
                    else "low"
                )

                rows.append((
                    record, position, "-", mismatches, site,
                    score, severity, ",".join(mismatch_positions)
                ))

        database.add_hits(job_id, rows)

        database.update_job(
            job_id,
            status="completed",
            progress=100,
            seconds=0
        )

        return {
            "job_id": job_id,
            "status": "completed",
            "filename": genome.filename,
            "guide": guide,
            "backend": active_backend(),
            "records": record_count,
            "total_bases": total_bases,
            "total_hits": len(rows),
            "results": database.get_hits(job_id)
        }

    except Exception as exc:
        database.update_job(
            job_id,
            status="failed",
            error=str(exc)
        )
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/jobs")
def jobs():
    return database.list_jobs()


@app.get("/api/jobs/{job_id}")
def job_details(job_id: int):
    job = database.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    return job


@app.get("/api/jobs/{job_id}/results")
def job_results(job_id: int):
    job = database.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    return {
        "job": job,
        "results": database.get_hits(job_id)
    }
