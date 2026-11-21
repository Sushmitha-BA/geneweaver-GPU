"""SQLite storage for jobs and off-target hits."""
import sqlite3, time
from .config import DB_PATH


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS jobs(
          id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, guide TEXT,
          max_mm INTEGER, status TEXT, progress REAL DEFAULT 0,
          backend TEXT, seconds REAL DEFAULT 0, error TEXT, created REAL);
        CREATE TABLE IF NOT EXISTS hits(
          id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER,
          record TEXT, position INTEGER, strand TEXT, mismatches INTEGER,
          site TEXT, score REAL, severity TEXT, mm_positions TEXT);
        """)


def count_active():
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]


def fail_stale_jobs():
    """After a server restart, jobs that were running can never finish."""
    with _conn() as c:
        c.execute("""
            UPDATE jobs SET status='failed',
            error='Server restarted', progress=0
            WHERE status IN ('queued','running')
        """)


def create_job(filename, guide, max_mm):
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO jobs(filename,guide,max_mm,status,created)"
            " VALUES(?,?,?,?,?)",
            (filename, guide, max_mm, "queued", time.time())
        )
        return cur.lastrowid


def update_job(job_id, **fields):
    allowed_fields = {
        "filename", "guide", "max_mm", "status", "progress",
        "backend", "seconds", "error", "created",
    }
    invalid_fields = fields.keys() - allowed_fields
    if invalid_fields:
        raise ValueError(f"Unsupported job fields: {', '.join(sorted(invalid_fields))}")
    if not fields:
        return
    keys = ", ".join(f"{key}=?" for key in fields)
    with _conn() as c:
        c.execute(f"UPDATE jobs SET {keys} WHERE id=?", (*fields.values(), job_id))


def add_hits(job_id, rows):
    with _conn() as c:
        c.executemany("INSERT INTO hits(job_id,record,position,strand,mismatches,"
                      "site,score,severity,mm_positions) VALUES(?,?,?,?,?,?,?,?,?)",
                      [(job_id, *row) for row in rows])


def get_job(job_id):
    with _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None


def list_jobs(limit=20):
    with _conn() as c:
        return [dict(row) for row in c.execute(
            "SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))]


def get_hits(job_id, limit=500):
    with _conn() as c:
        return [dict(row) for row in c.execute(
            "SELECT * FROM hits WHERE job_id=? ORDER BY score DESC LIMIT ?",
            (job_id, limit))]
