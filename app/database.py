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
