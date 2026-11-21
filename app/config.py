"""Central settings for GeneWeaver."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "geneweaver.db"
WEB_DIR = BASE_DIR / "web"

GUIDE_LENGTH = 20            # CRISPR guide (protospacer) length
PAM_LENGTH = 3               # NGG
SITE_LEN = GUIDE_LENGTH + PAM_LENGTH
CHUNK_SIZE = 100_000         # bases per chunk
CHUNK_OVERLAP = SITE_LEN - 1 # so no site is split between chunks
DEFAULT_MAX_MISMATCHES = 4
MAX_UPLOAD_MB = 25           # live-site safety limits
MAX_ACTIVE_JOBS = 2

for _d in (DATA_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)
