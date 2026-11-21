from app.sequence_utils import make_mock_genome, encode
from app.parser import iter_chunks, genome_stats
from app.cpu_baseline import scan_cpu

GUIDE = "GACGTTACCGGATTACAGTC"


def test_chunks_cover_genome(tmp_path):
    fa = make_mock_genome(tmp_path / "g.fa", 250_000)
    n, total = genome_stats(fa)
    assert (n, total) == (1, 250_000)
    chunks = list(iter_chunks(fa, chunk_size=100_000))
    assert len(chunks) == 3 and all(len(c) <= 100_000 for _, _, c in chunks)


def test_cpu_baseline_finds_planted_sites(tmp_path):
    fa = make_mock_genome(tmp_path / "g.fa", 200_000, GUIDE, plants=15)
    found = sum(len(scan_cpu(c, encode(GUIDE), 3)) for _, _, c in iter_chunks(fa))
    assert found >= 15
