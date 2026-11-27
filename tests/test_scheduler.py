
from app.scheduler import scan_parallel


def test_scan_parallel_empty_input():
    result = scan_parallel([], 0, 4)
    assert result == []


def test_scan_parallel_returns_results():
    from app.gpu_kernel import scan_chunk
    from app.sequence_utils import encode

    guide = "GACGTTACCGGATTACAGTC"
    genome = encode(guide + "AGG")

    expected = scan_chunk(genome, encode(guide), 0)
    actual = scan_parallel([genome], encode(guide), 0)

    assert actual == [expected]


def test_scan_parallel_multiple_chunks():
    from app.sequence_utils import encode

    guide = "GACGTTACCGGATTACAGTC"
    guide_enc = encode(guide)

    chunks = [
        encode(guide + "AGG"),
        encode("A" * 23),
    ]

    results = scan_parallel(chunks, guide_enc, 0)

    assert len(results) == 2
    assert results[0]
    assert results[1] == []