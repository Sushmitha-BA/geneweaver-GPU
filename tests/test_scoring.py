
import pytest
from app.scoring import score_hit


GUIDE = "GACGTTACCGGATTACAGTC"


def test_exact_match():
    score, severity, mismatches = score_hit(GUIDE, GUIDE)

    assert score == 100.0
    assert severity == "High"
    assert mismatches == []


def test_one_mismatch():
    site = "GACGTTACCGGATTACAGTA"

    score, severity, mismatches = score_hit(GUIDE, site)

    assert score == 95.0
    assert severity == "Medium"
    assert mismatches == [19]


def test_multiple_mismatches():
    site = "TACGTTACCGGATTACAGTA"

    score, severity, mismatches = score_hit(GUIDE, site)

    assert score == 90.0
    assert severity == "Medium"
    assert mismatches == [0, 19]


def test_invalid_guide_length():
    with pytest.raises(ValueError):
        score_hit("GACT", GUIDE)
        