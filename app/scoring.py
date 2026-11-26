
def score_hit(guide, site):
    """
    Calculate the off-target score based on mismatches.

    Returns:
        score: Similarity score from 0 to 100.
        severity: Risk category.
        mismatches: Positions where bases differ.
    """

    guide = guide.upper()
    site = site.upper()

    if len(guide) != 20 or len(site) < 20:
        raise ValueError("Guide must be 20 bases and site must contain at least 20 bases.")

    mismatches = []

    for i in range(20):
        if guide[i] != site[i]:
            mismatches.append(i)

    mismatch_count = len(mismatches)

    score = round(
        ((20 - mismatch_count) / 20) * 100,
        2
    )

    if mismatch_count == 0:
        severity = "High"
    elif mismatch_count <= 2:
        severity = "Medium"
    else:
        severity = "Low"

    return score, severity, mismatches
