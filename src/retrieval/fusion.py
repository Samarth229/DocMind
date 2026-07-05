def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Merge multiple ranked lists via Reciprocal Rank Fusion.

    Formula: rrf_score(doc) = sum over lists of 1 / (k + rank)
    where rank is 1-indexed position in each list.

    k=60 is the default from the original RRF paper (Cormack et al., 2009).
    No weights needed — RRF is robust to list quality variation without tuning.

    Returns a deduplicated, merged list sorted by rrf_score descending,
    with an added `rrf_score` field on each result dict.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            cid = doc["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in docs:
                docs[cid] = doc

    return sorted(
        [{**docs[cid], "rrf_score": score} for cid, score in scores.items()],
        key=lambda d: d["rrf_score"],
        reverse=True,
    )
