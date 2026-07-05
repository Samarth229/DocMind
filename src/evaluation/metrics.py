"""
Standard IR metrics implemented from scratch.

Plain-English meanings (worth reciting without notes in an interview):
  precision@k  — of the top-k chunks we returned, what fraction were relevant?
  recall@k     — of all relevant chunks in the corpus, what fraction did we find in top-k?
  MRR          — on average, how far down our ranked list is the FIRST relevant chunk?
                 (score = 1/rank, 0 if no hit; averaged across queries = MRR)
"""


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        # No relevant chunks exist — recall is undefined; return 1.0 so a
        # no-answer query doesn't penalize the retriever for not finding nothing.
        return 1.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 1.0  # same convention as recall: no relevant → not a miss
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0
