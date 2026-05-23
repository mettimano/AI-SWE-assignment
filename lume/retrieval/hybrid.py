"""Reciprocal Rank Fusion over BM25 + vector results with explicit weighting.

Weights are set by the Intent agent based on query characteristics
(e.g., brand/product name → higher bm25_weight; mood/occasion → higher vector_weight).
"""

from __future__ import annotations

_RRF_K = 60  # standard dampening constant


def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[str, float]],
    weights: list[float] | None = None,
    top_n: int = 20,
) -> list[str]:
    """Fuse ranked lists via weighted RRF.

    Args:
        ranked_lists: Each is [(product_id, score), ...] sorted by relevance.
        weights: One weight per list. Defaults to equal weights (1.0 each).
        top_n: Number of product_ids to return.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match number of ranked_lists")

    scores: dict[str, float] = {}
    for ranked, w in zip(ranked_lists, weights):
        for rank, (pid, _) in enumerate(ranked, start=1):
            scores[pid] = scores.get(pid, 0.0) + w / (_RRF_K + rank)

    return [pid for pid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:top_n]


def hybrid_search(
    bm25_results: list[tuple[str, float]],
    vector_results: list[tuple[str, float]],
    bm25_weight: float = 1.0,
    vector_weight: float = 1.0,
    top_n: int = 20,
) -> list[str]:
    """Fuse BM25 and vector results. Weights supplied by the Intent agent.

    Returns top_n product_ids sorted by fused RRF score.
    """
    return reciprocal_rank_fusion(
        bm25_results,
        vector_results,
        weights=[bm25_weight, vector_weight],
        top_n=top_n,
    )
