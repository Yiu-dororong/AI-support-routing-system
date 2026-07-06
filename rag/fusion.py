import re


def get_query_weights(query: str) -> tuple[float, float]:
    """
    Detect SKU patterns or code indicators in query (e.g., 'doc_23', 'V1', '100Wh')
    to dynamically shift weights.
    Returns: (bm25_weight, dense_weight)
    """
    if re.search(r"\b[a-z0-9]+(?:[-_][a-z0-9]+)+\b|\b\d+wh\b|\bv\d+\b", query.lower()):
        return 1.8, 0.8  # Heavy BM25 bias for structured queries
    return 1.2, 1.0  # Standard hybrid balance


def reciprocal_rank_fusion(
    dense_results: list[dict], sparse_results: list[dict], query: str, k: int = 60
) -> list[dict]:
    """
    Applies Reciprocal Rank Fusion (RRF) on dense and sparse retrieval results.
    Ranks are fused using query-adaptive weighting.
    """
    rrf_scores = {}
    bm25_weight, dense_weight = get_query_weights(query)

    # Add ranks for dense results
    for rank, doc in enumerate(dense_results, start=1):
        doc_id = doc["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {"doc": doc.copy(), "score": 0.0}
        rrf_scores[doc_id]["score"] += dense_weight / (k + rank)

    # Add ranks for sparse results
    for rank, doc in enumerate(sparse_results, start=1):
        doc_id = doc["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {"doc": doc.copy(), "score": 0.0}
        rrf_scores[doc_id]["score"] += bm25_weight / (k + rank)

    sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

    fused_docs = []
    for item in sorted_docs:
        doc = item["doc"]
        doc["rrf_score"] = item["score"]
        # Retain original similarity cosine score if present,
        # otherwise use RRF score
        if "similarity" not in doc or doc["similarity"] is None:
            doc["similarity"] = item["score"]
        fused_docs.append(doc)

    return fused_docs
