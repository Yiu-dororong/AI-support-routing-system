import numpy as np

from rag.fusion import reciprocal_rank_fusion


class RAGPipeline:
    """
    Orchestrates dense search, sparse search, rank-based fusion (RRF),
    and Cross-Encoder reranking into a modular, production-grade retrieval pipeline.
    """

    def __init__(self, chroma, bm25, reranker):
        self.chroma = chroma
        self.bm25 = bm25
        self.reranker = reranker

    def run(
        self, query: str | np.ndarray, user_role: str = "customer", n_results: int = 5
    ) -> list[dict]:
        """
        Retrieves hybrid documents using dense (Chroma) and sparse (BM25)
        systems, fuses them using query-adaptive RRF, and filters/reranks them
        using a Cross-Encoder. If query is an embedding vector (np.ndarray),
        it falls back to pure dense search.
        """
        if isinstance(query, np.ndarray):
            # Graceful fallback to pure dense search when string query is not available
            return self.chroma.search(query, user_role=user_role, n_results=n_results)

        dense_results = self.chroma.search(query, user_role=user_role, n_results=15)
        sparse_results = self.bm25.search(query, user_role=user_role, n_results=15)

        fused = reciprocal_rank_fusion(dense_results, sparse_results, query)

        reranked = self.reranker.rerank(query, fused, top_m=n_results)

        return reranked
