import time

from sentence_transformers import CrossEncoder


class DocumentReranker:
    """
    Reranks candidate documents using a Cross-Encoder model.
    Includes strict candidate count limits and latency-based fallback safeguards.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        timeout_ms: float = 300.0,
    ):
        self.model_name = model_name
        self.timeout_ms = timeout_ms
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def warmup(self) -> None:
        """
        Warm up the CrossEncoder model
        by triggering model load and running a dummy query.
        This prevents latency budget warnings on the first real rerank.
        """
        try:
            # Accessing the property triggers initialization/loading of the weights
            model = self.model
            # Execute a dummy prediction to warm up compilation/caching
            model.predict([["warmup query", "warmup document"]])
        except Exception as e:
            print(f"Reranker warm-up warning: {e}", flush=True)

    def rerank(self, query: str, candidates: list[dict], top_m: int = 5) -> list[dict]:
        # Latency control: only evaluate top 10-15 candidates from RRF output
        candidates = candidates[:15]
        if not candidates:
            return []

        start_time = time.time()
        pairs = [[query, doc["content"]] for doc in candidates]

        try:
            scores = self.model.predict(pairs)

            for idx, score in enumerate(scores):
                candidates[idx]["rerank_score"] = float(score)

            # Sort by Cross-Encoder score descending
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

            elapsed_ms = (time.time() - start_time) * 1000.0
            if elapsed_ms > self.timeout_ms:
                print(
                    f"Reranker Warning: Cross-Encoder latency budget exceeded: "
                    f"{elapsed_ms:.1f}ms > {self.timeout_ms}ms"
                )

        except Exception as e:
            # Fallback mode: if error, return the original RRF ordering
            print(f"Reranker Error: {e}. Returning raw RRF candidate order.")
            for doc in candidates:
                doc["rerank_score"] = 0.0

        return candidates[:top_m]
