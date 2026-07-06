from rank_bm25 import BM25Okapi

from rag.utils import tokenize


class BM25SearchEngine:
    """
    Lexical Search Engine utilizing BM25 Okapi for exact-keyword lookup.
    """

    def __init__(self, documents: list[dict]):
        self.raw_docs = (
            documents  # list of {"id": doc_id, "content": text, "metadata": meta}
        )
        self.corpus = [tokenize(doc["content"]) for doc in documents]
        self.bm25 = BM25Okapi(self.corpus)

    def search(
        self, query: str, user_role: str = "customer", n_results: int = 15
    ) -> list[dict]:
        """
        Retrieves matching documents based on BM25 lexical scores,
        with role-based filtering.
        """
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        doc_scores = []
        for idx, doc in enumerate(self.raw_docs):
            if doc["id"].startswith("faq_"):
                continue
            allowed_roles = doc.get("metadata", {}).get(
                "allowed_roles", ["customer", "employee"]
            )
            if user_role in allowed_roles:
                doc_scores.append((doc, scores[idx]))

        doc_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc, score in doc_scores[:n_results]:
            results.append(
                {
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "similarity": float(score),  # score as BM25 score
                }
            )
        return results
