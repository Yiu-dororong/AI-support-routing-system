import numpy as np


class ChromaRetriever:
    """
    Retrieves documents from ChromaDB with dynamic role-based query filtering.
    Supports both text queries and pre-computed embedding vector queries.
    Excludes FAQ documents from the RAG search space.
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def search(
        self,
        query: str | np.ndarray,
        user_role: str = "customer",
        n_results: int = 15,
        threshold: float = 0.0,
    ) -> list[dict]:
        """
        Queries ChromaDB using metadata-based allowed_roles filtering.
        Excludes faq_ prefixed IDs from RAG results.
        """
        metadata_filter = {"allowed_roles": {"$contains": user_role}}

        if isinstance(query, np.ndarray):
            # Query by embedding vector directly (fetching extra to account
            # for FAQ filters)
            res = self.vector_store._collection.query(
                query_embeddings=[query.tolist()],
                n_results=n_results + 15,
                where=metadata_filter,
            )
            docs = []
            if res and res["documents"] and len(res["documents"]) > 0:
                for i in range(len(res["documents"][0])):
                    doc_id = res["ids"][0][i]
                    if doc_id.startswith("faq_"):
                        continue
                    dist = res["distances"][0][i] if res["distances"] else 0.0
                    similarity = 1.0 - dist
                    if similarity >= threshold:
                        docs.append(
                            {
                                "id": doc_id,
                                "content": res["documents"][0][i],
                                "metadata": res["metadatas"][0][i],
                                "distance": dist,
                                "similarity": similarity,
                            }
                        )
            return docs[:n_results]
        else:
            # Query by semantic search text
            raw_results = self.vector_store.similarity_search_with_score(
                query, k=n_results + 15, filter=metadata_filter
            )
            docs = []
            for doc, score in raw_results:
                if doc.id.startswith("faq_"):
                    continue
                similarity = 1.0 - score
                if similarity >= threshold:
                    docs.append(
                        {
                            "id": doc.id,
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "distance": score,
                            "similarity": similarity,
                        }
                    )
            return docs[:n_results]
