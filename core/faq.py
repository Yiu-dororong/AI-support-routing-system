import numpy as np

from core.scope import cosine_similarity


class FAQHandler:
    """
    Handles deterministic match checking against a pre-embedded FAQ dataset.
    Does not make database queries or LLM calls.
    """

    def __init__(self, faq_embeddings: list[dict]):
        self.faq_embeddings = faq_embeddings

    def match(
        self, query_emb: np.ndarray, threshold: float = 0.8
    ) -> tuple[dict | None, list[dict]]:
        """
        Calculates cosine similarity of the query against FAQ embeddings.
        Returns: (best_match_dict_or_None, top_3_candidates_list)
        """
        matches = []
        for faq in self.faq_embeddings:
            score = cosine_similarity(query_emb, faq["embedding"])
            matches.append(
                {
                    "question": faq["question"],
                    "answer": faq["answer"],
                    "intent": faq["intent"],
                    "score": score,
                }
            )

        matches.sort(key=lambda x: x["score"], reverse=True)

        best_match = matches[0] if matches else None
        if best_match and best_match["score"] >= threshold:
            return best_match, matches[:3]
        return None, matches[:3]
