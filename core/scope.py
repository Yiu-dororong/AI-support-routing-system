import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class IntentClassifier:
    """
    Handles classification of user query intent using cosine similarity
    against precomputed centroids.
    """

    def __init__(self, intent_centroids: dict[str, np.ndarray]):
        self.intent_centroids = intent_centroids

    def classify(
        self, query_emb: np.ndarray, threshold: float = 0.4
    ) -> tuple[bool, str, dict[str, float]]:
        """
        Classifies the query embedding against the centroids.
        Returns: (in_scope, max_intent, similarities)
        """
        similarities = {}
        for intent_id, centroid in self.intent_centroids.items():
            similarities[intent_id] = cosine_similarity(query_emb, centroid)

        if not similarities:
            return False, "unknown", {}

        max_intent = max(similarities, key=similarities.get)
        max_score = similarities[max_intent]

        in_scope = max_score >= threshold
        return in_scope, max_intent, similarities
