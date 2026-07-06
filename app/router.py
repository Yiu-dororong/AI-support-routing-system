import numpy as np


class Router:
    """
    Orchestration only. Contains no business logic, no database operations,
    no vector DB calls, and no ranking calculations.
    """

    def __init__(self, intent_classifier, faq_handler, planner, rag_pipeline):
        self.intent_classifier = intent_classifier
        self.faq_handler = faq_handler
        self.planner = planner
        self.rag_pipeline = rag_pipeline

    def classify_intent(
        self, query_emb: np.ndarray, threshold: float = 0.4
    ) -> tuple[bool, str, dict[str, float]]:
        """
        Delegates intent classification to the intent_classifier.
        """
        return self.intent_classifier.classify(query_emb, threshold=threshold)

    def match_faq(
        self, query_emb: np.ndarray, threshold: float = 0.8
    ) -> tuple[dict | None, list[dict]]:
        """
        Delegates FAQ matching to the faq_handler.
        """
        return self.faq_handler.match(query_emb, threshold=threshold)

    def plan_routing(
        self, query: str, intent: str, callbacks=None, metadata: dict = None
    ):
        """
        Delegates multi-step planning and routing to the planner.
        """
        return self.planner.plan(query, intent, callbacks=callbacks, metadata=metadata)

    def retrieve_rag_documents(
        self, query: str, user_role: str = "customer", n_results: int = 5
    ) -> list[dict]:
        """
        Delegates RAG document retrieval to the rag_pipeline.
        """
        return self.rag_pipeline.run(query, user_role=user_role, n_results=n_results)
