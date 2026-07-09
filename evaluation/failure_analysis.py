# Offline system failure analysis and taxonomy logging.


def classify_failure(
    query: str, retrieved_docs: list[dict], generated_answer: str, ground_truth: str
) -> str:
    """
    Classifies RAG errors into standard taxonomies:
    - E1: Retrieval Failure (unretrieved context)
    - E2: Synthesis Hallucination (answer contains ungrounded claims)
    - E3: Guardrail Refusal (accidental refuse trigger)
    - E4: Intent Classifier Error (misrouted intent)
    """
    # Simple classification rule placeholder
    if not retrieved_docs:
        return "E1: Retrieval Failure"
    return "Unclassified failure"
