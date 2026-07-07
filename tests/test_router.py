from unittest.mock import MagicMock

import numpy as np
import pytest

from core.faq import FAQHandler

# Import modular components from new folder structure
from core.scope import IntentClassifier, cosine_similarity


def test_cosine_similarity():
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.0, 1.0])

    assert cosine_similarity(a, b) == pytest.approx(1.0)
    assert cosine_similarity(a, c) == pytest.approx(0.0)
    assert cosine_similarity(a, np.array([0.0, 0.0])) == 0.0


def test_intent_classifier():
    intent_centroids = {
        "shipping": np.array([1.0, 0.0]),
        "returns": np.array([0.0, 1.0]),
    }
    classifier = IntentClassifier(intent_centroids)

    # Matches shipping centroid
    in_scope, intent, similarities = classifier.classify(
        np.array([0.9, 0.1]), threshold=0.5
    )
    assert in_scope is True
    assert intent == "shipping"
    assert similarities["shipping"] > similarities["returns"]

    # Matches neither well
    in_scope, intent, similarities = classifier.classify(
        np.array([0.1, 0.1]), threshold=0.8
    )
    assert in_scope is False


def test_faq_handler():
    faq_embeddings = [
        {
            "question": "What is the return policy?",
            "answer": "30 days return window.",
            "intent": "returns",
            "embedding": np.array([1.0, 0.0]),
        },
        {
            "question": "Do you ship internationally?",
            "answer": "Yes, worldwide shipping.",
            "intent": "shipping",
            "embedding": np.array([0.0, 1.0]),
        },
    ]
    handler = FAQHandler(faq_embeddings)

    # Direct match returns
    best_match, top_3 = handler.match(np.array([0.95, 0.05]), threshold=0.8)
    assert best_match is not None
    assert best_match["intent"] == "returns"
    assert len(top_3) == 2

    # Under threshold
    best_match, top_3 = handler.match(np.array([0.5, 0.5]), threshold=0.8)
    assert best_match is None
    assert len(top_3) == 2


def test_support_router_integration(monkeypatch):
    # Mock HuggingFaceEmbeddings
    mock_emb = MagicMock()
    mock_emb.embed_documents.return_value = [[0.1] * 384]
    mock_emb.embed_query.return_value = [0.1] * 384
    monkeypatch.setattr(
        "router_logic.HuggingFaceEmbeddings", lambda model_name: mock_emb
    )

    # Mock hf_hub_download
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: "mock_path")

    # Mock OS checks
    monkeypatch.setattr("os.path.exists", lambda path: True)

    # Mock httpx server check
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: mock_resp)

    # Mock ChatOpenAI structured outputs
    mock_llm = MagicMock()
    monkeypatch.setattr("router_logic.ChatOpenAI", lambda **kwargs: mock_llm)

    # Mock ChromaDB
    mock_chroma = MagicMock()
    mock_chroma._collection.count.return_value = 10
    mock_chroma.get.return_value = {
        "ids": ["doc_01"],
        "documents": ["VoltVibe capacity 500Wh"],
        "metadatas": [
            {"title": "Title 1", "category": "general", "allowed_roles": '["customer"]'}
        ],
    }
    monkeypatch.setattr("router_logic.Chroma", lambda **kwargs: mock_chroma)

    # Instantiate SupportRouter (using mock chroma db and config)
    from router_logic import SupportRouter

    # Instantiate the class (will trigger constructor and populate modules)
    support_router = SupportRouter(chroma_path="mock_chroma_path")

    assert support_router.router is not None
    assert support_router.intent_classifier is not None
    assert support_router.faq_handler is not None
    assert support_router.planner is not None
    assert support_router.rag_pipeline is not None
    assert len(support_router.kb_documents) == 1
    assert support_router.kb_documents[0]["id"] == "doc_01"


def test_mixed_intent_classification():
    # Intent classifier with overlapping intent centroids
    intent_centroids = {
        "returns": np.array([1.0, 0.0, 0.0]),
        "shipping": np.array([0.0, 1.0, 0.0]),
        "payments": np.array([0.0, 0.0, 1.0]),
    }
    classifier = IntentClassifier(intent_centroids)

    # Query embedding lies exactly between returns and shipping
    query_emb = np.array([0.707, 0.707, 0.0])

    # Should resolve to either return or shipping since they are tied
    # as highest similarity
    in_scope, intent, similarities = classifier.classify(query_emb, threshold=0.5)
    assert in_scope is True
    assert intent in ["returns", "shipping"]
    assert similarities["returns"] == pytest.approx(0.707, abs=1e-3)
    assert similarities["shipping"] == pytest.approx(0.707, abs=1e-3)


def test_vague_and_empty_queries():
    # Test classifier handling of an all-zero embedding (e.g. empty or failed embedding)
    intent_centroids = {
        "returns": np.array([1.0, 0.0]),
        "shipping": np.array([0.0, 1.0]),
    }
    classifier = IntentClassifier(intent_centroids)

    # Zero vector
    in_scope, intent, similarities = classifier.classify(
        np.array([0.0, 0.0]), threshold=0.4
    )
    assert in_scope is False
