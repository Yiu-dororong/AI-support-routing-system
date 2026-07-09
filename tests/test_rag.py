from unittest.mock import MagicMock

import numpy as np

from rag.bm25 import BM25SearchEngine
from rag.pipeline import RAGPipeline
from rag.reranker import DocumentReranker
from rag.utils import tokenize


def test_bm25_search_engine():
    documents = [
        {
            "id": "doc_01",
            "content": "VoltVibe Titan power station has a capacity of 500Wh.",
            "metadata": {"allowed_roles": ["customer", "employee"]},
        },
        {
            "id": "doc_11_internal",
            "content": (
                "Internal forklift procedures specify a 15,000 lbs loading limit."
            ),
            "metadata": {"allowed_roles": ["employee"]},
        },
        {
            "id": "doc_dummy",
            "content": (
                "This is a random unrelated document to prevent "
                "zero-IDF scores on small collection size."
            ),
            "metadata": {"allowed_roles": ["customer", "employee"]},
        },
    ]
    bm25 = BM25SearchEngine(documents)

    # Tokenizer checks (SKU patterns should be preserved, bigrams added)
    tokens = tokenize("VoltVibe Titan-C")
    assert "voltvibe" in tokens
    assert "titan-c" in tokens
    assert "voltvibe_titan-c" in tokens  # bigram

    # Search under customer role
    results = bm25.search("Titan", user_role="customer")
    assert len(results) == 2
    assert results[0]["id"] == "doc_01"

    # Search under employee role finds internal docs
    results_emp = bm25.search("forklift limit", user_role="employee")
    assert len(results_emp) == 3
    assert results_emp[0]["id"] == "doc_11_internal"


def test_reranker_latency_fallback():
    reranker = DocumentReranker(timeout_ms=0.01)  # tiny timeout to force fallback
    # Mock model
    reranker._model = MagicMock()
    reranker._model.predict.return_value = [1.5, 0.5]

    candidates = [
        {"id": "doc_A", "content": "Text A", "similarity": 0.8},
        {"id": "doc_B", "content": "Text B", "similarity": 0.7},
    ]

    # Reranking with tiny timeout triggers fallback
    results = reranker.rerank("query", candidates, top_m=2)
    assert len(results) == 2
    # Fallback should preserve original sequence or raw RRF scores
    assert results[0]["id"] == "doc_A"


def test_reranker_warmup():
    reranker = DocumentReranker()
    reranker._model = MagicMock()
    reranker.warmup()
    reranker._model.predict.assert_called_once_with(
        [["warmup query", "warmup document"]]
        )


def test_rag_pipeline_graceful_fallback():
    chroma = MagicMock()
    bm25 = MagicMock()
    reranker = MagicMock()

    pipeline = RAGPipeline(chroma, bm25, reranker)

    # If np.ndarray vector query is passed, fallback to pure chroma search
    vector_query = np.array([0.5, 0.5])
    pipeline.run(vector_query, user_role="customer")

    chroma.search.assert_called_once_with(
        vector_query, user_role="customer", n_results=5
    )
    bm25.search.assert_not_called()


def test_tokenization_edge_cases():
    # Mixed casing, serial numbers with underscores, hyphens, and dots
    tokens = tokenize("VT-Titan_XL-99.A2")
    # vt-titan_xl-99 matches r"[a-z0-9]+(?:[-_][a-z0-9]+)*".
    # The dot (.) is excluded, splitting "a2" into a separate token.
    assert "vt-titan_xl-99" in tokens
    assert "a2" in tokens
    # Should build shingles for consecutive tokens
    assert "vt-titan_xl-99_a2" in tokens
