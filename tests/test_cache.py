import time

import numpy as np
import pytest

from core.cache import (
    DualLayerCacheManager,
    calculate_rbo,
    hash_prompt,
    normalize_query,
)
from core.types import RoutingDecision
from llm.generator import ResponseGenerator, SynthesisResponse


def test_normalize_query():
    q1 = "Can I return a damaged iPhone???"
    q2 = "can i return a damaged iphone"
    q3 = "  CAN I   RETURN A DAMAGED IPHONE!  "

    norm1 = normalize_query(q1)
    norm2 = normalize_query(q2)
    norm3 = normalize_query(q3)

    assert norm1 == "can i return a damaged iphone"
    assert norm1 == norm2 == norm3


def test_calculate_rbo():
    # Identical lists
    list_a = ["doc1", "doc2", "doc3"]
    list_b = ["doc1", "doc2", "doc3"]
    assert calculate_rbo(list_a, list_b) == pytest.approx(1.0)

    # Disjoint lists
    list_c = ["doc4", "doc5"]
    assert calculate_rbo(list_a, list_c) == pytest.approx(0.0)

    # Both empty
    assert calculate_rbo([], []) == 1.0

    # Top-rank match scores higher than lower-rank match
    top_match_b = ["doc1", "docX", "docY"]
    bottom_match_b = ["docX", "docY", "doc3"]
    score_top = calculate_rbo(list_a, top_match_b, p=0.8)
    score_bottom = calculate_rbo(list_a, bottom_match_b, p=0.8)
    assert score_top > score_bottom


def test_planner_cache_layer_1():
    cache = DualLayerCacheManager(ttl_planner_seconds=10.0)
    prompt_hash = hash_prompt("System prompt v1")

    query = "Can I return my order?"
    decision = RoutingDecision(
        path="rag_llm",
        reason="Requires policy synthesis",
    )

    # Store in Layer 1
    stored = cache.set_planner(
        query=query,
        user_role="customer",
        prompt_hash=prompt_hash,
        decision=decision,
        intent="return_policy",
    )
    assert stored is True

    # Exact query hit
    hit = cache.get_planner(query, "customer", prompt_hash)
    assert hit is not None
    assert hit.path == "rag_llm"

    # Normalized query hit (punctuation & upper case variant)
    hit_variant = cache.get_planner("CAN I RETURN MY ORDER???", "customer", prompt_hash)
    assert hit_variant is not None
    assert hit_variant.path == "rag_llm"

    # Role mismatch
    miss_role = cache.get_planner(query, "employee", prompt_hash)
    assert miss_role is None

    # Prompt hash mismatch (prompt updated)
    miss_prompt = cache.get_planner(query, "customer", "different_hash")
    assert miss_prompt is None


def test_planner_cache_reject_error_fallbacks():
    cache = DualLayerCacheManager()
    prompt_hash = hash_prompt("System prompt v1")

    # Error decision from except Exception fallback
    error_decision = RoutingDecision(
        path="rag",
        reason="""Local execution planning failed with error: llama-server crash.
        Defaulting to standard RAG lookup.""",
    )

    stored = cache.set_planner(
        query="Failed query",
        user_role="customer",
        prompt_hash=prompt_hash,
        decision=error_decision,
        intent="general",
    )
    assert stored is False

    hit = cache.get_planner("Failed query", "customer", prompt_hash)
    assert hit is None


def test_planner_cache_ttl_expiration():
    cache = DualLayerCacheManager(ttl_planner_seconds=0.1)
    prompt_hash = hash_prompt("System prompt v1")

    decision = RoutingDecision(path="rag", reason="Standard RAG")
    cache.set_planner("Query", "customer", prompt_hash, decision, "intent")

    # Immediate hit
    assert cache.get_planner("Query", "customer", prompt_hash) is not None

    # Wait for TTL to expire
    time.sleep(0.15)
    assert cache.get_planner("Query", "customer", prompt_hash) is None


def test_response_cache_layer_2_and_5point_gate():
    cache = DualLayerCacheManager(theta_read=0.85, theta_rbo=0.60)

    query = "What is the return window for electronics?"
    query_emb = np.array([1.0, 0.0, 0.0])
    doc_ids = ["doc_returns_p1", "doc_returns_p2"]

    # Save to Layer 2 cache
    stored = cache.set_response(
        query=query,
        query_embedding=query_emb,
        cacheable_answer="Our return window for electronics is 30 days.",
        user_role="customer",
        kb_version="v1",
        path="rag_llm",
        intent="return_policy",
        retrieved_doc_ids=doc_ids,
        tool_results=None,
    )
    assert stored is True

    # 1. Exact Hit (Tier 1)
    ans = cache.get_response_candidate(
        query=query,
        query_embedding=query_emb,
        user_role="customer",
        kb_version="v1",
        current_path="rag_llm",
        current_intent="return_policy",
        current_doc_ids=doc_ids,
    )
    assert ans == "Our return window for electronics is 30 days."

    # 2. Semantic Hit (Tier 2 Candidate Lookup - Near Duplicate Query)
    similar_query_emb = np.array([0.95, 0.1, 0.0])
    ans_semantic = cache.get_response_candidate(
        query="How long do I have to return an electronic item?",
        query_embedding=similar_query_emb,
        user_role="customer",
        kb_version="v1",
        current_path="rag_llm",
        current_intent="return_policy",
        current_doc_ids=doc_ids,
    )
    assert ans_semantic == "Our return window for electronics is 30 days."

    # 3. Fail Gate 1: Dissimilar query
    dissimilar_emb = np.array([0.0, 1.0, 0.0])
    ans_dissimilar = cache.get_response_candidate(
        query="What are shipping costs?",
        query_embedding=dissimilar_emb,
        user_role="customer",
        kb_version="v1",
        current_path="rag_llm",
        current_intent="return_policy",
        current_doc_ids=doc_ids,
    )
    assert ans_dissimilar is None

    # 4. Fail Gate 2: Planner path mismatch
    ans_path_mismatch = cache.get_response_candidate(
        query=query,
        query_embedding=query_emb,
        user_role="customer",
        kb_version="v1",
        current_path="clarify",
        current_intent="return_policy",
        current_doc_ids=doc_ids,
    )
    assert ans_path_mismatch is None

    # 5. Fail Gate 3: RBAC role mismatch
    ans_role_mismatch = cache.get_response_candidate(
        query=query,
        query_embedding=query_emb,
        user_role="employee",
        kb_version="v1",
        current_path="rag_llm",
        current_intent="return_policy",
        current_doc_ids=doc_ids,
    )
    assert ans_role_mismatch is None

    # 6. Fail Gate 4: KB version mismatch
    ans_kb_mismatch = cache.get_response_candidate(
        query=query,
        query_embedding=query_emb,
        user_role="customer",
        kb_version="v2",
        current_path="rag_llm",
        current_intent="return_policy",
        current_doc_ids=doc_ids,
    )
    assert ans_kb_mismatch is None

    # 7. Fail Gate 5: RBO evidence consistency fail (disjoint documents)
    disjoint_docs = ["doc_shipping_p1", "doc_shipping_p2"]
    ans_rbo_fail = cache.get_response_candidate(
        query=query,
        query_embedding=query_emb,
        user_role="customer",
        kb_version="v1",
        current_path="rag_llm",
        current_intent="return_policy",
        current_doc_ids=disjoint_docs,
    )
    assert ans_rbo_fail is None


def test_dynamic_data_gating_tool_results():
    cache = DualLayerCacheManager()

    # Query with stateful tool execution results (e.g. get_order_details)
    tool_results = {
        "get_order_details": {"order_id": "12345",
                              "status": "Shipped",
                              "days_bought": 20}
    }

    stored = cache.set_response(
        query="Where is my order #12345?",
        query_embedding=[1.0, 0.0],
        cacheable_answer="Orders are shipped via Express.",
        user_role="customer",
        kb_version="v1",
        path="rag_llm",
        intent="order_status",
        retrieved_doc_ids=["doc_shipping"],
        tool_results=tool_results,
    )

    # Must be strictly bypassed (not stored)
    assert stored is False


def test_response_generator_synthesis_schema():
    class DummyLLM:
        def with_structured_output(self, schema):
            class DummyStructured:
                def invoke(self, messages, config=None):
                    return SynthesisResponse(
                        cacheable_answer="Standard return window is 30 days.",
                        specific_answer="Your order #123 was placed 15 days ago."
                    )
            return DummyStructured()

        def invoke(self, messages, config=None):
            class DummyMessage:
                content = "Plain response string."
                additional_kwargs = {}
                response_metadata = {}
            return DummyMessage()

    generator = ResponseGenerator(
        synthesis_llm=DummyLLM(),
        server_exe=None,
        local_model_path=None
    )

    output, prompt = generator.generate(
        query="Can I return order #123?",
        retrieved_docs=[{"metadata": {"title": "Return Policy"},
                         "content": "30 days policy",
                         "similarity": 0.9}]
    )

    assert "Standard return window is 30 days." in output
    assert "Your order #123 was placed 15 days ago." in output
    assert generator.last_synthesis_response is not None
    resp = generator.last_synthesis_response
    assert resp.cacheable_answer == "Standard return window is 30 days."
    assert resp.specific_answer == "Your order #123 was placed 15 days ago."
