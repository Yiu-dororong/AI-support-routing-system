from rag.fusion import get_query_weights, reciprocal_rank_fusion


def test_rrf_weights():
    # Adaptive weights test
    bm25_w, dense_w = get_query_weights("Where is document doc_23?")
    assert bm25_w == 1.8
    assert dense_w == 0.8

    bm25_w_sem, dense_w_sem = get_query_weights("How do I return my device?")
    assert bm25_w_sem == 1.2
    assert dense_w_sem == 1.0


def test_reciprocal_rank_fusion():
    dense_results = [
        {
            "id": "doc_A",
            "content": "Text A",
            "metadata": {"title": "Title A"},
            "similarity": 0.9,
        },
        {
            "id": "doc_B",
            "content": "Text B",
            "metadata": {"title": "Title B"},
            "similarity": 0.8,
        },
    ]
    sparse_results = [
        {
            "id": "doc_B",
            "content": "Text B",
            "metadata": {"title": "Title B"},
            "similarity": 10.0,
        },
        {
            "id": "doc_C",
            "content": "Text C",
            "metadata": {"title": "Title C"},
            "similarity": 5.0,
        },
    ]

    fused = reciprocal_rank_fusion(dense_results, sparse_results, "query")
    assert len(fused) == 3
    # doc_B should rank highest because it is present in both lists
    assert fused[0]["id"] == "doc_B"
