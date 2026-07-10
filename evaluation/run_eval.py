import os
import sys

from evaluation.datasets import load_evaluation_dataset
from evaluation.failure_analysis import classify_failure
from router_logic import SupportRouter


# Disable Ragas background tracking/telemetry calls
# to avoid DNS resolution/network timeouts
os.environ["DO_NOT_TRACK"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"


# Prevent the evaluation directory from shadowing third-party packages
# (like HuggingFace datasets)
_script_dir = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == os.path.abspath(_script_dir):
    sys.path.pop(0)

# Ensure project root is at the front of sys.path
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)




def main():
    print(
        "Initializing SupportRouter (starting local llama-server if needed)...",
        flush=True,
    )
    router = SupportRouter()

    print("Loading evaluation dataset...", flush=True)
    dataset = load_evaluation_dataset()
    if not dataset:
        print("Error: Dataset is empty or not found.", flush=True)
        return

    print(f"Running evaluation on {len(dataset)} queries...\n", flush=True)

    total = len(dataset)
    correct_path_count = 0
    total_hit_rate = 0.0
    total_hit_rate_at_1 = 0.0
    retrieval_queries_count = 0
    failures = []

    # Process each query in the dataset
    for item in dataset:
        qid = item["id"]
        category = item["category"]
        query = item["query"]
        role = item["user_role"]
        expected_path = item["expected_path"]
        ground_truth = item["ground_truth_docs"]

        # 1. Get embedding
        query_emb = router.get_query_embedding(query)

        # 2. Check scope
        in_scope, intent, similarities = router.run_scope_filter(query_emb)

        # 3. Check FAQ
        faq_match, _ = router.run_faq_layer(query_emb)

        # 4. Determine path
        actual_path = "rag"
        if not in_scope:
            actual_path = "refuse"
        elif faq_match:
            actual_path = "faq_bypass"
        else:
            # Planner
            decision, _ = router.run_execution_planner(query, intent)
            actual_path = decision.path

        # Compare paths
        is_path_correct = actual_path == expected_path
        if is_path_correct:
            correct_path_count += 1

        # 5. Retrieval layer check using the role
        retrieved_docs, _ = router.run_retrieval_layer(
            query, n_results=5, user_role=role
        )

        # Calculate Hit@5 and Hit@1
        hit_score = 0.0
        hit_score_at_1 = 0.0
        is_retrieval_query = expected_path in ["rag", "rag_llm"] and bool(ground_truth)
        if is_retrieval_query:
            retrieval_queries_count += 1

            # Helper to check if doc matches ground truth
            def doc_matches_gt(doc_id, gt_docs):
                doc_base = doc_id.split("_page")[0]
                for gt in gt_docs:
                    gt_base = gt.split("_page")[0]
                    if doc_base.startswith(gt_base) or gt_base.startswith(doc_base):
                        return True
                return False

            # Hit@1 Check
            if retrieved_docs and doc_matches_gt(retrieved_docs[0]["id"], ground_truth):
                hit_score_at_1 = 1.0
            total_hit_rate_at_1 += hit_score_at_1

            # Hit@5 Check
            matched = False
            for doc in retrieved_docs[:5]:
                if doc_matches_gt(doc["id"], ground_truth):
                    matched = True
                    break
            hit_score = 1.0 if matched else 0.0
            total_hit_rate += hit_score
        else:
            # RAG bypassed, refused, or expecting empty context (e.g. RBAC checks)
            hit_score = 1.0

        # Log failures
        if not is_path_correct or (ground_truth and hit_score == 0.0):
            failure_type = classify_failure(
                query, retrieved_docs, "", ground_truth[0] if ground_truth else ""
            )
            failures.append(
                {
                    "id": qid,
                    "query": query,
                    "category": category,
                    "expected_path": expected_path,
                    "actual_path": actual_path,
                    "hit_score": hit_score,
                    "failure_type": failure_type,
                }
            )

    # Print summary
    path_accuracy = (correct_path_count / total) * 100
    avg_hit_rate = 0.0
    avg_hit_rate_at_1 = 0.0
    if retrieval_queries_count > 0:
        avg_hit_rate = (total_hit_rate / retrieval_queries_count) * 100
        avg_hit_rate_at_1 = (total_hit_rate_at_1 / retrieval_queries_count) * 100

    print("=" * 50, flush=True)
    print(" EVALUATION RESULTS", flush=True)
    print("=" * 50, flush=True)
    print(f"Total Queries evaluated: {total}", flush=True)
    print(
        f"Routing Path Accuracy:  {path_accuracy:.1f}% ({correct_path_count}/{total})",
        flush=True,
    )
    hit_rate_at_1_str = (
        f"Retrieval Hit@1 Rate (RAG only): {avg_hit_rate_at_1:.1f}% "
        f"({int(total_hit_rate_at_1)}/{retrieval_queries_count})"
    )
    hit_rate_str = (
        f"Retrieval Hit@5 Rate (RAG only): {avg_hit_rate:.1f}% "
        f"({int(total_hit_rate)}/{retrieval_queries_count})"
    )
    print(hit_rate_at_1_str, flush=True)
    print(hit_rate_str, flush=True)
    print("=" * 50, flush=True)

    if failures:
        print("\nTop Routing & Retrieval Failures:", flush=True)
        for f in failures[:5]:
            print(f"- [{f['id']}] ({f['category']}) Query: '{f['query']}'", flush=True)
            print(
                f"  Expected path: {f['expected_path']}, Got: {f['actual_path']}",
                flush=True,
            )
            print(
                f"  Hit Score: {f['hit_score']}, Failure Type: {f['failure_type']}",
                flush=True,
            )
    else:
        print("\nAll queries routed and retrieved perfectly!", flush=True)

    # Construct complete evaluation report dict
    report = {
        "routing_retrieval": {
            "total_queries": total,
            "routing_accuracy_pct": round(path_accuracy, 2),
            "routing_correct_count": correct_path_count,
            "routing_errors_count": total - correct_path_count,
            "retrieval_hit_at_1_pct": round(avg_hit_rate_at_1, 2),
            "retrieval_hit_at_1_correct_count": int(total_hit_rate_at_1),
            "retrieval_hit_rate_pct": round(avg_hit_rate, 2),
            "retrieval_correct_count": int(total_hit_rate),
            "retrieval_total_count": retrieval_queries_count,
            "failures_count": len(failures),
            "failures": failures,
        },
        "ragas": None
    }

    # 6. RAGAS Semantic Evaluation (optional, if key and packages are present)
    from evaluation.ragas_eval import (
        build_ragas_samples_from_eval_items,
        is_ragas_available,
        run_ragas_evaluation,
    )

    if is_ragas_available():
        print("\n" + "=" * 50, flush=True)
        print(" RUNNING RAGAS SEMANTIC EVALUATION", flush=True)
        print("=" * 50, flush=True)
        ragas_samples = build_ragas_samples_from_eval_items(dataset, router=router)
        if ragas_samples:
            # By default it will use model & batch size from environment vars
            # (RAGAS_LLM_MODEL, RAGAS_BATCH_SIZE)
            ragas_result = run_ragas_evaluation(ragas_samples)
            import dataclasses  # noqa: PLC0415
            import json  # noqa: PLC0415
            ragas_dict = dataclasses.asdict(ragas_result)
            report["ragas"] = ragas_dict
            print("\nRAGAS Evaluation Results (JSON):", flush=True)
            print(json.dumps(ragas_dict, indent=2), flush=True)
        else:
            print("No RAG/RAG_LLM queries to evaluate.", flush=True)
        print("=" * 50, flush=True)

    # Save complete report to a JSON file
    import json  # noqa: PLC0415
    from datetime import datetime  # noqa: PLC0415
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("data", "eval")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"ragas_results_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved complete evaluation report to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
