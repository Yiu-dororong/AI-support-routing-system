import os
import sys


# Ensure root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.datasets import load_evaluation_dataset
from evaluation.failure_analysis import classify_failure
from router_logic import SupportRouter


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

        # Calculate Hit@5
        hit_score = 0.0
        if expected_path not in ["rag", "rag_llm"]:
            # RAG bypassed or refused -> no documents expected
            hit_score = 1.0
            total_hit_rate += hit_score
        elif ground_truth:
            matched = False
            for gt in ground_truth:
                gt_base = gt.split("_page")[0]
                for doc in retrieved_docs:
                    doc_base = doc["id"].split("_page")[0]
                    if doc_base.startswith(gt_base) or gt_base.startswith(
                        doc_base
                    ):
                        matched = True
                        break
                if matched:
                    break
            hit_score = 1.0 if matched else 0.0
            total_hit_rate += hit_score
        else:
            # If no ground truth docs are expected, retrieval success is
            # 1.0 (empty expected)
            hit_score = 1.0
            total_hit_rate += hit_score

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
    avg_hit_rate = (total_hit_rate / total) * 100

    print("=" * 50, flush=True)
    print(" EVALUATION RESULTS", flush=True)
    print("=" * 50, flush=True)
    print(f"Total Queries evaluated: {total}", flush=True)
    print(
        f"Routing Path Accuracy:  {path_accuracy:.1f}% ({correct_path_count}/{total})",
        flush=True,
    )
    print(f"Retrieval Hit@5 Rate:    {avg_hit_rate:.1f}%", flush=True)
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


if __name__ == "__main__":
    main()
