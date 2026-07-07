import json
import os


def load_evaluation_dataset(path: str = None) -> list[dict]:
    """
    Loads evaluation queries, ground truths, and context records.
    """
    if not path:
        path = os.path.join("data", "eval", "dataset.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []
