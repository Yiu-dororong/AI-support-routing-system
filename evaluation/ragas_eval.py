"""
Evaluation logic for RAG pipeline — Hit@K (offline) +
RAGAS LLM-judged metrics (optional).

RAGAS Framework
---------------
RAGAS (Retrieval Augmented Generation Assessment) uses an LLM judge to
score four semantic attributes that exact-match evaluation cannot capture:

  • Faithfulness        – are all response claims grounded in context?
  • Answer Relevance   – does the response directly address the question?
  • Context Recall     – did retrieval capture every statement needed by
                          the ground-truth answer?
  • Context Precision  – are the most relevant chunks ranked at the top
                          of the context block?

Because these metrics require a high-capability generative model (GPT-4o or
equivalent), the RAGAS runner is **disabled by default** and only activates
when a valid LLM API key is present. Set one of the environment variables:

  RAGAS_LLM_API_KEY           → generic LLM API key (OpenAI or Gemini)
  OPENAI_API_KEY=sk-...       → uses OpenAI (gpt-4o, recommended)
  GEMINI_API_KEY=AIzaSy...    → uses Google Gemini (via OpenAI compatibility)
  RAGAS_LLM_MODEL             → override the model name (default: gpt-4o)

If no key is found, :func:`run_ragas_evaluation` returns ``None`` and
prints a clear warning. The rest of the evaluation suite (Hit@K) is
unaffected and runs without any API key.
"""

from __future__ import annotations

import os
import sys

from ragas import DiskCacheBackend


# Disable Ragas background tracking/telemetry calls
# to avoid DNS resolution/network timeouts
os.environ["DO_NOT_TRACK"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"

import types
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from dotenv import load_dotenv


load_dotenv()  # Load .env file if present


def _patch_ragas_vertexai_compat() -> None:
    """Patch sys.modules and sys.path so ragas can be imported cleanly.

    Two issues are addressed:

    1. **vertexai stub** – ragas 0.4.x hard-imports
       ``langchain_community.chat_models.vertexai`` (for ChatVertexAI), but
       that sub-module was removed in langchain-community 0.3+. We pre-register
       a minimal stub so Python never raises ModuleNotFoundError.

    2. **sys.path shadowing** – When this file is executed directly (e.g.
       ``python evaluation/ragas_eval.py``), Python inserts the ``evaluation/``
       directory at ``sys.path[0]``.  This causes ``from datasets import Dataset``
       inside ragas to resolve to *our local* ``evaluation/datasets.py`` instead
       of the HuggingFace ``datasets`` package.  We remove the evaluation
       directory from the front of sys.path before ragas is ever imported.

    This function is idempotent and safe to call multiple times.
    """
    import pathlib

    # --- Fix 1: remove evaluation/ from sys.path if it was auto-inserted ---
    _eval_dir = str(pathlib.Path(__file__).parent.resolve())
    if sys.path and pathlib.Path(sys.path[0]).resolve() == pathlib.Path(_eval_dir):
        sys.path.pop(0)

    # --- Fix 2: stub out the removed vertexai chat-model module ---
    stub_name = "langchain_community.chat_models.vertexai"
    if stub_name not in sys.modules:
        stub = types.ModuleType(stub_name)
        stub.ChatVertexAI = type("ChatVertexAI", (object,), {})  # type: ignore[attr-defined]
        sys.modules[stub_name] = stub

        try:
            import langchain_community.chat_models as _cm

            if not hasattr(_cm, "vertexai"):
                _cm.vertexai = stub
        except ImportError:
            pass


_patch_ragas_vertexai_compat()

if TYPE_CHECKING:
    from router_logic import SupportRouter
else:
    SupportRouter = None


class SupportRouterLike(Protocol):
    """
    Protocol defining the interface required by build_ragas_samples_from_eval_items.
    """

    def run_retrieval_layer(
        self,
        query_input: str | Any,
        n_results: int = 2,
        threshold: float = 0.0,
        user_role: str = "customer",
    ) -> tuple[list[dict[str, Any]], str | None]: ...

    def run_response_generation(
        self,
        query: str,
        retrieved_docs: list[dict[str, Any]],
        callbacks=None,
        metadata: dict = None,
    ) -> tuple[str, str]: ...


# ---------------------------------------------------------------------------
# Hit@K  (no external dependencies — always available)
# ---------------------------------------------------------------------------


def calculate_hit_at_k(
    retrieved_docs: list[dict], ground_truth_id: str, k: int = 5
) -> float:
    """
    Evaluates whether the correct document (``ground_truth_id``) appears
    in the top-K retrieved documents.

    Parameters
    ----------
    retrieved_docs:
        Ordered list of retrieved document dicts, each containing an ``"id"`` key.
    ground_truth_id:
        The document ID (or ID prefix) that must appear in the top-K results.
    k:
        How many top results to inspect (default 5).

    Returns
    -------
    float
        ``1.0`` if the ground-truth document is in the top-K, ``0.0`` otherwise.
    """
    top_k_ids = [doc["id"] for doc in retrieved_docs[:k]]
    return 1.0 if ground_truth_id in top_k_ids else 0.0


# ---------------------------------------------------------------------------
# RAGAS availability check
# ---------------------------------------------------------------------------

_RAGAS_DISABLED_REASON: str | None = None
_LLM_MODEL: str = os.environ.get("RAGAS_LLM_MODEL", "gpt-4o")
_LLM_BASE_URL = os.environ.get("RAGAS_LLM_BASE_URL", "")
_BATCH_SIZE = int(os.environ.get("RAGAS_BATCH_SIZE", 4))
_BATCH_SLEEP = int(os.environ.get("RAGAS_BATCH_SLEEP", 0))


def _detect_api_key() -> str | None:
    """
    Return the active OpenAI or Gemini API key,
    checking RAGAS-specific keys first.
    """
    return (
        os.environ.get("RAGAS_LLM_API_KEY")
        or os.environ.get("RAGAS_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("RAGAS_GEMINI_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )


def _check_ragas_available() -> bool:
    """
    Verify that:
      1. An LLM API key is set in the environment.
      2. The ``ragas`` and ``langchain_openai`` packages are importable.

    Sets ``_RAGAS_DISABLED_REASON`` and returns False if either check fails.
    """
    global _RAGAS_DISABLED_REASON

    if not _detect_api_key():
        _RAGAS_DISABLED_REASON = (
            "No LLM API key found. Set RAGAS_LLM_API_KEY, RAGAS_OPENAI_API_KEY, "
            "OPENAI_API_KEY, RAGAS_GEMINI_API_KEY, or GEMINI_API_KEY to "
            "enable RAGAS evaluation. A high-capability model (e.g. gpt-4o "
            "or gemini-1.5-pro) is required for accurate LLM-judged metrics."
        )
        return False

    _RAGAS_DISABLED_REASON = None
    return True


def is_ragas_available() -> bool:
    """
    Dynamically verify if RAGAS evaluation is available based on current
    environment variables and package imports.
    """
    return _check_ragas_available()


def __getattr__(name: str) -> Any:
    """
    Support accessing dynamic RAGAS_AVAILABLE module property
    for backwards compatibility.
    """
    if name == "RAGAS_AVAILABLE":
        return is_ragas_available()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class RAGASSample:
    """
    One evaluation record consumed by the RAGAS runner.

    Attributes
    ----------
    question:
        The original user query.
    answer:
        The synthesized LLM response to evaluate.
    contexts:
        List of retrieved document texts that were passed to the LLM.
    ground_truth:
        The reference / expected answer used for Context Recall scoring.
        Optional — Context Recall is skipped when not provided.
    """

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""


@dataclass
class RAGASResult:
    """
    Aggregated RAGAS metric scores over an evaluation dataset.

    All scores are in the range [0.0, 1.0] where higher is better.
    A value of ``None`` means the metric could not be computed (e.g. missing
    ground truths for Context Recall).
    """

    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_recall: float | None = None
    context_precision: float | None = None
    num_samples: int = 0
    scores: list[dict[str, Any]] | None = None
    skipped_reason: str | None = None

    def is_valid(self) -> bool:
        """True if at least one metric was successfully computed."""
        return any(
            v is not None
            for v in (
                self.faithfulness,
                self.answer_relevance,
                self.context_recall,
                self.context_precision,
            )
        )

    def summary(self) -> str:
        """Human-readable one-line summary suitable for console output."""
        if self.skipped_reason:
            return f"[RAGAS SKIPPED] {self.skipped_reason}"

        def fmt(v: float | None) -> str:
            return f"{v:.3f}" if v is not None else "n/a"

        return (
            f"Faithfulness={fmt(self.faithfulness)}  "
            f"AnswerRelevance={fmt(self.answer_relevance)}  "
            f"ContextRecall={fmt(self.context_recall)}  "
            f"ContextPrecision={fmt(self.context_precision)}  "
            f"(n={self.num_samples})"
        )


# ---------------------------------------------------------------------------
# RAGAS runner
# ---------------------------------------------------------------------------


def run_ragas_evaluation(
    samples: list[RAGASSample],
    *,
    model: str | None = None,
    batch_size: int = 4,
    raise_on_disabled: bool = False,
) -> RAGASResult:
    """
    Run LLM-judged RAGAS evaluation over a list of :class:`RAGASSample` records.

    The function is a no-op when RAGAS is disabled (no API key or missing
    packages). In that case it returns a :class:`RAGASResult` whose
    ``skipped_reason`` explains why.

    Parameters
    ----------
    samples:
        List of evaluation records (question, answer, contexts,
        optional ground_truth).
    model:
        OpenAI or Gemini model to use as LLM judge. Defaults to the ``RAGAS_LLM_MODEL``
        environment variable or ``"gpt-4o"``. Using a weaker model
        (e.g. gpt-3.5-turbo) will produce unreliable metric scores.
    batch_size:
        Number of samples to evaluate per API call batch. Lower values reduce
        the risk of hitting rate limits.
    raise_on_disabled:
        If True, raise ``RuntimeError`` when RAGAS is not available instead of
        returning a skipped result.

    Returns
    -------
    RAGASResult
        Aggregated metric scores, or a skipped result with an explanation.

    Raises
    ------
    RuntimeError
        Only when ``raise_on_disabled=True`` and RAGAS cannot run.
    """
    if not is_ragas_available():
        msg = _RAGAS_DISABLED_REASON or "RAGAS evaluation is not available."
        warnings.warn(f"[ragas_eval] {msg}", stacklevel=2)
        if raise_on_disabled:
            raise RuntimeError(msg)
        return RAGASResult(skipped_reason=msg)

    if not samples:
        return RAGASResult(skipped_reason="Empty sample list — nothing to evaluate.")

    llm_model = model or _LLM_MODEL
    api_key = _detect_api_key()

    # --- lazy imports (only reached when RAGAS is available) ---
    from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings
    from openai import AsyncOpenAI
    from ragas import evaluate
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from ragas.llms import llm_factory
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_precision import ContextPrecision
    from ragas.metrics._context_recall import ContextRecall
    from ragas.metrics._faithfulness import Faithfulness

    actual_batch_size = batch_size or _BATCH_SIZE
    _has_ground_truth = any(s.ground_truth.strip() for s in samples)

    # Detect provider: Gemini keys start with "AIzaSy"; model name is also a signal
    is_gemini = bool(
        (api_key and api_key.startswith("AIzaSy"))
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("RAGAS_GEMINI_API_KEY")
        or (llm_model and "gemini" in llm_model.lower())
    )

    # Determine base URL — Gemini uses an OpenAI-compatible REST endpoint
    actual_base_url = _LLM_BASE_URL
    if not actual_base_url and is_gemini:
        actual_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

    # LLM: llm_factory (modern ragas API, no deprecation warning)
    # Enforce strict 10s request timeouts and low max_retries
    # to prevent long hangs on the API proxy
    llm_client_kwargs: dict = {
        "api_key": api_key,
        "timeout": 60.0,
        "max_retries": 1,
    }
    if actual_base_url:
        llm_client_kwargs["base_url"] = actual_base_url

    cache = DiskCacheBackend()
    ragas_llm = llm_factory(
        model=llm_model,
        provider="openai",
        client=AsyncOpenAI(**llm_client_kwargs),
        max_tokens=4096,
        cache=cache,
    )

    # Embeddings: LangchainEmbeddingsWrapper is needed because old-style
    # AnswerRelevancy calls .embed_query() which only LangChain wrappers expose.
    # Suppress the wrapper's own DeprecationWarning — it's an internal ragas detail.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=".*LangchainEmbeddingsWrapper.*",
        )
        if is_gemini:
            from langchain_google_genai import (
                GoogleGenerativeAIEmbeddings,
            )

            lc_embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-2",
                task_type="retrieval_query",
                google_api_key=api_key,
            )
        else:
            embed_model = "text-embedding-3-small"
            embed_kwargs: dict = {"api_key": api_key}
            if actual_base_url:
                embed_kwargs["base_url"] = actual_base_url
            lc_embeddings = LCOpenAIEmbeddings(model=embed_model, **embed_kwargs)

        ragas_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)

    faithfulness = Faithfulness()
    answer_relevancy = AnswerRelevancy(strictness=1)
    context_precision = ContextPrecision()
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings
    context_precision.llm = ragas_llm

    _metrics: list = [faithfulness, answer_relevancy, context_precision]
    if _has_ground_truth:
        context_recall = ContextRecall()
        context_recall.llm = ragas_llm
        _metrics.append(context_recall)

    import time

    import pandas as pd
    from datasets import Dataset as HFDataset

    dfs = []
    total_samples = len(samples)

    for i in range(0, total_samples, actual_batch_size):
        chunk = samples[i : i + actual_batch_size]

        # Build HuggingFace Dataset with the column names ragas stable metrics expect
        data: dict = {
            "question": [s.question for s in chunk],
            "answer": [s.answer for s in chunk],
            "contexts": [s.contexts for s in chunk],
        }
        if _has_ground_truth:
            data["ground_truth"] = [s.ground_truth for s in chunk]
        hf_dataset = HFDataset.from_dict(data)

        batch_num = (i // actual_batch_size) + 1
        total_batches = (total_samples + actual_batch_size - 1) // actual_batch_size
        print(
            f"[ragas_eval] Running RAGAS batch {batch_num}/{total_batches} "
            f"({len(chunk)} samples) using judge model '{llm_model}' ..."
        )

        try:
            from ragas.run_config import RunConfig

            # Limit parallel workers to 4 to speed up evaluations
            # while staying under Gemini's 15 RPM limit
            run_config = RunConfig(max_workers=4, timeout=60, max_retries=3)
            result = evaluate(
                hf_dataset,
                metrics=_metrics,
                batch_size=len(chunk),
                run_config=run_config,
            )
            dfs.append(result.to_pandas())
        except Exception as exc:
            err = f"RAGAS evaluation failed on batch {batch_num}: {exc}"
            warnings.warn(f"[ragas_eval] {err}", stacklevel=2)
            return RAGASResult(skipped_reason=err, num_samples=len(samples))

        # Sleep between batches if not the last batch
        if i + actual_batch_size < total_samples and _BATCH_SLEEP > 0:
            print(f"[ragas_eval] Sleeping {_BATCH_SLEEP}s to respect rate limits...")
            time.sleep(_BATCH_SLEEP)

    if not dfs:
        return RAGASResult(
            skipped_reason="No results produced.", num_samples=len(samples)
        )

    df = pd.concat(dfs, ignore_index=True)

    def _mean(col: str) -> float | None:
        if col not in df.columns:
            return None
        vals = df[col].dropna()
        return float(vals.mean()) if not vals.empty else None

    # Convert individual sample scores to standard Python types
    scores = []
    if not df.empty:
        import numpy as np

        for record in df.to_dict(orient="records"):
            clean_record = {}
            for k, v in record.items():
                if isinstance(v, (list | np.ndarray)):
                    # Convert to list and clean elements
                    lst = v.tolist() if isinstance(v, np.ndarray) else v
                    clean_record[k] = [
                        (
                            None
                            if (
                                item is None
                                or (
                                    isinstance(item, (float | np.floating))
                                    and np.isnan(item)
                                )
                            )
                            else float(item)
                            if isinstance(item, (np.floating | float))
                            else int(item)
                            if isinstance(item, (np.integer | int))
                            else item
                        )
                        for item in lst
                    ]
                elif pd.isna(v):
                    clean_record[k] = None
                elif isinstance(v, (np.floating | float)):
                    clean_record[k] = float(v)
                elif isinstance(v, (np.integer | int)):
                    clean_record[k] = int(v)
                else:
                    clean_record[k] = v
            scores.append(clean_record)

    return RAGASResult(
        faithfulness=_mean("faithfulness"),
        answer_relevance=_mean("answer_relevancy"),
        context_recall=_mean("context_recall") if _has_ground_truth else None,
        context_precision=_mean("context_precision"),
        num_samples=len(samples),
        scores=scores if scores else None,
    )


# ---------------------------------------------------------------------------
# Dataset helper: convert run_eval.py dicts → RAGASSample
# ---------------------------------------------------------------------------


def build_ragas_samples_from_eval_items(
    eval_items: list[dict],
    *,
    router: SupportRouterLike,
    paths_to_include: tuple[str, ...] = ("rag", "rag_llm"),
) -> list[RAGASSample]:
    """
    Helper that drives the router over each ``eval_item`` and constructs a
    :class:`RAGASSample` from the live pipeline output.

    Only items whose ``expected_path`` is in ``paths_to_include`` are processed,
    since RAGAS metrics are only meaningful when a retrieval + synthesis step
    actually occurred.

    Parameters
    ----------
    eval_items:
        Raw dicts from :func:`evaluation.datasets.load_evaluation_dataset`.
    router:
        An object implementing the SupportRouterLike protocol.
    paths_to_include:
        Expected routing paths to include; defaults to RAG paths only.

    Returns
    -------
    list[RAGASSample]
        Samples ready for :func:`run_ragas_evaluation`.
    """
    samples: list[RAGASSample] = []

    for item in eval_items:
        if item.get("expected_path") not in paths_to_include:
            continue

        query: str = item["query"]
        role: str = item.get("user_role", "customer")
        ground_truth: str = item.get("ground_truth_answer", "")

        # Skip if ground truth is empty (e.g. RBAC checks, refusal/refutation tests)
        if not ground_truth.strip():
            continue

        # Retrieve context
        retrieved_docs, _ = router.run_retrieval_layer(
            query, n_results=5, user_role=role
        )
        contexts = [doc.get("text", doc.get("content", "")) for doc in retrieved_docs]

        # Skip if no context was retrieved (e.g. RBAC authorization blocked retrieval)
        if not contexts:
            continue

        # Synthesise answer using the router's LLM layer
        try:
            answer, _ = router.run_response_generation(query, retrieved_docs)
        except Exception as exc:
            # Synthesis/Response generation failed (e.g. local LLM not running)
            # — skip sample
            warnings.warn(
                f"[ragas_eval] Response generation failed for query '{query}': {exc}",
                stacklevel=2,
            )
            continue

        samples.append(
            RAGASSample(
                question=query,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
            )
        )

    return samples


# ---------------------------------------------------------------------------
# Quick self-test / smoke-test (python -m evaluation.ragas_eval)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("RAGAS Evaluation Module - Status Check")
    print("=" * 60)

    if is_ragas_available():
        print(f"[OK] RAGAS is ENABLED (judge model: {_LLM_MODEL})")
    else:
        print(f"[FAIL] RAGAS is DISABLED: {_RAGAS_DISABLED_REASON}")

    print()
    print("Hit@K (always available): testing with dummy data...")
    _docs = [{"id": "doc_1_page1"}, {"id": "doc_2_page3"}, {"id": "doc_3_page1"}]
    _score = calculate_hit_at_k(_docs, "doc_2_page3", k=5)
    assert _score == 1.0, "Hit@K self-test failed"
    print(f"  calculate_hit_at_k() -> {_score} [OK]")

    if is_ragas_available():
        print()
        print("Running mini RAGAS smoke-test (2 samples, batch_size=1)...")
        _samples = [
            RAGASSample(
                question="What is the return window?",
                answer="The return window is 30 days from the purchase date.",
                contexts=["Returns must be made within 30 days of purchase."],
                ground_truth="30 days from purchase date.",
            ),
            RAGASSample(
                question="How do I track my order?",
                answer=(
                    "You can track your order using the tracking link "
                    "sent in the confirmation email."
                ),
                contexts=[
                    "An email containing a tracking link is sent once the order ships."
                ],
                ground_truth=(
                    "Via the tracking link in the shipping confirmation email."
                ),
            ),
        ]
        _result = run_ragas_evaluation(_samples, batch_size=4)
        print(_result.summary())
        print("\nIndividual scores:")
        import json

        print(json.dumps(_result.scores, indent=2))
    else:
        print()
        print("Skipping RAGAS smoke-test (not available).")

    print("=" * 60)
