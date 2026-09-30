import hashlib
import time
import unicodedata

import numpy as np
from pydantic import BaseModel, Field

from core.types import RoutingDecision


def normalize_query(query: str) -> str:
    """
    Normalizes query string:
    - Lowercase
    - Unicode NFKD normalization
    - Collapse multiple whitespaces
    - Strip leading/trailing whitespace and punctuation (?, ., !, ,, ;, :)
    """
    if not query:
        return ""

    # NFKD normalization
    normalized = unicodedata.normalize("NFKD", query)
    normalized = normalized.lower()

    # Strip whitespace & trailing punctuation
    normalized = normalized.strip()
    punctuation_to_strip = "?.!:,;"
    while normalized and normalized[-1] in punctuation_to_strip:
        normalized = normalized[:-1].strip()

    # Collapse multiple spaces
    normalized = " ".join(normalized.split())
    return normalized


def hash_query(norm_query: str) -> str:
    """Computes SHA-256 hash of a normalized query string."""
    return hashlib.sha256(norm_query.encode("utf-8")).hexdigest()


def hash_prompt(prompt: str) -> str:
    """Computes short 8-char SHA-256 hash of prompt content for versioning."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]


def calculate_rbo(list_a: list[str], list_b: list[str], p: float = 0.8) -> float:
    """
    Calculates Rank-Biased Overlap (RBO) between two ordered document ID lists.
    Uses Webber et al. (2010) finite depth formula with residue:
    RBO(A, B, p, k) = (1 - p) * sum_{d=1}^k (p^(d-1) * A_d) + p^k * A_k
    Weight parameter p controls top-rank sensitivity (default 0.8).
    Returns float between 0.0 and 1.0.
    """
    if not list_a and not list_b:
        return 1.0
    if not list_a or not list_b:
        return 0.0

    k = max(len(list_a), len(list_b))
    rbo_sum = 0.0
    weight = 1.0 - p

    set_a = set()
    set_b = set()
    final_agreement = 0.0

    for d in range(1, k + 1):
        if d <= len(list_a):
            set_a.add(list_a[d - 1])
        if d <= len(list_b):
            set_b.add(list_b[d - 1])

        overlap = len(set_a.intersection(set_b))
        agreement = overlap / float(d)
        final_agreement = agreement
        rbo_sum += (p ** (d - 1)) * agreement

    return weight * rbo_sum + (p ** k) * final_agreement



class PlannerCacheEntry(BaseModel):
    query: str = ""
    query_hash: str = ""
    query_embedding: list[float] = Field(default_factory=list)
    decision: RoutingDecision
    intent: str
    user_role: str
    prompt_hash: str
    created_at: float = Field(default_factory=time.time)
    ttl_seconds: float = 86400.0

    def is_expired(self, current_time: float | None = None) -> bool:
        now = current_time or time.time()
        return (now - self.created_at) > self.ttl_seconds


class ResponseCacheEntry(BaseModel):
    query: str
    query_hash: str
    query_embedding: list[float]
    cacheable_answer: str
    user_role: str
    kb_version: str
    path: str
    intent: str
    retrieved_doc_ids: list[str]
    created_at: float = Field(default_factory=time.time)
    ttl_seconds: float = 86400.0

    def is_expired(self, current_time: float | None = None) -> bool:
        now = current_time or time.time()
        return (now - self.created_at) > self.ttl_seconds



class DualLayerCacheManager:
    """
    Production 2-Layer Cache Manager:
    - Layer 1: Planner / Classifier Cache (bypass execution planner LLM)
               Supports 2-Tier Lookup: Tier 1 Exact Hash & Tier 2 Vector Semantic Match.
    - Layer 2: Response Cache (bypass synthesis LLM grounded by RBO evidence)
    """

    def __init__(
        self,
        ttl_planner_seconds: float = 86400.0,
        theta_planner: float = 0.88,
        theta_read: float = 0.88,
        theta_rbo: float = 0.70,
    ):
        self.ttl_planner_seconds = ttl_planner_seconds
        self.theta_planner = theta_planner
        self.theta_read = theta_read
        self.theta_rbo = theta_rbo

        # Store keys: cache:planner:{user_role}:p{prompt_hash}:{query_hash}
        self._planner_cache: dict[str, PlannerCacheEntry] = {}

        # Store keys: cache:response:{user_role}:v{kb_ver}:{query_hash}
        self._response_cache: dict[str, ResponseCacheEntry] = {}
        self._response_entries: list[ResponseCacheEntry] = []

    # --- Layer 1: Planner Cache ---

    def get_planner(
        self,
        query: str,
        user_role: str,
        prompt_hash: str,
        query_embedding: np.ndarray | list[float] | None = None,
        theta_planner: float | None = None,
    ) -> tuple[RoutingDecision | None, str | None]:
        """
        Retrieves cached RoutingDecision via 2-Tier Lookup:
        - Tier 1: Exact Query Hash match (O(1))
        - Tier 2: Cosine Similarity >= theta_planner against candidate embeddings
        Returns (decision, hit_note) or (None, None).
        """
        norm_q = normalize_query(query)
        q_hash = hash_query(norm_q)
        cache_key = f"cache:planner:{user_role}:p{prompt_hash}:{q_hash}"

        # Tier 1 Exact Hash Check
        entry = self._planner_cache.get(cache_key)
        if entry:
            if entry.is_expired():
                del self._planner_cache[cache_key]
            else:
                return (
                    entry.decision,
                    f"""[Planner Cache Exact Hit]
                    Reused decision for path '{entry.decision.path}'.""",
                )

        # Tier 2 Vector Semantic Match Check
        if query_embedding is not None and len(self._planner_cache) > 0:
            thresh = theta_planner if theta_planner is not None else self.theta_planner
            query_emb_vec = np.array(query_embedding, dtype=np.float32)
            norm_query_emb = np.linalg.norm(query_emb_vec)
            if norm_query_emb > 0:
                query_emb_vec = query_emb_vec / norm_query_emb

            best_decision: RoutingDecision | None = None
            best_sim = -1.0

            for key, cand in list(self._planner_cache.items()):
                if cand.is_expired():
                    del self._planner_cache[key]
                    continue

                if cand.user_role != user_role or cand.prompt_hash != prompt_hash:
                    continue

                if not cand.query_embedding:
                    continue

                cand_emb_vec = np.array(cand.query_embedding, dtype=np.float32)
                norm_cand_emb = np.linalg.norm(cand_emb_vec)
                if norm_cand_emb > 0:
                    cand_emb_vec = cand_emb_vec / norm_cand_emb

                sim = float(np.dot(query_emb_vec, cand_emb_vec))
                if sim >= thresh and sim > best_sim:
                    best_sim = sim
                    best_decision = cand.decision

            if best_decision is not None:
                return (
                    best_decision,
                    (
                        f"[Planner Cache Semantic Hit (Similarity: {best_sim:.2f})] "
                        f"Reused decision for path '{best_decision.path}'."
                    ),
                )

        return None, None

    def set_planner(
        self,
        query: str,
        user_role: str,
        prompt_hash: str,
        decision: RoutingDecision,
        intent: str,
        query_embedding: np.ndarray | list[float] | None = None,
    ) -> bool:
        """
        Stores RoutingDecision in Layer 1 cache.
        Write-Side Gate: Rejects fallback error decisions.
        """
        # Reject error fallbacks
        reason_lower = decision.reason.lower()
        if (
            "local execution planning failed" in reason_lower
            or "error details" in reason_lower
        ):
            return False

        norm_q = normalize_query(query)
        q_hash = hash_query(norm_q)
        cache_key = f"cache:planner:{user_role}:p{prompt_hash}:{q_hash}"

        emb_list = []
        if query_embedding is not None:
            emb_list = (
                query_embedding.tolist()
                if isinstance(query_embedding, np.ndarray)
                else list(query_embedding)
            )

        entry = PlannerCacheEntry(
            query=query,
            query_hash=q_hash,
            query_embedding=emb_list,
            decision=decision,
            intent=intent,
            user_role=user_role,
            prompt_hash=prompt_hash,
            ttl_seconds=self.ttl_planner_seconds,
        )
        self._planner_cache[cache_key] = entry
        return True

    # --- Layer 2: Response Cache ---

    def get_response_candidate(
        self,
        query: str,
        query_embedding: np.ndarray | list[float],
        user_role: str,
        kb_version: str,
        current_path: str,
        current_intent: str,
        current_doc_ids: list[str],
        theta_read: float | None = None,
        theta_rbo: float | None = None,
    ) -> str | None:
        """
        Discovers candidate from Layer 2 Response Cache evaluating 5-Point Reuse Gate:
        1. Cosine similarity >= theta_read
        2. Planner compatibility (same path & intent)
        3. RBAC match (same user_role)
        4. Knowledge version match (same kb_version)
        5. Retrieval evidence consistency (RBO on ranked docs >= theta_rbo)
        """
        read_thresh = theta_read if theta_read is not None else self.theta_read
        rbo_thresh = theta_rbo if theta_rbo is not None else self.theta_rbo

        norm_q = normalize_query(query)
        q_hash = hash_query(norm_q)
        exact_key = f"cache:response:{user_role}:v{kb_version}:{q_hash}"

        query_emb_vec = np.array(query_embedding, dtype=np.float32)
        norm_query_emb = np.linalg.norm(query_emb_vec)
        if norm_query_emb > 0:
            query_emb_vec = query_emb_vec / norm_query_emb

        best_answer: str | None = None
        best_score = -1.0

        # Tier 1 Exact Hash Check
        tier1_rejected_hash: str | None = None
        exact_entry = self._response_cache.get(exact_key)
        if exact_entry:
            if exact_entry.is_expired():
                del self._response_cache[exact_key]
            elif (
                exact_entry.path == current_path
                and exact_entry.intent == current_intent
            ):
                rbo = calculate_rbo(exact_entry.retrieved_doc_ids, current_doc_ids)
                if rbo >= rbo_thresh:
                    return exact_entry.cacheable_answer
                tier1_rejected_hash = exact_entry.query_hash

        # Tier 2 Vector Candidate Discovery
        for entry in self._response_entries:
            if entry.is_expired():
                continue

            if tier1_rejected_hash and entry.query_hash == tier1_rejected_hash:
                continue

            if entry.user_role != user_role or entry.kb_version != kb_version:
                continue

            if entry.path != current_path or entry.intent != current_intent:
                continue

            cand_emb_vec = np.array(entry.query_embedding, dtype=np.float32)
            norm_cand_emb = np.linalg.norm(cand_emb_vec)
            if norm_cand_emb > 0:
                cand_emb_vec = cand_emb_vec / norm_cand_emb

            sim = float(np.dot(query_emb_vec, cand_emb_vec))
            if sim < read_thresh:
                continue

            rbo = calculate_rbo(entry.retrieved_doc_ids, current_doc_ids)
            if rbo < rbo_thresh:
                continue

            if sim > best_score:
                best_score = sim
                best_answer = entry.cacheable_answer

        return best_answer

    def set_response(
        self,
        query: str,
        query_embedding: np.ndarray | list[float],
        cacheable_answer: str,
        user_role: str,
        kb_version: str,
        path: str,
        intent: str,
        retrieved_doc_ids: list[str],
        tool_results: dict | None = None,
    ) -> bool:
        """
        Stores cacheable_answer in Layer 2 Response Cache.
        Dynamic Data Gate: Bypasses caching if tool_results is non-empty.
        """
        # Dynamic Data Gate: bypass caching if stateful tools were executed
        if tool_results and len(tool_results) > 0:
            return False

        if not cacheable_answer or not cacheable_answer.strip():
            return False

        norm_q = normalize_query(query)
        q_hash = hash_query(norm_q)
        cache_key = f"cache:response:{user_role}:v{kb_version}:{q_hash}"

        query_emb_list = (
            query_embedding.tolist()
            if isinstance(query_embedding, np.ndarray)
            else list(query_embedding)
        )

        entry = ResponseCacheEntry(
            query=query,
            query_hash=q_hash,
            query_embedding=query_emb_list,
            cacheable_answer=cacheable_answer,
            user_role=user_role,
            kb_version=kb_version,
            path=path,
            intent=intent,
            retrieved_doc_ids=retrieved_doc_ids,
        )

        self._response_cache[cache_key] = entry
        self._response_entries.append(entry)
        return True

    def clear(self):
        """Clears all in-memory cache entries."""
        self._planner_cache.clear()
        self._response_cache.clear()
        self._response_entries.clear()
