# Technical Documentation: Support Routing & Retrieval Internals

This document details the underlying engineering designs, ingestion methodologies, search algorithms, access control protocols, performance optimization patterns, and evaluation frameworks of the AI Support Routing System.

---

## 📂 Document Ingestion & Chunking Strategy

The system processes and indexes knowledge base data from two heterogeneous formats: structured JSON FAQ databases and formatted PDF policy guides.

### 1. Ingestion Mechanics
* **Curated FAQ JSON (`Ecommerce_FAQ_Chatbot_dataset.json`)**: Ingested on a **one-to-one record basis**. Each question-answer pair is parsed as a single, self-contained chunk formatted as `Question: {q}\nAnswer: {a}`. This guarantees that query-response mappings are preserved within a single embedding vector.
* **Policy PDFs (`data/documents/`)**: Parsed using IBM **Docling** (`DoclingLoader`), which extracts layout elements, headings, lists, and tables. The documents are partitioned using a **page-by-page chunking strategy** where each physical page maps to exactly one ChromaDB document chunk, with page-level IDs (e.g. `doc_24_page16`) to maintain provenance.

### 2. Trade-offs: Page-Level Ingestion vs. Standard Text Splitters

This strategy was chosen over conventional recursive text splitters with overlap (such as LangChain's `RecursiveCharacterTextSplitter`) due to specific e-commerce domain trade-offs:

| Ingestion Strategy | Pros | Cons |
| :--- | :--- | :--- |
| **Page-by-Page + Docling** (Current implementation) | <ul><li>**Preserves Table Integrity**: Table grids and rows are kept intact on a single page rather than being split in half mid-row.</li><li>**Reference Alignment**: Enables clean citations (*"Page 16 of Document 24"*) that match physical documentation layouts, aiding manual operator verification.</li><li>**Document Structure Preservation**: Headings and their corresponding sub-paragraphs are kept together within their natural context.</li></ul> | <ul><li>**Context Density Variations**: Pages vary in size (some contain 100 words, others 1000 + tables), leading to unbalanced vector chunks.</li><li>**Embedding Model Constraints**: Extremely dense pages can exceed the token limits of small embedding models, causing truncation or dilution of semantic vector scores.</li></ul> |
| **Recursive Text Splitters** (With token/character overlaps) | <ul><li>**Token Uniformity**: Guarantees highly consistent chunk sizes (e.g., exactly 500 characters), which aligns well with embedding model dimensions.</li><li>**Boundary Continuity**: Text overlap (e.g., 50–100 characters) prevents semantic information loss at the boundaries where splits occur.</li></ul> | <ul><li>**Layout & Table Destruction**: Systematically splits complex table grids and rows in half, rendering formatted layouts unreadable to the LLM.</li><li>**Semantic Context Fragmentation**: Fixed-length cutoffs frequently separate section headers and titles from their body text, leading to context drift.</li><li>**No Page/Reference Tracking**: Completely discards physical page boundaries, making it impossible to audit exact source pages.</li></ul> |

---

## 🔍 Advanced Hybrid Search & Re-ranking Architecture

Dense vector retrieval frequently fails to match exact string parameters (such as part numbers, SKUs, or error codes). The system addresses this with a multi-stage retrieval architecture:

```
Query 
  │
  ├──► [Dense Retrieval] (ChromaDB / all-MiniLM-L6-v2) ──► List A (Ranked Chunks)
  │                                                            │
  ├──► [Sparse Retrieval] (BM25 / rank_bm25) ─────────────► List B (Ranked Chunks)
  │                                                            │
  ▼                                                            ▼
[Reciprocal Rank Fusion] ◄─────────────────────────────────────┘
  │ (Fuses top K candidates based on rank position & weights)
  ▼
[Cross-Encoder Re-ranker] (ms-marco-MiniLM-L-6-v2)
  │ (Scores top candidate pairs with latency safeguards)
  ▼
[Top M Chunks] ──► Passed to Synthesis LLM
```

### 1. Lexical Search Integration (BM25)
To support exact lookups and prevent splitting product IDs, the system indexes documents with a custom sparse BM25 engine. Tokenization is performed using the regex pattern `r"[a-z0-9]+(?:[-_][a-z0-9]+)*"`, which preserves alphanumeric dashes and underscores (e.g., matching `VT-Titan_XL-99` intact) and extracts contiguous bigram phrases (shingles) to capture product name boundaries.

### 2. Reciprocal Rank Fusion (RRF) with Adaptive Weighting
Ranked candidate lists from ChromaDB and BM25 are merged using a **Query-Adaptive Reciprocal Rank Fusion (RRF)** mechanism:
* **Execution Model**: Dense and sparse retrievals run sequentially and synchronously on a single thread. Since the in-memory BM25 lexical search is extremely fast (~1–5ms), sequential execution minimizes software complexity without affecting latency.
* **Rationale**: Ordinal ranks are combined using weights that shift based on query style. For general queries, we balance dense semantic matching and exact keyword matching (`bm25_weight = 1.2`, `dense_weight = 1.0`).
* **Adaptive Boosting**: If a query is identified as containing model identifiers, version codes, or capacity units (e.g., `v1`, `Titan-C`, `100Wh`), the system dynamically increases lexical priority (`bm25_weight = 1.8`, `dense_weight = 0.8`) to ensure SKU precision.

### 3. Cross-Encoder Re-ranking with Latency Budget Guards
The top candidates returned by RRF are re-evaluated using a Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`) to capture deep query-context relevance. 
* **Proactive Latency Control**: Reranking latency scales linearly with candidate count. To protect the latency budget on CPU, the input is strictly pruned to the top **15 candidates** before prediction, preventing runaway CPU cycles.
* **Post-Hoc Timeout Warning**: If the reranking pass exceeds a **250ms** threshold, a latency budget warning is logged for performance monitoring. Because CPU-bound inference in Python cannot be preemptively interrupted mid-operation without prohibitive multiprocessing overhead, the system continues with the successfully computed reranked results (as the latency has already been paid) rather than discarding them.

---

## 🔒 Security, Access Control, and Document Classification

### 1. Document Access Control Heuristics
ChromaDB does not support native multi-tenant isolation. To simulate role-based authorization and prevent accidental leakage of internal files (employee guidelines, business strategies), the system segments document ingestion and retrieval using metadata tags:
* **Internal Scope**: Documents prefixed from `doc_11` to `doc_20`, plus `doc_22` and `doc_23`, are marked `allowed_roles: ["employee"]`.
* **Public Scope**: All other documents are marked `allowed_roles: ["customer", "employee"]`.
* **Retrieval Query Filter**: Queries are filtered at the database driver level:
  ```python
  where={"allowed_roles": {"$contains": user_role}}
  ```

### 2. Document Inventory Summary
The knowledge base comprises **25 document policies** parsed and partitioned into page chunks:
* **Public Customer Support Manuals (13 Documents)**: Publicly accessible documents covering warranties, loyalty points, return terms, restocking fees, and shipping details (standard page length: 1–6 pages).
* **Internal Employee Operations Policies (12 Documents)**: Internal policies covering warehouse safety, HR protocols, IT configurations, financial statements, and development strategies (standard page length: 1–20 pages).

### 3. Production Enterprise Database Migration Roadmap
Metadata filtering acts as a soft application constraint. To establish strict cryptographic and network security boundaries in production, we recommend migrating from local ChromaDB to one of the following enterprise architectures:

* **Option A: Qdrant Namespaces with JWT Scopes**
  * *Implementation*: Split documents into separate collections (`public_customer` and `internal_employee`).
  * *Access Control*: Issue signed JSON Web Tokens (JWTs) with scopes to client connections. Qdrant filters queries at the network/driver layer, blocking customer tokens from reading the internal namespace.

* **Option B: Milvus Multi-Tenancy & Partition Keys**
  * *Implementation*: Use partition keys or separate Milvus collections.
  * *Access Control*: Map application roles to database RBAC roles (`customer_role`, `employee_role`), ensuring physical access segregation.

---

## ⚡ Local Inference & Performance Optimizations

Running LLM planning and generation models locally on CPU introduces latency challenges. The system implements the following performance patterns:

### 1. Prefix Prompt Caching
To avoid re-evaluating static system instructions (~600 tokens) on CPU for every query, the prompt templates are structured to place static system prompts as a prefix and dynamic inputs (user query and context) as a trailing suffix. This **preserves the KV-cache**, reducing subsequent routing runs from 9.66s to 4.12s (a **2.3x speedup**).

### 2. Dual-Server Deployment Model (Recommended)
Generating reasoning tokens (Gemma's `<thought>` block) is compute-heavy. This prototype operates a single local `llama-server` instance with `--reasoning off` to optimize planning. In production, we recommend a dual-server layout:
1. **Planner Instance**: A low-latency instance running with **reasoning disabled** (`--reasoning off`, dropping routing times from 12.54s to 1.56s—an **8.0x speedup**).
2. **Synthesis Instance**: A separate instance running with **reasoning enabled** (`--reasoning on`) for executing RAG synthesis.

<details>
<summary><b>Common llama.cpp reasoning configuration options</b></summary>

```text
--reasoning [on|off|auto]              Use reasoning/thinking in the chat ('on', 'off', or 'auto')
                                        (env: LLAMA_ARG_REASONING)
--reasoning-budget N                    Token budget for thinking: -1 for unrestricted, 0 for immediate end
                                        (env: LLAMA_ARG_THINK_BUDGET)
--reasoning-budget-message MESSAGE      Message injected before the end-of-thinking tag when reasoning budget is exceeded
                                        (env: LLAMA_ARG_THINK_BUDGET_MESSAGE)
```
</details>

### 3. Model Warm-up
Because `llama-server` compiles JSON grammar syntax trees lazily on the first incoming request, the initial query faces a **~9.46s cold start**. The startup routine executes a dummy query during system initialization to warm up model weights and JSON parsing state, **reducing first-query latency to 5.01s** (a 4.5s speedup).

---

## 📈 System Evaluation & Performance Benchmarks

To establish a test-driven development loop, we run an offline evaluation runner `run_eval.py` against a golden dataset of **60 queries** spanning standard factual lookups, SKU matches, out-of-scope prompts, safety injections, and access-control edge cases.

### 1. Test Dataset Distribution

| Category | Queries | Expected Routing Path | Verification Target |
| :--- | :---: | :---: | :--- |
| **Standard Factual Lookup** | 15 | `rag` / `faq_bypass` | Basic policy retrieval accuracy and semantic recall. |
| **Exact Keyword / SKU / Model** | 10 | `rag` | Preserving alphanumeric codes and bigram tokenization. |
| **Multi-hop / Combining Info** | 5 | `rag_llm` | Multi-document retrieval and answer synthesis. |
| **Hidden Rule / Exception** | 5 | `rag_llm` | Extracting conditional rules and exception overrides. |
| **Ambiguous Query** | 5 | `clarify` | Routing incomplete or vague queries to clarification. |
| **Out-of-Scope** | 5 | `refuse` | Safety guardrails and out-of-domain refusals. |
| **Adversarial / Typo** | 5 | `refuse` or `rag` | Prompt injection resilience and typo robustness. |
| **Direct Human Escalation** | 5 | `escalate` | Separating action-oriented billing/account requests. |
| **Access Control (RBAC)** | 5 | `rag` (customer/employee) | Document leakage prevention (Customer vs. Employee). |
| **Total** | **60** | | |

### 2. Benchmark Performance Results
The evaluation was executed on local CPU hardware:

* **Routing Path Accuracy**: **73.3%** (44/60 queries correctly routed)
* **Retrieval Hit@5 Rate (RAG only)**: **97.3%** (36/37 queries expecting retrieval successfully retrieved the target context)

#### Performance Insights:
1. **Adversarial & Typo Resilience**: Lowering the scope threshold to `0.15` allowed minor-typo queries (e.g. `"retun window"`) to pass the semantic filter and route correctly, preventing false refusals.
2. **Access Control Verification**: Role metadata filters successfully blocked customer queries from retrieving internal documents.
3. **Cross-Encoder Latency Pruning**: Under CPU load, candidate pruning to 15 items capped cross-encoder inference time. When the 250ms budget was exceeded, the system retained the reranked results rather than discarding them—since the latency cost had already been paid.

### 3. Proposed Automated RAG Evaluation (RAGAS Framework - Roadmap)
> [!NOTE]
> **Implementation Status**: This is a planned upgrade for the offline evaluation suite. Currently, stubs are defined in [ragas_eval.py](file:///E:/agy/router/AI-support-routing-system/evaluation/ragas_eval.py) but are not executed in the active benchmark suite ([run_eval.py](file:///E:/agy/router/AI-support-routing-system/evaluation/run_eval.py)) to avoid the latency and API cost of running generative LLM judges on CPU.

RAGAS (Retrieval Augmented Generation Assessment) evaluates output quality by prompting an LLM judge to score semantic attributes rather than doing exact-match comparisons.

#### 3.1 LLM-Judged Metrics

| Metric | What it measures | How the evaluator scores it |
| :--- | :--- | :--- |
| **Faithfulness** | Whether the response is supported only by retrieved context. | Extracts claims from the response and verifies each against the retrieved chunks. |
| **Answer Relevance** | Whether the response directly addresses the question. | Generates hypothetical queries from the answer and measures similarity to the original input. |
| **Context Recall** | Whether retrieval captured all statements needed by the ground-truth answer. | Compares ground-truth sentences against retrieved chunks to identify missing links. |
| **Context Precision** | Whether the most relevant chunks are ranked at the top. | Analyzes relevance scores across rank positions in the context block. |

---

## ⚠️ Known Limitations

* **ChromaDB-based RBAC is not a secure access control mechanism**: Metadata filtering operates entirely at the application logic layer and lacks native hardware, container, or network separation.
* **Cross-encoder reranking introduces latency under high load**: Evaluating query-context pairs simultaneously requires substantial computational budget ($O(N)$ transformer forwards), which poses scalability challenges compared to simple inner-product vector indexing.
* **Evaluation dataset may not fully represent production query distribution**: Ground-truth test suites contain pre-engineered query-response pairs, which can fail to capture real-world drift, user syntax variance, or conversational follow-ups.
* **Chunking heuristics may still split semantic boundaries in complex documents**: Document parser models (e.g., Docling) or character/token splitter offsets can split key clauses across adjacent chunks, leading to degraded context recall.
