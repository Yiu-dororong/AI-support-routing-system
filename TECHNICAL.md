# Technical Documentation: Support Routing & Retrieval Internals

This document details the underlying engineering designs, ingestion methodologies, search algorithms, access control protocols, performance optimization patterns, and evaluation frameworks of the AI Support Routing System. For MCP, please see [MCP extension](#mcp-extension) below.

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

### 3. Cross-Encoder Re-ranking with Latency Budget

The top candidates returned by RRF are re-evaluated using a Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`) to capture deep query-context relevance.

* **Latency Budget**: The reranker is designed to operate within a **250–300 ms target latency** under normal CPU workloads. Because CrossEncoder inference executes as a synchronous transformer forward pass, it **cannot be preemptively interrupted once computation has begun**. Consequently, if inference exceeds the latency budget, the system can only emit a performance warning—the computational cost has already been incurred.

* **Cold-start Mitigation**: A dummy inference is executed during application startup to warm the model, eliminating the initial **7.3 s (7299 ms)** cold-start delay caused by model loading and runtime initialization.

* **Practical Mitigations**: Since runtime cancellation is impractical without introducing expensive multiprocessing or process isolation, latency is primarily controlled **before inference**. The current system reranks the **top 15** RRF candidates. When operating under higher CPU load or tighter latency budgets, this can be adaptively reduced (e.g., **15 → 10 → 5** candidates), sacrificing some reranking accuracy in exchange for more predictable latency.

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
| **Standard Factual Lookup** | 15 | `rag`(10) / `faq_bypass`(5) | Basic policy retrieval accuracy and semantic recall. |
| **Exact Keyword / SKU / Model** | 10 | `rag` | Preserving alphanumeric codes and bigram tokenization. |
| **Multi-hop / Combining Info** | 5 | `rag_llm` | Multi-document retrieval and answer synthesis. |
| **Hidden Rule / Exception** | 5 | `rag_llm` | Extracting conditional rules and exception overrides. |
| **Ambiguous Query** | 5 | `clarify` | Routing incomplete or vague queries to clarification. |
| **Out-of-Scope** | 5 | `refuse` | Safety guardrails and out-of-domain refusals. |
| **Adversarial / Typo** | 5 | `refuse` or `rag` | Prompt injection resilience and typo robustness. |
| **Direct Human Escalation** | 5 | `escalate` | Separating action-oriented billing/account requests. |
| **Access Control (RBAC)** | 5 | `rag` (customer/employee) | Document leakage prevention (Customer vs. Employee). |
| **Total** | **60** | | |

*Note: Although there is the 'Combining Info' category, it focuses on combining within the chunk rather than across multiple chunks.*

### 2. Benchmark Performance Results

### Routing Layer
* **Routing Path Accuracy**: **88.3%** (53/60 queries correctly routed)

### Retrieval Layer (`rag` + `rag_llm`)

The retrieval layer was evaluated across both retrieval paths. For `rag`, the top-ranked document is directly presented to the user without LLM synthesis, making retrieval ranking quality critical.

* **Retrieval Hit@1**: **96.9%** (31/32 retrieval queries ranked the target context first)
* **Retrieval Hit@5**: **100.0%** (32/32 retrieval queries successfully retrieved the target context)

*2 queries are designed to test RBAC that are expected to return nothing, so the [result](data/eval/ragas_results_20260710_115104.json) appears to be 34.*

#### Performance Insights:

1. **Retrieval Reliability**
   
   The retrieval layer consistently located relevant source material, achieving a 100% Hit@5 rate. Routing failures did not indicate retrieval weakness; failed routing cases that entered retrieval paths still retrieved the expected context (`hit_score = 1.0`).

2. **Routing Boundary Trade-offs**
   
   Remaining routing errors primarily occurred at ambiguous execution boundaries rather than retrieval failures:
   
   - **Over-escalation**: Account-related informational queries were classified as requiring human handoff instead of document retrieval.
   - **Incorrect synthesis selection**: The planner occasionally misclassified whether a query required LLM synthesis:
     - Simple policy lookups were unnecessarily routed to `rag_llm`.
     - Multi-condition policy questions were routed to `rag` despite requiring cross-document reasoning.
   - **Insufficient ambiguity detection**: Broad queries (e.g., "Where is it?", "Tell me about returns.") were answered through retrieval instead of requesting clarification.

3. **Adversarial & Typo Resilience**
   
   Lowering the scope threshold to `0.15` allowed minor-typo queries (e.g., `"retun window"`) to pass semantic filtering and avoid false refusals. However, this introduces a trade-off: a lower threshold increases recall for valid support queries while allowing more out-of-scope queries to reach downstream routing such as `eval_49` that tries to inject the system.

4. **Access Control Verification**
   
   Role metadata filtering successfully prevented customer queries from retrieving restricted internal documents.

### 3. Generation Layer Evaluation (`rag_llm` only)

To evaluate the quality of generated responses, RAGAS evaluation (LLM Judge: `gemini-3.1-flash-lite`) was applied only to the subset of queries routed through the `rag_llm` path. Retrieval-only responses (`rag`) bypass LLM synthesis and are evaluated using retrieval metrics instead.

| Metric | Score |
| --- | ---: |
| Faithfulness | 0.764 |
| Context Recall | 1.000 |
| Context Precision | 0.857 |
| Answer Relevance | 0.783 |

**Evaluation limitation:** The `rag_llm` evaluation contains only 7 test cases, so individual examples have a significant impact on the aggregate scores. In addition, **generation metrics (Faithfulness and Answer Relevance) are inherently dependent on the underlying model and prompt configuration**. These results were obtained using `Gemma-4-2B` with thinking disabled and should be interpreted as a baseline measurement of the local synthesis capability rather than a standalone measure of the routing architecture.

*Note: Ragas metrics are highly compute-intensive and slow to execute; for faster iteration and robust production scaling, **DeepEval** is recommended.*

---

## ⚠️ Known Limitations

* **ChromaDB-based RBAC is not a secure access control mechanism**: Metadata filtering operates entirely at the application logic layer and lacks native hardware, container, or network separation.
* **Cross-encoder reranking introduces latency under high load**: Evaluating query-context pairs simultaneously requires substantial computational budget. To maintain scalability under high load, candidate depth must be restricted prior to cross-encoding, reducing evaluation pools to cap execution time.
* **Evaluation dataset may not fully represent production query distribution**: Ground-truth test suites contain pre-engineered query-response pairs, which can fail to capture real-world drift, user syntax variance, or conversational follow-ups.
* **Demo-Constrained Knowledge Base**: The underlying knowledge base is constructed strictly for demonstration purposes, utilizing a mixture of public and synthetically generated data. Consequently, the retrieval corpus may contain factual inconsistencies, outdated information, or logical gaps that do not reflect a production enterprise data environment.

---

## MCP Extension 

### Architecture Overview
The MCP extension augments the hybrid RAG pipeline with live, structured data lookups while preserving the two-LLM architecture:
* **Separation of Sources**: Static company documentation is retrieved from ChromaDB, while transactional database records and operational CMS schedules are fetched from external MCP servers.
* **Concurrent Execution**: ChromaDB document retrieval and MCP tool dispatching run in parallel via `asyncio.gather()` to minimize request latency.
* **Single-Pass Planning**: The Execution Planner determines both the routing path and the target tool calls in a single inference pass.
* **Response Synthesis**: Combines retrieved RAG documents and structured JSON tool outputs (or connection error/timeout contexts) as separate, distinct prompt sections. On tool failure, the LLM is instructed to politely refuse that portion of the query, mask internal details, use successful outputs/documents, and append the RAG disclaimer: *"I can only answer questions based on the retrieved documents."*

### Tool Execution (MCP)
Managed by a persistent `MCPServicesContainer` initialized at startup. Registered business-level tools are mapped to their respective MCP servers via a `ToolRegistry`, keeping vendor-specific details hidden from the planner:
* **Custom PostgreSQL MCP Server**: Exposes transactional database lookups.
* **Official Notion MCP Server**: Exposes operational promo dates and marketing calendars.
* **Abstract Toolset**: The planner reasons over business capabilities (`get_order_details`, `get_customer_profile`, `get_event_details`, `search_events`) rather than direct database/API queries.
* **Concurrency**: Independent tool calls are dispatched in parallel via `asyncio.gather()`.

```text
Execution Planner ───> ToolExecutor ───> asyncio.gather()
                                                │
                                        ┌───────┴───────┐
                                        ▼               ▼
                                   PostgreSQL        Notion
                                   MCP Server      MCP Server
                                        │               │
                                        └───────┬───────┘
                                                ▼
                                     Structured Tool Results
                                                ▼
                                       Response Synthesis
```

### Evaluation & Results
To verify correct tool execution, path routing, and synthesis grounding, a suite of [11 test cases](data/mcp_test_scenarios.json) was evaluated. The results can be found in the [report](data/mcp_test_scenarios_report.md). The planner selected the correct routing path and tool set in **10 out of 11 cases (90.9%)**; scenarios testing Postgres transaction lookups, Notion promo searches, concurrent calls, and simulated timeouts all executed successfully. For a breakdown of the single failed case, see *Intent Ambiguity* below.

### Limitations & Trade-offs
* **Tool Scaling**: Exposing all tool schemas directly to the planner increases prompt size and decoding time. **Future roadmap**: Introduce a hierarchical **Tool Retriever** layer (choose scope first, then pick specific tools) to dynamically bind only the relevant tools.
* **Intent Ambiguity**: Ambiguity between instruction-seeking (RAG) and data-seeking (MCP) queries (e.g., *"Can you show me my recent purchase history?"*) can cause planner misrouting. Refining the query with explicit data-seeking terminology (e.g., *"What is my current loyalty points balance and the name on my profile?"*) achieves 100% routing success. Future prompt boundaries will enforce this distinction dynamically.
* **Trade-offs**: MCP adds architectural complexity and network latency. In exchange, the system gains critical access to live transactional and operational data that cannot be indexed semantically.