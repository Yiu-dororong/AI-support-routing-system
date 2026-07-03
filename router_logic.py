import json
import os
from typing import Any, Literal

import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import prompts


# Configuration Defaults (overridable by environment variables)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env if it exists
load_dotenv()

SENTENCE_TRANSFORMER_MODEL = os.environ.get(
    "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
)
LLM_REPO_ID = os.environ.get("LLM_REPO_ID", "unsloth/gemma-4-E2B-it-GGUF")
LLM_FILENAME = os.environ.get("LLM_FILENAME", "gemma-4-E2B-it-UD-Q4_K_XL.gguf")

DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_LLAMA_BIN_DIR = os.path.join(BASE_DIR, "llama_bin")
DEFAULT_CHROMA_PATH = os.path.join(DEFAULT_DATA_DIR, "chroma_db")

DATA_DIR = os.environ.get("SUPPORT_ROUTER_DATA_DIR", DEFAULT_DATA_DIR)
LLAMA_BIN_DIR = os.environ.get("LLAMA_BIN_DIR", DEFAULT_LLAMA_BIN_DIR)
CHROMA_PATH = os.environ.get("CHROMA_PATH", DEFAULT_CHROMA_PATH)

INTENTS_FILE = os.environ.get(
    "INTENTS_FILE", os.path.join(DATA_DIR, "intents.json")
)
FAQS_FILE = os.environ.get(
    "FAQS_FILE",
    os.path.join(DATA_DIR, "Ecommerce_FAQ_Chatbot_dataset.json"),
)

LLAMA_SERVER_HOST = os.environ.get("LLAMA_SERVER_HOST", "127.0.0.1")
LLAMA_SERVER_PORT = os.environ.get("LLAMA_SERVER_PORT", "8080")
LLAMA_SERVER_URL = os.environ.get(
    "LLAMA_SERVER_URL", f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"
)
LLAMA_ZIP_URL = os.environ.get(
    "LLAMA_ZIP_URL",
    (
        "https://github.com/ggml-org/llama.cpp/releases/download/"
        "b9840/llama-b9840-bin-win-cpu-x64.zip"
    ),
)

# Ensure directory structure exists
os.makedirs(DATA_DIR, exist_ok=True)


# Pydantic models for structured routing
class RoutingDecision(BaseModel):
    path: Literal["refuse", "clarify", "rag", "rag_llm", "escalate"] = Field(
        description=(
            "The routing path for the query. 'refuse' for unsafe/abusive "
            "content, 'clarify' for underspecified/vague queries, 'rag' for "
            "simple factual lookups, 'rag_llm' for queries requiring "
            "reasoning/synthesis, and 'escalate' for human handoff."
        )
    )
    reason: str = Field(description="A brief explanation for this routing decision.")
    clarification_question: str | None = Field(
        default=None,
        description=(
            "If path is 'clarify', the clarification question to ask the user. "
            "Otherwise null/None."
        ),
    )
    refusal_message: str | None = Field(
        default=None,
        description=(
            "If path is 'refuse', the refusal message to show. Otherwise null/None."
        ),
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SupportRouter:
    def __init__(self, chroma_path: str | None = None):
        chroma_path = chroma_path or CHROMA_PATH
        print(
            f"Initializing SentenceTransformer model ({SENTENCE_TRANSFORMER_MODEL})...",
            flush=True,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=SENTENCE_TRANSFORMER_MODEL
        )

        # Download local execution planner model weights
        from huggingface_hub import hf_hub_download

        print("Ensuring local execution planner model is downloaded...", flush=True)
        llm_dir = os.path.join(BASE_DIR, "llm")
        try:
            self.local_model_path = hf_hub_download(
                repo_id=LLM_REPO_ID,
                filename=LLM_FILENAME,
                local_dir=llm_dir,
            )
        except Exception as e:
            print(
                f"Failed to download local model, using cached fallback: {e}",
                flush=True,
            )
            import glob

            files = glob.glob(
                os.path.join(llm_dir, f"**/{LLM_FILENAME}"), recursive=True
            )
            self.local_model_path = files[0] if files else None

        self.llama_bin_dir = LLAMA_BIN_DIR
        self.server_exe = os.path.join(self.llama_bin_dir, "llama-server.exe")
        self.server_url = LLAMA_SERVER_URL

        if not os.path.exists(self.server_exe):
            print("Downloading llama.cpp CPU binaries...", flush=True)
            import urllib.request
            import zipfile

            os.makedirs(self.llama_bin_dir, exist_ok=True)
            zip_path = os.path.join(self.llama_bin_dir, "llama_cpu.zip")
            urllib.request.urlretrieve(LLAMA_ZIP_URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.llama_bin_dir)

        # Start llama-server if not already active
        import httpx

        self.started_server = False
        try:
            resp = httpx.get(f"{self.server_url}/health", timeout=1.0)
            if resp.status_code == 200:
                print(
                    "Local llama-server is already running on port "
                    f"{LLAMA_SERVER_PORT}. Reusing it.",
                    flush=True,
                )
        except Exception:
            print("Starting llama-server.exe as a background process...", flush=True)
            import subprocess

            self.server_process = subprocess.Popen(
                [
                    self.server_exe,
                    "-m",
                    self.local_model_path,
                    "--port",
                    LLAMA_SERVER_PORT,
                    "-c",
                    "4096",
                    "-t",
                    "4",
                    "-fa",
                    "on",
                    "--reasoning",
                    "off",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.started_server = True

            # Wait for health check
            import time

            print("Waiting for local server to initialize...", flush=True)
            for _ in range(30):
                try:
                    resp = httpx.get(f"{self.server_url}/health", timeout=1.0)
                    if resp.status_code == 200:
                        print("Local server is ready!", flush=True)
                        break
                except Exception:
                    time.sleep(1)

        # Register cleanup handler
        if (
            self.started_server
            and hasattr(self, "server_process")
            and self.server_process
        ):
            import atexit

            def cleanup_server(process):
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except Exception:
                    pass

            atexit.register(cleanup_server, self.server_process)

        # Setup LangChain models for routing and synthesis
        self.planner_llm = ChatOpenAI(
            base_url=self.server_url + "/v1",
            api_key="local-dummy-key",
            model="gemma-4-e2b",
            temperature=0.0,
        )
        self.structured_planner = self.planner_llm.with_structured_output(
            RoutingDecision
        )

        self.synthesis_llm = ChatOpenAI(
            base_url=self.server_url + "/v1",
            api_key="local-dummy-key",
            model="gemma-4-e2b",
            temperature=0.2,
        )

        # Load local data configuration
        self.intents = self._load_json(INTENTS_FILE)
        faqs_data = self._load_json(FAQS_FILE)
        if isinstance(faqs_data, dict) and "questions" in faqs_data:
            self.faqs = faqs_data["questions"]
        else:
            self.faqs = faqs_data

        # Compute intent centroids
        print("Computing intent centroids...", flush=True)
        self.intent_centroids = {}
        for intent_id, info in self.intents.items():
            examples = info.get("examples", [])
            if examples:
                embeddings = self.embeddings.embed_documents(examples)
                centroid = np.mean(embeddings, axis=0)
                self.intent_centroids[intent_id] = centroid

        # Compute FAQ embeddings
        print("Embedding curated FAQs...", flush=True)
        self.faq_embeddings = []
        for faq in self.faqs:
            q_emb = self.embeddings.embed_query(faq["question"])
            self.faq_embeddings.append(
                {
                    "question": faq["question"],
                    "answer": faq["answer"],
                    "intent": faq.get("intent", "general"),
                    "embedding": np.array(q_emb),
                }
            )

        # Set up ChromaDB via LangChain
        print("Initializing ChromaDB...", flush=True)
        self.vector_store = Chroma(
            collection_name="kb_documents",
            embedding_function=self.embeddings,
            persist_directory=chroma_path,
            collection_metadata={"hnsw:space": "cosine"},
        )

        # Force database refresh if legacy JSON format exists in the database
        try:
            doc_count = self.vector_store._collection.count()
            if doc_count > 0:
                results = self.vector_store.get(limit=1)
                is_legacy = (
                    results
                    and results["ids"]
                    and not results["ids"][0].endswith("_page1")
                )
                if is_legacy:
                    print("Clearing legacy JSON ChromaDB collection...", flush=True)
                    self.vector_store.delete_collection()
                    self.vector_store = Chroma(
                        collection_name="kb_documents",
                        embedding_function=self.embeddings,
                        persist_directory=chroma_path,
                        collection_metadata={"hnsw:space": "cosine"},
                    )
        except Exception as e:
            print(f"Chroma clean check warning: {e}", flush=True)

        # Populate collection if empty using langchain_docling
        try:
            if self.vector_store._collection.count() == 0:
                print(
                    "Ingesting PDF documents using langchain_docling...",
                    flush=True,
                )
                import glob

                from langchain_docling import DoclingLoader

                pdf_folder = os.path.join(DATA_DIR, "documents")
                pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

                kb_docs = []
                for pdf_file in pdf_files:
                    loader = DoclingLoader(file_path=pdf_file)
                    pages = loader.load()

                    doc_title = None
                    for i, page in enumerate(pages):
                        doc_id = (
                            f"{os.path.basename(pdf_file).replace('.pdf', '')}"
                            f"_page{i + 1}"
                        )

                        content = page.page_content.strip()
                        lines = [
                            line.strip()
                            for line in content.split("\n")
                            if line.strip()
                        ]

                        category = "general"
                        for line in lines:
                            if line.startswith("Category:"):
                                category = line.split(":", 1)[1].strip()
                                break

                        if i == 0:
                            title = "Support Guide"
                            if lines:
                                first_line = lines[0]
                                meta_lower = first_line.lower()
                                meta_keys = [
                                    "effective date:",
                                    "published:",
                                    "validity:",
                                    "effective:",
                                    "revision:",
                                    "updated:",
                                    "date:",
                                ]
                                is_meta = any(
                                    meta_lower.startswith(p) for p in meta_keys
                                )
                                if is_meta:
                                    base = os.path.basename(pdf_file).replace(
                                        ".pdf", ""
                                    )
                                    if (
                                        base.startswith("doc_")
                                        and len(base) > 7
                                        and base[6] == "_"
                                    ):
                                        base = base[7:]
                                    elif (
                                        base.startswith("doc_")
                                        and len(base) > 6
                                        and base[5] == "_"
                                    ):
                                        base = base[6:]
                                    title = base.replace("_", " ").title()
                                else:
                                    title = first_line
                            doc_title = title
                        else:
                            title = doc_title or "Support Guide"

                        kb_docs.append(
                            LCDocument(
                                page_content=content,
                                metadata={"title": title, "category": category},
                                id=doc_id,
                            )
                        )

                # Also ingest curated FAQs into the vector store as reference documents
                for idx, faq in enumerate(self.faqs):
                    faq_id = f"faq_{idx + 1}"
                    faq_content = (
                        f"Question: {faq['question']}\n"
                        f"Answer: {faq['answer']}"
                    )
                    faq_title = f"FAQ: {faq['question']}"
                    faq_category = faq.get("intent", "general")

                    kb_docs.append(
                        LCDocument(
                            page_content=faq_content,
                            metadata={
                                "title": faq_title,
                                "category": faq_category,
                            },
                            id=faq_id,
                        )
                    )

                if kb_docs:
                    self.vector_store.add_documents(kb_docs)
                    print(
                        f"Ingested {len(kb_docs)} documents into ChromaDB.",
                        flush=True,
                    )
        except Exception as e:
            print(f"Chroma ingestion warning/error: {e}", flush=True)

        # Populate self.kb_documents dynamically from database for UI Inspector
        self.kb_documents = []
        try:
            results = self.vector_store.get()
            if results and results["documents"]:
                for idx in range(len(results["documents"])):
                    self.kb_documents.append(
                        {
                            "id": results["ids"][idx],
                            "content": results["documents"][idx],
                            "title": results["metadatas"][idx]["title"],
                            "category": results["metadatas"][idx]["category"],
                        }
                    )
        except Exception:
            pass

        # Warm up the LLM to trigger server-side model loading and grammar
        # compilation during initialization
        try:
            print("Warming up local LLM model and compiling grammar...", flush=True)
            from langchain_core.messages import HumanMessage, SystemMessage
            dummy_messages = [
                SystemMessage(content=prompts.EXECUTION_PLANNER_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        prompts.EXECUTION_PLANNER_USER_TEMPLATE.format(
                            intent="general", query="ping"
                        )
                    )
                ),
            ]
            self.structured_planner.invoke(dummy_messages)
            print("LLM model and grammar warm-up complete!", flush=True)
        except Exception as e:
            print(f"LLM warm-up warning: {e}", flush=True)

        print("Initialization complete!", flush=True)

    def _load_json(self, path: str) -> Any:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def get_query_embedding(self, query: str) -> np.ndarray:
        return np.array(self.embeddings.embed_query(query))

    def run_scope_filter(
        self, query_emb: np.ndarray, threshold: float = 0.4
    ) -> tuple[bool, str, dict[str, float]]:
        """
        [0] Scope Filter: Check if the query is in-scope against intent centroids.
        """
        similarities = {}
        for intent_id, centroid in self.intent_centroids.items():
            similarities[intent_id] = cosine_similarity(query_emb, centroid)

        max_intent = max(similarities, key=similarities.get)
        max_score = similarities[max_intent]

        in_scope = max_score >= threshold
        return in_scope, max_intent, similarities

    def run_faq_layer(
        self, query_emb: np.ndarray, threshold: float = 0.8
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """
        [1] FAQ Layer: Look for a high-confidence match in the FAQ dataset.
        """
        matches = []
        for faq in self.faq_embeddings:
            score = cosine_similarity(query_emb, faq["embedding"])
            matches.append(
                {
                    "question": faq["question"],
                    "answer": faq["answer"],
                    "intent": faq["intent"],
                    "score": score,
                }
            )

        # Sort matches by score descending
        matches.sort(key=lambda x: x["score"], reverse=True)

        best_match = matches[0] if matches else None
        if best_match and best_match["score"] >= threshold:
            return best_match, matches[:3]  # Return best match and top 3 candidates
        return None, matches[:3]

    def run_execution_planner(
        self, query: str, intent: str, callbacks=None, metadata: dict = None
    ) -> tuple[RoutingDecision, str | None]:
        """
        [2] Execution Planner: Use a local Gemma-4-E2B model
        to determine the execution path.
        """
        if (
            not hasattr(self, "local_model_path")
            or not self.local_model_path
            or not os.path.exists(self.server_exe)
        ):
            # Fallback if local model is not initialized/found
            return RoutingDecision(
                path="rag",
                reason=(
                    "Local Gemma-4 model or llama.cpp binary not found. "
                    "Defaulting to standard RAG lookup."
                ),
            ), None

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=prompts.EXECUTION_PLANNER_SYSTEM_PROMPT),
            HumanMessage(
                content=prompts.EXECUTION_PLANNER_USER_TEMPLATE.format(
                    intent=intent, query=query
                )
            ),
        ]

        config = {"callbacks": callbacks}
        if metadata:
            config["metadata"] = metadata
            config["run_name"] = "support_router_query"

        try:
            decision = self.structured_planner.invoke(
                messages, config=config
            )
            raw_output = decision.model_dump_json(indent=2)
            return decision, raw_output
        except Exception as e:
            # Fallback on failure
            return RoutingDecision(
                path="rag",
                reason=(
                    f"Local execution planning failed with error: {str(e)}. "
                    "Defaulting to standard RAG lookup."
                ),
            ), f"Error details: {str(e)}"

    def run_retrieval_layer(
        self,
        query_input: str | np.ndarray,
        n_results: int = 2,
        threshold: float = 0.0,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """
        [3] Retrieval Layer: Query ChromaDB for top-k matching support documents.
        """
        try:
            if isinstance(query_input, np.ndarray):
                # Use raw chromadb collection query by embedding vector
                res = self.vector_store._collection.query(
                    query_embeddings=[query_input.tolist()],
                    n_results=n_results,
                )
                docs = []
                if res and res["documents"] and len(res["documents"]) > 0:
                    for i in range(len(res["documents"][0])):
                        dist = (
                            res["distances"][0][i]
                            if res["distances"]
                            else 0.0
                        )
                        similarity = 1.0 - dist
                        if similarity >= threshold:
                            docs.append(
                                {
                                    "id": res["ids"][0][i],
                                    "content": res["documents"][0][i],
                                    "metadata": res["metadatas"][0][i],
                                    "distance": dist,
                                    "similarity": similarity,
                                }
                            )
                return docs, None
            else:
                results = self.vector_store.similarity_search_with_score(
                    query_input, k=n_results
                )
                docs = []
                for doc, score in results:
                    # Cosine distance: similarity = 1.0 - score
                    similarity = 1.0 - score
                    if similarity >= threshold:
                        docs.append(
                            {
                                "id": doc.id,
                                "content": doc.page_content,
                                "metadata": doc.metadata,
                                "distance": score,
                                "similarity": similarity,
                            }
                        )
                return docs, None
        except Exception as e:
            return [], str(e)

    def run_response_generation(
        self,
        query: str,
        retrieved_docs: list[dict[str, Any]],
        callbacks=None,
        metadata: dict = None,
    ) -> tuple[str, str]:
        """
        [4] Response Generation: Synthesize the final answer using local
        Gemma-4-E2B, grounded only in the retrieved documents.
        """
        if (
            not hasattr(self, "local_model_path")
            or not self.local_model_path
            or not os.path.exists(self.server_exe)
        ):
            # Fallback if local model/binary is missing
            fallback_resp = prompts.LLM_UNAVAILABLE_FALLBACK_HEADER
            for doc in retrieved_docs:
                fallback_resp += (
                    f"**{doc['metadata']['title']}** "
                    f"(Confidence: {doc['similarity']:.2f})\n"
                    f"{doc['content']}\n\n"
                )
            return (
                fallback_resp,
                "Local model not initialized. Surfaced retrieved documents directly.",
            )

        # Prepare context from retrieved documents
        context_str = ""
        for i, doc in enumerate(retrieved_docs):
            context_str += (
                f"--- Document {i + 1}: {doc['metadata']['title']} ---\n"
                f"{doc['content']}\n\n"
            )

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=prompts.RESPONSE_SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(
                content=prompts.RESPONSE_SYNTHESIS_USER_TEMPLATE.format(
                    context_str=context_str, query=query
                )
            ),
        ]

        user_part = prompts.RESPONSE_SYNTHESIS_USER_TEMPLATE.format(
            context_str=context_str, query=query
        )
        prompt = (
            f"System:\n{prompts.RESPONSE_SYNTHESIS_SYSTEM_PROMPT}\n\n"
            f"User:\n{user_part}"
        )

        config = {"callbacks": callbacks}
        if metadata:
            config["metadata"] = metadata
            config["run_name"] = "support_router_query"

        try:
            response = self.synthesis_llm.invoke(
                messages, config=config
            )

            content = response.content
            reasoning = ""
            if hasattr(response, "additional_kwargs"):
                reasoning = response.additional_kwargs.get("reasoning_content", "")
            if not reasoning and response.response_metadata:
                reasoning = response.response_metadata.get("reasoning_content", "")

            if reasoning:
                raw_output = (
                    f"[Start thinking]\n{reasoning}\n[End thinking]\n\n{content}"
                )
            else:
                raw_output = content
            return raw_output, prompt
        except Exception as e:
            # Fallback on failure: Surface supporting articles directly
            fallback_resp = prompts.LLM_SYNTHESIS_FAILED_FALLBACK_HEADER
            for doc in retrieved_docs:
                fallback_resp += (
                    f"**{doc['metadata']['title']}** "
                    f"(Confidence: {doc['similarity']:.2f})\n"
                    f"{doc['content']}\n\n"
                )
            return (
                fallback_resp,
                (
                    "Local model synthesis failed with error: "
                    f"{str(e)}. Surfaced retrieved documents directly."
                ),
            )
