import json
import os
from typing import Any

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from app.router import Router
from config.models import (
    LLAMA_BIN_DIR,
    LLAMA_SERVER_PORT,
    LLAMA_SERVER_URL,
    LLAMA_ZIP_URL,
    LLM_FILENAME,
    LLM_REPO_ID,
    SENTENCE_TRANSFORMER_MODEL,
)

# Import refactored architectural configurations
from config.settings import CHROMA_PATH, DATA_DIR, FAQS_FILE, INTENTS_FILE
from core.faq import FAQHandler
from core.planner import ExecutionPlanner
from core.scope import IntentClassifier
from core.types import RoutingDecision
from llm.generator import ResponseGenerator
from rag.bm25 import BM25SearchEngine
from rag.chroma import ChromaRetriever
from rag.pipeline import RAGPipeline
from rag.reranker import DocumentReranker


class SupportRouter:
    def __init__(self, chroma_path: str | None = None):
        chroma_path = chroma_path or CHROMA_PATH
        print(
            f"Initializing SentenceTransformer model ({SENTENCE_TRANSFORMER_MODEL})...",
            flush=True,
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=SENTENCE_TRANSFORMER_MODEL)

        # Download local execution planner model weights
        from huggingface_hub import hf_hub_download

        print("Ensuring local execution planner model is downloaded...", flush=True)
        llm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm")
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
                            line.strip() for line in content.split("\n") if line.strip()
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

                        # Segment public documents from internal documents
                        is_internal = any(
                            doc_id.startswith(prefix)
                            for prefix in [f"doc_{idx:02d}" for idx in range(11, 21)]
                            + ["doc_22", "doc_23"]
                        )
                        allowed_roles = (
                            ["employee"] if is_internal else ["customer", "employee"]
                        )

                        kb_docs.append(
                            LCDocument(
                                page_content=content,
                                metadata={
                                    "title": title,
                                    "category": category,
                                    "allowed_roles": allowed_roles,
                                },
                                id=doc_id,
                            )
                        )

                # Also ingest curated FAQs into the vector store as reference documents
                for idx, faq in enumerate(self.faqs):
                    faq_id = f"faq_{idx + 1}"
                    faq_content = (
                        f"Question: {faq['question']}\nAnswer: {faq['answer']}"
                    )
                    faq_title = f"FAQ: {faq['question']}"
                    faq_category = faq.get("intent", "general")

                    kb_docs.append(
                        LCDocument(
                            page_content=faq_content,
                            metadata={
                                "title": faq_title,
                                "category": faq_category,
                                "allowed_roles": ["customer", "employee"],
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

        # Populate self.kb_documents dynamically from database
        # for UI Inspector and BM25 Search Engine
        self.kb_documents = []
        try:
            results = self.vector_store.get()
            if results and results["documents"]:
                for idx in range(len(results["documents"])):
                    meta = results["metadatas"][idx] or {}
                    allowed_roles = meta.get("allowed_roles", ["customer", "employee"])
                    if isinstance(allowed_roles, str):
                        try:
                            allowed_roles = json.loads(allowed_roles)
                        except Exception:
                            allowed_roles = [
                                r.strip() for r in allowed_roles.split(",") if r.strip()
                            ]

                    doc_meta = {
                        "title": meta.get("title", "Support Guide"),
                        "category": meta.get("category", "general"),
                        "allowed_roles": allowed_roles,
                    }
                    self.kb_documents.append(
                        {
                            "id": results["ids"][idx],
                            "content": results["documents"][idx],
                            "title": doc_meta["title"],
                            "category": doc_meta["category"],
                            "metadata": doc_meta,
                        }
                    )
        except Exception as e:
            print(f"Populate kb_documents warning: {e}", flush=True)

        # Instantiate modular components matching folder layout
        self.intent_classifier = IntentClassifier(self.intent_centroids)
        self.faq_handler = FAQHandler(self.faq_embeddings)
        self.planner = ExecutionPlanner(self.structured_planner)

        self.chroma = ChromaRetriever(self.vector_store)
        self.bm25 = BM25SearchEngine(self.kb_documents)
        self.reranker = DocumentReranker(timeout_ms=250.0)
        self.rag_pipeline = RAGPipeline(self.chroma, self.bm25, self.reranker)

        self.router = Router(
            intent_classifier=self.intent_classifier,
            faq_handler=self.faq_handler,
            planner=self.planner,
            rag_pipeline=self.rag_pipeline,
        )

        self.generator = ResponseGenerator(
            synthesis_llm=self.synthesis_llm,
            server_exe=self.server_exe,
            local_model_path=self.local_model_path,
        )

        # Warm up the LLM to trigger server-side model loading and grammar
        # compilation during initialization
        try:
            print("Warming up local LLM model and compiling grammar...", flush=True)
            from langchain_core.messages import HumanMessage, SystemMessage

            from llm import prompts as llm_prompts

            dummy_messages = [
                SystemMessage(content=llm_prompts.EXECUTION_PLANNER_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        llm_prompts.EXECUTION_PLANNER_USER_TEMPLATE.format(
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
        self, query_emb: np.ndarray, threshold: float = 0.15
    ) -> tuple[bool, str, dict[str, float]]:
        """
        [0] Scope Filter: Check if the query is in-scope against intent centroids.
        """
        return self.router.classify_intent(query_emb, threshold=threshold)

    def run_faq_layer(
        self, query_emb: np.ndarray, threshold: float = 0.8
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """
        [1] FAQ Layer: Look for a high-confidence match in the FAQ dataset.
        """
        return self.router.match_faq(query_emb, threshold=threshold)

    def run_execution_planner(
        self, query: str, intent: str, callbacks=None, metadata: dict = None
    ) -> tuple[RoutingDecision, str | None]:
        """
        [2] Execution Planner: Use a local Gemma-4-E2B model
        to determine the execution path.
        """
        return self.router.plan_routing(
            query, intent, callbacks=callbacks, metadata=metadata
        )

    def run_retrieval_layer(
        self,
        query_input: str | np.ndarray,
        n_results: int = 2,
        threshold: float = 0.0,
        user_role: str = "customer",
    ) -> tuple[list[dict[str, Any]], str | None]:
        """
        [3] Retrieval Layer: Query RAGPipeline for top-k matching support documents.
        """
        try:
            docs = self.router.retrieve_rag_documents(
                query_input, user_role=user_role, n_results=n_results
            )
            filtered_docs = []
            for doc in docs:
                similarity = doc.get("similarity", 1.0)
                if similarity >= threshold:
                    filtered_docs.append(doc)
            return filtered_docs, None
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
        return self.generator.generate(
            query=query,
            retrieved_docs=retrieved_docs,
            callbacks=callbacks,
            metadata=metadata,
        )
