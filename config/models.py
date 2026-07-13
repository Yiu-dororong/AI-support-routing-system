import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SENTENCE_TRANSFORMER_MODEL = os.environ.get(
    "SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
RERANKER_MODEL = os.environ.get(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"
)
LLM_REPO_ID = os.environ.get("LLM_REPO_ID", "unsloth/gemma-4-E2B-it-GGUF")
LLM_FILENAME = os.environ.get("LLM_FILENAME", "gemma-4-E2B-it-UD-Q4_K_XL.gguf")

DEFAULT_LLAMA_BIN_DIR = os.path.join(BASE_DIR, "llama_bin")
LLAMA_BIN_DIR = os.environ.get("LLAMA_BIN_DIR", DEFAULT_LLAMA_BIN_DIR)

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
