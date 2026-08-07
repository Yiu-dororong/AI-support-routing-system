import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_CHROMA_PATH = os.path.join(DEFAULT_DATA_DIR, "chroma_db")

DATA_DIR = os.environ.get("SUPPORT_ROUTER_DATA_DIR", DEFAULT_DATA_DIR)
CHROMA_PATH = os.environ.get("CHROMA_PATH", DEFAULT_CHROMA_PATH)

INTENTS_FILE = os.environ.get("INTENTS_FILE", os.path.join(DATA_DIR, "intents.json"))
FAQS_FILE = os.environ.get(
    "FAQS_FILE",
    os.path.join(DATA_DIR, "Ecommerce_FAQ_Chatbot_dataset.json"),
)

# LLM Inference Configuration Settings
USE_LOCAL_LLM = os.environ.get("USE_LOCAL_LLM",
                               "true").lower().strip() in ("true", "1", "yes")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower().strip()

# Gemini Cloud LLM Settings
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_PLANNER_MODEL = os.environ.get("GEMINI_PLANNER_MODEL",
                                      "gemini-2.5-flash")
GEMINI_SYNTHESIS_MODEL = os.environ.get("GEMINI_SYNTHESIS_MODEL",
                                        "gemini-3.1-flash-lite")

# Generic Cloud OpenAI Settings
LLM_CLOUD_BASE_URL = os.environ.get("LLM_CLOUD_BASE_URL", "https://api.openai.com/v1")
LLM_CLOUD_API_KEY = os.environ.get("LLM_CLOUD_API_KEY", "")
LLM_CLOUD_PLANNER_MODEL = os.environ.get("LLM_CLOUD_PLANNER_MODEL", "gpt-4o-mini")
LLM_CLOUD_SYNTHESIS_MODEL = os.environ.get("LLM_CLOUD_SYNTHESIS_MODEL", "gpt-4o-mini")
