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
