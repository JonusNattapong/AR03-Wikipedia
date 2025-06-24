# app/config.py
import os

# ==================== ENVIRONMENT VARIABLES ====================

WIKI_API_URL = os.environ.get("WIKI_API_URL", "https://en.wikipedia.org/w/api.php")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/mt5-small"
VECTOR_DIM = 384

# ==================== DATA PATHS ====================

DATA_DIR = os.environ.get("DATA_DIR", "./data/")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_data.parquet")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.faiss")

# ==================== SUPERVISION ====================

LABEL_CARDINALITY = 3
LFS = []  # list of label functions will be registered at runtime
