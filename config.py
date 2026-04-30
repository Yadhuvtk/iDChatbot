"""
Central configuration for SOP Chatbot (V5 — Accuracy-First).
All tunable parameters in one place.
"""
import os
import torch

# ── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Data Files
SOP_DATA_FILE = os.path.join(DATA_DIR, "sop_data.json")
RAW_DATA_FILE = os.path.join(DATA_DIR, "data.txt")
CONVERTED_JSON_PATH = os.path.join(DATA_DIR, "converted.json") # Added for pipeline compatibility

# Index Files
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "sop_index.faiss")
ID_MAP_FILE = os.path.join(DATA_DIR, "id_map.json")
BM25_INDEX_FILE = os.path.join(DATA_DIR, "bm25_index.pkl")
FUSION_MODEL_FILE = os.path.join(DATA_DIR, "fusion_model.pkl")
CALIBRATOR_FILE = os.path.join(DATA_DIR, "calibrator.pkl")

# ── Hardware Settings ──────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Using device: {DEVICE}")

# ── Embedding Model (V5: Single Strong Retriever) ─────
# Check for local manual download first
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "bge-m3")
if os.path.exists(LOCAL_MODEL_PATH):
    EMBEDDING_MODEL = LOCAL_MODEL_PATH
    print(f"📂 Using local embedding model: {EMBEDDING_MODEL}")
else:
    # BGE-M3: dense + sparse in one model (downloads from HF)
    EMBEDDING_MODEL = "BAAI/bge-m3"

RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIM = 1024  # BGE-M3 dense dimension

# Prefix for BGE queries (used by fallback mode)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Retrieval (V5) ────────────────────────────────────
# The threshold below which we reject answers (Zero Hallucination)
SIMILARITY_THRESHOLD = 0.45   

TOP_K_RETRIEVAL = 20          # Candidates for reranking
TOP_K_FINAL = 1               # Final output

# Fusion default weights: [dense, sparse, bm25]
FUSION_WEIGHTS = [0.5, 0.2, 0.3]

# ── LLM Rewrite Style ──────────────────────
USE_LLM_REWRITE = False
OLLAMA_BASE_URL = "http://localhost:11434"
# Using Qwen 2.5 (1.5B) - fast and accurate for rewriting
OLLAMA_MODEL = "mistral" 

# ── Messages ──────────────────────────────────────────
REJECTION_MESSAGE = (
    "This question is not provided in Database.\n"
    "Please Contact Ai Team for further assistance."
)

# ── Web UI ────────────────────────────────────────────
APP_HOST = "0.0.0.0"
APP_PORT = 7860

# ── COMPATIBILITY ALIASES ─────────────────────────────
# These ensure compatibility with core/pipeline.py and core/rewriter.py
FALLBACK_MESSAGE = REJECTION_MESSAGE
RETRIEVAL_K = TOP_K_RETRIEVAL
MODEL_NAME = EMBEDDING_MODEL 
INDEX_PATH = FAISS_INDEX_FILE
METADATA_PATH = ID_MAP_FILE
