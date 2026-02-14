import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.pipeline import SOPPipeline

if __name__ == "__main__":
    print("🔧 Building FAISS index from converted.json...")
    SOPPipeline().build()
    print("Completed")
