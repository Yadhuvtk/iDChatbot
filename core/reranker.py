# core/reranker.py
# Cross-encoder reranker wrapper

from __future__ import annotations
from typing import List, Tuple

try:
    from sentence_transformers import CrossEncoder
except Exception as e:
    raise ImportError(
        "sentence-transformers is required for CrossEncoder reranker.\n"
        "Install with:\n"
        "  <runtime_python> -m pip install -U sentence-transformers\n"
        f"Original error: {e}"
    )


class CrossEncoderReranker:
    """
    Reranks (query, text) pairs using a cross-encoder.
    Returns float relevance scores (higher = more relevant).
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str | None = None):
        self.model = CrossEncoder(model_name, device=device)

    def score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]
