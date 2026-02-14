# core/bge_m3_embed.py
# BGE-M3 embedder wrapper that returns numpy float32 embeddings

from __future__ import annotations

import numpy as np

try:
    from FlagEmbedding import BGEM3FlagModel
except Exception as e:
    raise ImportError(
        "FlagEmbedding is not installed. Install with:\n"
        "  <runtime_python> -m pip install -U FlagEmbedding\n"
        f"Original error: {e}"
    )


class BgeM3Embedder:
    """
    Uses BGE-M3 to generate dense embeddings.
    Returns: np.ndarray shape (N, dim) float32
    """
    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True, device: str | None = None):
        # BGEM3FlagModel picks device automatically if device=None
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)

        out = self.model.encode(
            texts,
            batch_size=16,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        dense = out["dense_vecs"]
        dense = np.array(dense, dtype=np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12
        dense = dense / norms
        return dense
