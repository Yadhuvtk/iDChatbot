# core/index.py
# Simple FAISS cosine index with id mapping

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import faiss
except Exception as e:
    raise ImportError(
        "faiss is not installed. Install with:\n"
        "  <runtime_python> -m pip install -U faiss-cpu\n"
        f"Original error: {e}"
    )


@dataclass
class FaissIndex:
    index: "faiss.Index"
    id_map: List[str]

    @staticmethod
    def build(vectors: np.ndarray, ids: List[str]) -> "FaissIndex":
        """
        vectors: (N, dim) float32, already L2-normalized for cosine.
        ids:     list of string ids (len N)
        """
        if vectors is None or len(vectors) == 0:
            raise ValueError("No vectors provided to build index.")
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids length must match vectors rows.")

        vecs = np.asarray(vectors, dtype=np.float32)
        dim = int(vecs.shape[1])

        # Cosine similarity via inner product on normalized vectors
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)

        return FaissIndex(index=index, id_map=list(ids))

    def save(self, faiss_path: Path, map_path: Path) -> None:
        faiss_path = Path(faiss_path)
        map_path = Path(map_path)
        faiss_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(faiss_path))
        map_path.write_text(json.dumps(self.id_map, indent=2), encoding="utf-8")

    @staticmethod
    def load(faiss_path: Path, map_path: Path) -> "FaissIndex":
        faiss_path = Path(faiss_path)
        map_path = Path(map_path)

        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS file not found: {faiss_path}")
        if not map_path.exists():
            raise FileNotFoundError(f"ID map file not found: {map_path}")

        index = faiss.read_index(str(faiss_path))
        id_map = json.loads(map_path.read_text(encoding="utf-8"))
        return FaissIndex(index=index, id_map=id_map)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        query_vec: (dim,) float32, already L2-normalized
        returns: [(id, score), ...] (score is cosine similarity)
        """
        if top_k <= 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)

        scores, idxs = self.index.search(q, top_k)

        out: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            if idx >= len(self.id_map):
                continue
            out.append((self.id_map[idx], float(score)))
        return out
