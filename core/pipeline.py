# core/pipeline.py  (FAISS-only version)
# ✅ Dense retrieval (BGE-M3) + FAISS cosine search
# ✅ Cross-encoder reranker
# ✅ Accept / Reject threshold
# ✅ Returns: answer + score + matched_question

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.bge_m3_embed import BgeM3Embedder
from core.index import FaissIndex
from core.reranker import CrossEncoderReranker


@dataclass
class SOPItem:
    id: str
    tag: str
    question: str
    answer: str
    source_id: str = ""


class SOPPipeline:
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        index_name: str = "sop_index",
        top_k_dense: int = 12,
        top_k_rerank: int = 5,
        accept_score: float = 0.55,   # you can tune (0.50–0.65)
    ):
        self.base_dir = base_dir or Path(__file__).resolve().parents[1]
        self.data_dir = data_dir or (self.base_dir / "data")

        self.index_name = index_name
        self.top_k_dense = top_k_dense
        self.top_k_rerank = top_k_rerank
        self.accept_score = accept_score

        self.sop_json_path = self.data_dir / "converted.json"

        self.faiss_path = self.data_dir / f"{self.index_name}.faiss"
        self.map_path = self.data_dir / f"{self.index_name}_map.json"

        self.embedder: Optional[BgeM3Embedder] = None
        self.index: Optional[FaissIndex] = None
        self.reranker: Optional[CrossEncoderReranker] = None
        self.items: List[SOPItem] = []

    # -----------------------------
    # LOAD / BUILD
    # -----------------------------
    def load(self) -> "SOPPipeline":
        """Load SOP data + FAISS index + models."""
        self._load_items()
        self.embedder = BgeM3Embedder()
        self.index = FaissIndex.load(self.faiss_path, self.map_path)
        self.reranker = CrossEncoderReranker()
        return self

    def build(self) -> None:
        """Build FAISS index from sop_data.json (run one-time or after data update)."""
        self._load_items()
        self.embedder = BgeM3Embedder()

        texts = [it.question for it in self.items]
        vecs = self.embedder.embed(texts)  # shape: (N, dim)

        self.index = FaissIndex.build(vecs, [it.id for it in self.items])
        self.index.save(self.faiss_path, self.map_path)
        print(f"✅ Built FAISS index: {self.faiss_path}  (items={len(self.items)})")

    # -----------------------------
    # QUERY
    # -----------------------------
    def query(self, user_text: str) -> Dict[str, Any]:
        if not self.index or not self.embedder or not self.reranker:
            raise RuntimeError("Pipeline not loaded. Call SOPPipeline().load() first.")

        qvec = self.embedder.embed([user_text])[0]

        # Dense retrieval
        hits = self.index.search(qvec, top_k=self.top_k_dense)  # [(id, score), ...]
        if not hits:
            return {"rejected": True, "score": 0.0}

        # Map ids -> items
        cand_items: List[SOPItem] = []
        for sid, _ in hits:
            it = self._get_item_by_id(sid)
            if it:
                cand_items.append(it)

        if not cand_items:
            return {"rejected": True, "score": 0.0}

        # Rerank candidates by cross-encoder
        pairs = [(user_text, it.question) for it in cand_items]
        rr_scores = self.reranker.score(pairs)

        # pick best
        best_idx = int(max(range(len(rr_scores)), key=lambda i: rr_scores[i]))
        best_item = cand_items[best_idx]
        best_score = float(rr_scores[best_idx])

        rejected = best_score < float(self.accept_score)

        return {
            "rejected": rejected,
            "score": best_score,
            "tag": best_item.tag,
            "matched_question": best_item.question,
            "answer": best_item.answer,
            "id": best_item.id,
            "source_id": best_item.source_id,
        }

    # -----------------------------
    # INTERNAL
    # -----------------------------
    def _load_items(self) -> None:
        if not self.sop_json_path.exists():
            raise FileNotFoundError(f"sop_data.json not found: {self.sop_json_path}")

        data = json.loads(self.sop_json_path.read_text(encoding="utf-8"))
        items: List[SOPItem] = []
        for row in data:
            items.append(
                SOPItem(
                    id=str(row.get("id", "")).strip(),
                    tag=str(row.get("tag", "")).strip(),
                    question=str(row.get("question", "")).strip(),
                    answer=str(row.get("answer", "")).strip(),
                    source_id=str(row.get("source_id", "")).strip(),
                )
            )
        self.items = [it for it in items if it.id and it.question and it.answer]
        if not self.items:
            raise ValueError("sop_data.json loaded but no valid items found.")

    def _get_item_by_id(self, sid: str) -> Optional[SOPItem]:
        # For speed you can build a dict map; for now linear is OK for small/medium KB.
        for it in self.items:
            if it.id == sid:
                return it
        return None
