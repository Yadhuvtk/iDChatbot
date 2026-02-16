# core/pipeline.py
from __future__ import annotations

import json
import re
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- BEN-CHAT INTEGRATION IMPORTS ---
import config
from core import rewriter
# ------------------------------------

from core.bge import BgeM3Embedder
from core.index import FaissIndex
from core.reranker import CrossEncoderReranker

from rapidfuzz import fuzz, process


@dataclass
class SOPItem:
    id: str
    tag: str
    question: str
    answer: str
    source_id: str = ""


# -------------------------
# TEXT NORMALIZATION (KEPT AS IS)
# -------------------------
def _basic_clean(text: str) -> str:
    t = (text or "").strip().lower()
    keep = "@"
    punct = "".join([p for p in string.punctuation if p not in keep])
    t = t.translate(str.maketrans({c: " " for c in punct}))
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [x for x in text.split(" ") if x]


def _build_vocab_from_items(items: List[SOPItem]) -> List[str]:
    vocab = set()
    def add_tokens(s: str):
        s2 = _basic_clean(s)
        for tok in _tokenize(s2):
            vocab.add(tok)
    for it in items:
        add_tokens(it.question)
        add_tokens(it.answer)
        add_tokens(it.tag)
    vocab.update(["pms", "pantone", "pt", "pts", "inch", "inches", "mm", "cm", "dpi", "rgb", "cmyk"])
    return sorted(vocab)


def _merge_split_words(tokens: List[str], vocab_set: set) -> List[str]:
    out = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            merged = tokens[i] + tokens[i + 1]
            if merged in vocab_set:
                out.append(merged)
                i += 2
                continue
        out.append(tokens[i])
        i += 1
    return out


def _split_merged_words(tokens: List[str], vocab_set: set) -> List[str]:
    out = []
    for t in tokens:
        if t in vocab_set:
            out.append(t)
            continue
        split_done = False
        for k in range(2, len(t) - 2):
            a = t[:k]
            b = t[k:]
            if a in vocab_set and b in vocab_set:
                out.extend([a, b])
                split_done = True
                break
        if not split_done:
            out.append(t)
    return out


# -------------------------
# INTENT / KEYWORD BOOST (KEPT AS IS)
# -------------------------
SPEC_TRIGGERS = {
    "spec", "specs", "specification", "specifications",
    "minimum", "min", "minimun",
    "stroke", "line", "lineweight", "thickness",
    "font", "fontsize", "text",
    "pt", "pts",
}
SPEC_SIGNAL_WORDS = {
    "spec", "specs", "specification", "minimum", "stroke", "thickness",
    "font", "text", "pt", "pts", "positive", "negative",
    "sans", "serif",
}
OFFTOPIC_FOR_SPEC = {
    "rgb", "cmyk", "device", "dependent", "color shift", "shift", "monitor",
}


def _contains_any(text: str, words: set) -> bool:
    t = _basic_clean(text)
    for w in words:
        if w in t:
            return True
    return False


def _keyword_overlap_score(query: str, candidate_text: str) -> float:
    q = _basic_clean(query)
    c = _basic_clean(candidate_text)
    q_tokens = [t for t in _tokenize(q) if len(t) >= 3]
    if not q_tokens:
        return 0.0
    hit = 0
    for tok in q_tokens:
        if tok in c:
            hit += 1
    return hit / max(1, len(q_tokens))


# -------------------------
# PIPELINE
# -------------------------
class SOPPipeline:
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        index_name: str = "sop_index",
        top_k_dense: int = 14,
        top_k_rerank: int = 6,
        accept_score: float = 0.55,
        typo_threshold: int = 88,
        max_token_len: int = 35,
        w_rerank: float = 1.0,
        w_overlap: float = 0.25,
        w_spec_boost: float = 0.25,
        w_offtopic_penalty: float = 0.35,
    ):
        self.base_dir = base_dir or Path(__file__).resolve().parents[1]
        self.data_dir = data_dir or (self.base_dir / "data")
        self.index_name = index_name
        self.top_k_dense = top_k_dense
        self.top_k_rerank = top_k_rerank
        self.accept_score = float(accept_score)
        self.typo_threshold = int(typo_threshold)
        self.max_token_len = int(max_token_len)
        self.w_rerank = float(w_rerank)
        self.w_overlap = float(w_overlap)
        self.w_spec_boost = float(w_spec_boost)
        self.w_offtopic_penalty = float(w_offtopic_penalty)

        self.sop_json_path = self.data_dir / "converted.json"
        self.faiss_path = self.data_dir / f"{self.index_name}.faiss"
        self.map_path = self.data_dir / f"{self.index_name}_map.json"

        self.embedder: Optional[BgeM3Embedder] = None
        self.index: Optional[FaissIndex] = None
        self.reranker: Optional[CrossEncoderReranker] = None

        self.items: List[SOPItem] = []
        self.vocab: List[str] = []
        self.vocab_set: set = set()
        self._fix_cache: Dict[str, str] = {}
        self._id_map: Dict[str, SOPItem] = {}

    def load(self) -> "SOPPipeline":
        self._load_items()
        self.vocab = _build_vocab_from_items(self.items)
        self.vocab_set = set(self.vocab)
        self.embedder = BgeM3Embedder()
        self.index = FaissIndex.load(self.faiss_path, self.map_path)
        self.reranker = CrossEncoderReranker()
        return self

    def build(self) -> None:
        self._load_items()
        self.vocab = _build_vocab_from_items(self.items)
        self.vocab_set = set(self.vocab)
        self.embedder = BgeM3Embedder()
        texts = [it.question for it in self.items]
        vecs = self.embedder.embed(texts)
        self.index = FaissIndex.build(vecs, [it.id for it in self.items])
        self.index.save(self.faiss_path, self.map_path)
        print(f"✅ Built FAISS index: {self.faiss_path} (items={len(self.items)})")

    def fix_token(self, tok: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        t = (tok or "").strip().lower()
        if not t: return "", None
        if t in self._fix_cache:
            fixed = self._fix_cache[t]
            return fixed, ({"from": t, "to": fixed, "score": 100} if fixed != t else None)
        
        if re.fullmatch(r"\d+(\.\d+)?", t) or len(t) > self.max_token_len or t in self.vocab_set:
            self._fix_cache[t] = t
            return t, None

        match = process.extractOne(t, self.vocab, scorer=fuzz.WRatio)
        if match:
            best_word, best_score, _ = match
            if int(best_score) >= self.typo_threshold:
                self._fix_cache[t] = best_word
                return best_word, {"from": t, "to": best_word, "score": int(best_score)}

        self._fix_cache[t] = t
        return t, None

    def normalize_query(self, user_text: str) -> Dict[str, Any]:
        original = (user_text or "").strip()
        if original.startswith("@"):
            return {"normalized_query": original, "corrected_query": original, "did_you_mean": None, "corrections": []}

        normalized = _basic_clean(original)
        toks = _tokenize(normalized)
        toks = _split_merged_words(toks, self.vocab_set)
        toks = _merge_split_words(toks, self.vocab_set)

        corrections = []
        fixed_tokens = []
        for t in toks:
            fixed, meta = self.fix_token(t)
            fixed_tokens.append(fixed)
            if meta: corrections.append(meta)

        corrected = " ".join([x for x in fixed_tokens if x]).strip()
        did_you_mean = corrected if corrected and corrected != normalized else None
        return {"normalized_query": normalized, "corrected_query": corrected or normalized, "did_you_mean": did_you_mean, "corrections": corrections}

    # -----------------------------
    # EDITED QUERY FUNCTION
    # -----------------------------
    def query(self, user_text: str) -> Dict[str, Any]:
        """
        Executes the search pipeline with Ben-Chat's Zero Hallucination Logic.
        """
        start_time = time.time()
        
        if not self.index or not self.embedder or not self.reranker:
            raise RuntimeError("Pipeline not loaded. Call SOPPipeline().load() first.")

        norm = self.normalize_query(user_text)
        qtext = norm["corrected_query"]
        qclean = _basic_clean(qtext)

        ask_spec = _contains_any(qclean, SPEC_TRIGGERS)

        # 1. Dense Retrieval
        qvec = self.embedder.embed([qtext])[0]
        hits = self.index.search(qvec, top_k=self.top_k_dense)
        
        # --- EARLY REJECTION CHECK ---
        if not hits:
            return self._build_rejection_response(norm, 0.0)

        cand_items: List[SOPItem] = []
        for sid, _ in hits:
            it = self._get_item_by_id(sid)
            if it: cand_items.append(it)

        if not cand_items:
            return self._build_rejection_response(norm, 0.0)

        # 2. Rerank
        cand_items = cand_items[: max(1, self.top_k_rerank)]
        pairs = [(qtext, it.question) for it in cand_items]
        rr_scores = self.reranker.score(pairs)

        # 3. Boost Scores
        best_item = None
        best_final = -1e9
        best_rr = 0.0
        best_dbg = {}

        for i, it in enumerate(cand_items):
            rr = float(rr_scores[i])
            cand_text = f"{it.tag} {it.question} {it.answer}"
            overlap = _keyword_overlap_score(qtext, cand_text)

            spec_boost = 0.0
            off_penalty = 0.0

            if ask_spec:
                if _contains_any(cand_text, SPEC_SIGNAL_WORDS): spec_boost = 1.0
                if _contains_any(cand_text, OFFTOPIC_FOR_SPEC): off_penalty = 1.0

            final_score = (
                self.w_rerank * rr
                + self.w_overlap * overlap
                + self.w_spec_boost * spec_boost
                - self.w_offtopic_penalty * off_penalty
            )

            if final_score > best_final:
                best_final = final_score
                best_item = it
                best_rr = rr
                best_dbg = {
                    "rerank": rr,
                    "overlap": overlap,
                    "ask_spec": ask_spec,
                    "spec_boost": spec_boost,
                    "offtopic_penalty": off_penalty,
                    "final_score": final_score,
                }

        # --- FINAL BEN-CHAT LOGIC ---
        
        # 4. Strict Hallucination Check
        if not best_item or best_rr < float(self.accept_score):
            return self._build_rejection_response(norm, best_rr)

        # 5. LLM Rewriting (The "Human" Touch)
        raw_answer = best_item.answer
        final_response = raw_answer # Default to raw text

        if config.USE_LLM_REWRITE:
            # We pass the question and the found fact to the LLM
            # The LLM is instructed ONLY to rewrite, not invent.
            final_response = rewriter.generate_response(qtext, raw_answer)

        return {
            "response": final_response, # The main answer for the UI
            "rejected": False,
            "score": best_rr,
            "final_score": best_final,
            "tag": best_item.tag,
            "matched_question": best_item.question,
            "source_text": raw_answer, # Original text for debugging
            "id": best_item.id,
            "source_id": best_item.source_id,
            "debug": best_dbg,
            "time": time.time() - start_time,
            **norm,
        }

    def _build_rejection_response(self, norm_data, score):
        """Helper to return a standardized rejection"""
        return {
            "response": config.FALLBACK_MESSAGE,
            "rejected": True,
            "score": score,
            "final_score": 0.0,
            **norm_data
        }

    def _load_items(self) -> None:
        if not self.sop_json_path.exists():
            raise FileNotFoundError(f"converted.json not found: {self.sop_json_path}")
        data = json.loads(self.sop_json_path.read_text(encoding="utf-8"))
        items = [SOPItem(id=str(r.get("id","")), tag=str(r.get("tag","")), question=str(r.get("question","")), answer=str(r.get("answer","")), source_id=str(r.get("source_id",""))) for r in data]
        self.items = [it for it in items if it.id and it.question and it.answer]
        if not self.items: raise ValueError("No valid items found.")
        self._id_map = {it.id: it for it in self.items}

    def _get_item_by_id(self, sid: str) -> Optional[SOPItem]:
        return self._id_map.get(sid)