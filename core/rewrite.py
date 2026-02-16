# core/rewriter.py
from __future__ import annotations

import re
import requests
from typing import List, Dict, Optional

import config


def _extract_key_tokens(text: str) -> set[str]:
    """
    Extract important tokens that must be preserved in rewrite:
    - Pantone/PMS codes, numbers with units, percentages, etc.
    """
    tokens = set()

    # Pantone / PMS (e.g., Pantone 877, PMS 871)
    for m in re.findall(r"\b(?:pantone|pms)\s*\d+\b", text, flags=re.I):
        tokens.add(m.lower().replace(" ", ""))  # normalize

    # Standalone numbers (avoid too strict; keep main digits)
    for m in re.findall(r"\b\d+(?:\.\d+)?\b", text):
        tokens.add(m)

    # Units like pt, mm, cm, in, inch, %, lpi
    for m in re.findall(r"\b\d+(?:\.\d+)?\s*(pt|mm|cm|in|inch|inches|%|lpi)\b", text, flags=re.I):
        tokens.add((m[0] if isinstance(m, tuple) else m).lower())

    return tokens


def generate_response(
    user_msg: str,
    raw_answer: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Strict rewrite:
    - Only rephrase raw_answer (authoritative)
    - Must preserve key facts (numbers, codes)
    - If rewrite looks unsafe -> return raw_answer
    """

    if not raw_answer or not raw_answer.strip():
        return raw_answer

    system = (
        "You are a strict answer rewriter for a QA knowledge-base chatbot.\n"
        "Rewrite the provided raw answer to be clearer and shorter.\n"
        "Rules:\n"
        "- Do NOT add new facts.\n"
        "- Do NOT remove important facts.\n"
        "- Keep ALL numbers, sizes, and codes exactly (Pantone/PMS values, pt, mm, %, LPI, etc.).\n"
        "- If the raw answer contains any Pantone/PMS codes, they MUST appear in the rewrite.\n"
        "- No apologies. No extra suggestions. Just the rewritten answer.\n"
    )

    prompt = (
        f"User question:\n{user_msg}\n\n"
        f"Raw answer (authoritative, do not change facts):\n{raw_answer}\n\n"
        "Rewrite the raw answer clearly (same meaning):"
    )

    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
        },
    }

    try:
        r = requests.post(config.OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        rewritten = (data.get("response") or "").strip()

        if not rewritten:
            return raw_answer

        # Safety checks: reject if it becomes an apology / refusal / too long
        bad_starts = ("sorry", "i'm sorry", "i cannot", "i can't", "i don’t", "i don't")
        if rewritten.lower().startswith(bad_starts):
            return raw_answer

        if len(rewritten) > max(600, len(raw_answer) * 3):
            return raw_answer

        # Must preserve key tokens from raw answer
        must_keep = _extract_key_tokens(raw_answer)
        if must_keep:
            norm_rewritten = rewritten.lower().replace(" ", "")
            for tok in must_keep:
                # for pantone/pms tokens we normalized spaces out
                if tok.isdigit():
                    if tok not in rewritten:
                        return raw_answer
                else:
                    if tok.replace(" ", "") not in norm_rewritten and tok not in rewritten.lower():
                        return raw_answer

        return rewritten

    except Exception:
        # If Ollama not running / request fails, fallback to raw answer
        return raw_answer
