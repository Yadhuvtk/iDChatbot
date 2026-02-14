"""
Optional LLM rewriter using Ollama.
The LLM NEVER generates answers — it ONLY rewrites/reformats retrieved SOP answers.
If Ollama is unavailable, returns the verbatim answer as fallback.
"""
import requests
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


REWRITE_PROMPT = """You are an SOP assistant.
Rewrite the following answer clearly and concisely.
Do NOT add, remove, or assume any information.
If the answer is empty, reply: "This is not in the SOP."

Answer:
{retrieved_answer}"""


def rewrite_answer(answer: str) -> str:
    """
    Use Ollama to rewrite an SOP answer for clarity.
    The LLM does NOT generate new content — only reformats.

    Falls back to verbatim answer if Ollama is unavailable.
    """
    if not config.USE_LLM_REWRITE:
        return answer

    if not answer or answer.strip() == "":
        return config.REJECTION_MESSAGE

    try:
        prompt = REWRITE_PROMPT.format(retrieved_answer=answer)

        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,  # Deterministic — no creativity
                    "top_p": 1.0,
                    "num_predict": 200,  # Short answers only
                },
            },
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            rewritten = result.get("response", "").strip()
            if rewritten:
                return rewritten

        # Fallback to verbatim
        print("⚠️  Ollama returned empty response, using verbatim answer")
        return answer

    except requests.exceptions.ConnectionError:
        print("⚠️  Ollama not available, using verbatim answer")
        return answer
    except Exception as e:
        print(f"⚠️  Rewrite error: {e}, using verbatim answer")
        return answer


def check_ollama_status() -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            model_available = any(config.OLLAMA_MODEL in m for m in models)
            return model_available
        return False
    except Exception:
        return False