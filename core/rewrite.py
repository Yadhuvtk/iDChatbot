"""
Optional LLM rewriter using Ollama.
The LLM NEVER generates answers from scratch — it ONLY rewrites retrieved SOP answers.
If Ollama is unavailable, returns the verbatim answer as fallback.
"""
import requests
import json
import time

# Attempt to import config. 
# We use a try/except block to handle different running environments.
try:
    import config
except ImportError:
    # If running directly inside core/, we need to look up one level
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import config

# --- CONFIGURATION DEFAULTS ---
# Use values from config.py, or defaults if missing
OLLAMA_URL = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = getattr(config, "OLLAMA_MODEL", "mistral")
USE_LLM = getattr(config, "USE_LLM_REWRITE", True)

def generate_response(question: str, context: str, history: list = []) -> str:
    """
    Rewrites the retrieved text to answer the specific question naturally.
    
    Args:
        question (str): The user's input question.
        context (str): The raw text found in the SOP documents.
        history (list): Chat history (optional, currently unused for strict RAG).
        
    Returns:
        str: The rewritten answer OR the raw context if Ollama fails.
    """
    
    # 1. Check if LLM is disabled in config
    if not USE_LLM:
        return context

    # 2. Input Validation
    if not context or not context.strip():
        return "I am sorry, but I cannot find an answer to your question in the provided documents."

    # 3. Construct the Strict Prompt
    # We provide the Context and the Question so the LLM knows WHAT to answer.
    system_prompt = (
        "You are a helpful and strict assistant for Standard Operating Procedures (SOPs).\n"
        "Your goal is to answer the user's question using ONLY the context provided below.\n"
        "1. Do NOT add any outside information or general knowledge.\n"
        "2. If the answer is not in the context, say 'I don't know'.\n"
        "3. Keep the answer concise and professional."
    )
    
    user_prompt = f"""
    Context Information:
    {context}

    User Question:
    {question}

    Answer:
    """

    # 4. Call Ollama API
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Very low temperature = Factual/Deterministic
                "num_predict": 300   # Limit length
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        # We use the /api/chat endpoint for better prompting
        response = requests.post(
            f"{OLLAMA_URL}/api/chat", 
            json=payload, 
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            rewritten_text = result.get("message", {}).get("content", "").strip()
            
            # If LLM returns something valid, use it.
            if rewritten_text:
                return rewritten_text
        
        # If we got here, Ollama responded but empty. Fallback.
        print(f"⚠️  Ollama returned empty response. Using raw text.")
        return context

    except requests.exceptions.ConnectionError:
        # Ollama is probably not running.
        print(f"⚠️  Ollama not connected at {OLLAMA_URL}. Using raw text.")
        return context
    except Exception as e:
        print(f"⚠️  LLM Error: {e}. Using raw text.")
        return context

def check_status() -> bool:
    """Helper to check if Ollama is actually running."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return resp.status_code == 200
    except:
        return False