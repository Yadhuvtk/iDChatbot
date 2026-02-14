# app.py (FULL BACKEND FILE)
# ✅ BenChat SOPPipeline (FAISS + embed + rerank + typo normalization)
# ✅ Employee details ONLY when user explicitly uses "@"
# ✅ Gratitude replies (thanks/thank you/etc.) bypass KB + employee
# ✅ LLM rewrites answers but guarantees key facts (Pantone, numbers)
# ✅ Keeps tooltip keys: source, matched_question, raw_answer
# ✅ NEW: returns corrected_query + did_you_mean + corrections

import random
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.pipeline import SOPPipeline

# -----------------------------
# CONFIGURATION
# -----------------------------
BASE_DIR = Path(__file__).parent

EMPLOYEE_EXCEL_PATH = BASE_DIR / "data" / "Employe.xlsx"
EMPLOYEE_SHEET_NAME = "Sheet1"

MIN_SCORE = 0.55

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

# -----------------------------
# MEMORY STORAGE
# -----------------------------
CHAT_HISTORY: Dict[str, List[str]] = {}

# -----------------------------
# GLOBALS
# -----------------------------
EMP_DF: Optional[pd.DataFrame] = None
PIPELINE: Optional[SOPPipeline] = None

# -----------------------------
# HELPERS
# -----------------------------
def clean_text(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()

def safe_val(v: Any) -> str:
    if v is None:
        return ""
    try:
        if isinstance(v, float) and pd.isna(v):
            return ""
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()

def as_int_str(v: Any) -> str:
    s = safe_val(v)
    if not s:
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except Exception:
        return s

# -----------------------------
# GRATITUDE / SMALL TALK
# -----------------------------
THANK_KEYWORDS = {
    "thanks", "thank you", "thx", "tq", "thanku", "ty",
    "ok thanks", "thanks a lot", "thanks!", "cool thanks",
    "great thanks", "awesome thanks", "ok thank you"
}

def is_thank_you_message(msg: str) -> bool:
    m = (msg or "").lower().strip()
    return any(k in m for k in THANK_KEYWORDS)

def thank_you_response() -> str:
    return random.choice([
        "You're welcome! 😊",
        "Happy to help!",
        "Anytime! Let me know if you need anything else.",
        "Glad I could help 👍",
        "No problem at all!",
    ])

# -----------------------------
# LLM SAFETY
# -----------------------------
def contains_apology(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    bad_phrases = [
        "i am sorry", "i'm sorry", "i dont have", "i don't have", "no information",
        "cannot find", "can't find", "don't know", "do not know", "not in my database",
        "i couldn't find", "couldn't find"
    ]
    return any(p in t for p in bad_phrases)

def extract_key_tokens(reference_answer: str) -> List[str]:
    ans = (reference_answer or "").lower()
    keys: List[str] = []

    for m in re.findall(r"\b\d+(?:\.\d+)?\b", ans):
        if m not in keys:
            keys.append(m)

    for w in ["pantone", "pms", "pt", "inch", "inches"]:
        if w in ans and w not in keys:
            keys.append(w)

    return keys

def rewritten_keeps_key_facts(rewritten: str, reference_answer: str) -> bool:
    keys = extract_key_tokens(reference_answer)
    if not keys:
        return True
    r = (rewritten or "").lower()
    return all(k.lower() in r for k in keys)

def rewrite_answer_with_llm(reference_answer: str, user_query: str, history: List[str], intent_tag: str = "") -> str:
    reference_answer = (reference_answer or "").strip()
    if not reference_answer:
        return "I am sorry, I don't have that information."

    history_text = "\n".join(history[-6:])
    tag_hint = f'INTENT TAG: "{intent_tag}"\n' if intent_tag else ""

    prompt = f"""
You are a factual assistant.

Task:
Rewrite the ANSWER into a single clear sentence.

Inputs:
{tag_hint}QUESTION: "{user_query}"
ANSWER (source of truth): "{reference_answer}"
CHAT CONTEXT: {history_text}

STRICT RULES:
- Use ONLY the information in ANSWER.
- Keep all numbers/codes exactly (example: 877 must stay 877).
- Do NOT add new facts.
- Do NOT say you don't have information.
- Output ONLY the rewritten sentence.
""".strip()

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 80},
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=10)
        res = (r.json().get("response", "") or "").strip().strip('"').strip("'")

        if contains_apology(res):
            return reference_answer

        if len(res) > max(240, len(reference_answer) * 5):
            return reference_answer

        if not rewritten_keeps_key_facts(res, reference_answer):
            return reference_answer

        if res and res[-1] not in ".!?":
            res += "."

        return res if res else reference_answer
    except Exception:
        return reference_answer

# -----------------------------
# EMPLOYEE LOADING
# -----------------------------
def load_employees() -> Optional[pd.DataFrame]:
    if not EMPLOYEE_EXCEL_PATH.exists():
        print(f"WARNING: Employee file not found at {EMPLOYEE_EXCEL_PATH}")
        return None

    try:
        df = pd.read_excel(EMPLOYEE_EXCEL_PATH, sheet_name=EMPLOYEE_SHEET_NAME)
    except Exception as e:
        print(f"Error reading Employee Excel: {e}")
        return None

    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]

    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ["employeeid", "employee id", "empid", "emp id"]:
            col_map[c] = "EmployeeID"
        elif lc in ["name", "employee name"]:
            col_map[c] = "Name"
        elif lc in ["email", "mail", "e-mail"]:
            col_map[c] = "Email"
        elif lc in ["base salary", "basesalary", "salary", "base"]:
            col_map[c] = "Base Salary"
        elif lc in ["availableleave", "available leave", "leave balance", "balance leave"]:
            col_map[c] = "AvailableLeave"
        elif lc in ["under", "manager", "reporting to", "reports to"]:
            col_map[c] = "Under"
        elif lc in ["designation", "role", "title"]:
            col_map[c] = "Designation"

    df = df.rename(columns=col_map)

    for r in ["EmployeeID", "Name"]:
        if r not in df.columns:
            print(f"WARNING: Employee file missing required column: {r}")
            return None

    df = df.dropna(how="all").reset_index(drop=True)
    df["EmployeeID"] = df["EmployeeID"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()

    if "Email" in df.columns:
        df["Email"] = df["Email"].astype(str).str.strip()

    print(f"SUCCESS: Loaded {len(df)} employees.")
    return df

def build_employee_one_line(row: pd.Series) -> str:
    name = safe_val(row.get("Name"))
    designation = safe_val(row.get("Designation"))
    salary = as_int_str(row.get("Base Salary"))
    leave = as_int_str(row.get("AvailableLeave"))
    manager = safe_val(row.get("Under"))

    parts = []
    if name and designation:
        parts.append(f"{name} is a {designation}")
    elif name:
        parts.append(f"{name} is an employee")
    else:
        parts.append("This employee")

    if salary:
        parts.append(f"earning ₹{salary}")
    if leave:
        parts.append(f"with {leave} days of leave remaining")
    if manager:
        parts.append(f"reporting to {manager}")

    sentence = ", ".join(parts).strip()
    if not sentence.endswith("."):
        sentence += "."
    return sentence

def find_employee_by_name_best(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    q = re.sub(r"\s+", " ", name.strip()).lower()
    if not q:
        return None

    best = None
    best_score = 0
    for _, row in df.iterrows():
        nm = str(row.get("Name", "")).strip().lower()
        if not nm:
            continue
        score = 0
        for token in q.split():
            if token in nm:
                score += 1
        if score > best_score:
            best_score = score
            best = row
    return best if best_score > 0 else None

def handle_employee_query_only_at(df: pd.DataFrame, user_msg: str) -> Optional[Dict[str, Any]]:
    msg = clean_text(user_msg)
    if not msg.startswith("@"):
        return None

    name = msg[1:].strip()
    if not name:
        return {
            "content": "Please type an employee name after @ (example: @Muhammed Aneesh).",
            "match_type": "employee",
        }

    row = find_employee_by_name_best(df, name)
    if row is None:
        return {
            "content": "I couldn't find that employee. Please check the name or use the @ dropdown.",
            "match_type": "employee_not_found",
        }

    return {
        "content": build_employee_one_line(row),
        "match_type": "employee",
        "employee_id": safe_val(row.get("EmployeeID")),
        "name": safe_val(row.get("Name")),
    }

# -----------------------------
# API SETUP
# -----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatIn(BaseModel):
    message: str
    session_id: str = "default"

@app.on_event("startup")
def startup_event():
    global EMP_DF, PIPELINE

    EMP_DF = load_employees()

    # ✅ Pipeline now includes typo/punctuation normalization internally
    PIPELINE = SOPPipeline(accept_score=MIN_SCORE).load()

    print("✅ Startup complete.")

@app.post("/chat")
def chat(payload: ChatIn) -> Dict[str, Any]:
    user_msg = payload.message.strip()
    session_id = payload.session_id

    if not user_msg:
        return {"content": "Hello! How can I help?", "match_type": "none"}

    if session_id not in CHAT_HISTORY:
        CHAT_HISTORY[session_id] = []
    current_history = CHAT_HISTORY[session_id]

    # ✅ 0) THANKS handler
    if is_thank_you_message(user_msg):
        reply = thank_you_response()
        CHAT_HISTORY[session_id].append(f"User: {user_msg}")
        CHAT_HISTORY[session_id].append(f"AI: {reply}")
        return {"content": reply, "match_type": "small_talk"}

    # ✅ 1) Employee ONLY when "@"
    if EMP_DF is not None:
        emp_res = handle_employee_query_only_at(EMP_DF, user_msg)
        if emp_res is not None:
            CHAT_HISTORY[session_id].append(f"User: {user_msg}")
            CHAT_HISTORY[session_id].append(f"AI: {emp_res.get('content','')}")
            return emp_res

    # ✅ 2) SOP Pipeline
    if not PIPELINE:
        return {"content": "System Error: SOP pipeline not loaded.", "match_type": "error"}

    result = PIPELINE.query(user_msg)

    if result.get("rejected", False):
        return {
            "content": "I am sorry, I don't have information about that in my database.",
            "match_type": "none",
            "score": result.get("score", 0.0),
            "suggestions": [],
            # typo info
            "normalized_query": result.get("normalized_query", ""),
            "corrected_query": result.get("corrected_query", ""),
            "did_you_mean": result.get("did_you_mean", None),
            "corrections": result.get("corrections", []),
        }

    raw_answer = (result.get("answer", "") or "").strip()

    final_response = rewrite_answer_with_llm(
        reference_answer=raw_answer,
        user_query=user_msg,
        history=current_history,
        intent_tag=result.get("tag", "")
    )

    CHAT_HISTORY[session_id].append(f"User: {user_msg}")
    CHAT_HISTORY[session_id].append(f"AI: {final_response}")

    return {
        "content": final_response,
        "match_type": "intent",
        "subject": result.get("tag", ""),
        "score": result.get("score", 0.0),
        "source": result.get("tag", ""),
        "matched_question": result.get("matched_question", ""),
        "raw_answer": raw_answer,

        # ✅ NEW: typo fix info (optional to show / log)
        "normalized_query": result.get("normalized_query", ""),
        "corrected_query": result.get("corrected_query", ""),
        "did_you_mean": result.get("did_you_mean", None),
        "corrections": result.get("corrections", []),
    }

@app.get("/employees")
def employees(q: str = Query(default=""), limit: int = Query(default=30, ge=1, le=200)):
    global EMP_DF
    if EMP_DF is None or EMP_DF.empty:
        return {"ok": True, "data": []}

    df = EMP_DF.copy()
    for col in ["EmployeeID", "Name", "Email"]:
        if col not in df.columns:
            df[col] = ""

    q_norm = (q or "").strip().lower()
    if q_norm:
        s = (
            df["EmployeeID"].astype(str).str.lower().fillna("") + " " +
            df["Name"].astype(str).str.lower().fillna("") + " " +
            df["Email"].astype(str).str.lower().fillna("")
        )
        df = df[s.str.contains(re.escape(q_norm), na=False)]

    df = df.head(limit)

    out = []
    for _, r in df.iterrows():
        out.append({
            "employee_id": safe_val(r.get("EmployeeID")),
            "name": safe_val(r.get("Name")),
            "email": safe_val(r.get("Email")),
        })

    return {"ok": True, "data": out}

# Run:
# & 'e:\Yadhu Projects\Chatbot\runtime\python.exe' -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
