# app.py (FULL BACKEND FILE)
# ✅ BenChat SOPPipeline (FAISS + embed + rerank + typo normalization)
# ✅ Employee details ONLY when user explicitly uses "@"
# ✅ Gratitude replies bypass KB + employee
# ✅ Frustration replies bypass KB + employee (fixed: runs BEFORE pipeline)
# ✅ Keyword-only guard: if user types just "screenprint" / single keyword -> ask clarification (NO KB answer)
# ✅ LLM rewrites ALWAYS, but NEVER allowed to add facts
# ✅ If pipeline truth answer is missing, uses deterministic fallback truth text
# ✅ Keeps tooltip keys: source, matched_question, raw_answer
# ✅ Returns normalized_query + corrected_query + did_you_mean + corrections
# ✅ Fixes indentation + unreachable code from your pasted file

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

# Your original MIN_SCORE was 0.25 (very low).
# Keeping it as-is, because you requested "full code", not tuning.
MIN_SCORE = 0.25

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
# GRATITUDE / SMALL TALK (SAFE)
# -----------------------------
THANK_PATTERNS = [
    r"\bthanks\b",
    r"\bthank\s*you\b",
    r"\bthx\b",
    r"\btq\b",
    r"\bthanku\b",
    r"\bty\b",
    r"\bok\s+thanks\b",
    r"\bthanks\s+a\s+lot\b",
    r"\bgreat\s+thanks\b",
    r"\bawesome\s+thanks\b",
    r"\bcool\s+thanks\b",
]
_thank_re = re.compile("|".join(THANK_PATTERNS), re.IGNORECASE)


def is_thank_you_message(msg: str) -> bool:
    m = (msg or "").strip().lower()
    m = re.sub(r"[^a-z0-9\s]", " ", m)
    m = re.sub(r"\s+", " ", m).strip()
    return bool(_thank_re.search(m))


def thank_you_response() -> str:
    return random.choice(
        [
            "You're welcome! 😊",
            "Happy to help!",
            "Anytime! Let me know if you need anything else.",
            "Glad I could help 👍",
            "No problem at all!",
        ]
    )


# -----------------------------
# FRUSTRATION / ANNOYED MESSAGES
# -----------------------------
FRUSTRATION_PATTERNS = [
    r"\bsorry\b",
    r"\bsoory\b",
    r"\bsry\b",
    r"\bnot\s*working\b",
    r"\bdoesn[’']?t\s*work\b",
    r"\bwrong\b",
    r"\bincorrect\b",
    r"\bbad\b",
    r"\buseless\b",
    r"\bwaste\b",
    r"\bfrustrat(ed|ing)\b",
    r"\bangry\b",
    r"\birritat(ed|ing)\b",
    r"\bwhat\s+the\s+hell\b",
    r"\bwtf\b",
]
_frustration_re = re.compile("|".join(FRUSTRATION_PATTERNS), re.IGNORECASE)


def is_frustrated_message(msg: str) -> bool:
    m = (msg or "").strip().lower()
    m = re.sub(r"[^a-z0-9\s]", " ", m)
    m = re.sub(r"\s+", " ", m).strip()
    return bool(_frustration_re.search(m))


def frustrated_response() -> str:
    return "Thank you for your patience. I’m still training. Please enter your review."




# -----------------------------
# GREETINGS / HELLO (BYPASS KB)
# -----------------------------
GREET_PATTERNS = [
    r"\bhi\b",
    r"\bhello\b",
    r"\bhey\b",
    r"\bhai\b",
    r"\bheya\b",
    r"\byo\b",
    r"\bgm\b",
    r"\bgood\s*morning\b",
    r"\bgn\b",
    r"\bgood\s*night\b",
    r"\bga\b",
    r"\bgood\s*afternoon\b",
    r"\bge\b",
    r"\bgood\s*evening\b",
    r"\bhow\s*are\s*you\b",
    r"\bhow\s*r\s*u\b",
    r"\bhow\s*are\s*u\b",
    r"\bsup\b",
    r"\bwhat'?s\s*up\b",
]
_greet_re = re.compile("|".join(GREET_PATTERNS), re.IGNORECASE)

def normalize_for_greeting(msg: str) -> str:
    m = (msg or "").strip().lower()
    # remove punctuation/emojis, keep words
    m = re.sub(r"[^a-z0-9\s]", " ", m)
    m = re.sub(r"\s+", " ", m).strip()
    return m

def is_greeting_message(msg: str) -> bool:
    m = normalize_for_greeting(msg)
    if not m:
        return False

    # If message is mostly greeting words and very short, treat as greeting
    tokens = m.split()
    if len(tokens) <= 4:
        return bool(_greet_re.search(m))

    # Also allow greeting at start like "hi benchat"
    if tokens and tokens[0] in {"hi", "hello", "hey", "hai"}:
        return True

    return bool(_greet_re.search(m)) and len(tokens) <= 7

def greeting_response() -> str:
    return random.choice([
        "Hey! 👋 I’m iD Chat AI.",
        "Hello! 😊 ",
        "Hi there! 👋 What can I help you with today?",
        "Hey! ",
        "Hello! ",
    ])




# -----------------------------
# KEYWORD-ONLY GUARD (YOUR REQUEST)
# -----------------------------
QUESTION_HINTS = {
    "spec",
    "specs",
    "size",
    "minimum",
    "min",
    "font",
    "pt",
    "point",
    "stroke",
    "line",
    "thickness",
    "pms",
    "pantone",
    "color",
    "area",
    "imprint",
    "method",
    "screenprint",
    "screen",
    "print",
    "embroidery",
    "engrave",
    "laser",
    "dtf",
    "dtg",
}


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", (text or "").lower().strip())


def is_keyword_only(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    tokens = tokenize(t)
    # 1 token like "screenprint" -> keyword only
    if len(tokens) <= 1:
        return True
    return False


def needs_clarification(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    tokens = set(tokenize(t))
    # 1 token always clarify
    if len(tokens) <= 1:
        return True
    # short + no "?" + no hints => clarify
    if len(t) < 12 and "?" not in t and len(tokens.intersection(QUESTION_HINTS)) == 0:
        return True
    return False


def clarification_message(keyword: str) -> str:
    kw = (keyword or "").strip()
    return (
        f"I got **{kw}**. What exactly do you need?\n\n"
        "Try one of these:\n"
        "What is the spec for screenprint\n"
        "What is screenprint?\n"
    )


# -----------------------------
# LLM SAFETY
# -----------------------------
def contains_apology(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    bad_phrases = [
        "i am sorry",
        "i'm sorry",
        "i dont have",
        "i don't have",
        "no information",
        "cannot find",
        "can't find",
        "don't know",
        "do not know",
        "not in my database",
        "i couldn't find",
        "couldn't find",
    ]
    return any(p in t for p in bad_phrases)


def extract_key_tokens(reference_answer: str) -> List[str]:
    ans = (reference_answer or "").lower()
    keys: List[str] = []

    # numbers like 877 / 1.5 etc
    for m in re.findall(r"\b\d+(?:\.\d+)?\b", ans):
        if m not in keys:
            keys.append(m)

    # important keywords
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


def choose_truth_text(result: Dict[str, Any]) -> str:
    """
    Always return a NON-EMPTY truth string.
    Priority:
      1) answer / raw_answer / response / content
      2) deterministic fallback (no hallucination)
    """
    candidates = [
        result.get("answer"),
        result.get("raw_answer"),
        result.get("response"),
        result.get("content"),
    ]
    for c in candidates:
        t = (c or "").strip()
        if t:
            return t

    mq = (result.get("matched_question") or "").strip()
    tag = (result.get("tag") or "").strip()

    if mq:
        return f"I could not find a stored answer for: {mq}"
    if tag:
        return f"I could not find a stored answer in category: {tag}"
    return "I could not find a stored answer in the database."


def rewrite_answer_with_llm(
    reference_answer: str, user_query: str, history: List[str], intent_tag: str = ""
) -> str:
    # ✅ ALWAYS rewrite (never return apology here)
    reference_answer = (reference_answer or "").strip()
    if not reference_answer:
        reference_answer = "I could not find a stored answer in the database."

    history_text = "\n".join(history[-6:])
    tag_hint = f'INTENT TAG: "{intent_tag}"\n' if intent_tag else ""

    prompt = f"""
You are a factual assistant.

Task:
Rewrite the ANSWER into a single clear sentence.

Inputs:
{tag_hint}QUESTION: "{user_query}"
ANSWER (source of truth): "{reference_answer}"
CHAT CONTEXT:
{history_text}

STRICT RULES:
- Use ONLY the information in ANSWER.
- Keep all numbers/codes exactly (example: 877 must stay 877).
- Do NOT add new facts.
- Output ONLY the rewritten sentence.
""".strip()

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 80},
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=15)
        res = (r.json().get("response", "") or "").strip().strip('"').strip("'")

        # If LLM gives apology / empty -> fallback to truth answer
        if contains_apology(res):
            res = reference_answer

        # Too long -> fallback
        if len(res) > max(240, len(reference_answer) * 5):
            res = reference_answer

        # Must keep key facts
        if not rewritten_keeps_key_facts(res, reference_answer):
            res = reference_answer

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

    PIPELINE = SOPPipeline(accept_score=MIN_SCORE).load()

    print("✅ Startup complete.")
    print("BASE_DIR:", BASE_DIR)
    print("HAS converted.json:", (BASE_DIR / "data" / "converted.json").exists())
    print("HAS sop_index.faiss:", (BASE_DIR / "data" / "sop_index.faiss").exists())


@app.post("/chat")
def chat(payload: ChatIn) -> Dict[str, Any]:
    user_msg = (payload.message or "").strip()
    session_id = (payload.session_id or "default").strip()

    if not user_msg:
        return {"content": "Hello! How can I help?", "match_type": "none"}

    # init history
    if session_id not in CHAT_HISTORY:
        CHAT_HISTORY[session_id] = []
    current_history = CHAT_HISTORY[session_id]

    # store user line (for rewrite context)
    current_history.append(f"User: {user_msg}")

    # 0) THANKS handler (bypass everything)
    if is_thank_you_message(user_msg):
        reply = thank_you_response()
        current_history.append(f"AI: {reply}")
        return {"content": reply, "match_type": "small_talk"}
    
        # 0.25) GREETING handler (bypass everything)
    if is_greeting_message(user_msg):
        reply = greeting_response()
        current_history.append(f"AI: {reply}")
        return {"content": reply, "match_type": "small_talk"}
    
    # 0.5) FRUSTRATION handler (bypass everything)
    if is_frustrated_message(user_msg):
        reply = frustrated_response()
        current_history.append(f"AI: {reply}")
        return {
            "content": reply,
            "match_type": "intent",  # keep "intent" so frontend shows it normally
            "subject": "feedback_request",
            "score": 1.0,
            "source": "feedback_request",
            "matched_question": user_msg,
            "raw_answer": reply,
        }

    # ✅ Keyword-only guard (YOUR MAIN REQUIREMENT)
    # If user types just a keyword like "screenprint" -> do NOT answer from KB.
    if is_keyword_only(user_msg) or needs_clarification(user_msg):
        reply = clarification_message(user_msg)
        current_history.append(f"AI: {reply}")
        return {
            "content": reply,
            "match_type": "none",
            "score": 0.0,
            "source": "clarification",
            "matched_question": "",
            "raw_answer": reply,
            "normalized_query": user_msg,
            "corrected_query": user_msg,
            "did_you_mean": None,
            "corrections": [],
        }

    # 1) Employee ONLY when "@"
    if EMP_DF is not None:
        emp_res = handle_employee_query_only_at(EMP_DF, user_msg)
        if emp_res is not None:
            current_history.append(f"AI: {emp_res.get('content','')}")
            return emp_res

    # 2) SOP Pipeline
    if not PIPELINE:
        err = "System Error: SOP pipeline not loaded."
        current_history.append(f"AI: {err}")
        return {"content": err, "match_type": "error"}

    result = PIPELINE.query(user_msg)

    # If rejected: still rewrite deterministic truth
    if result.get("rejected", False):
        truth_answer = choose_truth_text(result)
        final_response = rewrite_answer_with_llm(
            reference_answer=truth_answer,
            user_query=user_msg,
            history=current_history,
            intent_tag=result.get("tag", ""),
        )

        current_history.append(f"AI: {final_response}")

        return {
            "content": final_response,
            "match_type": "none",
            "score": result.get("score", 0.0),
            "suggestions": [],
            "source": result.get("tag", ""),
            "subject": result.get("tag", ""),
            "matched_question": result.get("matched_question", ""),
            "raw_answer": truth_answer,
            "normalized_query": result.get("normalized_query", ""),
            "corrected_query": result.get("corrected_query", ""),
            "did_you_mean": result.get("did_you_mean", None),
            "corrections": result.get("corrections", []),
        }

    # ✅ ALWAYS build a non-empty truth answer
    truth_answer = choose_truth_text(result)

    # ✅ ALWAYS rewrite via LLM
    final_response = rewrite_answer_with_llm(
        reference_answer=truth_answer,
        user_query=user_msg,
        history=current_history,
        intent_tag=result.get("tag", ""),
    )

    current_history.append(f"AI: {final_response}")

    return {
        "content": final_response,
        "match_type": "intent",
        "subject": result.get("tag", ""),
        "score": result.get("score", 0.0),
        "source": result.get("tag", ""),
        "matched_question": result.get("matched_question", ""),
        "raw_answer": truth_answer,
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
            df["EmployeeID"].astype(str).str.lower().fillna("")
            + " "
            + df["Name"].astype(str).str.lower().fillna("")
            + " "
            + df["Email"].astype(str).str.lower().fillna("")
        )
        df = df[s.str.contains(re.escape(q_norm), na=False)]

    df = df.head(limit)

    out = []
    for _, r in df.iterrows():
        out.append(
            {
                "employee_id": safe_val(r.get("EmployeeID")),
                "name": safe_val(r.get("Name")),
                "email": safe_val(r.get("Email")),
            }
        )

    return {"ok": True, "data": out}


# Run:
# & 'e:\Yadhu Projects\Chatbot\runtime\python.exe' -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
