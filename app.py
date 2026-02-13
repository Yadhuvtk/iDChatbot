# app.py (FULL BACKEND FILE)
# ✅ Intent answers: LLM rewrites EVERY time, but is forced to keep key facts (Pantone codes, numbers, etc.)
# ✅ If LLM output is unsafe (apology, missing key code, too long), fallback to raw Excel answer.
# ✅ Employee: one-line summary from Excel (deterministic)
# ✅ Fix: KB queries like "gold standard pms value" will not be routed to employee

import random
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# CONFIGURATION
# -----------------------------
BASE_DIR = Path(__file__).parent

EXCEL_PATH = BASE_DIR / "data" / "data.xlsx"
SHEET_NAME = "Sheet1"

EMPLOYEE_EXCEL_PATH = BASE_DIR / "data" / "Employe.xlsx"
EMPLOYEE_SHEET_NAME = "Sheet1"



TXT_PATH = BASE_DIR / "data" / "data.txt"


MIN_SCORE = 0.55

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

# -----------------------------
# MEMORY STORAGE
# -----------------------------
CHAT_HISTORY: Dict[str, List[str]] = {}

# -----------------------------
# HELPERS
# -----------------------------
def clean_text(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()

def norm(s: Any) -> str:
    return clean_text(s).lower()

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



def load_txt_intents() -> List[Dict[str, Any]]:
    if not TXT_PATH.exists():
        print(f"INFO: data.txt not found at {TXT_PATH}")
        return []

    try:
        text = TXT_PATH.read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"Error reading data.txt: {e}")
        return []

    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    intents = []

    for i, block in enumerate(blocks):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 2:
            continue

        question = lines[0]
        answer = " ".join(lines[1:])

        intents.append({
            "tag": f"txt_{i}",
            "patterns": [question],
            "responses": [answer],
        })

    print(f"SUCCESS: Loaded {len(intents)} TXT intents.")
    return intents



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
    """
    Extract key tokens that MUST be preserved in rewritten output.
    - numbers like 877, 871, 1.5, 12 etc.
    - keywords like pantone, pms, pt
    """
    ans = (reference_answer or "").lower()
    keys = []

    # numbers
    for m in re.findall(r"\b\d+(?:\.\d+)?\b", ans):
        if m not in keys:
            keys.append(m)

    # key terms
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

# -----------------------------
# INTENTS LOADING
# -----------------------------
def load_intents():
    if not EXCEL_PATH.exists():
        print(f"CRITICAL ERROR: File not found at {EXCEL_PATH}")
        return []

    try:
        df_raw = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=None)
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return []

    def hclean(x):
        return str(x).replace("\ufeff", "").strip().lower() if x else ""

    header_row_index = 0
    found_header = False

    for i in range(min(20, len(df_raw))):
        row_values = [hclean(x) for x in df_raw.iloc[i].tolist()]
        if "tag" in row_values and ("question" in row_values or any("pattern" in c for c in row_values)):
            header_row_index = i
            found_header = True
            break

    if not found_header:
        print("ERROR: Headers (Tag, Question, Answer) not found in Excel.")
        return []

    headers = [hclean(x) for x in df_raw.iloc[header_row_index].tolist()]
    df = df_raw.iloc[header_row_index + 1:].copy()
    df.columns = headers
    df = df.dropna(how="all").reset_index(drop=True)

    question_cols = [c for c in df.columns if "question" in c or "pattern" in c]
    answer_cols = [c for c in df.columns if "answer" in c or "response" in c]
    tag_col = "tag"

    intents = []
    for index, row in df.iterrows():
        tag = clean_text(row.get(tag_col))
        questions = [clean_text(row.get(c)) for c in question_cols if clean_text(row.get(c))]
        answers = [clean_text(row.get(c)) for c in answer_cols if clean_text(row.get(c))]

        if not questions or not answers:
            continue
        if not tag or tag.lower() == "nan":
            tag = f"auto_tag_{index}"

        intents.append({"tag": tag, "patterns": questions, "responses": answers})

    print(f"SUCCESS: Loaded {len(intents)} intents.")
    return intents

# -----------------------------
# EMPLOYEE LOADING
# -----------------------------
EMP_DF: Optional[pd.DataFrame] = None

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
        elif lc in ["takeleave", "take leave", "taken leave", "leavetaken", "leave taken"]:
            col_map[c] = "TakeLeave"
        elif lc in ["availableleave", "available leave", "leave balance", "balance leave"]:
            col_map[c] = "AvailableLeave"
        elif lc in ["under", "manager", "reporting to", "reports to"]:
            col_map[c] = "Under"
        elif lc in ["designation", "role", "title"]:
            col_map[c] = "Designation"
        elif lc in ["sl", "sno", "sr", "serial"]:
            col_map[c] = "SL"

    df = df.rename(columns=col_map)

    required = ["EmployeeID", "Name"]
    for r in required:
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

# -----------------------------
# SEMANTIC MATCHING (INTENTS)
# -----------------------------
class IntentMatcher:
    def __init__(self, intents: List[dict]):
        self.intents = intents
        self.pattern_texts = []
        self.pattern_meta = []
        if not intents:
            return

        print("Loading AI Model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        for i, it in enumerate(intents):
            for p in it["patterns"]:
                self.pattern_texts.append(p)
                self.pattern_meta.append((i, p))

        self.pattern_embeddings = self.model.encode(self.pattern_texts, convert_to_tensor=True)
        print("AI Model Ready.")

    def match(self, user_text: str):
        if not self.intents:
            return None, 0.0, ""
        query_embedding = self.model.encode(user_text, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.pattern_embeddings)[0]
        best_idx = int(cos_scores.argmax())
        return self.intents[self.pattern_meta[best_idx][0]], float(cos_scores[best_idx]), self.pattern_meta[best_idx][1]

    def top_suggestions(self, user_text: str):
        if not self.intents:
            return []
        query_embedding = self.model.encode(user_text, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.pattern_embeddings)[0]
        top_results = cos_scores.argsort(descending=True)

        out, seen = [], set()
        for idx in top_results:
            if float(cos_scores[idx]) < 0.2:
                break
            intent = self.intents[self.pattern_meta[int(idx)][0]]
            if intent["tag"] in seen:
                continue
            seen.add(intent["tag"])
            out.append({"intent": intent["tag"], "example": intent["patterns"][0]})
            if len(out) >= 3:
                break
        return out

# -----------------------------
# LLM REWRITE (FORCE REWRITE + GUARANTEED)
# -----------------------------
def rewrite_answer_with_llm(reference_answer: str, user_query: str, history: List[str], intent_tag: str = "") -> str:
    """
    ✅ Always rewrites (tries), but always stays correct via safety gates.
    ✅ If rewrite fails gates -> fallback to reference_answer.
    """
    reference_answer = (reference_answer or "").strip()
    if not reference_answer:
        return "I am sorry, I don't have that information."

    history_text = "\n".join(history[-6:])

    # Optional label helps LLM create a nicer sentence ("Silver Standard" etc.)
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

        # ---- Safety gates ----
        if contains_apology(res):
            return reference_answer

        if len(res) > max(240, len(reference_answer) * 5):
            return reference_answer

        if not rewritten_keeps_key_facts(res, reference_answer):
            return reference_answer

        # Ensure it ends like a sentence
        if res and res[-1] not in ".!?":
            res += "."

        return res if res else reference_answer
    except Exception:
        return reference_answer

# -----------------------------
# EMPLOYEE QUERY LOGIC
# -----------------------------
EMP_KEYWORDS = {
    "employee", "emp", "salary", "base salary", "pay", "ctc", "leave", "availableleave",
    "takeleave", "email", "designation", "role", "manager", "under", "report", "reporting"
}
EMP_ID_REGEX = re.compile(r"\b[a-z]{2,10}\d+[a-z]?\d*\b", re.IGNORECASE)

def looks_like_employee_question(msg: str) -> bool:
    m = norm(msg)
    return any(k in m for k in EMP_KEYWORDS) or (EMP_ID_REGEX.search(msg) is not None)

def find_employee_by_id(df: pd.DataFrame, msg: str) -> Optional[pd.Series]:
    m = EMP_ID_REGEX.search(msg)
    if not m:
        return None
    emp_id = m.group(0).strip()
    hits = df[df["EmployeeID"].str.lower() == emp_id.lower()]
    if len(hits) == 1:
        return hits.iloc[0]
    return None

# Strict name-only detection (avoid KB text being treated as name)
NOT_A_NAME_KEYWORDS = {
    "pms", "pantone", "standard", "value", "code", "color",
    "minimum", "stroke", "thickness", "font", "size",
    "screen", "screenprint", "screen printing", "embroidery",
    "silver", "gold"
}
NAME_ONLY_REGEX = re.compile(r"^[a-zA-Z\s\.\-']+$")

def is_name_only_message(msg: str) -> bool:
    s = clean_text(msg)
    if not s:
        return False
    s_low = s.lower()
    if any(k in s_low for k in NOT_A_NAME_KEYWORDS):
        return False
    if "?" in s:
        return False
    if len(s) > 40:
        return False
    if not NAME_ONLY_REGEX.match(s):
        return False
    words = [w for w in re.split(r"\s+", s.strip()) if w]
    if not (2 <= len(words) <= 4):
        return False
    cap_words = sum(1 for w in words if len(w) >= 2 and w[0].isupper())
    return cap_words >= 2

def find_employees_by_name_scored(df: pd.DataFrame, msg: str) -> List[Tuple[int, pd.Series]]:
    STOPWORDS = {
        "what", "is", "the", "of", "a", "an", "to", "for", "please", "tell", "me",
        "give", "show", "find", "employee", "emp", "id", "details", "info", "information",
        "salary", "email", "leave", "designation", "manager", "under"
    }
    text = re.sub(r"[^a-zA-Z\s]", " ", msg).lower()
    tokens = [t for t in text.split() if len(t) >= 2 and t not in STOPWORDS]
    if not tokens:
        return []
    scored: List[Tuple[int, pd.Series]] = []
    for _, row in df.iterrows():
        name = str(row.get("Name", "")).lower()
        score = sum(1 for t in tokens if t in name)
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

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

def handle_employee_query(df: pd.DataFrame, user_msg: str, history: List[str]) -> Optional[Dict[str, Any]]:
    if df is None or df.empty:
        return None

    msg = clean_text(user_msg)

    allow_employee_mode = (
        looks_like_employee_question(msg)
        or msg.strip().startswith("@")
        or is_name_only_message(msg)
    )
    if not allow_employee_mode:
        return None

    if msg.strip().startswith("@"):
        msg = msg.strip()[1:].strip()

    row = find_employee_by_id(df, msg)
    if row is not None:
        return {
            "content": build_employee_one_line(row),
            "match_type": "employee",
            "employee_id": safe_val(row.get("EmployeeID")),
            "name": safe_val(row.get("Name")),
        }

    scored = find_employees_by_name_scored(df, msg)
    if not scored:
        return {
            "content": "I am sorry, I couldn't find that employee in my database. Please provide the EmployeeID.",
            "match_type": "employee_not_found",
        }

    best_score, best_row = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0

    if is_name_only_message(msg):
        if best_score < 2 or (len(scored) > 1 and best_score == second_score):
            short_list = []
            for s, r in scored[:5]:
                short_list.append({
                    "EmployeeID": safe_val(r.get("EmployeeID")),
                    "Name": safe_val(r.get("Name")),
                    "Designation": safe_val(r.get("Designation")) if "Designation" in r.index else ""
                })
            return {
                "content": "I found multiple employees matching that name. Please specify the EmployeeID.",
                "match_type": "employee_multiple",
                "matches": short_list,
            }

    return {
        "content": build_employee_one_line(best_row),
        "match_type": "employee",
        "employee_id": safe_val(best_row.get("EmployeeID")),
        "name": safe_val(best_row.get("Name")),
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

INTENTS: List[Dict[str, Any]] = []
MATCHER: Optional[IntentMatcher] = None

@app.on_event("startup")
def startup_event():
    global INTENTS, MATCHER, EMP_DF

    excel_intents = load_intents()
    txt_intents = load_txt_intents()

    INTENTS = excel_intents + txt_intents

    MATCHER = IntentMatcher(INTENTS) if INTENTS else None
    EMP_DF = load_employees()


class ChatIn(BaseModel):
    message: str
    session_id: str = "default"

@app.post("/chat")
def chat(payload: ChatIn) -> Dict[str, Any]:
    user_msg = payload.message.strip()
    session_id = payload.session_id

    if not user_msg:
        return {"content": "Hello! How can I help?", "match_type": "none"}

    if session_id not in CHAT_HISTORY:
        CHAT_HISTORY[session_id] = []
    current_history = CHAT_HISTORY[session_id]

    # 1) Employee handler
    if EMP_DF is not None:
        emp_res = handle_employee_query(EMP_DF, user_msg, current_history)
        if emp_res is not None:
            CHAT_HISTORY[session_id].append(f"User: {user_msg}")
            CHAT_HISTORY[session_id].append(f"AI: {emp_res.get('content','')}")
            return emp_res

    # 2) Intent matcher
    if not MATCHER:
        return {"content": "System Error: Intents database not loaded.", "match_type": "error"}

    intent, score, matched_pattern = MATCHER.match(user_msg)

    if score < MIN_SCORE:
        suggestions = MATCHER.top_suggestions(user_msg)
        return {
            "content": "I am sorry, I don't have information about that in my database.",
            "match_type": "none",
            "score": score,
            "suggestions": suggestions,
        }

    raw_response = random.choice(intent["responses"])

    # ✅ ALWAYS rewrite (but safe + fallback)
    final_response = rewrite_answer_with_llm(
        reference_answer=raw_response,
        user_query=user_msg,
        history=current_history,
        intent_tag=intent.get("tag", "")
    )

    CHAT_HISTORY[session_id].append(f"User: {user_msg}")
    CHAT_HISTORY[session_id].append(f"AI: {final_response}")

    return {
        "content": final_response,
        "match_type": "intent",
        "subject": intent["tag"],
        "score": score,
        "source": intent["tag"],
        "matched_question": matched_pattern,
        "raw_answer": raw_response,  # keeps tooltip correct
    }

# -----------------------------
# EMPLOYEE SEARCH ENDPOINT (for @ mention dropdown)
# -----------------------------
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



#& 'e:\Yadhu Projects\Chatbot\runtime\python.exe' -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

