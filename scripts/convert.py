import json
import re
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

EXCEL_PATH = DATA_DIR / "data.xlsx"
SHEET_NAME = "Sheet1"
TXT_PATH = DATA_DIR / "data.txt"

OUT_JSON = DATA_DIR / "converted.json"

def clean(s):
    if s is None:
        return ""
    return str(s).strip()

def load_excel():
    df_raw = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=None)

    def hclean(x):
        return str(x).replace("\ufeff", "").strip().lower() if x else ""

    header_row_index = None
    for i in range(min(20, len(df_raw))):
        row = [hclean(x) for x in df_raw.iloc[i].tolist()]
        if "tag" in row and ("question" in row or any("pattern" in c for c in row)):
            header_row_index = i
            break

    if header_row_index is None:
        raise ValueError("Header row not found. Need Tag + Question/Pattern + Answer/Response columns.")

    headers = [hclean(x) for x in df_raw.iloc[header_row_index].tolist()]
    df = df_raw.iloc[header_row_index + 1:].copy()
    df.columns = headers
    df = df.dropna(how="all").reset_index(drop=True)

    q_cols = [c for c in df.columns if "question" in c or "pattern" in c]
    a_cols = [c for c in df.columns if "answer" in c or "response" in c]

    out = []
    idx = 0
    for _, row in df.iterrows():
        tag = clean(row.get("tag")) or f"auto_{idx}"
        questions = [clean(row.get(c)) for c in q_cols if clean(row.get(c))]
        answers = [clean(row.get(c)) for c in a_cols if clean(row.get(c))]
        if not questions or not answers:
            continue

        # one entry per question (best for retrieval)
        for q in questions:
            out.append({
                "id": f"xl_{idx}",
                "tag": tag,
                "question": q,
                "answer": answers[0],
                "source_id": f"xl_{tag}",
            })
            idx += 1

    return out

def load_txt():
    if not TXT_PATH.exists():
        return []

    text = TXT_PATH.read_text(encoding="utf-8").strip()
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    out = []
    for i, block in enumerate(blocks):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        q = lines[0]
        a = " ".join(lines[1:])
        out.append({
            "id": f"txt_{i}",
            "tag": f"txt_{i}",
            "question": q,
            "answer": a,
            "source_id": f"txt_{i}",
        })
    return out

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    items += load_excel()
    items += load_txt()

    OUT_JSON.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Created {OUT_JSON} with {len(items)} Q&A entries")

if __name__ == "__main__":
    main()
