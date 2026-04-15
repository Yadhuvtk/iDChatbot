# Chatbot Service Runbook

This folder contains the standalone chatbot API used by the frontend at `AiFrontend-web/src/V2/Chatbot/chatbot.jsx`.

The service is a FastAPI app with:
- Q&A retrieval from FAISS index
- optional LLM rewriting via Ollama (`mistral`)
- employee lookup for `@name` queries

## 1) Paths and Ports

- Service folder: `E:\GITHUB\UPSCALE\Chatbot`
- Chatbot API port: `8033`
- Main endpoint base URL: `http://<host>:8033`
- Frontend default chatbot base URL: `http://10.1.17.201:8033`

## 2) How It Is Wired in This Repo

- API code: `Chatbot/app.py`
- Knowledge conversion: `Chatbot/scripts/convert.py`
- FAISS build: `Chatbot/build.py`
- Knowledge files:
  - `Chatbot/data/data.txt`
  - `Chatbot/data/data.xlsx`
  - `Chatbot/data/converted.json` (generated)
  - `Chatbot/data/sop_index.faiss` (generated)
  - `Chatbot/data/sop_index_map.json` (generated)
- Employee source: `Chatbot/data/Employe.xlsx`

Root startup script already includes chatbot startup:
- `START_SERVER.bat` calls `:ensure_chatbot` and runs `uvicorn` on port `8033`.

## 3) Start Options

### Option A: Start full stack (recommended)

From repo root:

```powershell
cd E:\GITHUB\UPSCALE
cmd /c START_SERVER.bat
```

This starts:
- MongoDB
- root AI API (`8765`)
- chatbot API (`8033`)
- backend (`5000`)
- frontend (`3000`)

### Option B: Start chatbot only

From repo root:

```powershell
cd E:\GITHUB\UPSCALE\Chatbot
$env:PYTHONIOENCODING = "utf-8"
.\runtime\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8033 --reload
```

If `Chatbot\runtime\python.exe` is unavailable, use system Python:

```powershell
python -m uvicorn app:app --host 0.0.0.0 --port 8033 --reload
```

## 4) API Endpoints

### `POST /chat`

Request body:

```json
{
  "message": "What is the PMS value of silver standard?",
  "session_id": "user_abc123"
}
```

Typical response fields:
- `content`
- `match_type`
- `score`
- `source`
- `subject`
- `matched_question`
- `raw_answer`
- `normalized_query`
- `corrected_query`
- `did_you_mean`
- `corrections`

### `GET /employees?q=<text>&limit=<n>`

Example:

```text
GET /employees?q=aneesh&limit=30
```

Response:

```json
{
  "ok": true,
  "data": [
    {
      "employee_id": "12345",
      "name": "Employee Name",
      "email": "name@company.com"
    }
  ]
}
```

## 5) Data Format (Recommended)

For best retrieval quality, use Q&A format with paragraph answers.

Preferred input in `Chatbot/data/data.txt`:

```text
Question text?
Paragraph answer text...

Next question?
Next answer...
```

Rules:
- first line of each block = question
- following line(s) = answer
- one blank line between blocks

`data.xlsx` is also supported. It should contain columns including:
- `Tag`
- `Question` (or `Pattern`)
- `Answer` (or `Response`)

## 6) Update Knowledge Base (required after data edits)

Run after changing `data.txt` or `data.xlsx`:

```powershell
cd E:\GITHUB\UPSCALE
$env:PYTHONIOENCODING = "utf-8"
.\Chatbot\runtime\python.exe .\Chatbot\scripts\convert.py
.\Chatbot\runtime\python.exe .\Chatbot\build.py
```

Then restart chatbot service (or rerun `START_SERVER.bat`).

## 7) Adding DOCX Content

When source content is `.docx`:
- convert it into Q&A blocks
- paste into `Chatbot/data/data.txt`
- run convert + build commands above

Do not paste raw document paragraphs without question lines. This pipeline retrieves primarily by question text.

## 8) Employee Data Notes

Employee file:
- `Chatbot/data/Employe.xlsx`

Required columns:
- `EmployeeID`
- `Name`

Optional columns:
- `Email`
- `Designation`
- `Base Salary`
- `AvailableLeave`
- `Under` (reporting manager)

Employee response is triggered only for messages that start with `@`.

## 9) Frontend Integration Notes

Frontend chatbot uses:
- `VITE_API_BASE_URL` for chatbot API (`/chat` and `/employees`)
- `VITE_LOGGER_BASE_URL` for chat logging backend (`/chat/log` on port `5000`)

Current fallback in frontend:
- `VITE_API_BASE_URL` fallback: `http://10.1.17.201:8033`
- `VITE_LOGGER_BASE_URL` fallback: `http://10.1.17.201:5000`

## 10) Quick Health Tests

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8033/docs | Select-Object StatusCode
```

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8033/chat -ContentType "application/json" -Body '{"message":"What is your name?","session_id":"test_1"}'
```

```powershell
Invoke-RestMethod -Method Get -Uri "http://localhost:8033/employees?q=a&limit=5"
```

## 11) Troubleshooting

- Port `8033` already in use:
  - stop existing process or change port in startup command.
- `UnicodeEncodeError` on Windows console:
  - set `PYTHONIOENCODING=utf-8` before running Python scripts.
- Missing answers after data update:
  - rerun both `convert.py` and `build.py`
  - restart chatbot service.
- Slow first build/start:
  - model downloads may occur on first run (`BAAI/bge-m3`, reranker model).
- Ollama not running:
  - chatbot still works; it falls back to source answer text without rewrite.

