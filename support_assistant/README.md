# Support Assistant — Zepto RAG Pipeline

A customer support assistant for Zepto built with **ChromaDB**, **LangGraph**, and **FastAPI**. It answers policy questions by searching a knowledge base of 8 policy documents and returning a structured JSON response.

---

## Project Structure

```
support_assistant/
├── Dockerfile            # Container setup (Python 3.11-slim, port 7860)
├── requirements.txt      # Python dependencies
├── main.py               # Pipeline + FastAPI app
├── llm agent.ipynb       # Development notebook
└── docs/
    ├── doc_01.txt        # Delivery Policy
    ├── doc_02.txt        # Returns & Refunds
    ├── doc_03.txt        # Membership Tiers
    ├── doc_04.txt        # Order Tracking
    ├── doc_05.txt        # Order Cancellation
    ├── doc_06.txt        # Damaged or Missing Items
    ├── doc_07.txt        # Gift Cards
    └── doc_08.txt        # Customer Support Hours
```

---

## How It Works — RAG Pipeline

The pipeline runs four stages in order every time the app starts and a query arrives.

```
docs/ (8 .txt files)
      │
      ▼
[1] INGESTION          load_documents() + chunk_documents()
      │                Reads files, splits into paragraphs
      ▼
[2] EMBEDDING          ChromaDB collection "zepto_policies"
      │                Auto-embeds chunks via all-MiniLM-L6-v2
      ▼
[3] RETRIEVAL          collection.query(n_results=3)
      │                Finds top-3 most relevant chunks for the query
      ▼
[4] GENERATION         LangGraph StateGraph
      │                classify_intent → retrieve_and_answer / direct_answer
      ▼
  POST /ask  →  AnswerResponse (JSON)
```

### Stage 1 — Ingestion

`load_documents()` reads all `.txt` files from `docs/`. `chunk_documents()` splits each document on blank lines (`\n\n`) into paragraph-level chunks. All chunks are collected into a single `all_chunks` list.

### Stage 2 — Embedding & Storage

All chunks are loaded into an in-memory ChromaDB collection called `"zepto_policies"` via `collection.add()`. ChromaDB automatically embeds each chunk using `all-MiniLM-L6-v2`. Each chunk gets a unique ID (`chunk_0`, `chunk_1`, …) and its source filename as metadata.

### Stage 3 — Retrieval

When a query is a policy question, `retrieve_and_answer()` calls `collection.query(n_results=3)` to fetch the 3 most semantically similar chunks. The chunk texts and their IDs are passed to the generation step. For non-policy questions, retrieval is skipped entirely.

### Stage 4 — Generation (LangGraph)

A `StateGraph` with three nodes handles every query:

1. **`classify_intent`** — checks for keywords (`delivery`, `refund`, `return`, `cancel`, `membership`, `tracking`, `gift card`, `support hours`). Match → `policy_question`; no match → `general_question`. Routing is **not** affected by `MOCK_LLM`.

2. **`retrieve_and_answer`** *(policy questions only)* — runs retrieval, then builds the answer. In mock mode, returns the first 200 characters of the top chunk. In real-LLM mode, injects chunks into the prompt template and calls the model.

3. **`direct_answer`** *(general questions)* — In mock mode, returns a fixed fallback. In real-LLM mode, calls the model directly with no context.

Both nodes return an `AnswerResponse` Pydantic object: `answer`, `sources` (chunk IDs), and `confidence`.

### MOCK_LLM Toggle

Set via the `MOCK_LLM` environment variable (default `"1"`). Only the answer-generation step changes — ingestion, embedding, retrieval, and routing are identical in both modes.

| | `MOCK_LLM=1` (default) | `MOCK_LLM=0` (real LLM) |
|---|---|---|
| `retrieve_and_answer` | First 200 chars of top chunk | Full LLM answer from prompt template |
| `direct_answer` | Fixed fallback string | LLM answer without context |
| `sources` | Retrieved chunk IDs | Retrieved chunk IDs |
| `confidence` | Always `1.0` | Model-derived score |

---

## API

### `POST /ask`

**Request**
```json
{ "query": "What is the refund policy?" }
```

**Response**
```json
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours...",
  "sources": ["chunk_1", "chunk_5", "chunk_3"],
  "confidence": 1.0
}
```

**Example — policy question (retrieval triggered)**
```bash
curl -X POST http://localhost:7860/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the refund policy?"}'
```
```json
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unopened, resalable condition. Approved refunds are credited",
  "sources": ["chunk_1", "chunk_5", "chunk_3"],
  "confidence": 1.0
}
```

**Example — general question (no retrieval)**
```bash
curl -X POST http://localhost:7860/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of India?"}'
```
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

---

## Running Locally

**Requirements:** Python 3.11+

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --host 0.0.0.0 --port 7860
```

- API: `http://localhost:7860`
- Swagger UI: `http://localhost:7860/docs`

---

## Docker

### Build

```bash
docker build -t support-assistant .
```

### Run (mock mode — default)

```bash
docker run -p 7860:7860 support-assistant
```

### Run (real-LLM mode — optional)

```bash
docker run -p 7860:7860 -e MOCK_LLM=0 support-assistant
```

### Other commands

```bash
# View logs
docker logs <container_id>

# Stop the container
docker stop <container_id>
```

The Dockerfile uses `python:3.11-slim`, sets `PYTHONDONTWRITEBYTECODE=1` and `PYTHONUNBUFFERED=1` for clean logs, installs dependencies, copies source code, exposes port `7860`, and starts the app with `uvicorn`.

---

## Dependencies

| Package     | Purpose |
|-------------|---------|
| `fastapi`   | Hosts the `POST /ask` HTTP endpoint |
| `uvicorn`   | ASGI server that runs FastAPI |
| `pydantic`  | Validates request and response schemas |
| `chromadb`  | In-memory vector store for policy chunks |
| `langgraph` | Orchestrates the classify → retrieve/answer graph |
