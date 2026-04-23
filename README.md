# ShopEase RAG Customer Support Assistant

> A production-ready, full-stack **Retrieval-Augmented Generation (RAG)** application built with FastAPI, LangGraph, ChromaDB, and Groq. It provides an AI-powered customer support interface for e-commerce, with a complete **Human-in-the-Loop (HITL)** escalation system for queries the AI cannot confidently answer.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Technology Stack](#technology-stack)
5. [Project Structure](#project-structure)
6. [How It Works — The RAG Pipeline](#how-it-works--the-rag-pipeline)
   - [LangGraph State Machine (6 Nodes)](#langgraph-state-machine-6-nodes)
   - [Intent Classification](#intent-classification)
   - [Routing Logic](#routing-logic)
   - [Embedding & Vector Search](#embedding--vector-search)
   - [HITL Lifecycle](#hitl-lifecycle)
7. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Quick Start (run.sh)](#quick-start-runsh)
   - [Manual Setup](#manual-setup)
8. [Configuration Reference (.env)](#configuration-reference-env)
9. [API Reference](#api-reference)
   - [Knowledge Base Endpoints](#knowledge-base-endpoints)
   - [Query Endpoints](#query-endpoints)
   - [HITL Agent Endpoints](#hitl-agent-endpoints)
   - [System Endpoints](#system-endpoints)
10. [Module Reference](#module-reference)
11. [Demo Mode](#demo-mode)
12. [Customisation Guide](#customisation-guide)
13. [Troubleshooting](#troubleshooting)

---

## Overview

ShopEase RAG Support Assistant is a self-contained customer support AI system. You load your product documentation, FAQs, or policy documents into its knowledge base, and it answers customer questions in real-time by retrieving the most relevant passages from those documents and feeding them to an LLM.

When the AI's retrieval confidence falls below a threshold — or when a customer asks a legal, fraud-related, or emotionally charged question — the system automatically escalates to a human agent queue. A built-in agent dashboard lets human agents read the full context and respond.

The system runs entirely locally (except for the Groq API call) and requires no cloud database, no authentication service, and no Docker. A single `bash run.sh` starts everything.

---

## Key Features

| Feature | Detail |
|---|---|
| **RAG Pipeline** | Retrieve → Rank → Generate grounded answers from your own documents |
| **LangGraph Orchestration** | 6-node state machine with typed state and conditional routing |
| **Intent Classification** | Keyword-regex classifier covering 7 intents (ESCALATE, COMPLAINT, RETURN, ORDER, PAYMENT, ACCOUNT, FAQ) |
| **HITL Escalation** | 4 escalation triggers: keyword, intent, low confidence, LLM self-admission |
| **ChromaDB Vector Store** | Local, persistent, cosine-similarity search with HNSW index |
| **TF-IDF Embedder** | Pure-NumPy offline embedder; no model download required |
| **Document Ingestion** | Supports PDF (via PyMuPDF), TXT, and Markdown with recursive chunking |
| **Demo Mode** | Works without a Groq API key — shows retrieved context instead |
| **Single-Page Dashboard** | Unified chat + admin + agent dashboard in one HTML file |
| **Auto-generated API Docs** | FastAPI Swagger UI at `/docs` |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser Client                       │
│           frontend/index.html  (Chat + Dashboard)        │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP / REST
┌───────────────────────▼─────────────────────────────────┐
│                   FastAPI  (main.py)                     │
│   /api/query   /api/ingest/*   /api/hitl/*   /api/stats  │
└────────┬──────────────────────────────────┬─────────────┘
         │                                  │
┌────────▼─────────────┐        ┌───────────▼────────────┐
│   graph_engine.py    │        │    hitl_handler.py      │
│  LangGraph Pipeline  │        │  File-based HITL Queue  │
│  ┌────────────────┐  │        │  hitl_queue/*.json      │
│  │  input_node    │  │        └────────────────────────┘
│  │  retrieval_node│  │
│  │  router_node   │  │   ┌────────────────────────────┐
│  │  generation_node│─┼──▶│   Groq API  (LLM)          │
│  │  hitl_node     │  │   │   llama-3.1-8b-instant      │
│  │  output_node   │  │   └────────────────────────────┘
│  └────────────────┘  │
└────────┬─────────────┘
         │
┌────────▼──────────────────────────────────────────────┐
│               vector_store.py + embedder.py            │
│   ChromaDB (chroma_db/)  ←  TF-IDF 2000-dim vectors   │
└────────┬──────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────┐
│             document_processor.py                      │
│   PDF (PyMuPDF) / TXT / MD  →  Recursive Chunking     │
└───────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Library / Tool | Purpose |
|---|---|---|
| Web Framework | `fastapi 0.115`, `uvicorn` | REST API + static file serving |
| AI Orchestration | `langgraph 0.2`, `langchain 0.3` | State machine & pipeline |
| LLM Provider | `openai 1.51` (Groq endpoint) | Text generation |
| Vector DB | `chromadb 0.5` | Persistent local vector store |
| Embeddings | Pure NumPy (TF-IDF, built-in) | Offline, zero-download embeddings |
| PDF Parsing | `pymupdf 1.24`, `pypdf 4.3` | Extract text from PDFs |
| Validation | `pydantic 2.9` | Request/response model validation |
| Config | `python-dotenv 1.0` | `.env` file management |
| HTTP Client | `httpx 0.27` | Async HTTP requests |

> **Embedding note:** `requirements.txt` includes `sentence-transformers` as an optional upgrade path. The current implementation uses the built-in pure-NumPy TF-IDF embedder that works completely offline. See the [Customisation Guide](#customisation-guide) to swap in neural embeddings.

---

## Project Structure

```
rag_project/
│
├── main.py                     # FastAPI app — all routes and middleware
├── config.py                   # Central config, reads from .env
├── run.sh                      # One-command startup script
├── requirements.txt            # All Python dependencies
├── .env                        # Your secrets (git-ignored)
├── .env.example                # Template — copy to .env
│
├── backend/                    # All application logic
│   ├── __init__.py
│   ├── graph_engine.py         # LangGraph 6-node state machine
│   ├── query_processor.py      # Intent classifier & query validator
│   ├── document_processor.py   # File loader, text cleaner, chunker
│   ├── embedder.py             # TF-IDF embedder (pure NumPy, offline)
│   ├── vector_store.py         # ChromaDB CRUD operations
│   └── hitl_handler.py         # HITL queue (file-based JSON store)
│
├── frontend/
│   └── index.html              # Complete single-page app (chat + admin)
│
├── data/
│   └── sample_kb.txt           # Sample e-commerce knowledge base
│
├── chroma_db/                  # (auto-created) Persisted vector database
├── hitl_queue/                 # (auto-created) Escalated query JSON files
└── uploads/                    # (auto-created) Temp storage for file uploads
```

---

## How It Works — The RAG Pipeline

### LangGraph State Machine (6 Nodes)

Every customer query is processed through a typed `GraphState` object flowing through six nodes in sequence, with conditional branching after each decision point.

```
[input_node] ──→ [retrieval_node] ──→ [router_node]
                                           │
                          ┌────────────────┤────────────────┐
                          ▼                ▼                ▼
                  [generation_node]   [hitl_node]   [output_node (error)]
                          │                │
                          └────────────────┘
                                   │
                           [output_node] ──→ END
```

| Node | Responsibility |
|---|---|
| `input_node` | Validates the query (length, non-empty), classifies intent, initialises state |
| `retrieval_node` | Embeds the query and retrieves top-K chunks from ChromaDB |
| `router_node` | Decides: generate an answer OR escalate to HITL |
| `generation_node` | Builds a structured prompt and calls the Groq LLM API |
| `hitl_node` | Writes an escalation record to the file-based HITL queue |
| `output_node` | Formats the final response with sources, confidence score, and latency |

The response returned to the API caller always contains:
```json
{
  "query_id":   "uuid",
  "answer":     "The answer text...",
  "sources":    ["filename.pdf (p.2)"],
  "confidence": 0.7812,
  "escalated":  false,
  "intent":     "RETURN_REQUEST",
  "latency_ms": 423,
  "error":      null
}
```

---

### Intent Classification

The `query_processor.py` module classifies each incoming query using **keyword-regex patterns** before retrieval. This is fast, deterministic, and requires no model.

| Intent | Example Triggers | Action |
|---|---|---|
| `ESCALATE` | "sue", "fraud", "chargeback", "attorney", "lawsuit" | Immediate HITL escalation |
| `COMPLAINT` | "terrible", "furious", "worst", "disgusting" | HITL escalation |
| `RETURN_REQUEST` | "return", "refund", "exchange", "money back" | Proceed to RAG |
| `ORDER_STATUS` | "track", "where is my order", "shipment" | Proceed to RAG |
| `PAYMENT` | "billing", "invoice", "promo code", "price match" | Proceed to RAG |
| `ACCOUNT` | "password", "login", "rewards", "profile" | Proceed to RAG |
| `GENERAL_FAQ` | *(catch-all)* | Proceed to RAG |

---

### Routing Logic

After retrieval, `router_node` applies four escalation checks in priority order:

1. **Error in state** → Short-circuit to `output_node`
2. **Hard escalation keywords** (legal/fraud) → `hitl_node`
3. **Intent is ESCALATE or COMPLAINT** → `hitl_node`
4. **Max retrieval score < `CONFIDENCE_THRESHOLD` (default 0.55)** → `hitl_node`
5. **Zero chunks retrieved** → `hitl_node`
6. **All checks pass** → `generation_node`

Additionally, after LLM generation, if the model's output contains phrases like `"don't have enough information"` or `"connect you with our support"`, the answer is also escalated.

---

### Embedding & Vector Search

**Embedder (`embedder.py`):**
- Implements a pure-NumPy **TF-IDF** pipeline — no internet download, no GPU required.
- Maintains a shared vocabulary of the top 2000 terms by IDF across all ingested documents.
- Vectors are L2-normalised so cosine similarity = dot product.
- Embeddings are cached by SHA-256 hash for repeated queries.
- Gracefully degrades to a random unit vector before any corpus is loaded.

**Vector Store (`vector_store.py`):**
- Uses ChromaDB's `PersistentClient` with cosine HNSW space.
- Chunks are upserted in batches of 100 using the chunk's SHA-256 hash as the unique ID (prevents duplicates on re-ingestion).
- Distance scores (range 0–2) are converted to similarity (range 0–1): `score = 1 - distance / 2`.

---

### HITL Lifecycle

The Human-in-the-Loop queue is **file-based** — each escalated query becomes a JSON file in `hitl_queue/`. This makes it trivially inspectable and requires no database.

```
Customer Query
     │
     ▼  (escalation triggered)
hitl_queue/<query_id>.json   ← status: "PENDING"
     │
     ▼  (agent opens dashboard)
GET /api/hitl/queue          ← returns all PENDING entries
     │
     ▼  (agent submits response)
POST /api/hitl/resolve       ← status: "RESOLVED", human_response set
```

**Status lifecycle:** `PENDING` → *(optional)* `IN_REVIEW` → `RESOLVED`

Each queue entry stores:
- `query_id`, `session_id`, `user_query`
- `intent` and `escalation_reason` (e.g. `"low_confidence:0.31"`, `"intent:complaint"`)
- `retrieved_chunks` — the context that was found (useful for the human agent)
- `timestamp`, `status`, `human_response`, `agent_id`, `resolved_at`

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **pip**
- A free [Groq API Key](https://console.groq.com) — *(optional but recommended; the app runs in Demo Mode without one)*

### Quick Start (run.sh)

```bash
# 1. Navigate to the project directory
cd /path/to/rag_project

# 2. Run the startup script
bash run.sh

# 3. Open your browser
open http://localhost:8000
```

The script will:
- Check for Python 3
- Copy `.env.example` → `.env` if no `.env` exists
- Install all dependencies via pip
- Create required directories (`chroma_db/`, `hitl_queue/`, `uploads/`)
- Start the FastAPI server with auto-reload

### Manual Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and set GROQ_API_KEY

# 4. Start the server
python3 main.py
# or with uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Configuration Reference (.env)

Copy `.env.example` to `.env` and adjust as needed.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(empty)* | Your Groq API key. Get one free at console.groq.com. Without this, the app runs in Demo Mode. |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Groq model name. Options below. |
| `LLM_MAX_TOKENS` | `512` | Max tokens in the LLM response. |
| `LLM_TEMPERATURE` | `0.3` | LLM sampling temperature (0 = deterministic, 1 = creative). |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Reserved for future neural embedding swap-in. |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Path where ChromaDB persists its data. |
| `CHUNK_SIZE` | `500` | Max characters per document chunk. |
| `CHUNK_OVERLAP` | `50` | Characters of overlap between adjacent chunks. |
| `TOP_K` | `4` | Number of chunks to retrieve per query. |
| `CONFIDENCE_THRESHOLD` | `0.55` | Min similarity score; below this triggers HITL. |
| `HITL_QUEUE_DIR` | `./hitl_queue` | Directory for escalated query JSON files. |
| `HOST` | `0.0.0.0` | Server bind host. |
| `PORT` | `8000` | Server bind port. |

**Available Groq Models (all free):**

| Model | Best For |
|---|---|
| `llama-3.1-8b-instant` | Fastest, great for support (default) |
| `llama-3.3-70b-versatile` | Smarter, handles complex queries |
| `mixtral-8x7b-32768` | Long documents, large context window |
| `gemma2-9b-it` | Lightweight Google alternative |

---

## API Reference

Interactive docs are available at **[http://localhost:8000/docs](http://localhost:8000/docs)** (Swagger UI) and **[http://localhost:8000/redoc](http://localhost:8000/redoc)** (ReDoc) while the server is running.

### Knowledge Base Endpoints

#### `POST /api/ingest/file`
Upload a PDF, TXT, or MD file to be chunked and indexed.

```bash
curl -X POST http://localhost:8000/api/ingest/file \
  -F "file=@/path/to/manual.pdf"
```

**Response:**
```json
{
  "status": "success",
  "filename": "manual.pdf",
  "chunks_ingested": 42,
  "message": "Successfully indexed 42 chunks from 'manual.pdf'"
}
```

#### `POST /api/ingest/text`
Ingest raw text directly (useful for testing or scripting).

```bash
curl -X POST http://localhost:8000/api/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Our return policy allows 30-day returns.", "source": "policy_v2"}'
```

#### `POST /api/ingest/sample`
Load the bundled sample ShopEase knowledge base (`data/sample_kb.txt`).

```bash
curl -X POST http://localhost:8000/api/ingest/sample
```

#### `GET /api/kb/stats`
Returns the number of indexed chunks and all source filenames.

```json
{"total_chunks": 87, "sources": ["sample_kb.txt", "returns_policy.pdf"], "status": "ready"}
```

#### `DELETE /api/kb/clear`
Wipes the entire ChromaDB collection and starts fresh.

---

### Query Endpoints

#### `POST /api/query`
Submit a user question through the full RAG pipeline.

**Request:**
```json
{
  "user_query": "What is your return policy for electronics?",
  "session_id": "optional-session-uuid"
}
```

**Response:**
```json
{
  "query_id":   "3fa85f64-...",
  "answer":     "Electronics can be returned within 15 days...",
  "sources":    ["sample_kb.txt (p.3)"],
  "confidence": 0.7812,
  "escalated":  false,
  "intent":     "RETURN_REQUEST",
  "latency_ms": 387,
  "error":      null
}
```

> If `escalated` is `true`, the `query_id` can be polled to check for a human response.

#### `GET /api/query/{query_id}/status`
Poll the resolution status of an escalated query.

```json
{
  "query_id":       "3fa85f64-...",
  "status":         "RESOLVED",
  "human_response": "Please contact us at support@shopease.com",
  "resolved_at":    "2026-04-23T10:45:00Z"
}
```

---

### HITL Agent Endpoints

#### `GET /api/hitl/queue`
Returns all `PENDING` escalated queries. Used by the agent dashboard.

#### `GET /api/hitl/all`
Returns all HITL entries regardless of status.

#### `POST /api/hitl/resolve`
Submit a human agent's response to close a ticket.

```json
{
  "query_id":       "3fa85f64-...",
  "human_response": "Your order #12345 has shipped. Tracking: XYZ.",
  "agent_id":       "agent_jane"
}
```

#### `GET /api/hitl/stats`
Returns ticket counts by status.

```json
{"PENDING": 3, "IN_REVIEW": 0, "RESOLVED": 14}
```

---

### System Endpoints

#### `GET /health`
Basic health check. Returns `{"status": "ok", "version": "1.0.0"}`.

#### `GET /api/stats`
Full system stats combining KB stats, HITL stats, query counters, and config values.

```json
{
  "kb":         {"total_chunks": 87, "sources": [...], "status": "ready"},
  "hitl":       {"PENDING": 1, "IN_REVIEW": 0, "RESOLVED": 5},
  "queries":    {"total": 42, "escalated": 6},
  "llm_model":  "llama-3.1-8b-instant",
  "demo_mode":  false,
  "embedding":  "all-MiniLM-L6-v2",
  "threshold":  0.55
}
```

---

## Module Reference

### `main.py`
FastAPI application entry point. Defines all routes, mounts CORS middleware, and serves the frontend HTML. All heavy imports (backend modules) are deferred inside route handlers to keep startup fast.

### `config.py`
Reads all configuration from the `.env` file via `python-dotenv`. Exposes constants (`GROQ_API_KEY`, `LLM_MODEL`, `CHUNK_SIZE`, etc.) consumed across all backend modules. Also defines the `SYSTEM_PROMPT` that instructs the LLM to answer only from the provided context.

### `backend/graph_engine.py`
The core of the application. Defines the `GraphState` TypedDict with 14 fields, implements all 6 node functions, and uses `langgraph.graph.StateGraph` to wire them together with conditional edges. The compiled graph is cached as a module-level singleton. The public entry point is `run_query(user_query, session_id)`.

### `backend/query_processor.py`
Provides three functions:
- `classify_intent(query)` — returns one of 7 intent labels using regex patterns
- `has_hard_escalation(query)` — returns True for legal/fraud keywords
- `validate_query(query)` — checks length (3–2000 chars)

### `backend/document_processor.py`
- `load_pdf(path)` — uses PyMuPDF (`fitz`) to extract per-page text, skipping pages with fewer than 30 characters
- `load_text(path)` — splits on section separators (`━━━`, `===`, `---`) to create pseudo-pages
- `_recursive_split(text, chunk_size, overlap)` — custom recursive splitter (paragraph → sentence → word → character fallback)
- `_clean_text(text)` — removes null bytes, normalises line endings, collapses spaces
- All chunks include metadata: `source`, `page`, `chunk_index`, `chunk_hash`, `char_count`

### `backend/embedder.py`
Pure-NumPy TF-IDF implementation:
- `update_corpus(texts)` — updates the shared IDF model after new documents are ingested
- `embed_documents(texts)` → `List[List[float]]` — batch embed for ingestion
- `embed_query(text)` → `List[float]` — single embed with caching for queries
- Vocabulary capped at 2000 terms; vectors are L2-normalised

### `backend/vector_store.py`
Wraps ChromaDB with lazy initialisation:
- `ingest_chunks(chunks)` — calls `update_corpus`, embeds, and upserts in batches of 100
- `retrieve(query, k)` — embeds query, queries ChromaDB, converts distances to similarity scores
- `get_stats()` — returns chunk count and unique source list
- `clear_collection()` — deletes and recreates the ChromaDB collection

### `backend/hitl_handler.py`
File-based queue — each ticket is a JSON file:
- `enqueue(state)` — creates a new `PENDING` JSON file, returns `query_id`
- `get_entry(query_id)` — reads a single ticket by ID
- `resolve(query_id, human_response, agent_id)` — updates status to `RESOLVED`
- `list_pending()` / `list_all()` — sorted by timestamp descending
- `get_stats()` — counts by status

### `frontend/index.html`
A single-file ~42KB vanilla HTML/CSS/JS application with three panels:
1. **Chat** — sends queries to `/api/query`, displays answers with source citations, confidence badges, and latency
2. **Admin** — file upload, text ingestion, KB stats, load sample KB, clear KB
3. **Agent Dashboard** — lists pending HITL tickets, allows agents to type and submit responses

---

## Demo Mode

If `GROQ_API_KEY` is not set in `.env`, the system runs in **Demo Mode**:

- Retrieval still works normally — documents are indexed and searched
- Instead of calling the Groq API, the response shows a preview of the top retrieved chunk
- A banner is displayed indicating Demo Mode is active and how to enable real AI

Demo Mode is useful for testing document ingestion and retrieval quality without consuming API credits.

---

## Customisation Guide

### Swap to Neural Embeddings
Replace the TF-IDF embedder with `sentence-transformers` for better semantic recall:

1. In `backend/embedder.py`, add:
```python
from sentence_transformers import SentenceTransformer
_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_documents(texts):
    return _model.encode(texts, normalize_embeddings=True).tolist()

def embed_query(text):
    return _model.encode([text], normalize_embeddings=True)[0].tolist()
```
2. Remove the `update_corpus` call from `vector_store.ingest_chunks` (not needed for neural models).

### Add a New Intent
In `backend/query_processor.py`, add a new key to `INTENT_PATTERNS`:
```python
"PRODUCT_AVAILABILITY": [
    r"\bin stock\b", r"\bavailable\b", r"\bout of stock\b",
],
```

### Adjust Escalation Sensitivity
- **Lower** `CONFIDENCE_THRESHOLD` (e.g., `0.35`) → escalate less, answer more questions
- **Raise** `CONFIDENCE_THRESHOLD` (e.g., `0.70`) → escalate more, only answer high-confidence queries

### Use a Different LLM
The system uses OpenAI's Python client pointed at Groq's base URL. Any OpenAI-compatible provider works. Update `config.py`:
```python
GROQ_BASE_URL = "https://api.openai.com/v1"  # or any compatible endpoint
LLM_MODEL     = "gpt-4o-mini"
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `chromadb` install fails | Ensure you have Python 3.10+ and try `pip install chromadb --upgrade` |
| `fitz` not found | Install PyMuPDF: `pip install pymupdf` |
| All queries escalate | Check that you have loaded a knowledge base first. Use `POST /api/ingest/sample`. |
| Low confidence on all queries | Try lowering `CONFIDENCE_THRESHOLD` in `.env` to `0.35` |
| Demo Mode even with API key | Ensure `.env` has `GROQ_API_KEY=sk-...` (no quotes, no trailing spaces) |
| Port 8000 already in use | Change `PORT=8001` in `.env` |
| Slow first query | The TF-IDF model and ChromaDB initialise lazily on the first request — subsequent queries are faster |
