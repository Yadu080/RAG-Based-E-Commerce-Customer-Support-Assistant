"""
main.py
FastAPI application: serves the frontend and all API routes.
"""
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG E-Commerce Customer Support Assistant",
    description="Retrieval-Augmented Generation with LangGraph & HITL",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.HITL_QUEUE_DIR, exist_ok=True)

# Query counter for stats
_query_counter = {"total": 0, "escalated": 0}

# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    user_query: str
    session_id: Optional[str] = None

class HITLResolveRequest(BaseModel):
    query_id:       str
    human_response: str
    agent_id:       Optional[str] = "human_agent"

class IngestTextRequest(BaseModel):
    text:   str
    source: Optional[str] = "manual_entry"


# ── Serve frontend ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "frontend" / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Frontend not found. Place index.html in /frontend/</h1>", status_code=404)
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


# ── Knowledge Base endpoints ───────────────────────────────────────────────────

@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """Upload and ingest a PDF or TXT file into ChromaDB."""
    allowed_types = {".pdf", ".txt", ".md"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_types:
        raise HTTPException(status_code=400, detail=f"File type '{ext}' not supported. Use PDF or TXT.")

    # Save upload
    save_path = os.path.join(config.UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        from backend.document_processor import process_file
        from backend.vector_store import ingest_chunks
        chunks = process_file(save_path)
        count  = ingest_chunks(chunks)
        return {
            "status":          "success",
            "filename":        file.filename,
            "chunks_ingested": count,
            "message":         f"Successfully indexed {count} chunks from '{file.filename}'"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/ingest/text")
async def ingest_text(req: IngestTextRequest):
    """Ingest raw text directly (for quick testing)."""
    import tempfile, os
    from backend.document_processor import process_file
    from backend.vector_store import ingest_chunks

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp.write(req.text)
        tmp_path = tmp.name

    try:
        chunks = process_file(tmp_path)
        count  = ingest_chunks(chunks)
        return {"status": "success", "chunks_ingested": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/api/ingest/sample")
async def ingest_sample_kb():
    """Load the bundled sample ShopEase knowledge base."""
    sample_path = Path(__file__).parent / "data" / "sample_kb.txt"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample KB not found.")
    try:
        from backend.document_processor import process_file
        from backend.vector_store import ingest_chunks
        chunks = process_file(str(sample_path))
        count  = ingest_chunks(chunks)
        return {
            "status":          "success",
            "filename":        "sample_kb.txt",
            "chunks_ingested": count,
            "message":         f"Sample ShopEase knowledge base loaded! {count} chunks indexed."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kb/stats")
async def kb_stats():
    """Return knowledge base statistics."""
    from backend.vector_store import get_stats
    return get_stats()


@app.delete("/api/kb/clear")
async def clear_kb():
    """Clear the entire vector store."""
    from backend.vector_store import clear_collection
    ok = clear_collection()
    if ok:
        return {"status": "success", "message": "Knowledge base cleared successfully."}
    raise HTTPException(status_code=500, detail="Failed to clear knowledge base.")


# ── Query endpoint ────────────────────────────────────────────────────────────

@app.post("/api/query")
async def query(req: QueryRequest):
    """Process a user query through the full RAG pipeline."""
    global _query_counter
    try:
        from backend.graph_engine import run_query
        result = run_query(
            user_query=req.user_query,
            session_id=req.session_id or str(uuid.uuid4()),
        )
        _query_counter["total"] += 1
        if result.get("escalated"):
            _query_counter["escalated"] += 1
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/api/query/{query_id}/status")
async def query_status(query_id: str):
    """Poll HITL status for an escalated query."""
    from backend.hitl_handler import get_entry
    entry = get_entry(query_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Query ID not found.")
    return {
        "query_id":       entry["query_id"],
        "status":         entry["status"],
        "human_response": entry.get("human_response"),
        "resolved_at":    entry.get("resolved_at"),
    }


# ── HITL agent endpoints ───────────────────────────────────────────────────────

@app.get("/api/hitl/queue")
async def hitl_queue():
    """Return all pending HITL entries (for the agent dashboard panel)."""
    from backend.hitl_handler import list_pending
    return list_pending()


@app.get("/api/hitl/all")
async def hitl_all():
    """Return all HITL entries (for full dashboard view)."""
    from backend.hitl_handler import list_all
    return list_all()


@app.post("/api/hitl/resolve")
async def hitl_resolve(req: HITLResolveRequest):
    """Human agent submits a response to an escalated query."""
    from backend.hitl_handler import resolve
    ok = resolve(req.query_id, req.human_response, req.agent_id or "human_agent")
    if ok:
        return {"status": "resolved", "query_id": req.query_id}
    raise HTTPException(status_code=404, detail="Query ID not found in HITL queue.")


@app.get("/api/hitl/stats")
async def hitl_stats():
    from backend.hitl_handler import get_stats
    return get_stats()


# ── System stats ──────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def system_stats():
    from backend.vector_store import get_stats as kb_stats_fn
    from backend.hitl_handler import get_stats as hitl_stats_fn
    kb   = kb_stats_fn()
    hitl = hitl_stats_fn()
    return {
        "kb":           kb,
        "hitl":         hitl,
        "queries":      _query_counter,
        "llm_model":    config.LLM_MODEL,
        "demo_mode":    not bool(config.GROQ_API_KEY),
        "embedding":    config.EMBEDDING_MODEL,
        "threshold":    config.CONFIDENCE_THRESHOLD,
    }


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)
