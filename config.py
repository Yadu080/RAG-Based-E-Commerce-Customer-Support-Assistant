"""Central configuration — reads from .env file."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

# ── LLM ───────────────────────────────────────────────────────────────────────
# Groq is free — get your key at https://console.groq.com
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL   = "https://api.groq.com/openai/v1"

# Free Groq models (fast):
#   llama-3.1-8b-instant   ← fastest, great for support
#   llama-3.3-70b-versatile ← smarter, still free
#   mixtral-8x7b-32768     ← good for longer context
#   gemma2-9b-it           ← lightweight alternative
LLM_MODEL       = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR  = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))
CHROMA_COLLECTION   = "ecommerce_support_kb"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K                = int(os.getenv("TOP_K", "4"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))

# ── HITL ──────────────────────────────────────────────────────────────────────
HITL_QUEUE_DIR  = os.getenv("HITL_QUEUE_DIR", str(BASE_DIR / "hitl_queue"))
HITL_TIMEOUT_S  = int(os.getenv("HITL_TIMEOUT_S", "3600"))

# ── Uploads ───────────────────────────────────────────────────────────────────
UPLOAD_DIR = str(BASE_DIR / "uploads")

# ── Server ────────────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# ── System prompt ─────────────────────────────────────────────────────────────
# Convenience alias used in graph_engine.py
LLM_API_KEY  = GROQ_API_KEY
LLM_BASE_URL = GROQ_BASE_URL

SYSTEM_PROMPT = """You are a helpful and professional e-commerce customer support assistant.

IMPORTANT RULES:
1. Answer ONLY using the provided context from the knowledge base.
2. If the context does not contain enough information, respond with exactly:
   "I don't have enough information to answer this from our knowledge base. Let me connect you with our support team."
3. Be concise and friendly (2-4 sentences).
4. Never invent order numbers, policies, prices, or dates.
5. If the customer seems distressed, acknowledge their feelings first.
6. Always cite which part of the knowledge base your answer comes from.
"""
