#!/bin/bash
# ─── ShopEase RAG Support Assistant — Startup Script ─────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   ShopEase RAG Customer Support Assistant  v1.0.0       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
  echo "❌ Python 3 not found. Please install Python 3.10+"
  exit 1
fi

# Create .env if not exists
if [ ! -f ".env" ]; then
  echo "📋 Creating .env from .env.example..."
  cp .env.example .env
  echo "   ⚠️  Edit .env and add your GROQ_API_KEY for real AI responses."
  echo "   Get a FREE key at: https://console.groq.com (no credit card needed)"
  echo "   Without it, the system runs in DEMO MODE (shows retrieved context only)."
  echo ""
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install fastapi uvicorn[standard] python-multipart python-dotenv chromadb \
  langgraph pymupdf openai pydantic numpy --break-system-packages -q
echo "✅ Dependencies ready"
echo ""

# Create required directories
mkdir -p chroma_db hitl_queue uploads

# Start server
echo "🚀 Starting server at http://localhost:8000"
echo "   Open your browser → http://localhost:8000"
echo "   Press Ctrl+C to stop"
echo ""
python3 main.py
