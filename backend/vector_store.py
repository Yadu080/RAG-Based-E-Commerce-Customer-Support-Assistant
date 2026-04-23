"""
vector_store.py
ChromaDB operations: ingest, retrieve, delete, stats.
"""
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from backend.embedder import embed_documents, embed_query, update_corpus

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        import chromadb
        os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        _collection = _client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[VectorStore] Collection '{config.CHROMA_COLLECTION}' ready. "
              f"Count: {_collection.count()}")
    return _collection


def ingest_chunks(chunks: List[Dict[str, Any]]) -> int:
    """Embed and upsert chunks into ChromaDB. Returns number of new chunks added."""
    col = _get_collection()
    texts = [c["page_content"] for c in chunks]
    # Update TF-IDF corpus vocabulary before embedding
    update_corpus(texts)
    embeddings = embed_documents(texts)

    ids         = [c["metadata"]["chunk_hash"] for c in chunks]
    metadatas   = [c["metadata"] for c in chunks]

    # Upsert in batches of 100 (ChromaDB recommendation)
    batch_size = 100
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch_ids   = ids[i:i+batch_size]
        batch_embs  = embeddings[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]
        batch_meta  = metadatas[i:i+batch_size]
        col.upsert(
            ids=batch_ids,
            embeddings=batch_embs,
            documents=batch_texts,
            metadatas=batch_meta,
        )
        total += len(batch_ids)

    print(f"[VectorStore] Upserted {total} chunks. Total: {col.count()}")
    return total


def retrieve(query: str, k: int = None) -> List[Dict[str, Any]]:
    """
    Similarity search. Returns list of dicts with keys:
      text, score, source, page, chunk_index
    """
    k = k or config.TOP_K
    col = _get_collection()
    if col.count() == 0:
        return []

    q_emb = embed_query(query)
    results = col.query(
        query_embeddings=[q_emb],
        n_results=min(k, col.count()),
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        # ChromaDB cosine distance is in [0, 2]; convert to similarity [0, 1]
        score = max(0.0, 1.0 - dist / 2.0)
        chunks.append({
            "text":        doc,
            "score":       round(score, 4),
            "source":      meta.get("source", "unknown"),
            "page":        meta.get("page", 0),
            "chunk_index": meta.get("chunk_index", 0),
        })

    # Sort by score descending
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks


def get_stats() -> Dict[str, Any]:
    """Return collection statistics."""
    try:
        col = _get_collection()
        count = col.count()
        # Get unique sources
        if count > 0:
            sample = col.get(limit=min(count, 1000), include=["metadatas"])
            sources = list({m.get("source", "unknown") for m in sample["metadatas"]})
        else:
            sources = []
        return {"total_chunks": count, "sources": sources, "status": "ready"}
    except Exception as e:
        return {"total_chunks": 0, "sources": [], "status": f"error: {str(e)}"}


def clear_collection() -> bool:
    """Delete and recreate the collection."""
    global _collection
    try:
        import chromadb
        col = _get_collection()
        _client.delete_collection(config.CHROMA_COLLECTION)
        _collection = _client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        print("[VectorStore] Collection cleared.")
        return True
    except Exception as e:
        print(f"[VectorStore] Clear failed: {e}")
        return False
