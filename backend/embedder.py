"""
embedder.py
Pure-NumPy TF-IDF embedder — works offline, no model downloads needed.
For production: swap _tfidf_embed() for sentence-transformers or OpenAI ada-002.

How it works:
  1. All ingested texts build a shared vocabulary (top 2000 terms by IDF).
  2. Each text is TF-IDF encoded into a 2000-dim float32 vector.
  3. Vectors are L2-normalised before storage so cosine sim == dot product.

Accuracy note: Neural embeddings (all-MiniLM-L6-v2) outperform TF-IDF on
  paraphrase/semantic queries. TF-IDF excels on exact and keyword-based queries.
  For the demo, TF-IDF is sufficient to demonstrate the full RAG pipeline.
"""
import re
import math
import hashlib
import numpy as np
from collections import Counter
from typing import List, Dict

# ── State (survives across calls in the same process) ─────────────────────────
_vocab:     Dict[str, int] = {}   # term → column index
_idf:       np.ndarray     = None  # shape (V,)
_corpus_tfs: List[Dict]   = []    # list of term-freq dicts for IDF update
_dim = 2000                        # output dimensionality

_cache: Dict[str, List[float]] = {}


# ── Tokenisation ──────────────────────────────────────────────────────────────
def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    # Simple stopword removal
    stops = {
        "the","a","an","is","it","in","of","to","for","and","or","but","not",
        "with","on","at","by","from","as","be","was","are","were","this","that",
        "have","has","had","do","does","did","will","would","can","could","should",
        "may","might","shall","its","their","our","your","my","we","you","i",
        "he","she","they","them","him","her","us","me","his","what","which",
        "how","when","where","who","why","so","if","then","than","any","all",
        "been","no","up","out","about","into","through","after","before","over",
    }
    return [t for t in tokens if t not in stops and len(t) > 1]


def _tf(tokens: List[str]) -> Dict[str, float]:
    """Raw term frequency (not normalised — IDF does the heavy lifting)."""
    cnt = Counter(tokens)
    n   = max(len(tokens), 1)
    return {t: c / n for t, c in cnt.items()}


# ── Vocabulary & IDF management ───────────────────────────────────────────────
def _rebuild_vocab(tfs: List[Dict]) -> None:
    global _vocab, _idf
    N = len(tfs)
    if N == 0:
        return
    # Document frequency
    df: Dict[str, int] = {}
    for tf_dict in tfs:
        for term in tf_dict:
            df[term] = df.get(term, 0) + 1
    # IDF: log((N+1)/(df+1)) + 1
    all_terms = sorted(df.keys(), key=lambda t: -df[t])
    # Keep top _dim terms
    top = all_terms[:_dim]
    _vocab = {t: i for i, t in enumerate(top)}
    idf_arr = np.zeros(len(_vocab), dtype=np.float32)
    for t, i in _vocab.items():
        idf_arr[i] = math.log((N + 1) / (df.get(t, 0) + 1)) + 1.0
    _idf = idf_arr


def _tfidf_vector(text: str) -> np.ndarray:
    """Convert text to a normalised TF-IDF vector in the current vocabulary."""
    if _idf is None or len(_vocab) == 0:
        # Before any corpus is loaded: return random unit vector for graceful degradation
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(_dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    tokens = _tokenize(text)
    tf_dict = _tf(tokens)
    V = len(_vocab)
    vec = np.zeros(V, dtype=np.float32)
    for term, tf_val in tf_dict.items():
        if term in _vocab:
            vec[_vocab[term]] = tf_val * _idf[_vocab[term]]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    else:
        # Out-of-vocabulary query: small uniform signal
        vec = np.ones(V, dtype=np.float32) / math.sqrt(V)
    return vec


# ── Public API ────────────────────────────────────────────────────────────────
def update_corpus(texts: List[str]) -> None:
    """
    Call this after ingesting new documents to update the IDF model.
    Adds new TF dicts to the corpus and rebuilds vocab.
    """
    global _corpus_tfs
    for text in texts:
        tokens = _tokenize(text)
        _corpus_tfs.append(_tf(tokens))
    _rebuild_vocab(_corpus_tfs)


def _key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def embed_query(text: str) -> List[float]:
    """Embed a single query string, with caching."""
    k = _key(text)
    if k in _cache:
        return _cache[k]
    vec = _tfidf_vector(text).tolist()
    _cache[k] = vec
    return vec


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Batch embed texts, using cache where possible."""
    results = []
    for text in texts:
        k = _key(text)
        if k not in _cache:
            _cache[k] = _tfidf_vector(text).tolist()
        results.append(_cache[k])
    return results
