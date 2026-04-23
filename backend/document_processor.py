"""
document_processor.py
Handles PDF and text file loading, cleaning, and chunking.
"""
import sys
import os
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ── Chunking helper (no LangChain dependency for splitting) ───────────────────
def _recursive_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks using recursive paragraph-first strategy."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks = []

    def split_with_sep(txt, seps):
        if not seps:
            return [txt[i:i+chunk_size] for i in range(0, len(txt), chunk_size-overlap)]
        sep = seps[0]
        parts = txt.split(sep)
        current, current_len = [], 0
        result = []
        for part in parts:
            part_len = len(part)
            if current_len + part_len + len(sep) <= chunk_size:
                current.append(part)
                current_len += part_len + len(sep)
            else:
                if current:
                    result.append(sep.join(current))
                if part_len > chunk_size:
                    result.extend(split_with_sep(part, seps[1:]))
                    current, current_len = [], 0
                else:
                    current = [part]
                    current_len = part_len
        if current:
            result.append(sep.join(current))
        return result

    raw_chunks = split_with_sep(text, separators)

    # Apply overlap: each chunk gets the tail of the previous chunk prepended
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and overlap > 0:
            prev = raw_chunks[i - 1]
            tail = prev[-overlap:] if len(prev) > overlap else prev
            chunk = tail + " " + chunk
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def _chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _clean_text(text: str) -> str:
    """Remove noise: multiple blank lines, page artifacts, non-printable chars."""
    text = re.sub(r'\x00', '', text)                     # null bytes
    text = re.sub(r'\r\n', '\n', text)                   # normalize line endings
    text = re.sub(r'[ \t]+', ' ', text)                  # collapse spaces
    text = re.sub(r'\n{3,}', '\n\n', text)               # max 2 blank lines
    text = text.strip()
    return text


# ── Document loading ──────────────────────────────────────────────────────────

def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Load a PDF file and return list of page dicts."""
    import fitz  # PyMuPDF
    pages = []
    doc = fitz.open(file_path)
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        text = _clean_text(text)
        if len(text.strip()) > 30:   # skip nearly-empty pages
            pages.append({
                "page_content": text,
                "metadata": {
                    "source": Path(file_path).name,
                    "page": page_num + 1,
                    "doc_id": _chunk_hash(file_path + str(page_num)),
                }
            })
    doc.close()
    return pages


def load_text(file_path: str) -> List[Dict[str, Any]]:
    """Load a plain text file as a single 'page'."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    raw = _clean_text(raw)
    # Split by sections (━━━ or ===) to create pseudo-pages
    section_pattern = re.compile(r'(?=\n(?:━{10,}|={10,}|\-{10,}))', re.MULTILINE)
    sections = section_pattern.split(raw)
    pages = []
    for i, section in enumerate(sections):
        section = section.strip()
        if len(section) > 50:
            pages.append({
                "page_content": section,
                "metadata": {
                    "source": Path(file_path).name,
                    "page": i + 1,
                    "doc_id": _chunk_hash(file_path + str(i)),
                }
            })
    return pages


def chunk_documents(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split page documents into overlapping chunks."""
    chunks = []
    chunk_index = 0
    for page in pages:
        text = page["page_content"]
        splits = _recursive_split(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for split in splits:
            if len(split.strip()) < 20:
                continue
            chunk = {
                "page_content": split,
                "metadata": {
                    **page["metadata"],
                    "chunk_index": chunk_index,
                    "chunk_hash": _chunk_hash(split),
                    "char_count": len(split),
                }
            }
            chunks.append(chunk)
            chunk_index += 1
    return chunks


def process_file(file_path: str) -> List[Dict[str, Any]]:
    """Main entry: load + chunk a PDF or text file."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        pages = load_pdf(file_path)
    elif ext in (".txt", ".md"):
        pages = load_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .txt")
    return chunk_documents(pages)
