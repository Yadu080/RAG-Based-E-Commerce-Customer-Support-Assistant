"""
hitl_handler.py
File-based HITL queue management.
Status lifecycle: PENDING → IN_REVIEW → RESOLVED
"""
import sys
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

os.makedirs(config.HITL_QUEUE_DIR, exist_ok=True)


def _path(query_id: str) -> str:
    return os.path.join(config.HITL_QUEUE_DIR, f"{query_id}.json")


def enqueue(state: Dict[str, Any]) -> str:
    """Create a HITL queue entry. Returns query_id."""
    query_id = state.get("query_id") or str(uuid.uuid4())
    payload = {
        "query_id":         query_id,
        "session_id":       state.get("session_id", ""),
        "user_query":       state.get("user_query", ""),
        "intent":           state.get("intent", "UNKNOWN"),
        "escalation_reason":state.get("escalation_reason", "unspecified"),
        "retrieved_chunks": state.get("retrieved_chunks", []),
        "timestamp":        datetime.utcnow().isoformat() + "Z",
        "status":           "PENDING",
        "human_response":   None,
        "agent_id":         None,
        "resolved_at":      None,
    }
    with open(_path(query_id), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[HITL] Queued query {query_id} (reason: {payload['escalation_reason']})")
    return query_id


def get_entry(query_id: str) -> Optional[Dict[str, Any]]:
    """Read a queue entry by ID."""
    p = _path(query_id)
    if not os.path.exists(p):
        return None
    with open(p, "r") as f:
        return json.load(f)


def resolve(query_id: str, human_response: str, agent_id: str = "human_agent") -> bool:
    """Mark a HITL entry as resolved with the human's response."""
    entry = get_entry(query_id)
    if not entry:
        return False
    entry["status"]         = "RESOLVED"
    entry["human_response"] = human_response
    entry["agent_id"]       = agent_id
    entry["resolved_at"]    = datetime.utcnow().isoformat() + "Z"
    with open(_path(query_id), "w") as f:
        json.dump(entry, f, indent=2)
    print(f"[HITL] Resolved {query_id} by {agent_id}")
    return True


def list_pending() -> list:
    """Return all PENDING entries for the agent dashboard."""
    pending = []
    for fname in os.listdir(config.HITL_QUEUE_DIR):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(config.HITL_QUEUE_DIR, fname)) as f:
                    entry = json.load(f)
                if entry.get("status") == "PENDING":
                    pending.append(entry)
            except Exception:
                pass
    return sorted(pending, key=lambda x: x.get("timestamp", ""), reverse=True)


def list_all() -> list:
    """Return all HITL entries."""
    entries = []
    for fname in os.listdir(config.HITL_QUEUE_DIR):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(config.HITL_QUEUE_DIR, fname)) as f:
                    entries.append(json.load(f))
            except Exception:
                pass
    return sorted(entries, key=lambda x: x.get("timestamp", ""), reverse=True)


def get_stats() -> Dict[str, int]:
    all_entries = list_all()
    counts = {"PENDING": 0, "IN_REVIEW": 0, "RESOLVED": 0}
    for e in all_entries:
        s = e.get("status", "PENDING")
        counts[s] = counts.get(s, 0) + 1
    return counts
