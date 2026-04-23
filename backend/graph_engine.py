"""
graph_engine.py
LangGraph state machine: 6 nodes, conditional routing, HITL support.
"""
import sys
import uuid
import time
from pathlib import Path
from typing import TypedDict, List, Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from backend import query_processor, vector_store, hitl_handler

# ── State schema ──────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    query_id:          str
    session_id:        str
    user_query:        str
    intent:            str
    retrieved_chunks:  List[Dict[str, Any]]
    max_score:         float
    prompt:            str
    llm_response:      str
    human_response:    Optional[str]
    escalated:         bool
    escalation_reason: str
    error:             Optional[str]
    final_answer:      str
    sources:           List[str]
    confidence:        float
    latency_ms:        int
    _start_time:       float


# ── Node functions ─────────────────────────────────────────────────────────────

def input_node(state: GraphState) -> GraphState:
    """Validate input and classify intent."""
    try:
        state["_start_time"] = time.time()
        query = state["user_query"].strip()

        is_valid, err = query_processor.validate_query(query)
        if not is_valid:
            state["error"] = err
            state["final_answer"] = err
            return state

        state["intent"] = query_processor.classify_intent(query)
        state["query_id"] = state.get("query_id") or str(uuid.uuid4())
        state["escalated"] = False
        state["escalation_reason"] = ""
        state["retrieved_chunks"] = []
        state["max_score"] = 0.0
        state["error"] = None
    except Exception as e:
        state["error"] = str(e)
        state["final_answer"] = "An error occurred processing your query. Please try again."
    return state


def retrieval_node(state: GraphState) -> GraphState:
    """Retrieve top-k relevant chunks from ChromaDB."""
    try:
        if state.get("error"):
            return state
        chunks = vector_store.retrieve(state["user_query"], k=config.TOP_K)
        state["retrieved_chunks"] = chunks
        state["max_score"] = chunks[0]["score"] if chunks else 0.0
        state["confidence"] = state["max_score"]
    except Exception as e:
        state["error"] = f"Retrieval error: {str(e)}"
        state["retrieved_chunks"] = []
        state["max_score"] = 0.0
    return state


def router_node(state: GraphState) -> str:
    """
    Conditional routing function (returns next node name).
    Called by LangGraph after input_node and after retrieval.
    """
    if state.get("error"):
        return "output_node"

    # Hard escalation: legal/fraud keywords
    if query_processor.has_hard_escalation(state["user_query"]):
        state["escalation_reason"] = "keyword:legal_or_fraud"
        return "hitl_node"

    # Intent-based escalation
    if state["intent"] in ("ESCALATE", "COMPLAINT"):
        state["escalation_reason"] = f"intent:{state['intent'].lower()}"
        return "hitl_node"

    # Confidence-based escalation (post-retrieval)
    if state.get("max_score", 0) < config.CONFIDENCE_THRESHOLD:
        state["escalation_reason"] = f"low_confidence:{state.get('max_score', 0):.2f}"
        return "hitl_node"

    if len(state.get("retrieved_chunks", [])) < 1:
        state["escalation_reason"] = "no_relevant_chunks"
        return "hitl_node"

    return "generation_node"


def _build_prompt(chunks: List[Dict], query: str) -> str:
    """Assemble the LLM prompt from retrieved chunks and the user query."""
    context_blocks = []
    for i, c in enumerate(chunks, 1):
        src = f"[Source: {c['source']} | Page {c['page']} | Relevance: {c['score']:.0%}]"
        context_blocks.append(f"{src}\n{c['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    return (
        f"{config.SYSTEM_PROMPT}\n\n"
        f"=== KNOWLEDGE BASE CONTEXT ===\n{context}\n"
        f"=== END CONTEXT ===\n\n"
        f"Customer Question: {query}\n\n"
        f"Support Assistant:"
    )


def generation_node(state: GraphState) -> GraphState:
    """Call the LLM via OpenRouter to generate a grounded answer."""
    try:
        if state.get("error"):
            return state

        state["prompt"] = _build_prompt(state["retrieved_chunks"], state["user_query"])

        if not config.GROQ_API_KEY:
            # Demo mode — no API key configured
            chunks = state["retrieved_chunks"]
            if chunks:
                state["llm_response"] = (
                    "⚠️ DEMO MODE — Add your free GROQ_API_KEY to .env to get real AI responses.\n\n"
                    "Get a free key at https://console.groq.com (no credit card needed).\n\n"
                    f"Retrieved context preview:\n{chunks[0]['text'][:300]}..."
                )
            else:
                state["llm_response"] = (
                    "⚠️ DEMO MODE — No API key set and no relevant chunks found. "
                    "Add GROQ_API_KEY to .env and upload a knowledge base document."
                )
            state["final_answer"] = state["llm_response"]
            return state

        from openai import OpenAI
        client = OpenAI(
            api_key=config.GROQ_API_KEY,
            base_url=config.GROQ_BASE_URL,
        )
        retries = 2
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": config.SYSTEM_PROMPT},
                        {"role": "user",   "content":
                            f"Context from knowledge base:\n"
                            + "\n\n".join(
                                f"[{c['source']} p.{c['page']}] {c['text']}"
                                for c in state["retrieved_chunks"]
                            )
                            + f"\n\nCustomer question: {state['user_query']}"
                        }
                    ],
                    max_tokens=config.LLM_MAX_TOKENS,
                    temperature=config.LLM_TEMPERATURE,
                )
                state["llm_response"] = response.choices[0].message.content.strip()
                break
            except Exception as e:
                if attempt == retries - 1:
                    raise e
                time.sleep(1)

        # Check if LLM admitted insufficient info → escalate
        no_info_phrases = ["don't have enough information", "cannot answer", "connect you with our support"]
        if any(p in state["llm_response"].lower() for p in no_info_phrases):
            state["escalation_reason"] = "llm_insufficient_context"
            state["escalated"] = True
            qid = hitl_handler.enqueue(state)
            state["query_id"] = qid
            state["final_answer"] = (
                "I don't have enough information in our knowledge base to answer this accurately. "
                "I've escalated your query to our support team — you'll receive a response shortly."
            )
        else:
            state["final_answer"] = state["llm_response"]

    except Exception as e:
        state["error"] = f"LLM error: {str(e)}"
        state["final_answer"] = (
            "I'm temporarily unable to generate a response. "
            "Please try again or contact our support team."
        )
    return state


def hitl_node(state: GraphState) -> GraphState:
    """Escalate to human agent queue."""
    try:
        state["escalated"] = True
        qid = hitl_handler.enqueue(state)
        state["query_id"] = qid
        state["final_answer"] = (
            "Your query requires attention from our support team. "
            "A human agent will review your message and respond shortly. "
            "Your ticket ID is: **" + qid[:8].upper() + "**"
        )
    except Exception as e:
        state["error"] = f"HITL error: {str(e)}"
        state["final_answer"] = "Your query has been escalated to our support team."
    return state


def output_node(state: GraphState) -> GraphState:
    """Format final response with citations and metrics."""
    # Build source list
    sources = []
    seen = set()
    for c in state.get("retrieved_chunks", []):
        key = f"{c['source']} (p.{c['page']})"
        if key not in seen:
            sources.append(key)
            seen.add(key)
    state["sources"] = sources
    state["confidence"] = round(state.get("max_score", 0.0), 4)
    state["latency_ms"] = int((time.time() - state.get("_start_time", time.time())) * 1000)
    return state


# ── Build LangGraph ───────────────────────────────────────────────────────────

def _build_graph():
    from langgraph.graph import StateGraph, END

    builder = StateGraph(GraphState)

    # Register nodes
    builder.add_node("input_node",      input_node)
    builder.add_node("retrieval_node",  retrieval_node)
    builder.add_node("generation_node", generation_node)
    builder.add_node("hitl_node",       hitl_node)
    builder.add_node("output_node",     output_node)

    # Entry point
    builder.set_entry_point("input_node")

    # After input: if error, go directly to output; otherwise retrieve first
    def after_input(state: GraphState) -> str:
        if state.get("error"):
            return "output_node"
        return "retrieval_node"

    builder.add_conditional_edges(
        "input_node",
        after_input,
        {"retrieval_node": "retrieval_node", "output_node": "output_node"}
    )

    # After retrieval: router decides generate or escalate
    builder.add_conditional_edges(
        "retrieval_node",
        router_node,
        {
            "generation_node": "generation_node",
            "hitl_node":       "hitl_node",
            "output_node":     "output_node",
        }
    )

    # Generation and HITL both go to output
    builder.add_edge("generation_node", "output_node")
    builder.add_edge("hitl_node",       "output_node")
    builder.add_edge("output_node",     END)

    return builder.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
        print("[GraphEngine] LangGraph compiled successfully.")
    return _graph


def run_query(user_query: str, session_id: str = "") -> Dict[str, Any]:
    """Main entry point: run a query through the full LangGraph pipeline."""
    graph = get_graph()

    initial_state: GraphState = {
        "query_id":          str(uuid.uuid4()),
        "session_id":        session_id or str(uuid.uuid4()),
        "user_query":        user_query,
        "intent":            "",
        "retrieved_chunks":  [],
        "max_score":         0.0,
        "prompt":            "",
        "llm_response":      "",
        "human_response":    None,
        "escalated":         False,
        "escalation_reason": "",
        "error":             None,
        "final_answer":      "",
        "sources":           [],
        "confidence":        0.0,
        "latency_ms":        0,
        "_start_time":       time.time(),
    }

    result = graph.invoke(initial_state)
    return {
        "query_id":    result["query_id"],
        "answer":      result["final_answer"],
        "sources":     result["sources"],
        "confidence":  result["confidence"],
        "escalated":   result["escalated"],
        "intent":      result["intent"],
        "latency_ms":  result["latency_ms"],
        "error":       result.get("error"),
    }
