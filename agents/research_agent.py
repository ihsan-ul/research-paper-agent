"""
agents/research_agent.py
LangGraph multi-agent orchestrator.

Graph nodes:
  guardrail  → checks input safety
  router     → decides which agent(s) to invoke
  rag_agent  → retrieves from ChromaDB + reranks
  web_agent  → searches the web via Tavily
  synthesizer→ merges outputs into a final answer

State flows through the graph as a typed dict.
LangSmith traces every run automatically.
"""

from typing import TypedDict, Annotated, List, Optional
import operator

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

from config.settings import GROQ_API_KEY, GROQ_MODEL
from guardrails.prompt_guard import check_input
from agents.tools.rag_tool import rag_retrieval_tool
from agents.tools.web_search_tool import web_search_tool
from agents.tools.summarizer_tool import summarize_paper_tool


# ── Shared graph state ────────────────────────────────────────────────────────

class AgentState(TypedDict):
    user_input: str
    chat_history: str                           # formatted conversation turns
    indexed_sources: List[str]                  # filenames in the vector store
    guard_passed: bool
    guard_reason: Optional[str]
    route: Optional[str]                        # "rag" | "web" | "both" | "summarize"
    rag_context: Optional[str]
    web_context: Optional[str]
    final_answer: Optional[str]


# ── LLM helper ────────────────────────────────────────────────────────────────

def _llm(temperature: float = 0.3) -> ChatGroq:
    return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=temperature)


# ── Node 1: Guardrail ─────────────────────────────────────────────────────────

def guardrail_node(state: AgentState) -> AgentState:
    result = check_input(state["user_input"])
    state["guard_passed"] = result.is_safe
    state["guard_reason"] = result.reason
    return state


# ── Node 2: Router ────────────────────────────────────────────────────────────

_ROUTER_SYSTEM = """You are a routing agent for a research paper assistant.
Given the user's message and the list of indexed papers, decide the best action:

- "rag"       → answer using the uploaded papers only
- "web"       → answer using a web search only (no relevant papers uploaded)
- "both"      → answer using BOTH papers and web search
- "summarize" → user is asking for a summary of one of the uploaded papers
- "chat"      → general conversation, no retrieval needed

Reply with ONLY one of those five words.
"""

def router_node(state: AgentState) -> AgentState:
    sources_str = ", ".join(state["indexed_sources"]) if state["indexed_sources"] else "none"
    prompt = (
        f"Indexed papers: {sources_str}\n\n"
        f"Conversation so far:\n{state['chat_history']}\n\n"
        f"User message: {state['user_input']}"
    )
    response = _llm(temperature=0).invoke([
        SystemMessage(content=_ROUTER_SYSTEM),
        HumanMessage(content=prompt),
    ])
    route = response.content.strip().lower()
    if route not in {"rag", "web", "both", "summarize", "chat"}:
        route = "rag" if state["indexed_sources"] else "web"
    state["route"] = route
    return state


# ── Node 3: RAG Agent ─────────────────────────────────────────────────────────

def rag_agent_node(state: AgentState) -> AgentState:
    context = rag_retrieval_tool.invoke(state["user_input"])
    state["rag_context"] = context
    return state


# ── Node 4: Web Agent ─────────────────────────────────────────────────────────

def web_agent_node(state: AgentState) -> AgentState:
    context = web_search_tool.invoke(state["user_input"])
    state["web_context"] = context
    return state


# ── Node 5: Summarizer Agent ──────────────────────────────────────────────────

def summarizer_node(state: AgentState) -> AgentState:
    """Detects which paper the user is asking about and summarizes it."""
    sources = state["indexed_sources"]
    if not sources:
        state["final_answer"] = "No papers are uploaded yet. Please upload a PDF first."
        return state

    # Ask the LLM to pick the correct filename
    pick_prompt = (
        f"The user asked: '{state['user_input']}'\n"
        f"Available papers: {sources}\n"
        f"Reply with ONLY the filename that best matches the user's request. "
        f"If unclear, reply with the first filename."
    )
    response = _llm(temperature=0).invoke([HumanMessage(content=pick_prompt)])
    filename = response.content.strip()
    if filename not in sources:
        filename = sources[0]

    state["final_answer"] = summarize_paper_tool.invoke(filename)
    return state


# ── Node 6: Synthesizer ───────────────────────────────────────────────────────

_SYNTH_SYSTEM = """You are an expert academic research assistant.
Your role is to give precise, well-grounded answers about research papers.

Rules:
- Always ground your answer in the provided context.
- If citing a paper, mention the source filename and page number.
- Do NOT hallucinate — if you don't know, say so.
- Be concise but complete.
- Use markdown formatting (headers, bullets) for clarity.
"""

def synthesizer_node(state: AgentState) -> AgentState:
    context_parts = []
    if state.get("rag_context"):
        context_parts.append(f"[From uploaded papers]\n{state['rag_context']}")
    if state.get("web_context"):
        context_parts.append(f"[From web search]\n{state['web_context']}")

    if not context_parts:
        # Pure chat mode
        response = _llm().invoke([
            SystemMessage(content=_SYNTH_SYSTEM),
            HumanMessage(content=f"History:\n{state['chat_history']}\n\nUser: {state['user_input']}"),
        ])
        state["final_answer"] = response.content
        return state

    combined_context = "\n\n".join(context_parts)
    prompt = (
        f"Conversation history:\n{state['chat_history']}\n\n"
        f"Retrieved context:\n{combined_context}\n\n"
        f"User question: {state['user_input']}"
    )
    response = _llm().invoke([
        SystemMessage(content=_SYNTH_SYSTEM),
        HumanMessage(content=prompt),
    ])
    state["final_answer"] = response.content
    return state


# ── Routing logic ─────────────────────────────────────────────────────────────

def after_guard(state: AgentState) -> str:
    return "router" if state["guard_passed"] else END

def after_router(state: AgentState) -> str:
    route = state.get("route", "rag")
    if route == "rag":
        return "rag_agent"
    elif route == "web":
        return "web_agent"
    elif route == "both":
        return "rag_agent"      # rag runs first; both handled via sequential edges
    elif route == "summarize":
        return "summarizer"
    else:
        return "synthesizer"    # "chat" — skip retrieval

def after_rag(state: AgentState) -> str:
    return "web_agent" if state.get("route") == "both" else "synthesizer"


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("guardrail", guardrail_node)
    graph.add_node("router", router_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("web_agent", web_agent_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("guardrail")

    graph.add_conditional_edges("guardrail", after_guard, {"router": "router", END: END})
    graph.add_conditional_edges("router", after_router, {
        "rag_agent": "rag_agent",
        "web_agent": "web_agent",
        "summarizer": "summarizer",
        "synthesizer": "synthesizer",
    })
    graph.add_conditional_edges("rag_agent", after_rag, {
        "web_agent": "web_agent",
        "synthesizer": "synthesizer",
    })
    graph.add_edge("web_agent", "synthesizer")
    graph.add_edge("summarizer", END)
    graph.add_edge("synthesizer", END)

    return graph.compile()


# ── Public entry point ────────────────────────────────────────────────────────

_compiled_graph = None

def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent(user_input: str, chat_history: str, indexed_sources: List[str]) -> dict:
    """
    Main entry point called by the Streamlit frontend.
    Returns the full final state dict.
    """
    graph = get_graph()
    initial_state: AgentState = {
        "user_input": user_input,
        "chat_history": chat_history,
        "indexed_sources": indexed_sources,
        "guard_passed": False,
        "guard_reason": None,
        "route": None,
        "rag_context": None,
        "web_context": None,
        "final_answer": None,
    }
    result = graph.invoke(initial_state)
    return result
