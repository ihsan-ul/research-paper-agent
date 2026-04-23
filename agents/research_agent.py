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

def _llm():
    """Returns the Groq LLM instance with retry logic."""
    return ChatGroq(
        model="llama-3.1-70b-versatile", 
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        max_retries=3,  
    )


# ── Node 1: Guardrail ─────────────────────────────────────────────────────────

def guardrail_node(state: AgentState) -> AgentState:
    result = check_input(state["user_input"])
    state["guard_passed"] = result.is_safe
    state["guard_reason"] = result.reason
    return state


# ── Node 2: Router ────────────────────────────────────────────────────────────

_ROUTER_SYSTEM = """You are a routing agent for a research paper assistant.
Given the user's message and the list of indexed papers, decide the best action:

- "both"      → ALWAYS use this for complex research questions, comparisons, 
                or when synthesis between papers and general knowledge is needed.
- "rag"       → use ONLY if the user asks for a very specific detail known to be in a paper.
- "web"       → use if no relevant papers are uploaded.
- "summarize" → user asks for a summary of a specific uploaded paper.
- "chat"      → general greeting or conversation.

Reply with ONLY one of those five words. Prioritize "both" for deep research.
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

_SYNTH_SYSTEM = """You are an expert academic research assistant specializing in synthesis.
Your goal is to provide a comprehensive answer by cross-referencing multiple sources.

Rules:
- Compare and contrast information found in the uploaded papers with information from the web.
- Highlight where the sources agree or where the web provides more recent context.
- Always ground your answer in the provided context and cite filenames/pages for papers.
- If the papers and web provide conflicting info, report both perspectives.
- Do NOT hallucinate. Use markdown (headers, bolding, bullets) for professional structure.
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
        return "rag_agent" # Sequence: RAG -> then Web
    elif route == "summarize":
        return "summarizer"
    else:
        return "synthesizer"

def after_rag(state: AgentState) -> str:
    """
    Decides whether to proceed to web search.
    - Proceeds if route is 'both'.
    - FALLBACK: Proceeds if RAG found no relevant content.
    """
    if state.get("route") == "both":
        return "web_agent"
    
    # Check if RAG tool returned its 'not found' message
    rag_out = state.get("rag_context", "")
    if "No relevant content found" in rag_out:
        return "web_agent"
        
    return "synthesizer"


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
