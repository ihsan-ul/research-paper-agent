"""
memory/session_memory.py
Manages per-session conversation history stored in Streamlit's session state.
Provides LangChain-compatible message history for the agent.
"""

from typing import List
from langchain.schema import BaseMessage, HumanMessage, AIMessage


def get_history(session_state: dict) -> List[BaseMessage]:
    """Returns the current conversation history as LangChain messages."""
    if "chat_history" not in session_state:
        session_state["chat_history"] = []
    return session_state["chat_history"]


def append_user_message(session_state: dict, content: str) -> None:
    history = get_history(session_state)
    history.append(HumanMessage(content=content))


def append_ai_message(session_state: dict, content: str) -> None:
    history = get_history(session_state)
    history.append(AIMessage(content=content))


def clear_history(session_state: dict) -> None:
    session_state["chat_history"] = []


def format_history_for_prompt(session_state: dict, max_turns: int = 6) -> str:
    """
    Formats the last N turns as a string for injection into the agent prompt.
    Keeps context window manageable.
    """
    history = get_history(session_state)
    recent = history[-(max_turns * 2):]   # each turn = 1 human + 1 AI message

    lines = []
    for msg in recent:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")

    return "\n".join(lines)
