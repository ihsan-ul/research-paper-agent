"""
config/settings.py
Central configuration — loads from .env and exposes typed settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Tracing ───────────────────────────────────────────────────────────────────
LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "research-paper-agent")

# Set env vars that LangSmith reads automatically
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# ── Web Search ────────────────────────────────────────────────────────────────
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# ── Reranking ─────────────────────────────────────────────────────────────────
COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")

# ── Vector Store ──────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "research_papers")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL: int = 6       # docs fetched from ChromaDB
TOP_K_RERANKED: int = 3        # docs kept after Cohere reranking

# ── Guardrails ────────────────────────────────────────────────────────────────
MAX_INPUT_CHARS: int = 4000    # reject suspiciously huge inputs
BLOCKED_PATTERNS: list[str] = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard your",
    "forget your instructions",
    "you are now",
    "act as if",
    "jailbreak",
    "pretend you are",
    "override",
    "system prompt",
    "bypass",
    "do anything now",
    "dan mode",
]
