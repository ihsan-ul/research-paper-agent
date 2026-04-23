"""
core/embeddings.py
Loads HuggingFace sentence-transformers embeddings locally (no API key needed).
Cached so the model is only loaded once per session.
"""

from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns a cached HuggingFace embedding model.
    First call downloads the model (~90 MB), subsequent calls are instant.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity ready
    )
