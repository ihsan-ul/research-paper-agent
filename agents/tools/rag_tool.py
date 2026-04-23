"""
agents/tools/rag_tool.py
LangChain tool that retrieves relevant chunks from ChromaDB + Cohere reranking.
"""

from langchain.tools import tool
from core.vector_store import retrieve_and_rerank


@tool
def rag_retrieval_tool(query: str) -> str:
    """
    Search the uploaded research papers for information relevant to the query.
    Use this tool when the user asks about content from their uploaded PDFs,
    specific paper findings, methods, results, or citations.

    Input: a focused search query (not the full user message).
    Output: relevant excerpts from the papers with source + page metadata.
    """
    docs = retrieve_and_rerank(query)

    if not docs:
        return "No relevant content found in the uploaded papers for this query."

    results = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        results.append(
            f"[Source {i}: {source}, Page {page}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(results)
