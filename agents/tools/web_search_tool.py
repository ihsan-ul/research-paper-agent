"""
agents/tools/web_search_tool.py
LangChain tool wrapping Tavily for live web search.
Used when the user asks about recent papers, external context, or
information not present in the uploaded PDFs.
"""

from langchain.tools import tool
from tavily import TavilyClient
from config.settings import TAVILY_API_KEY


@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for recent research, papers, or background information
    that may not be in the uploaded PDFs. Use this when:
    - The user asks about recent publications or news
    - The uploaded papers don't contain enough context
    - The user explicitly asks to search the web

    Input: a focused web search query.
    Output: a summary of the top web results.
    """
    if not TAVILY_API_KEY:
        return "Web search is unavailable (TAVILY_API_KEY not set)."

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=4,
            include_answer=True,
        )

        output_parts = []

        # Tavily's synthesized answer (best quality)
        if response.get("answer"):
            output_parts.append(f"Summary: {response['answer']}")

        # Individual results
        for i, result in enumerate(response.get("results", []), start=1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "")[:400]
            output_parts.append(f"[Result {i}] {title}\n{url}\n{content}")

        return "\n\n".join(output_parts) if output_parts else "No results found."

    except Exception as e:
        return f"Web search failed: {str(e)}"
