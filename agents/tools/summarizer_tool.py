"""
agents/tools/summarizer_tool.py
Dedicated tool that produces a structured summary of an uploaded paper.
Retrieves all chunks for a given filename and synthesises a structured summary.
"""

from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

from core.vector_store import get_vector_store
from config.settings import GROQ_API_KEY, GROQ_MODEL


_SUMMARIZER_SYSTEM = """You are an expert academic assistant.
Given excerpts from a research paper, produce a structured summary with these sections:
1. **Problem / Motivation** — what problem does this paper address?
2. **Proposed Method / Approach** — what do the authors propose?
3. **Key Results** — main findings and metrics.
4. **Limitations** — what the authors acknowledge as limitations.
5. **Takeaway** — one-sentence bottom line.

Be concise but precise. Use bullet points within each section.
"""


@tool
def summarize_paper_tool(filename: str) -> str:
    """
    Generate a structured summary of a specific uploaded research paper.
    Use this when the user asks to summarize or get an overview of a paper.

    Input: the filename of the uploaded paper (e.g. 'attention_is_all_you_need.pdf').
    Output: a structured summary covering problem, method, results, and limitations.
    """
    store = get_vector_store()

    # Retrieve all chunks for this specific paper
    try:
        collection = store._collection
        result = collection.get(
            where={"source": filename},
            include=["documents"],
        )
        chunks = result.get("documents", [])
    except Exception as e:
        return f"Could not retrieve chunks for '{filename}': {e}"

    if not chunks:
        return f"No content found for '{filename}'. Make sure the file is uploaded."

    # Use up to ~3000 chars to stay within context limits
    combined_text = "\n\n".join(chunks)[:3000]

    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.2)
    response = llm.invoke([
        SystemMessage(content=_SUMMARIZER_SYSTEM),
        HumanMessage(content=f"Here are excerpts from '{filename}':\n\n{combined_text}"),
    ])

    return response.content
