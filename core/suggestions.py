from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from config.settings import GROQ_API_KEY

_SUGGESTION_SYSTEM = """You are an academic assistant. 
Read the provided excerpt from a research paper (usually the title and abstract). 
Suggest exactly 3 insightful questions a user could ask to explore this paper further. 
Return ONLY the 3 questions, one per line, starting each with a hyphen (-). Do not include any intro or outro text."""

def generate_suggested_questions(text_content: str) -> list[str]:
    """Generates 3 suggested questions based on the first few pages of a PDF."""
    try:
        # Using the faster/cheaper 8b model for a quick UX response
        llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0.3)
        response = llm.invoke([
            SystemMessage(content=_SUGGESTION_SYSTEM),
            HumanMessage(content=f"Paper excerpt:\n{text_content[:2500]}")
        ])
        
        # Parse the bullet points into a list
        lines = response.content.strip().split('\n')
        questions = [line.strip('- ').strip() for line in lines if line.strip().startswith('-')]
        
        return questions[:3]
    except Exception as e:
        print(f"Failed to generate questions: {e}")
        return []