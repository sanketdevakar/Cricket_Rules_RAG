from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class RAGState(TypedDict):
    """State for RAG workflow with built-in message history"""
    messages: Annotated[list, add_messages]  # Built-in message handling
    user_question: str
    retrieved_chunks: list
    context_summary: str  # Short summary text carried between turns
    answer: Optional[str]  # Final answer to be displayed