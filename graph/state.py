from dataclasses import dataclass,field
from typing import List, Dict, Any

@dataclass
class RAGState:
    user_question: str = ""
    retrieved_chunks: List[Dict[str, Any]] = None
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    answer: str = ""