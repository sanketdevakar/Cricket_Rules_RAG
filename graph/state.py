from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class RAGState:
    user_question: str = ""
    retrieved_chunks: List[Dict[str, Any]] = None
    answer: str = ""