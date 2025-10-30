from langgraph.graph import StateGraph
from retriever.milvus_retriever import milvus_retrieve
from graph.state import RAGState
import requests
import json
from typing import Dict, Generator

def llm_answer_node(state: RAGState) -> Dict:
    """
    LLM node that generates answers using retrieved chunks from state.
    Returns a dictionary with state updates including the answer stream.
    """
    retrieved_chunks = state.retrieved_chunks
    user_question = state.user_question
    model_name = "llama3.2:3b"  # Using a more capable model
    url = "http://localhost:11434/api/generate"

    context_parts = []
    for c in retrieved_chunks:
        snippet = c.get("text", "").strip()
        context_parts.append(f"LAW {c.get('law_number')} - {c.get('law_title')}\n{snippet}")
    context = "\n\n".join(context_parts)

    prompt = f"""
You are a precise cricket laws assistant. Use ONLY the following CONTEXT to answer the QUESTION. 
Cite the relevant law numbers (for example: Law 34) in your answer where applicable.
If the context does not contain the answer, respond with "I don't know from the given rules." 
Do not hallucinate or add facts outside the provided context.

CONTEXT:
{context}

QUESTION:
{user_question}

FINAL ANSWER (be concise, cite laws inline or at the end):
"""

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True
    }

    def generate_stream() -> Generator[str, None, None]:
        with requests.post(url, json=payload, stream=True) as response:
            if response.status_code != 200:
                raise RuntimeError(f"Ollama Error [{response.status_code}]: {response.text}")
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                except json.JSONDecodeError:
                    continue

    # Return state updates with the stream
    return {
        "user_question": state.user_question,
        "retrieved_chunks": state.retrieved_chunks,
        "answer": generate_stream()
    }

# Define the graph architecture
rag_graph = StateGraph(RAGState)

# Add nodes
rag_graph.add_node("retrieve", milvus_retrieve)
rag_graph.add_node("llm", llm_answer_node)

# Define the entry point and edges
rag_graph.set_entry_point("retrieve")
rag_graph.add_edge("retrieve", "llm")
rag_graph.set_finish_point("llm")

# Compile the graph
workflow = rag_graph.compile()