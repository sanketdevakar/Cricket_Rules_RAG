import os
from langgraph.graph import StateGraph
from retriever.milvus_retriever import milvus_retrieve
from graph.state import RAGState
import requests
import json
from typing import Dict, Generator, Optional, Union
from dotenv import load_dotenv

load_dotenv()

class LLMCaller:
    def __init__(self, model_name: str = "llama-3.1-8b-instant", api_key: str = None):
        """
        Initialize Groq LLM caller
        Args:
            model_name: Name of the Groq model to use
            api_key: Groq API key (will fall back to environment variable)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key not found. Set GROQ_API_KEY environment variable or pass it to LLMCaller")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def call_llm(self, prompt: str, stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """
        Call Groq LLM API
        Args:
            prompt: The prompt to send
            stream: Whether to stream the response
        Returns:
            Either a string response or a generator for streaming
        """
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": 0.7,
        }

        if not stream:
            response = requests.post(self.url, headers=self.headers, json=payload)
            if response.status_code != 200:
                raise RuntimeError(f"Groq Error [{response.status_code}]: {response.text}")
            data = response.json()
            return data["choices"][0]["message"]["content"]

        def generate_stream():
            with requests.post(self.url, headers=self.headers, json=payload, stream=True) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"Groq Error [{response.status_code}]: {response.text}")
                for line in response.iter_lines(decode_unicode=True):
                    if not line or line.strip() == "":
                        continue
                    if line.startswith("data: "):
                        line = line[6:]  # Remove "data: " prefix
                    if line.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(line)
                        if content := data["choices"][0]["delta"].get("content"):
                            yield content
                    except json.JSONDecodeError:
                        continue

        return generate_stream()


# Create LLM instance to be used across nodes
llm_caller = LLMCaller()

def grader_node(state: RAGState) -> Dict:
    """
    Grader node calls an llm to grade each retrieved chunk individually.
    Returns updated state with filtered retrieved chunks.
    """
    retrieved_chunks = state.retrieved_chunks
    user_question = state.user_question
    filtered_chunks = []

    # Grade each chunk individually
    for chunk in retrieved_chunks:
        snippet = chunk.get("text", "").strip()
        context = f"LAW {chunk.get('law_number')} - {chunk.get('law_title')}\n{snippet}"

        prompt = f"""
You are an expert at evaluating the relevance of cricket laws to specific questions.
Given the LAW TEXT and QUESTION below, rate the relevance on a scale from 1 to 10.
Only provide a single number as response.

LAW TEXT:
{context}

QUESTION:
{user_question}

RELEVANCE SCORE (just the number 1-10):
"""
        
        response = llm_caller.call_llm(prompt, stream=False)
        print(f"Grader response for chunk {chunk.get('law_number')}: {response}")
        try:
            relevance_score = int(response.strip())
            # Add relevance score to chunk metadata
            chunk["relevance_score"] = relevance_score
            if relevance_score >= 3:  # threshold for relevance
                filtered_chunks.append(chunk)
        except ValueError:
            print(f"Warning: Invalid relevance score received: {response}")
            continue

    # Sort chunks by relevance score
    filtered_chunks.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    return {
        "user_question": state.user_question,
        "retrieved_chunks": filtered_chunks
    }

def llm_answer_node(state: RAGState) -> Dict:
    """
    LLM node that generates answers using retrieved chunks from state.
    Returns a dictionary with state updates including the answer stream.
    """
    retrieved_chunks = state.retrieved_chunks
    user_question = state.user_question

    context_parts = []
    for c in retrieved_chunks:
        snippet = c.get("text", "").strip()
        context_parts.append(f"LAW {c.get('law_number')} - {c.get('law_title')}\n{snippet}")
    context = "\n\n".join(context_parts)

    prompt = f"""
You are a precise cricket laws assistant. Use ONLY the following CONTEXT to answer the QUESTION.
And present the answer in numerical points if applicable. 
Cite the relevant law numbers (for example: Law 34) in your answer where applicable. Only cite the laws whose points you have used in your answer.
If the context does not contain the answer, respond with "I don't know from the given rules." 
Do not hallucinate or add facts outside the provided context.

CONTEXT:
{context}

QUESTION:
{user_question}

FINAL ANSWER (be concise, cite laws inline or at the end):
"""

    # Get streaming response using the LLMCaller instance
    print(" \n -------- FINAL ANSWER -------\n")
    answer_stream = llm_caller.call_llm(prompt, stream=True)

    return {
        "user_question": state.user_question,
        "retrieved_chunks": state.retrieved_chunks,
        "answer": answer_stream
    }


# Define the graph architecture
rag_graph = StateGraph(RAGState)

# Add nodes
rag_graph.add_node("retrieve", milvus_retrieve)
rag_graph.add_node("grader",grader_node)
rag_graph.add_node("llm", llm_answer_node)

# Define the entry point and edges
rag_graph.set_entry_point("retrieve")
rag_graph.add_edge("retrieve", "grader")
rag_graph.add_edge("grader", "llm")
rag_graph.set_finish_point("llm")

# Compile the graph
workflow = rag_graph.compile()