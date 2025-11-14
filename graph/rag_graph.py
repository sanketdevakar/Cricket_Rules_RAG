import os
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from retriever.milvus_retriever import milvus_retrieve
from graph.state import RAGState
import requests
import json
from typing import Dict, Generator, Optional, Union
from dotenv import load_dotenv
import time

load_dotenv()


class LLMCaller:
    def __init__(self, api_key: str = None):
        """
        Initialize Groq LLM caller.

        Args:
            api_key: Groq API key (will fall back to environment variable)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY environment variable or pass it to LLMCaller."
            )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.url = "https://api.groq.com/openai/v1/chat/completions"

        # Default models
        self.grader_model = "llama-3.1-8b-instant"
        self.answer_model = "llama-3.3-70b-versatile"

    def call_llm(
        self,
        prompt: str,
        model: str = None,
        stream: bool = False,
        retries: int = 5,
        base_delay: int = 5,
        temperature: float = 0.7,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Call Groq LLM API with retry handling and optional streaming.

        Args:
            prompt: Text prompt to send
            model: Model to use ("grader" or "answer" or custom Groq model name)
            stream: Whether to stream the response
            retries: Number of retries for rate-limit errors
            base_delay: Base wait time for retry backoff
            temperature: Sampling temperature

        Returns:
            String response or generator stream
        """

        # Choose model
        if model is None:
            model = self.answer_model  # default to strong one
        elif model.lower() == "grader":
            model = self.grader_model
        elif model.lower() == "answer":
            model = self.answer_model

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
        }

        # Retry loop
        for attempt in range(retries):
            try:
                if not stream:
                    response = requests.post(self.url, headers=self.headers, json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        return data["choices"][0]["message"]["content"].strip()
                    elif response.status_code == 429:
                        wait_time = base_delay * (attempt + 1)
                        print(f"⚠️ Rate limit hit for {model}. Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError(
                            f"❌ Groq Error [{response.status_code}]: {response.text}"
                        )

                # Stream mode
                def generate_stream():
                    with requests.post(
                        self.url, headers=self.headers, json=payload, stream=True
                    ) as response:
                        if response.status_code == 429:
                            wait_time = base_delay * (attempt + 1)
                            print(
                                f"⚠️ Rate limit hit for {model}. Waiting {wait_time}s before retry..."
                            )
                            time.sleep(wait_time)
                            return self.call_llm(
                                prompt, model=model, stream=True, retries=retries - 1
                            )

                        if response.status_code != 200:
                            raise RuntimeError(
                                f"❌ Groq Error [{response.status_code}]: {response.text}"
                            )

                        for line in response.iter_lines(decode_unicode=True):
                            if not line or line.strip() == "":
                                continue
                            if line.startswith("data: "):
                                line = line[6:]
                            if line.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(line)
                                if content := data["choices"][0]["delta"].get("content"):
                                    yield content
                            except json.JSONDecodeError:
                                continue

                return generate_stream()

            except Exception as e:
                if attempt < retries - 1:
                    wait_time = base_delay * (attempt + 1)
                    print(f"⚠️ Error calling {model}, retrying in {wait_time}s... ({e})")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"❌ Fatal LLM error with {model}: {e}")

        raise RuntimeError(f"Exceeded max retries for model {model}")


# Create LLM instance to be used across nodes
llm_caller = LLMCaller()

# ✅ Initialize LangGraph MemorySaver
memory = MemorySaver()


def grader_node(state: RAGState) -> Dict:
    """
    Grader node calls an llm to grade each retrieved chunk individually.
    Returns updated state with filtered retrieved chunks.
    """
    retrieved_chunks = state["retrieved_chunks"]
    user_question = state["user_question"]
    filtered_chunks = []

    # Grade each chunk individually
    for chunk in retrieved_chunks:
        snippet = chunk.get("text_chunk", "").strip()
        source = chunk.get("source", "Unknown Source")
        page = chunk.get("page_num", "N/A")

        context = f"Source: {source}\nPage: {page}\n\n{snippet}"

        prompt = f"""
You are an expert in the official rules and regulations of international cricket.
Given the LAW EXCERPT and the USER QUESTION below, rate how relevant the law excerpt is
to answering the user's question on a scale of 1 to 10.

- 1 means "completely irrelevant"
- 10 means "directly and highly relevant"

Only respond with a single integer between 1 and 10 — no explanations.

LAW EXCERPT:
{context}

USER QUESTION:
{user_question}

RELEVANCE SCORE (just the number 1–10):
"""

        response = llm_caller.call_llm(prompt, model="grader", stream=False)
        print(f"Grader response for chunk from {source} (page {page}): {response}")

        try:
            relevance_score = int(response.strip())
            chunk["relevance_score"] = relevance_score

            # Keep only chunks above threshold (e.g. >= 3)
            if relevance_score >= 3:
                filtered_chunks.append(chunk)
        except ValueError:
            print(f"Warning: Invalid relevance score received: {response}")
            continue

    # Sort chunks by score (highest first)
    filtered_chunks.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    return {
        "retrieved_chunks": filtered_chunks
    }


def llm_answer_node(state: RAGState) -> Dict:
    """
    LLM node that uses LangGraph's built-in message history for conversation context.
    - Uses messages from state for continuity.
    - Summarizes retrieved chunks if too many.
    - Returns full answer text (not stream, since we need to save to messages).
    """

    retrieved_chunks = state.get("retrieved_chunks", [])
    user_question = state["user_question"]
    messages = state.get("messages", [])

    # ---------- FORMAT CHAT HISTORY FROM MESSAGES ----------
    history_str = ""
    if messages:
        history_str = "Recent conversation:\n"
        # Get last 6 messages (3 turns)
        recent_messages = messages[-6:] if len(messages) > 6 else messages
        for msg in recent_messages:
            if isinstance(msg, dict):
                role = "Human" if msg.get("type") == "human" else "Assistant"
                content = msg.get("content", "")
            else:
                role = "Human" if msg.type == "human" else "Assistant"
                content = msg.content
            history_str += f"{role}: {content}\n"

    # ---------- BUILD CONTEXT FROM RETRIEVED CHUNKS ----------
    context_parts = []
    for c in retrieved_chunks:
        snippet = c.get("text_chunk", "").strip()
        source = c.get("source", "Unknown Source")
        page = c.get("page_num", "N/A")
        score = c.get("relevance_score", "N/A")
        context_parts.append(f"Source: {source} | Page: {page} | Relevance: {score}\n{snippet}")
    full_context = "\n\n".join(context_parts)

    # ---------- STEP 1: CONDITIONAL CONDENSATION ----------
    if len(retrieved_chunks) > 3:
        condenser_prompt = f"""
You are a summarization expert for the official laws of cricket.
Summarize the following information into 3–5 precise and relevant points.
Focus strictly on content that helps answer the question below.

QUESTION:
{user_question}

CONTEXT TO SUMMARIZE:
{full_context}

Return a concise, factual summary suitable for answering the question.
"""
        print("\n -------- CONDENSING CONTEXT (too many chunks) -------\n")
        condensed_context = llm_caller.call_llm(condenser_prompt, model="grader", stream=False).strip()
    else:
        condensed_context = full_context

    # ---------- STEP 2: FINAL ANSWER PROMPT WITH HISTORY ----------
    answer_prompt = f"""
You are an expert assistant specializing in the official laws and rules of international cricket.

Use the provided CONTEXT, along with recent conversation history, to answer the QUESTION precisely.

Guidelines:
- Use only the given CONTEXT.
- Provide a structured, concise answer in bullet points or short paragraphs.
- If the answer cannot be found in the context, say: "I don't know from the given rules."
- Maintain conversational continuity using the conversation history.

{history_str}

CONTEXT:
{condensed_context}

QUESTION:
{user_question}

FINAL ANSWER (concise, structured, cite sources inline or at the end):
"""

    print("\n -------- GENERATING FINAL ANSWER -------\n")

    # ---------- GET FULL ANSWER (non-streaming) ----------
    full_answer = llm_caller.call_llm(answer_prompt, model="answer", stream=False)

    # ---------- UPDATE MESSAGES ----------
    # Add user question and assistant response to messages
    new_messages = [
        {"type": "human", "content": user_question},
        {"type": "ai", "content": full_answer}
    ]

    return {
        "messages": new_messages,
        "answer": full_answer
    }


# Define the graph architecture
rag_graph = StateGraph(RAGState)

# Add nodes
rag_graph.add_node("retrieve", milvus_retrieve)
rag_graph.add_node("grader", grader_node)
rag_graph.add_node("llm", llm_answer_node)

# Define the entry point and edges
rag_graph.set_entry_point("retrieve")
rag_graph.add_edge("retrieve", "grader")
rag_graph.add_edge("grader", "llm")
rag_graph.set_finish_point("llm")

# ✅ Compile the graph with memory checkpointer
workflow = rag_graph.compile(checkpointer=memory)