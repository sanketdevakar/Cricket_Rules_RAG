from graph.rag_graph import workflow
from graph.state import RAGState
import uuid

def format_citations(retrieved_chunks):
    """
    Build unique citation list (Source and Page) preserving order of first appearance.
    """
    seen = set()
    citations = []
    for c in retrieved_chunks:
        source = c.get("source", "Unknown Source").strip()
        page = c.get("page_num", "N/A").strip()
        key = (source, page)
        if key not in seen:
            seen.add(key)
            citations.append(f"{source} (Page {page})")
    return citations


def run_cli():
    print("⚾ Cricket Rules RAG (LangGraph) - Streaming mode (Groq)")
    print("Ask questions about the Laws of Cricket. Type 'exit' to quit.\n")

    # ✅ Create a unique thread_id for this conversation session
    # Each user/session should have its own thread_id
    thread_id = str(uuid.uuid4())
    print(f"[Session ID: {thread_id}]\n")

    # ✅ Configuration for LangGraph memory
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:
        while True:
            query = input("\nQuestion > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("👋 Goodbye.")
                break

            # ✅ Create state for this query
            state = {
                "user_question": query,
                "retrieved_chunks": [],
                "messages": [],
                "context_summary": ""
            }

            print("\nAnswer (streaming):\n")
            result = None

            try:
                # ✅ Run the workflow with config for memory persistence
                result = workflow.invoke(state, config=config)

                # ✅ Display the answer
                if "answer" in result and result["answer"]:
                    answer_text = result["answer"]
                    # Display with streaming effect (optional)
                    for char in answer_text:
                        print(char, end="", flush=True)
                        # Optional: add tiny delay for streaming effect
                        # import time
                        # time.sleep(0.01)
                else:
                    print("[No answer generated or empty response.]")

            except KeyboardInterrupt:
                print("\n\n[⚠️ Interrupted by user]\n")
            except Exception as e:
                print(f"\n[❌ Error during generation: {e}]\n")
                import traceback
                traceback.print_exc()

            # ---------- Sources ----------
            print("\n\n--- Sources ---")
            if result and "retrieved_chunks" in result:
                citations = format_citations(result.get("retrieved_chunks", []))
                if citations:
                    for c in citations:
                        print("•", c)
                else:
                    print("No sources retrieved.")
            else:
                print("No valid result to display sources.")

            print("\n----------------\n")

    except (KeyboardInterrupt, EOFError):
        print("\nExiting gracefully...")


if __name__ == "__main__":
    run_cli()