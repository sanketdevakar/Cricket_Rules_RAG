from graph.rag_graph import workflow
from graph.state import RAGState

def format_citations(retrieved_chunks):
    """
    Build unique citation list (Law X - TITLE) preserving order of first appearance.
    """
    seen = set()
    citations = []
    for c in retrieved_chunks:
        key = (int(c.get("law_number", -1)), c.get("law_title", "").strip())
        if key not in seen:
            seen.add(key)
            citations.append(f"Law {key[0]} - {key[1]}")
    return citations

def run_cli():
    print("Cricket Rules RAG (LangGraph) - streaming mode (Ollama)")
    print("Type questions about the Laws of Cricket. Type 'exit' to quit.\n")

    try:
        while True:
            query = input("Question > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Goodbye.")
                break

            print("\nAnswer (streaming):\n")
            # Create initial state with just the query
            initial_state = RAGState(user_question=query)
            
            # Run the workflow
            result = workflow.invoke(initial_state)

            # Handle streaming response
            try:
                answer_stream = result["answer"]
                for chunk in answer_stream:
                    print(chunk, end="", flush=True)
            except KeyboardInterrupt:
                print("\n\n[Interrupted streaming by user]\n")
            except Exception as e:
                print(f"\n\n[Error during generation] {e}\n")

            print("\n\n--- Sources ---")
            citations = format_citations(result.get("retrieved_chunks", []))
            if citations:
                for c in citations:
                    print(c)
            else:
                print("No sources retrieved.")
            print("\n----------------\n")

    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")

if __name__ == "__main__":
    run_cli()