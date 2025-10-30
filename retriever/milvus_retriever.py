import os
from dotenv import load_dotenv
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
from graph.state import RAGState

load_dotenv()

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "cricket_rules_subchunks")

class MilvusRetriever:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.model = SentenceTransformer("BAAI/bge-large-en")
        self.model.max_seq_length = 512

        connections.connect(
            alias="default",
            uri=ZILLIZ_URI,
            token=ZILLIZ_TOKEN,
            secure=True
        )
        self.collection = Collection(COLLECTION_NAME)

    def embed(self, text: str):
        return self.model.encode([text], normalize_embeddings=True)[0]

    def query(self, user_query: str):
        query_vec = self.embed(user_query)

        results = self.collection.search(
            data=[query_vec],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=self.top_k,
            output_fields=["law_number", "law_title", "text"]
        )

        final_chunks = []
        for hit in results[0]:
            final_chunks.append({
                "law_number": hit.entity.get("law_number"),
                "law_title": hit.entity.get("law_title"),
                "text": hit.entity.get("text"),
                "score": hit.distance
            })

        return final_chunks

def milvus_retrieve(state: RAGState) -> dict:
    """
    Retrieves relevant chunks from Milvus using the question in state.
    Args:
        state: RAGState containing user_question
    Returns:
        Dict with updates to state
    """
    retriever = MilvusRetriever(top_k=5)
    retrieved_chunks = retriever.query(state.user_question)
    
    # Return updates as a dictionary
    return {
        "user_question": state.user_question,
        "retrieved_chunks": retrieved_chunks
    }