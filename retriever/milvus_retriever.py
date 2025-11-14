import os
from dotenv import load_dotenv
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
from graph.state import RAGState

load_dotenv()

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

class MilvusRetriever:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.model = SentenceTransformer(EMBED_MODEL)
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
            output_fields=["source", "page_num","text_chunk"]
        )

        final_chunks = []
        for hit in results[0]:
            final_chunks.append({
                "source": hit.entity.get("source"),
                "page_num": hit.entity.get("page_num"),
                "text_chunk": hit.entity.get("text_chunk"),
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
    # ✅ Access state as dictionary
    user_question = state["user_question"]
    
    retriever = MilvusRetriever(top_k=5)
    retrieved_chunks = retriever.query(user_question)
    
    # Return updates as a dictionary
    return {
        "retrieved_chunks": retrieved_chunks
    }