
import os
import json
from dotenv import load_dotenv
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

load_dotenv()

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "cricket_rules_subchunks")

# ✅ BGE-Large (1024 dim) embed model
embed_model = SentenceTransformer("BAAI/bge-large-en")
embed_model.max_seq_length = 512

def connect():
    connections.connect(
        alias="default",
        uri=ZILLIZ_URI,
        token=ZILLIZ_TOKEN,
        secure=True
    )
    print("[INFO] Connected to Zilliz Cloud")

def embed_query(query: str):
    emb = embed_model.encode([query], normalize_embeddings=True)
    return emb[0]

def search(query: str, top_k: int = 5):
    collection = Collection(COLLECTION_NAME)
    query_emb = embed_query(query)
    results = collection.search(
        data=[query_emb],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"ef": 64}},
        limit=top_k,
        output_fields=["law_number", "law_title", "text"]
    )
    return results[0]

if __name__ == "__main__":
    connect()
    query = input("Enter a cricket rule question: ")
    hits = search(query, top_k=5)

    for idx, hit in enumerate(hits):
        print(f"\n--- Result #{idx+1} ---")
        print(f"Law Number : {hit.entity.get('law_number')}")
        print(f"Law Title  : {hit.entity.get('law_title')}")
        print(f"Text       : {hit.entity.get('text')[:200]}...")
        print(f"Score      : {hit.distance:.4f}")
