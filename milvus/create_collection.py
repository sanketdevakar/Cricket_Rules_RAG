
import os
from dotenv import load_dotenv
from pymilvus import (
    connections, FieldSchema, CollectionSchema,
    DataType, Collection, utility
)

load_dotenv()

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "cricket_rules_subchunks")
DIMENSION = 1024

INDEX_PARAMS = {
    "index_type": "HNSW",
    "metric_type": "IP",
    "params": {"M": 8, "efConstruction": 64}
}

def connect():
    connections.connect(
        alias="default",
        uri=ZILLIZ_URI,
        token=ZILLIZ_TOKEN,
        secure=True
    )
    print("[INFO] Connected to Zilliz Cloud")

def create_collection():
    if utility.has_collection(COLLECTION_NAME):
        print(f"[INFO] Collection '{COLLECTION_NAME}' already exists.")
        return Collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="law_number", dtype=DataType.INT64),
        FieldSchema(name="law_title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
    ]
    schema = CollectionSchema(fields, description="Cricket laws sub-chunk embeddings")

    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print("[INFO] Created new collection on Zilliz cloud.")

    collection.create_index(field_name="embedding", index_params=INDEX_PARAMS)
    print("[INFO] Index created (HNSW + IP)")

    collection.load()
    print("[INFO] Collection loaded.")

    return collection

if __name__ == "__main__":
    connect()
    create_collection()
