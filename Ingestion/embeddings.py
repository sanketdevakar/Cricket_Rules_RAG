import json
import os
from sentence_transformers import SentenceTransformer
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_JSON_PATH = os.path.join(PROJECT_ROOT, "data/extracted/chunks.json")
EMBEDDINGS_JSON_PATH = os.path.join(PROJECT_ROOT, "data/extracted/embeddings.json")
os.makedirs(os.path.dirname(EMBEDDINGS_JSON_PATH), exist_ok=True)

MODEL_NAME = "BAAI/bge-large-en"
model = SentenceTransformer(MODEL_NAME)

def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def split_law_into_subchunks(law_text):
    subrule_pattern = r"(\d+\.\d+)\s+"
    parts = re.split(subrule_pattern, law_text)
    chunks = []
    buffer = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.match(r"\d+\.\d+", part):
            if buffer:
                chunks.append(buffer.strip())
            buffer = part
        else:
            buffer += " " + part
    if buffer:
        chunks.append(buffer.strip())
    
    # fallback: split by sentences if only 1 chunk
    if len(chunks) <= 1:
        sentences = re.split(r'(?<=[.?!])\s+', law_text)
        chunks = [s.strip() for s in sentences if s.strip()]
    return chunks

def generate_subchunk_embeddings(law_chunks):
    embeddings_list = []
    for law in law_chunks:
        subchunks = split_law_into_subchunks(law["text"])
        for sub in subchunks:
            emb = model.encode(sub, normalize_embeddings=True).tolist()
            embeddings_list.append({
                "law_number": law["law_number"],
                "law_title": law["law_title"],
                "text": sub,
                "embedding": emb
            })
    return embeddings_list

def save_embeddings(embeddings, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, indent=4, ensure_ascii=False)
    print(f"[INFO] Saved {len(embeddings)} sub-chunk embeddings to {path}")

def main():
    law_chunks = load_chunks(CHUNKS_JSON_PATH)
    embeddings = generate_subchunk_embeddings(law_chunks)
    save_embeddings(embeddings, EMBEDDINGS_JSON_PATH)

if __name__ == "__main__":
    main()
