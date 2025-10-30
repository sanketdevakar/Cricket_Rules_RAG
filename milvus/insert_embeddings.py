def insert_embeddings(inputs):
    """
    Inserts embeddings and metadata into an existing Zilliz Cloud collection.

    Args:
        inputs (dict): {
            "embeddings": List[List[float]],
            "law_numbers": List[int],
            "law_titles": List[str],
            "texts": List[str]
        }
    Returns:
        dict: {"status": "success", "count": int}
    """
    import os
    from dotenv import load_dotenv
    from pymilvus import Collection, connections

    load_dotenv()
    ZILLIZ_URI = os.getenv("ZILLIZ_URI")
    ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "cricket_rules_subchunks")

    connections.connect(
        alias="default",
        uri=ZILLIZ_URI,
        token=ZILLIZ_TOKEN,
        secure=True
    )
    collection = Collection(COLLECTION_NAME)

    # Prepare data for insertion
    embeddings = inputs.get("embeddings", [])
    law_numbers = inputs.get("law_numbers", [])
    law_titles = inputs.get("law_titles", [])
    texts = inputs.get("texts", [])

    entities = [
        law_numbers,
        law_titles,
        texts,
        embeddings
    ]
    result = collection.insert(entities)
    count = len(embeddings)
    return {"status": "success", "count": count}