import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import uuid

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(collection_name: str, vector_size: int):
    """Ensure collection exists for this customer."""
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def save_embeddings(collection_name: str, file_id: str, filename: str, chunks: list, embeddings: list):
    """Save embeddings and their metadata to Qdrant."""
    points = []
    for i, emb in enumerate(embeddings):
        points.append(
            PointStruct(
                id=uuid.uuid4().hex,
                vector=emb,
                payload={
                    "file_id": file_id,
                    "chunk_index": i,
                    "filename": filename,
                    "text": chunks[i],
                },
            )
        )
    client.upsert(collection_name=collection_name, points=points)


def search_similar(collection_name: str, query_vector: list, file_ids: list, top_k: int):
    """Search in Qdrant collection."""
    query_filter = None
    if file_ids:
        query_filter = {"must": [{"key": "file_id", "match": {"any": file_ids}}]}

    return client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        query_filter=query_filter,
    )
