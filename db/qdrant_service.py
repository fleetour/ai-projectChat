import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import uuid
import logging
from contextlib import contextmanager

load_dotenv()

logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Global client instance (singleton pattern)
_client = None

def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client instance with connection pooling."""
    global _client
    if _client is None:
        try:
            _client = QdrantClient(
                host=QDRANT_HOST, 
                port=QDRANT_PORT,
                timeout=30,  # 30 second timeout
                # Qdrant client manages connections internally, no need to close manually
            )
            # Test connection
            _client.get_collections()
            logger.info(f"✅ Qdrant client connected to {QDRANT_HOST}:{QDRANT_PORT}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {e}")
            raise
    return _client

@contextmanager
def qdrant_session():
    """Context manager for Qdrant operations (optional, for explicit control)."""
    client = get_qdrant_client()
    try:
        yield client
    except Exception as e:
        logger.error(f"Qdrant operation failed: {e}")
        raise
    # Note: We don't close the client here as it's meant to be reused

def ensure_collection(collection_name: str, vector_size: int):
    """Ensure collection exists for this customer."""
    client = get_qdrant_client()
    try:
        existing = [c.name for c in client.get_collections().collections]
        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"✅ Created collection: {collection_name}")
        else:
            logger.info(f"✅ Collection exists: {collection_name}")
    except Exception as e:
        logger.error(f"❌ Error ensuring collection {collection_name}: {e}")
        raise

def save_embeddings(collection_name: str, file_id: str, filename: str, chunks: list, embeddings: list):
    """Save embeddings and their metadata to Qdrant."""
    client = get_qdrant_client()
    
    points = []
    for i, emb in enumerate(embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),  # Use string UUID
                vector=emb,
                payload={
                    "file_id": file_id,
                    "chunk_index": i,
                    "filename": filename,
                    "text": chunks[i],
                },
            )
        )
    
    try:
        client.upsert(collection_name=collection_name, points=points)
        logger.info(f"✅ Saved {len(points)} embeddings to collection: {collection_name}")
    except Exception as e:
        logger.error(f"❌ Error saving embeddings to {collection_name}: {e}")
        raise

def search_similar(collection_name: str, query_vector: list, file_ids: list, top_k: int = 5):
    """Search in Qdrant collection."""
    client = get_qdrant_client()
    
    query_filter = None
    if file_ids:
        query_filter = {"must": [{"key": "file_id", "match": {"any": file_ids}}]}

    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )
        logger.info(f"✅ Search found {len(results)} results from {collection_name}")
        return results
    except Exception as e:
        logger.error(f"❌ Error searching in {collection_name}: {e}")
        raise

def delete_file_embeddings(collection_name: str, file_id: str):
    """Delete all embeddings for a specific file."""
    client = get_qdrant_client()
    
    try:
        client.delete(
            collection_name=collection_name,
            points_selector={
                "filter": {
                    "must": [
                        {"key": "file_id", "match": {"value": file_id}}
                    ]
                }
            }
        )
        logger.info(f"✅ Deleted embeddings for file: {file_id}")
    except Exception as e:
        logger.error(f"❌ Error deleting embeddings for file {file_id}: {e}")
        raise

def get_collection_info(collection_name: str):
    """Get information about a collection."""
    client = get_qdrant_client()
    
    try:
        collection_info = client.get_collection(collection_name=collection_name)
        return {
            "points_count": collection_info.points_count,
            "vectors_count": collection_info.vectors_count,
            "status": collection_info.status
        }
    except Exception as e:
        logger.error(f"❌ Error getting info for collection {collection_name}: {e}")
        raise

def close_connection():
    """Close the Qdrant connection (call this during application shutdown)."""
    global _client
    if _client is not None:
        # QdrantClient doesn't have an explicit close method, but we can nullify it
        _client = None
        logger.info("✅ Qdrant connection closed")