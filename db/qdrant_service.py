import asyncio
from functools import partial
import os
from typing import List, Optional
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
_client_lock = asyncio.Lock()


async def get_qdrant_client_async() -> QdrantClient:
    """Get or create Qdrant client instance asynchronously."""
    global _client
    
    if _client is None:
        async with _client_lock:
            if _client is None:  # Double-check pattern
                try:
                    print("🔄 Creating Qdrant client asynchronously...")
                    
                    # Run synchronous Qdrant client creation in thread pool
                    loop = asyncio.get_event_loop()
                    
                    # Create client and test connection in thread pool
                    _client = await loop.run_in_executor(
                        None,  # Default thread pool executor
                        partial(QdrantClient,
                            host=QDRANT_HOST,
                            port=QDRANT_PORT,
                            timeout=30
                        )
                    )
                    
                    # Test connection asynchronously
                    await loop.run_in_executor(None, _client.get_collections)
                    
                    logger.info(f"✅ Qdrant client connected to {QDRANT_HOST}:{QDRANT_PORT}")
                except Exception as e:
                    logger.error(f"❌ Failed to connect to Qdrant: {e}")
                    # Clear the client on failure
                    _client = None
                    raise
    
    return _client

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

def ensure_cosine_collection(qdrant_client: QdrantClient, collection_name: str, vector_size: int = 384):
    """Ensure collection uses cosine distance - CORRECTED"""
    try:
        # Try to get existing collection
        collection_info = qdrant_client.get_collection(collection_name=collection_name)
        current_distance = collection_info.config.params.vectors.distance
        
        print(f"📋 Current collection distance: {current_distance}")
        
        if current_distance != "Cosine":
            print(f"❌ Wrong distance metric: {current_distance}. Recreating collection...")
            qdrant_client.delete_collection(collection_name=collection_name)
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print("✅ Recreated collection with Cosine distance")
        else:
            print("✅ Collection already uses Cosine distance")
            
    except Exception as e:
        # Collection doesn't exist, create it
        print(f"ℹ️ Collection doesn't exist, creating new one: {e}")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print("✅ Created new collection with Cosine distance")

def ensure_collection(collection_name: str, vector_size: int = 384):
    """Wrapper that calls ensure_cosine_collection"""
    qdrant_client = get_qdrant_client()
    ensure_cosine_collection(qdrant_client, collection_name, vector_size)


async def ensure_collection_async(collection_name: str, vector_size: int):
    """Async version of ensure_collection."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        ensure_collection,  # Your existing sync function
        collection_name,
        vector_size
    )
    

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


def search_similar_with_project_grouping(collection_name: str, query_vector: list, 
                                         file_ids: Optional[List[str]] = None, 
                                         project_name: Optional[str] = None, 
                                         top_k: int = 5):
    """Search and group results by project."""
    client = get_qdrant_client()
    
    # Build query filter (your existing logic)
    query_filter = None
    if file_ids and project_name:
        query_filter = {
            "should": [
                {"key": "file_id", "match": {"any": file_ids}},
                {"key": "project", "match": {"value": project_name}}
            ]
        }
    elif file_ids:
        query_filter = {"must": [{"key": "file_id", "match": {"any": file_ids}}]}
    elif project_name:
        query_filter = {"must": [{"key": "project", "match": {"value": project_name}}]}
    
    try:
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k * 3,  # Get more results to group by project
            query_filter=query_filter,
            with_payload=True,
        )
        
        grouped_results = {}
        # for result in results:
        #     project_id = result.payload.get("project") or "Unknown Project"
        #     if project_id not in grouped_results:
        #         grouped_results[project_id] = []
        #     grouped_results[project_id].append(result)
        
        for scored_point in results.points:  # Use descriptive variable name
            payload = scored_point.payload
            project_id = payload.get("project") or "Unknown Project"
            
            if project_id not in grouped_results:
                grouped_results[project_id] = []

            grouped_results[project_id].append(scored_point)

        logger.info(f"✅ Search found {len(results.points)} results grouped into {len(grouped_results)} projects")
        return grouped_results
        
    except Exception as e:
        logger.error(f"❌ Error searching in {collection_name}: {e}")
        raise

def search_similar(collection_name: str, query_vector: list, file_ids: Optional[List[str]] = None, 
                   project_name: Optional[str] = None, top_k: int = 5):
    """Search in Qdrant collection."""
    client = get_qdrant_client()
    
    # Build query filter based on provided parameters
    query_filter = None
    filter_conditions = []
    
    if file_ids and project_name:
        # If both file_ids and project_name are provided, use OR logic
        query_filter = {
            "should": [
                {"key": "file_id", "match": {"any": file_ids}},
                {"key": "project_name", "match": {"value": project_name}}
            ]
        }
    elif file_ids:
        # Only file_ids filter
        query_filter = {"must": [{"key": "file_id", "match": {"any": file_ids}}]}
    elif project_name:
        # Only project_name filter
        query_filter = {"must": [{"key": "project_name", "match": {"value": project_name}}]}
    # If neither is provided, query_filter remains None (search all)
    
    print(f"filter: {query_filter}")

    try:
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True, 
        )
        
        # FIX: Return results.points, not results
        points = results.points
        logger.info(f"✅ Search found {len(points)} results from {collection_name}")
        
        # Debug logging
        if query_filter:
            logger.debug(f"🔍 Query filter applied: {query_filter}")
        
        return points  # Return the list of ScoredPoint objects
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


