import asyncio
import os
from docx import Document
from qdrant_client import QdrantClient
import requests
from dotenv import load_dotenv
from mistralai import Mistral
from typing import Any, Dict, List, Optional
from config import FILES_DIR
from datetime import datetime
import uuid 
from qdrant_client.models import PointStruct
import numpy as np
import logging
from db.qdrant_service import get_qdrant_client
from services.local_llma_service import LocalLlamaService
from services.utils import normalize_vector

load_dotenv()
logger = logging.getLogger(__name__)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/embeddings"

# Initialize Mistral client
try:
    client = Mistral(api_key=MISTRAL_API_KEY)  # Make sure to set your API key
except ImportError:
    print("Warning: mistralai package not installed")
    client = None



# def get_embeddings_from_mistral(texts: list):
#     """Get embeddings from Mistral Cloud API for a list of text chunks."""
#     headers = {
#         "Authorization": f"Bearer {MISTRAL_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     response = requests.post(
#         MISTRAL_API_URL,
#         json={"model": "mistral-embed", "input": texts},
#         headers=headers
#     )

#     response.raise_for_status()
#     data = response.json()

#     print("Mistral embeddings response:", data)  # Debugging line
#     # Return embeddings in same order
#     return [item["embedding"] for item in data["data"]]


    
# async def get_mistral_chat_completion(prompt: str, model: str = "mistral-medium-latest") -> str:
#     """
#     Get chat completion from Mistral API asynchronously
#     """
#     if client is None:
#         raise Exception("Mistral client not configured. Please install mistralai package and set MISTRAL_API_KEY")
    
#     try:
#         response = await client.chat.complete_async(
#             model=model,
#             messages=[{"role": "user", "content": prompt}]
#         )
        
#         return response.choices[0].message.content
        
#     except Exception as e:
#         raise Exception(f"Failed to get Mistral chat completion: {str(e)}")

async def get_embeddings_async(texts: List[str], model: str) -> List[List[float]]:
    """Async embedding generation."""
    # Example with OpenAI (if using):
    # import openai
    # response = await openai.Embedding.acreate(
    #     model="text-embedding-ada-002",
    #     input=texts
    # )
    # return [item['embedding'] for item in response['data']]
    
    # For your Llama setup, use ThreadPoolExecutor
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        get_embeddings_from_llama,  # Your existing function
        texts,
        model
    )

def get_embeddings_from_llama(texts: List[str], model: str = "mistral:7b") -> List[List[float]]:
    """Get embeddings using local Llama"""

    if not model.strip():
        raise Exception(f"Model must be defined.")
    local_llama = LocalLlamaService(model=model)
    return local_llama.get_embeddings(texts)

async def get_llama_chat_completion(prompt: str, model: str) -> str:
    """Get chat completion using local Llama"""
    if not model.strip():
        raise Exception(f"Model must be defined.")
    local_llama = LocalLlamaService(model=model)
    return await local_llama.get_chat_completion(prompt)

def ensure_cosine_collection(qdrant_client: QdrantClient, collection_name: str, vector_size: int = 4096):
    """Ensure collection uses cosine distance"""
    from qdrant_client.models import Distance, VectorParams
    
    try:
        # Try to get existing collection
        collection_info = qdrant_client.get_collection(collection_name=collection_name)
        current_distance = collection_info.config.params.vectors.distance
        
        if current_distance != "Cosine":
            print(f"❌ Collection has wrong distance: {current_distance}. Recreating...")
            qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print("✅ Recreated collection with Cosine distance")
        else:
            print("✅ Collection already uses Cosine distance")
            
    except Exception:
        # Collection doesn't exist, create it
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print("✅ Created new collection with Cosine distance")


async def save_embeddings_async(
    collection_name: str,
    file_id: str,
    filename: str,
    chunks: List[str],
    embeddings: List[List[float]],
    target_path: str,
    target_project: str,
    blob_metadata: Dict[str, Any],
    auto_generated: bool = False,
    source_template_id: Optional[str] = None
) -> bool:
    """Async save embeddings to Qdrant."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        save_embeddings_with_path,  # Your existing sync function
        collection_name,
        file_id,
        filename,
        chunks,
        embeddings,
        target_path,
        target_project,
        auto_generated,
        source_template_id,
        blob_metadata
    )



def save_embeddings_with_path(
    collection_name: str, 
    file_id: str, 
    filename: str, 
    chunks: List[str], 
    embeddings: List[List[float]], 
    target_path: str,
    target_project: str,
    auto_generated: bool = False,
    source_template_id: Optional[str] = None,
    # Azure Blob specific parameters
    blob_metadata: Optional[dict] = None,
    storage_type: str = "azure_blob"
):
    """
    Save embeddings to Qdrant with path information for Azure Blob Storage
    
    Args:
        qdrant_client: Qdrant client instance
        collection_name: Name of the Qdrant collection
        file_id: Unique file identifier
        filename: Original filename
        chunks: List of text chunks
        embeddings: List of embedding vectors
        target_path: Path within the project
        target_project: Project name
        auto_generated: Whether the content was auto-generated
        source_template_id: ID of source template (if any)
        blob_metadata: Azure Blob metadata (from upload_file result)
        storage_type: Storage type ("azure_blob", "local", etc.)
    """

    qdrant_client = get_qdrant_client()
    points = []
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Normalize the embedding for cosine similarity
        normalized_embedding = normalize_vector(embedding)
        
        # Create UUID for each point
        point_id = str(uuid.uuid4())
        
        # Build full file path based on storage type
        if storage_type == "azure_blob" and blob_metadata:
            # Use Azure Blob path from metadata
            full_file_path = blob_metadata.get("full_file_path", "")
            blob_name = blob_metadata.get("blob_name", "")
            blob_url = blob_metadata.get("blob_url", "")
            container = blob_metadata.get("container", "")
        else:
            # Fallback to local path (for backward compatibility)
            if target_path and target_path.strip():
                full_file_path = f"{FILES_DIR}/{target_project}/{target_path}/{file_id}_{filename}"
            else:
                full_file_path = f"{FILES_DIR}/{target_project}/{file_id}_{filename}"
            blob_name = ""
            blob_url = ""
            container = ""
        
        # Base payload with all metadata
        payload = {
            "file_id": file_id,
            "filename": filename,
            "chunk_index": i,
            "text": chunk,
            "total_chunks": len(chunks),
            "upload_path": target_path,
            "full_file_path": full_file_path,
            "upload_time": datetime.now().isoformat(),
            "original_point_id": f"{file_id}_{i}",
            "project": target_project,
            "storage_type": storage_type,
            
            # Azure Blob specific fields
            "blob_name": blob_name,
            "blob_url": blob_url,
            "container": container,
            "file_size": blob_metadata.get("size") if blob_metadata else None,
            "content_type": blob_metadata.get("content_type") if blob_metadata else None,
        }
        
        # Add blob metadata if available
        if blob_metadata:
            payload.update({
                "azure_metadata": {
                    k: v for k, v in blob_metadata.items() 
                    if k not in ["file_id", "filename", "full_file_path", "size", "content_type"]
                }
            })
        
        # Add optional parameters if provided
        if auto_generated:
            payload["auto_generated"] = True
        
        if source_template_id:
            payload["source_template_id"] = source_template_id
        
        # Add base folder information if available
        if blob_metadata and "base_folder" in blob_metadata:
            payload["base_folder"] = blob_metadata["base_folder"]
        
        point = PointStruct(
            id=point_id,
            vector=normalized_embedding,
            payload=payload
        )
        points.append(point)
    
    # Batch upload to Qdrant
    try:
        operation_info = qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True  # Wait for confirmation
        )
        
        # Log the additional parameters if provided
        additional_info = []
        if auto_generated:
            additional_info.append("auto_generated")
        if source_template_id:
            additional_info.append(f"source_template: {source_template_id}")
        
        info_suffix = f" ({', '.join(additional_info)})" if additional_info else ""
        
        # Log based on storage type
        if storage_type == "azure_blob":
            logger.info(f"✅ Saved {len(points)} chunks for file {filename} to Azure Blob")
            if blob_metadata and "blob_name" in blob_metadata:
                logger.info(f"   Blob path: {blob_metadata['blob_name']}")
                logger.info(f"   Container: {blob_metadata.get('container', 'N/A')}")
        else:
            logger.info(f"✅ Saved {len(points)} chunks for file {filename} in path {target_path}{info_suffix}")
        
        # Verify the upload worked
        if hasattr(operation_info, 'status') and operation_info.status == 'completed':
            logger.info(f"✅ Upload confirmed for {filename}")
        else:
            logger.warning(f"⚠️  Upload status uncertain for {filename}")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save embeddings for {filename}: {e}")
        return False