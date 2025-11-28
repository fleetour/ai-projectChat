import os
from docx import Document
from qdrant_client import QdrantClient
import requests
from dotenv import load_dotenv
from mistralai import Mistral
from typing import List
from services.local_llma_service import local_llama
from datetime import datetime
import uuid 
from qdrant_client.models import PointStruct
import numpy as np

from services.utils import normalize_vector

load_dotenv()

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



def extract_text_from_file(file_path: str) -> str:
    """Extract text from .docx, .pdf, or .txt files."""
    ext = file_path.lower()
    if ext.endswith(".pdf"):
        from pdfminer.high_level import extract_text
        return extract_text(file_path)
    elif ext.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext.endswith(".txt"):
        with open(file_path, "r") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
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
    
def get_embeddings_from_llama(texts: List[str]) -> List[List[float]]:
    """Get embeddings using local Llama"""
    return local_llama.get_embeddings(texts)

async def get_llama_chat_completion(prompt: str) -> str:
    """Get chat completion using local Llama"""
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

def save_embeddings_with_path(
    qdrant_client: QdrantClient,
    collection_name: str, 
    file_id: str, 
    filename: str, 
    chunks: List[str], 
    embeddings: List[List[float]], 
    target_path: str,
    target_project: str
):
    """
    Save embeddings to Qdrant with path information
    """
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Normalize the embedding for cosine similarity
        normalized_embedding = normalize_vector(embedding)
        
        # Create UUID for each point
        point_id = str(uuid.uuid4())
        
        point = PointStruct(
            id=point_id,
            vector=normalized_embedding,  # ✅ Use NORMALIZED embedding
            payload={
                "file_id": file_id,
                "filename": filename,
                "chunk_index": i,
                "text": chunk,
                "total_chunks": len(chunks),
                "upload_path": target_path,
                "full_file_path": f"{target_path}/{file_id}_{filename}" if target_path else f"{file_id}_{filename}",
                "upload_time": datetime.now().isoformat(),
                "original_point_id": f"{file_id}_{i}",
                "project": target_project
            }
        )
        points.append(point)
    
    # Batch upload to Qdrant
    operation_info = qdrant_client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True  # Wait for confirmation
    )
    
    print(f"✅ Saved {len(points)} NORMALIZED chunks for file {filename} in path {target_path}")
    
    # Verify the upload worked
    if hasattr(operation_info, 'status') and operation_info.status == 'completed':
        print(f"✅ Upload confirmed for {filename}")
    else:
        print(f"⚠️  Upload status uncertain for {filename}")