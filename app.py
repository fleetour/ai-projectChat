from fastapi import FastAPI, Form, HTTPException, Response, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List
import os
import uuid
import json
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from services.embeddings_service import extract_text_from_file, get_embeddings_from_llama, get_llama_chat_completion, save_embeddings_with_path
from services.file_service import  build_file_structure_tree, normalize_target_path
import logging
from db.qdrant_service import ensure_collection, get_qdrant_client,  search_similar
from services.embeddings_service import  extract_text_from_file
from qdrant_client import models 
from typing import Optional
from pydantic import BaseModel

from services.local_llma_service import LocalLlamaService

# === CONFIGURATION ===
FILES_DIR = "uploaded_files"
CUSTOMER_ID = 1
VECTOR_SIZE = 1024  # depends on your Mistral model

os.makedirs(FILES_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Local Qdrant RAG with Cloud Embeddings")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    fileIds: List[str]
    top_k: int = 5



class UploadRequest(BaseModel):
    target_path: Optional[str] = ""  # Relativer Pfad innerhalb des Upload-Verzeichnisses

@app.post("/upload")
async def upload_files(
    target_path: str = Form(""),  # Pfadparameter als Form-Feld
    files: List[UploadFile] = File(...)
):
    """
    Upload files with optional target path within the upload directory
    
    Args:
        target_path: Relative path within FILES_DIR where files should be stored
                    e.g., "projects/design", "documents/contracts"
        files: List of files to upload
    """
    collection_name = f"customer_{CUSTOMER_ID}_documents"
    ensure_collection(collection_name, vector_size=VECTOR_SIZE)

    results = []
    errors = []

    # Validate and normalize target path
    target_path = normalize_target_path(target_path)
    
    # Create target directory if it doesn't exist
    full_target_dir = os.path.join(FILES_DIR, target_path)
    os.makedirs(full_target_dir, exist_ok=True)
    qdrant_client = get_qdrant_client()

    for file in files:
        try:
            file_id = str(uuid.uuid4())
            
            # Create file path with target directory
            file_path = os.path.join(full_target_dir, f"{file_id}_{file.filename}")
            
            # Store relative path for database (without FILES_DIR)
            relative_file_path = os.path.join(target_path, f"{file_id}_{file.filename}")

            # Save the file
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            # Extract text
            text = extract_text_from_file(file_path)
            
            if not text or len(text.strip()) < 10:
                errors.append(f"File {file.filename}: No readable text content found")
                # Remove the empty file
                os.remove(file_path)
                continue

            # Split text into chunks
            chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
            chunks = [chunk for chunk in chunks if len(chunk.strip()) > 10]

            if not chunks:
                errors.append(f"File {file.filename}: No valid text chunks created")
                # Remove the file since no chunks were created
                os.remove(file_path)
                continue

            logger.info(f"Processing {file.filename}: {len(chunks)} chunks, path: {target_path}")

            # Get embeddings from local Llama
            embeddings = get_embeddings_from_llama(chunks)

            if not embeddings or len(embeddings) != len(chunks):
                errors.append(f"File {file.filename}: Embedding count mismatch")
                # Remove the file since embedding failed
                os.remove(file_path)
                continue
           
            # Save to Qdrant with path information
            save_embeddings_with_path(
                qdrant_client=qdrant_client,
                collection_name=collection_name, 
                file_id=file_id, 
                filename=file.filename, 
                chunks=chunks, 
                embeddings=embeddings, 
                target_path=target_path
            )
            results.append({
                "fileId": file_id, 
                "filename": file.filename, 
                "chunks": len(chunks),
                "path": target_path,
                "fullPath": relative_file_path
            })
            
        except Exception as e:
            errors.append(f"File {file.filename}: Processing failed - {str(e)}")
            logger.error(f"Error processing {file.filename}: {e}")
            # Cleanup: Remove file if it was partially created
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)

    response_content = {
        "uploaded": results,
        "targetPath": target_path,
        "fullTargetDir": full_target_dir
    }
    if errors:
        response_content["errors"] = errors

    return JSONResponse(content=response_content)


@app.post("/query")
async def query_docs(request: QueryRequest):
    collection_name = f"customer_{CUSTOMER_ID}_documents"

    try:
        # Get embedding for the query using local Llama
        query_emb = get_embeddings_from_llama([request.query])[0]
        
        # Search for similar chunks in Qdrant
        results = search_similar(collection_name, query_emb, request.fileIds, request.top_k)

        # Extract the actual TEXT from top results to form context
        top_chunks = [r.payload["text"] for r in results]
        context = "\n\n".join(top_chunks)

        # Create prompt
        prompt = f"Answer the question based on the following text:\n\n{context}\n\nQuestion: {request.query}\nAnswer:"

        logger.info(f"Query: {request.query}, Found {len(results)} relevant chunks")

        # Use local Llama for chat completion
        answer = await get_llama_chat_completion(prompt)

        # Return both the AI answer and the source documents for reference
        formatted_results = [
            {
                "file_id": r.payload["file_id"],
                "filename": r.payload["filename"],
                "score": r.score,
                "text": r.payload["text"]
            }
            for r in results
        ]
        
        return JSONResponse(content={
            "answer": answer,
            "sources": formatted_results,
            "question": request.query
        })
        
    except Exception as e:
        logger.error(f"Error in query processing: {e}")
        return JSONResponse(
            content={"error": f"Failed to process query: {str(e)}"}, 
            status_code=500
        )
    
@app.post("/query-stream")
async def query_docs_stream(request: QueryRequest, response: Response):
    """Streaming version of query endpoint"""
    collection_name = f"customer_{CUSTOMER_ID}_documents"
    
    async def generate():
        try:
            # Get embedding for the query
            query_emb = get_embeddings_from_llama([request.query])[0]
            
            # Search for similar chunks
            results = search_similar(collection_name, query_emb, request.fileIds, request.top_k)
            
            # Extract context
            top_chunks = [r.payload["text"] for r in results]
            context = "\n\n".join(top_chunks)
            
            # Create prompt
            prompt = f"Answer the question based on the following text:\n\n{context}\n\nQuestion: {request.query}\nAnswer:"
            
            logger.info(f"Starting stream for query: {request.query}")
            
            # Stream response from Llama
            full_answer = ""
            async for chunk in local_llama.get_llama_stream_completion(request.query, context):
                full_answer += chunk
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
            
            # Send sources
            formatted_results = [
                {
                    "file_id": r.payload["file_id"],
                    "filename": r.payload["filename"],
                    "score": r.score,
                    "text": r.payload["text"]
                }
                for r in results
            ]
            
            yield f"data: {json.dumps({'type': 'sources', 'sources': formatted_results})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'content': full_answer})}\n\n"
            
            logger.info(f"✅ Stream completed successfully. Response length: {len(full_answer)}")
            
        except Exception as e:
            error_msg = f"Error in stream processing: {str(e)}"
            logger.error(error_msg)
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no"  # Important for streaming
        }
    )




@app.get("/files")
async def get_files():
    """Get the file structure from uploaded files and Qdrant as tree structure"""
    try:
        collection_name = f"customer_{CUSTOMER_ID}_documents"
        file_structure = await build_file_structure_tree(FILES_DIR, collection_name)
        return file_structure
        
    except Exception as e:
        logger.error(f"Error retrieving files: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving files: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        collection_name = f"customer_{CUSTOMER_ID}_documents"
        ensure_collection(collection_name, vector_size=VECTOR_SIZE)
        return {
            "status": "healthy", 
            "qdrant": "connected",
            "llama": "available"
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "qdrant": "error",
            "llama": "error",
            "error": str(e)
        }

local_llama = LocalLlamaService(model="llama3:8b")