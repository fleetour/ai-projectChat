from pathlib import Path
from fastapi import APIRouter, FastAPI, Form, HTTPException, Response, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
from pydantic import BaseModel
from typing import List
import os
import uuid
import json
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from services.embeddings_service import ensure_cosine_collection, extract_text_from_file, get_embeddings_from_llama, get_llama_chat_completion, normalize_vector, save_embeddings_with_path
from services.file_service import  build_file_structure_tree, normalize_target_path
import logging
from db.qdrant_service import ensure_collection, get_qdrant_client,  search_similar, test_exact_vector_search
from services.embeddings_service import  extract_text_from_file
from qdrant_client import models 
from typing import Optional
from pydantic import BaseModel
from services.projects_handler import projects_handler, ProjectInfo

from services.local_llma_service import LocalLlamaService


# Create router

# === CONFIGURATION ===
FILES_DIR = "Projects"
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


# Request/Response models
class CreateProjectRequest(BaseModel):
    name: str

class ProjectResponse(BaseModel):
    projects: List[ProjectInfo]

class CreateProjectResponse(BaseModel):
    success: bool
    project: Optional[ProjectInfo] = None
    message: str = ""

class DeleteProjectResponse(BaseModel):
    success: bool
    message: str = ""



# ====== PROJECTS MODELS ======
class ProjectInfo(BaseModel):
    name: str
    path: str
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    file_count: int = 0
    size: int = 0

class ProjectResponse(BaseModel):
    projects: List[ProjectInfo]

class CreateProjectRequest(BaseModel):
    name: str

class CreateProjectResponse(BaseModel):
    success: bool
    project: Optional[ProjectInfo] = None
    message: str = ""

# ====== PROJECTS ENDPOINTS ======
@app.get("/api/projects", response_model=ProjectResponse)
async def get_all_projects():
    """Get all projects from the Projects directory"""
    try:
        projects_base_path = Path("Projects")
        projects = []
        
        # Create Projects directory if it doesn't exist
        projects_base_path.mkdir(exist_ok=True)
        
        # List all directories in Projects folder
        for item in projects_base_path.iterdir():
            if item.is_dir():
                project = ProjectInfo(
                    name=item.name,
                    path=str(item.absolute())
                )
                projects.append(project)
        
        return ProjectResponse(projects=projects)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get projects: {str(e)}")

@app.post("/api/projects", response_model=CreateProjectResponse)
async def create_project(request: CreateProjectRequest):
    """Create a new project folder"""
    try:
        projects_base_path = Path("Projects")
        projects_base_path.mkdir(exist_ok=True)
        
        project_path = projects_base_path / request.name
        
        if project_path.exists():
            return CreateProjectResponse(
                success=False,
                message="Project already exists"
            )
        
        project_path.mkdir()
        
        project = ProjectInfo(
            name=request.name,
            path=str(project_path.absolute())
        )
        
        return CreateProjectResponse(
            success=True,
            project=project,
            message="Project created successfully"
        )
        
    except Exception as e:
        return CreateProjectResponse(
            success=False,
            message=f"Failed to create project: {str(e)}"
        )

@app.get("/api/projects/{project_name}")
async def get_project(project_name: str):
    """Get a specific project by name"""
    try:
        project_path = Path("Projects") / project_name
        
        if not project_path.exists() or not project_path.is_dir():
            raise HTTPException(status_code=404, detail="Project not found")
        
        return ProjectInfo(
            name=project_name,
            path=str(project_path.absolute())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get project: {str(e)}")

@app.delete("/api/projects/{project_name}", response_model=DeleteProjectResponse)
async def delete_project(project_name: str):
    """Delete a project"""
    try:
        success = projects_handler.delete_project(project_name)
        if success:
            return DeleteProjectResponse(
                success=True,
                message="Project deleted successfully"
            )
        else:
            return DeleteProjectResponse(
                success=False,
                message="Project not found or could not be deleted"
            )
            
    except Exception as e:
        logger.error(f"Error in delete_project endpoint: {e}")
        return DeleteProjectResponse(
            success=False,
            message=f"Failed to delete project: {str(e)}"
        )


@app.post("/upload")
async def upload_files(
    target_project: str = Form(""),
    target_path: str = Form(""), 
    files: List[UploadFile] = File(...)
):
    """
    Upload files with optional target path within the upload directory
    """
    collection_name = f"customer_{CUSTOMER_ID}_documents"
    ensure_collection(collection_name, vector_size=VECTOR_SIZE)
    if (not target_project) or target_project.strip() == "":
        raise HTTPException(status_code=400, detail="Target project must be specified")
    
    upload_results = []  # CHANGED: Renamed to avoid conflict
    errors = []
    
    # Validate and normalize target path
    target_path = normalize_target_path(target_path)
    
    # Create target directory if it doesn't exist
    full_target_dir = os.path.join(FILES_DIR, target_project, target_path)
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
                os.remove(file_path)
                continue

            # Split text into chunks
            chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
            chunks = [chunk for chunk in chunks if len(chunk.strip()) > 10]

            if not chunks:
                errors.append(f"File {file.filename}: No valid text chunks created")
                os.remove(file_path)
                continue

            logger.info(f"Processing {file.filename}: {len(chunks)} chunks, path: {target_path}")

            # Get embeddings from local Llama
            embeddings = get_embeddings_from_llama(chunks)

            if not embeddings or len(embeddings) != len(chunks):
                errors.append(f"File {file.filename}: Embedding count mismatch")
                os.remove(file_path)
                continue
            
            # Before saving embeddings, ensure collection is correct
            ensure_cosine_collection(qdrant_client, collection_name, vector_size=4096)

            # Save to Qdrant with path information
            save_embeddings_with_path(
                qdrant_client=qdrant_client,
                collection_name=collection_name, 
                file_id=file_id, 
                filename=file.filename, 
                chunks=chunks, 
                embeddings=embeddings, 
                target_path=target_path,
                target_project=target_project
            )

            # KEEP the search test for debugging but don't include in response
            print(f"🧪 IMMEDIATE SEARCH TEST FOR {file.filename}:")
            for i, embedding in enumerate(embeddings[:3]):  # Test first 3 chunks
                normalized_embedding = normalize_vector(embedding)
                
                # CHANGED: Use a different variable name for search results
                search_results = qdrant_client.search(  # CHANGED: search_results instead of results
                    collection_name=collection_name,
                    query_vector=normalized_embedding,
                    limit=3,
                    with_payload=True,
                    score_threshold=0.0
                )
                
                print(f"   Chunk {i} search:")
                if search_results:  # CHANGED: search_results instead of results
                    for j, result in enumerate(search_results):  # CHANGED: search_results instead of results
                        match_type = "EXACT" if result.score > 0.99 else "SIMILAR" if result.score > 0.8 else "LOW"
                        print(f"      {j+1}. Score: {result.score:.6f} [{match_type}], File: {result.payload.get('filename', 'N/A')}")
                else:
                    print("      ❌ No results!")
            
            # CHANGED: Use upload_results instead of results
            upload_results.append({
                "fileId": file_id, 
                "filename": file.filename, 
                "chunks": len(chunks),
                "path": target_path,
                "fullPath": relative_file_path
            })
            
        except Exception as e:
            errors.append(f"File {file.filename}: Processing failed - {str(e)}")
            logger.error(f"Error processing {file.filename}: {e}")
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)

    # CHANGED: Use upload_results instead of results
    response_content = {
        "uploaded": upload_results,  # CHANGED: upload_results instead of results
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
            print(f"🎯 STARTING QUERY STREAM: '{request.query}'")
            
            # Get embedding for the query
            query_emb = get_embeddings_from_llama([request.query])[0]
            
            # Search for similar chunks
            results = search_similar(collection_name, query_emb, request.fileIds, request.top_k)
            
            print(f"🔍 SEARCH FOUND: {len(results)} results")
            
            # Extract context
            top_chunks = [r.payload["text"] for r in results]
            context = "\n\n".join(top_chunks)
            
            print(f"📝 Context: {len(context)} chars from {len(top_chunks)} chunks")
            
            # Stream response from Llama
            full_answer = ""
            async for chunk in local_llama.get_llama_stream_completion(request.query, context):
                full_answer += chunk
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
            
            print(f"✅ LLAMA COMPLETED: {len(full_answer)} chars")
            
            # Format sources
            formatted_results = []
            for r in results:
                formatted_result = {
                    "file_id": str(r.payload.get("file_id", "")),
                    "filename": str(r.payload.get("filename", "")),
                    "score": round(float(r.score), 6),
                    "text": str(r.payload.get("text", ""))[:500] + "..." if len(r.payload.get("text", "")) > 500 else str(r.payload.get("text", ""))
                }
                formatted_results.append(formatted_result)
            
            print(f"📊 SENDING {len(formatted_results)} SOURCES")
            
            # Send sources
            sources_data = {
                'type': 'sources', 
                'sources': formatted_results
            }
            yield f"data: {json.dumps(sources_data)}\n\n"
            
            # Send completion
            yield f"data: {json.dumps({'type': 'done', 'content': full_answer})}\n\n"
            
            print(f"🎉 STREAM COMPLETED SUCCESSFULLY")
            
        except Exception as e:
            error_msg = f"Error in stream processing: {str(e)}"
            print(f"💥 STREAM ERROR: {error_msg}")
            logger.error(error_msg)
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no"
        }
    )




@app.get("/api/projects/{project_name}/files")
async def get_files(project_name: str):
    """Get the file structure from uploaded files and Qdrant as tree structure"""
    try:
        collection_name = f"customer_{CUSTOMER_ID}_documents"
        project_path = f"Projects/{project_name}"
        file_structure = await build_file_structure_tree(project_name, project_path, collection_name)
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
    
@app.post("/test-embeddings")
async def testembeedings():
    collection_name = f"customer_{CUSTOMER_ID}_documents"

    try:
        # Get embedding for the query using local Llama
        query_emb = test_exact_vector_search(collection_name)
        
        
        
        return JSONResponse(query_emb)
        
    except Exception as e:
        logger.error(f"Error in query processing: {e}")
        return JSONResponse(
            content={"error": f"Failed to process query: {str(e)}"}, 
            status_code=500
        )
    
@app.get("/diagnose-collection")
async def diagnose_collection():
    """Comprehensive collection diagnosis"""
    collection_name = f"customer_{CUSTOMER_ID}_documents"
    qdrant_client = get_qdrant_client()
    
    diagnostics = {}
    
    try:
        # 1. Collection info
        collection_info = qdrant_client.get_collection(collection_name=collection_name)
        diagnostics["collection_info"] = {
            "status": collection_info.status,
            "points_count": collection_info.points_count,
            "vectors_count": collection_info.vectors_count,
            "distance": str(collection_info.config.params.vectors.distance),
            "vector_size": collection_info.config.params.vectors.size
        }
        
        # 2. Sample points
        points, _ = qdrant_client.scroll(
            collection_name=collection_name,
            limit=10,
            with_payload=True,
            with_vectors=True
        )
        
        diagnostics["sample_points"] = []
        for point in points:
            point_info = {
                "id": point.id,
                "has_vector": point.vector is not None,
                "vector_length": len(point.vector) if point.vector else 0,
                "vector_norm": float(np.linalg.norm(point.vector)) if point.vector else 0,
                "payload": {
                    "filename": point.payload.get("filename"),
                    "project": point.payload.get("project"),
                    "file_id": point.payload.get("file_id")
                }
            }
            diagnostics["sample_points"].append(point_info)
        
        # 3. Test search with simple vector
        if points and points[0].vector:
            test_vector = points[0].vector
            search_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=test_vector,
                limit=5,
                with_payload=True
            )
            
            diagnostics["test_search"] = [
                {
                    "score": float(result.score),
                    "id": result.id,
                    "filename": result.payload.get("filename")
                }
                for result in search_results
            ]
        
        return diagnostics
        
    except Exception as e:
        return {"error": str(e)}


@app.post("/test-search-only")
async def test_search_only(request: QueryRequest):
    """Test just the search functionality without streaming"""
    collection_name = f"customer_{CUSTOMER_ID}_documents"
    
    try:
        print(f"🧪 TEST SEARCH: '{request.query}'")
        
        # Get embedding for the query
        query_emb = get_embeddings_from_llama([request.query])[0]
        
        # Search for similar chunks
        results = search_similar(collection_name, query_emb, request.fileIds, request.top_k)
        
        # Convert to JSON-serializable format
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append({
                "rank": i + 1,
                "score": float(result.score),
                "filename": result.payload.get("filename", "N/A"),
                "file_id": result.payload.get("file_id", "N/A"),
                "text_preview": result.payload.get("text", "")[:100] + "..." if result.payload.get("text") else "No text",
                "chunk_index": result.payload.get("chunk_index", "N/A")
            })
        
        return {
            "success": True,
            "query": request.query,
            "results_count": len(results),
            "results": formatted_results,
            "has_results": len(results) > 0
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

local_llama = LocalLlamaService(model="llama3:8b")