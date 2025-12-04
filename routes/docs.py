from typing import Literal
from fastapi import APIRouter, Form, UploadFile, File, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import os
import uuid
import json
import logging
import numpy as np

from qdrant_client import models

from services.embeddings_service import (
    ensure_cosine_collection,
    get_embeddings_from_llama,
    get_llama_chat_completion,
    normalize_vector,
    save_embeddings_with_path,
)
from services.file_service import build_file_structure_tree, find_file_path_by_id, intelligent_chunking_simple, normalize_target_path
from db.qdrant_service import ensure_collection, get_qdrant_client, search_similar, test_exact_vector_search
from services.local_llma_service import LocalLlamaService
from routes.schemas import QueryRequest
from config import FILES_DIR, CUSTOMER_ID, VECTOR_SIZE
from services.utils import calculate_adaptive_top_k, extract_text_from_file, get_collection_name, get_content_type

router = APIRouter(tags=["documents"])
logger = logging.getLogger(__name__)

local_llama = LocalLlamaService(model="llama3:8b")


@router.post("/upload")
async def upload_files(
    target_project: str = Form(""),
    target_path: str = Form(""),
    files: list[UploadFile] = File(...),
):
    """Verbesserte Datei-Upload-Funktion mit intelligentem Chunking."""
    
    collection_name = f"customer_{CUSTOMER_ID}_documents"
    ensure_collection(collection_name, vector_size=VECTOR_SIZE)
    
    if (not target_project) or target_project.strip() == "":
        raise HTTPException(status_code=400, detail="Target project must be specified")

    upload_results = []
    errors = []
    target_path = normalize_target_path(target_path)
    full_target_dir = os.path.join(FILES_DIR, target_project, target_path)
    os.makedirs(full_target_dir, exist_ok=True)
    qdrant_client = get_qdrant_client()

    for file in files:
        file_id = str(uuid.uuid4())
        file_path = None
        
        try:
            # 1. Datei speichern
            file_path = os.path.join(full_target_dir, f"{file_id}_{file.filename}")
            relative_file_path = os.path.join(target_path, f"{file_id}_{file.filename}")
            
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            # 2. Text extrahieren
            text = extract_text_from_file(file_path)
            
            if not text or len(text.strip()) < 10:
                errors.append(f"File {file.filename}: No readable text content found")
                os.remove(file_path)
                continue

            # 3. INTELLIGENTES CHUNKING (statt fix 500 Zeichen)
            chunks = intelligent_chunking_simple(
                text=text,
                max_chunk_size=1500,
                overlap=200
            )
            
            if not chunks:
                errors.append(f"File {file.filename}: No valid text chunks created")
                os.remove(file_path)
                continue

            logger.info(f"✅ Processing {file.filename}: {len(chunks)} chunks (vorher: {len(text) // 500})")

            # 4. Embeddings berechnen (in Batches für große Dateien)
            all_embeddings = []
            batch_size = 10  # Kleinere Batch-Größe für Stabilität
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_embeddings = get_embeddings_from_llama(batch_chunks)
                
                if not batch_embeddings:
                    raise Exception(f"Embedding batch {i//batch_size + 1} returned None")
                
                if len(batch_embeddings) != len(batch_chunks):
                    logger.warning(f"Embedding mismatch: expected {len(batch_chunks)}, got {len(batch_embeddings)}")
                    # Nimm so viele wie möglich
                    all_embeddings.extend(batch_embeddings[:len(batch_chunks)])
                else:
                    all_embeddings.extend(batch_embeddings)
                
                logger.debug(f"  Processed embedding batch {i//batch_size + 1}")

            # 5. In Qdrant speichern (mit deiner Funktion)
            if len(all_embeddings) != len(chunks):
                logger.warning(f"Final embedding mismatch for {file.filename}: chunks={len(chunks)}, embeddings={len(all_embeddings)}")
                # Anpassen: Nimm nur so viele Chunks wie Embeddings vorhanden
                min_length = min(len(chunks), len(all_embeddings))
                chunks = chunks[:min_length]
                all_embeddings = all_embeddings[:min_length]

            # DEINE FUNKTION VERWENDEN
            save_embeddings_with_path(
                qdrant_client=qdrant_client,
                collection_name=collection_name,
                file_id=file_id,
                filename=file.filename,
                chunks=chunks,
                embeddings=all_embeddings,
                target_path=target_path,
                target_project=target_project,
                auto_generated=False,  # Kannst du anpassen
                source_template_id=None  # Kannst du anpassen
            )

            upload_results.append({
                "fileId": file_id,
                "filename": file.filename,
                "chunks": len(chunks),
                "path": target_path,
                "fullPath": relative_file_path,
            })

        except Exception as e:
            error_msg = f"File {file.filename}: Processing failed - {str(e)}"
            errors.append(error_msg)
            logger.error(f"❌ Error processing {file.filename}: {e}", exc_info=True)
            
            if file_path and os.path.exists(file_path):
                os.remove(file_path)

    response_content = {
        "uploaded": upload_results, 
        "targetPath": target_path, 
        "fullTargetDir": full_target_dir
    }
    
    if errors:
        response_content["errors"] = errors
        
    return JSONResponse(content=response_content)


@router.post("/query")
async def query_docs(request: QueryRequest):
    collection_name = f"customer_{CUSTOMER_ID}_documents"
    try:
        query_emb = get_embeddings_from_llama([request.query])[0]
        results = search_similar(collection_name, query_emb, request.fileIds, request.top_k)
        top_chunks = [r.payload["text"] for r in results]
        context = "\n\n".join(top_chunks)
        prompt = f"Answer the question based on the following text:\n\n{context}\n\nQuestion: {request.query}\nAnswer:"
        logger.info(f"Query: {request.query}, Found {len(results)} relevant chunks")
        answer = await get_llama_chat_completion(prompt)
        formatted_results = [
            {"file_id": r.payload["file_id"], "filename": r.payload["filename"], "score": r.score, "text": r.payload["text"]}
            for r in results
        ]
        return JSONResponse(content={"answer": answer, "sources": formatted_results, "question": request.query})
    except Exception as e:
        logger.error(f"Error in query processing: {e}")
        return JSONResponse(content={"error": f"Failed to process query: {str(e)}"}, status_code=500)


@router.post("/query-stream")
async def query_docs_stream(request: QueryRequest, response: Response):
    collection_name = f"customer_{CUSTOMER_ID}_documents"

    async def generate():
        try:
            print(f"🎯 STARTING QUERY STREAM: '{request.query}'")

            adaptive_top_k = calculate_adaptive_top_k(request.query)
            effective_top_k = max(request.top_k, adaptive_top_k)  # Use whichever is larger
            

            query_emb = get_embeddings_from_llama([request.query])[0]
            results = search_similar(collection_name, query_emb, request.fileIds, request.project_name, effective_top_k)
            
           
            top_chunks = [r.payload["text"] for r in results]
            context = "\n\n".join(top_chunks)
           

            full_answer = ""
            async for chunk in local_llama.get_llama_stream_completion(request.query, context):
                full_answer += chunk
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

            formatted_results = []
            for r in results:
                formatted_result = {
                    "file_id": str(r.payload.get("file_id", "")),
                    "filename": str(r.payload.get("filename", "")),
                    "project_name": str(r.payload.get("project_name", "")),
                    "score": round(float(r.score), 6),
                    "text": str(r.payload.get("text", ""))[:500] + "..." if len(r.payload.get("text", "")) > 500 else str(r.payload.get("text", "")),
                }
                formatted_results.append(formatted_result)

            sources_data = {"type": "sources", "sources": formatted_results}
            yield f"data: {json.dumps(sources_data)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'content': full_answer})}\n\n"
        except Exception as e:
            error_msg = f"Error in stream processing: {str(e)}"
            logger.error(error_msg)
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",
        },
    )



@router.get("/download/{collection_type}/{file_id}")
async def download_file(
    collection_type: Literal["documents", "templates"],
    file_id: str
):
    """
    Download a file by its file_id from specified collection type
    
    Args:
        collection_type: Either "documents" or "templates" (required)
        file_id: The ID of the file to download
    """
    # Get the full collection name
    collection_name = get_collection_name(collection_type)
    
    print(f"🔍 Searching for file {file_id} in collection: {collection_name}")
    
    # Find file path
    file_path = await find_file_path_by_id(collection_name, file_id)
    
    # Get file info
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    
    # Determine content type
    content_type = get_content_type(filename)
    
    print(f"✅ Found file: {filename} at path: {file_path}")
    
    # Return the file
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Length": str(file_size)
        }
    )

@router.get("/stream/{collection_type}/{file_id}")
async def download_file_stream(
    collection_type: Literal["documents", "templates"],
    file_id: str
):
    """
    Stream download a file by its file_id
    Useful for large files
    """
    # Get the full collection name
    collection_name = get_collection_name(collection_type)
    
    # Find file path
    file_path = await find_file_path_by_id(collection_name, file_id)
    filename = os.path.basename(file_path)
    
    # Stream the file
    def iterfile():
        with open(file_path, mode="rb") as file_like:
            yield from file_like
    
    content_type = get_content_type(filename)
    
    return Response(
        iterfile(),
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )

@router.get("/info/{collection_type}/{file_id}")
async def get_file_info(
    collection_type: Literal["documents", "templates"],
    file_id: str
):
    """
    Get file information without downloading
    """
    # Get the full collection name
    collection_name = get_collection_name(collection_type)
    
    try:
        client = get_qdrant_client()
        
        # Search for the file
        scroll_result = client.scroll(
            collection_name=collection_name,
            scroll_filter={
                "must": [
                    {"key": "file_id", "match": {"value": file_id}}
                ]
            },
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        
        if not scroll_result[0]:
            raise HTTPException(
                status_code=404,
                detail=f"File with ID {file_id} not found in collection {collection_name}"
            )
        
        point = scroll_result[0][0]
        payload = point.payload
        
        # Get file path
        file_path = await find_file_path_by_id(collection_name, file_id)
        
        # Get file stats
        file_exists = os.path.exists(file_path)
        file_stats = {}
        
        if file_exists:
            stat_info = os.stat(file_path)
            import datetime
            file_stats = {
                "size": stat_info.st_size,
                "created": datetime.datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                "modified": datetime.datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                "exists": True
            }
        else:
            file_stats = {"exists": False}
        
        # Extract filename from payload or file path
        filename = payload.get("filename") or os.path.basename(file_path)
        
        return {
            "file_id": file_id,
            "filename": filename,
            "collection_type": collection_type,
            "collection_name": collection_name,
            "file_path": file_path,
            "file_exists": file_exists,
            "stats": file_stats,
            "metadata": {
                "upload_path": payload.get("upload_path"),
                "project": payload.get("project"),
                "upload_date": payload.get("upload_time"),
                "total_chunks": payload.get("total_chunks", 0),
                "auto_generated": payload.get("auto_generated", False),
                "source_template_id": payload.get("source_template_id"),
                "category": payload.get("category")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting file info: {str(e)}"
        )

@router.get("/health")
async def health_check():
    try:
        collection_name = f"customer_{CUSTOMER_ID}_documents"
        ensure_collection(collection_name, vector_size=VECTOR_SIZE)
        return {"status": "healthy", "qdrant": "connected", "llama": "available"}
    except Exception as e:
        return {"status": "unhealthy", "qdrant": "error", "llama": "error", "error": str(e)}


@router.post("/test-embeddings")
async def testembeedings():
    collection_name = f"customer_{CUSTOMER_ID}_documents"
    try:
        query_emb = test_exact_vector_search(collection_name)
        return JSONResponse(query_emb)
    except Exception as e:
        logger.error(f"Error in query processing: {e}")
        return JSONResponse(content={"error": f"Failed to process query: {str(e)}"}, status_code=500)


@router.get("/diagnose-collection")
async def diagnose_collection():
    collection_name = f"customer_{CUSTOMER_ID}_documents"
    qdrant_client = get_qdrant_client()
    diagnostics = {}
    try:
        collection_info = qdrant_client.get_collection(collection_name=collection_name)
        diagnostics["collection_info"] = {
            "status": collection_info.status,
            "points_count": collection_info.points_count,
            "vectors_count": collection_info.vectors_count,
            "distance": str(collection_info.config.params.vectors.distance),
            "vector_size": collection_info.config.params.vectors.size,
        }
        points, _ = qdrant_client.scroll(collection_name=collection_name, limit=10, with_payload=True, with_vectors=True)
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
                    "file_id": point.payload.get("file_id"),
                },
            }
            diagnostics["sample_points"].append(point_info)

        if points and points[0].vector:
            test_vector = points[0].vector
            search_results = qdrant_client.search(collection_name=collection_name, query_vector=test_vector, limit=5, with_payload=True)
            diagnostics["test_search"] = [
                {"score": float(result.score), "id": result.id, "filename": result.payload.get("filename")} for result in search_results
            ]

        return diagnostics
    except Exception as e:
        return {"error": str(e)}


@router.post("/test-search-only")
async def test_search_only(request: QueryRequest):
    collection_name = f"customer_{CUSTOMER_ID}_documents"
    try:
        print(f"🧪 TEST SEARCH: '{request.query}'")
        query_emb = get_embeddings_from_llama([request.query])[0]
        results = search_similar(collection_name, query_emb, request.fileIds, request.top_k)
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append(
                {
                    "rank": i + 1,
                    "score": float(result.score),
                    "filename": result.payload.get("filename", "N/A"),
                    "file_id": result.payload.get("file_id", "N/A"),
                    "text_preview": result.payload.get("text", "")[:100] + "..." if result.payload.get("text") else "No text",
                    "chunk_index": result.payload.get("chunk_index", "N/A"),
                }
            )

        return {"success": True, "query": request.query, "results_count": len(results), "results": formatted_results, "has_results": len(results) > 0}
    except Exception as e:
        return {"success": False, "error": str(e)}
