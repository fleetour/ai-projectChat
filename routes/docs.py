import asyncio
from datetime import datetime
import io
from pathlib import Path
from typing import Literal
from fastapi import APIRouter, Form, UploadFile, File, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import os
import uuid
import json
import logging
import numpy as np

from qdrant_client import models
import urllib

from services.azure_blob_service_async import get_async_blob_service
from services.conversation_service import AsyncConversationService, get_conversation_service
from services.embeddings_service import (
    ensure_cosine_collection,
    get_embeddings_async,
    get_embeddings_from_llama,
    get_llama_chat_completion,
    normalize_vector,
    save_embeddings_async,
    save_embeddings_with_path,
)
from services.file_service import build_file_structure_tree, find_file_path_by_id, find_file_payload, intelligent_chunking_simple, intelligent_chunking_simple_async, normalize_target_path
from db.qdrant_service import ensure_collection, ensure_collection_async, get_qdrant_client, search_similar
from services.local_llma_service import LocalLlamaService
from routes.schemas import QueryRequest
from config import FILES_DIR, CUSTOMER_ID, VECTOR_SIZE
from services.utils import calculate_adaptive_top_k, extract_text_from_bytes, extract_text_from_bytes_async, extract_text_from_file, format_results_for_history, get_collection_name, get_content_type

router = APIRouter(tags=["documents"])
logger = logging.getLogger(__name__)




@router.post("/upload")
async def upload_files(
    target_project: str = Form(""),
    target_path: str = Form(""),
    files: list[UploadFile] = File(...),
):
    """Async file upload function with Azure Blob Storage and parallel processing."""
    
    collection_name = f"customer_{CUSTOMER_ID}_documents"
    
    if (not target_project) or target_project.strip() == "":
        raise HTTPException(status_code=400, detail="Target project must be specified")

    # Get async blob service
    blob_service = await get_async_blob_service(base_folder="Projects")
    
    upload_results = []
    errors = []
    target_path = normalize_target_path(target_path)
    
    # Ensure collection exists (async if needed)
    await ensure_collection_async(collection_name, vector_size=VECTOR_SIZE)
    
    # Process files in parallel with semaphore to limit concurrency
    semaphore = asyncio.Semaphore(3)  # Process 3 files concurrently
    
    async def process_single_file(file: UploadFile) -> dict:
        """Process a single file asynchronously."""
        async with semaphore:
            return await _process_file(
                file=file,
                blob_service=blob_service,
                target_project=target_project,
                target_path=target_path,
                collection_name=collection_name
            )
    
    # Create tasks for all files
    processing_tasks = [process_single_file(file) for file in files]
    
    # Process files concurrently
    results = await asyncio.gather(*processing_tasks, return_exceptions=True)
    
    # Collect results
    for i, result in enumerate(results):
        filename = files[i].filename if i < len(files) else f"file_{i}"
        
        if isinstance(result, Exception):
            errors.append(f"File {filename}: Processing failed - {str(result)}")
            logger.error(f"❌ Error processing {filename}: {result}", exc_info=True)
        elif result.get("error"):
            errors.append(f"File {filename}: {result['error']}")
        else:
            upload_results.append(result)
            logger.info(f"✅ Successfully processed: {filename}")
    
    response_content = {
        "uploaded": upload_results, 
        "targetPath": target_path, 
        "targetProject": target_project,
        "container": f"customer_{CUSTOMER_ID}",
        "totalProcessed": len(files),
        "successful": len(upload_results),
        "failed": len(errors)
    }
    
    if errors:
        response_content["errors"] = errors
        
    return JSONResponse(content=response_content)


async def _process_file(
    file: UploadFile,
    blob_service,
    target_project: str,
    target_path: str,
    collection_name: str
) -> dict:
    """Process a single file asynchronously."""
    file_id = str(uuid.uuid4())
    
    try:
        # 1. Read file content
        content = await file.read()
     
        if target_path:
        # If target_path is provided, join with target_project
            target_path = str(Path(target_project) / target_path)
        else:
            # If target_path is empty, use just target_project
            target_path = target_project

        # 2. Upload to Azure Blob (async)
        upload_result = await blob_service.upload_file(
            customer_id=str(CUSTOMER_ID),
            file_content=content,
            filename=file.filename,
            target_path=target_path,
            content_type=file.content_type,
            metadata={
                "original_filename": file.filename,
                "project": target_project,
                "upload_path": target_path
            }
        )
        
        # 3. Extract text from content (async)
        text = await extract_text_from_bytes_async(content, filename=file.filename)
        
        if not text or len(text.strip()) < 10:
            # Delete from Azure if no text
            try:
                await blob_service.delete_file(str(CUSTOMER_ID), upload_result["blob_name"])
            except Exception as delete_error:
                logger.warning(f"Failed to delete blob: {delete_error}")
            
            return {"error": "No readable text content found"}
        
        # 4. Intelligent chunking (async)
        chunks = await intelligent_chunking_simple_async(
            text=text,
            max_chunk_size=1500,
            overlap=200
        )
        
        if not chunks:
            try:
                await blob_service.delete_file(str(CUSTOMER_ID), upload_result["blob_name"])
            except Exception as delete_error:
                logger.warning(f"Failed to delete blob: {delete_error}")
            
            return {"error": "No valid text chunks created"}
        
        logger.info(f"✅ Processing {file.filename}: {len(chunks)} chunks")
        
        # 5. Get embeddings asynchronously in batches
        all_embeddings = []
        batch_size = 10
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = await get_embeddings_async(
                texts=batch_chunks,
                model="mistral:7b"
            )
            
            if not batch_embeddings:
                # Clean up Azure blob
                try:
                    await blob_service.delete_file(str(CUSTOMER_ID), upload_result["blob_name"])
                except:
                    pass
                return {"error": f"Embedding batch {i//batch_size + 1} failed"}
            
            if len(batch_embeddings) != len(batch_chunks):
                logger.warning(f"Embedding mismatch: expected {len(batch_chunks)}, got {len(batch_embeddings)}")
                all_embeddings.extend(batch_embeddings[:len(batch_chunks)])
            else:
                all_embeddings.extend(batch_embeddings)
            
            logger.debug(f"  Processed embedding batch {i//batch_size + 1}")
        
        # 6. Adjust chunks if embedding mismatch
        if len(all_embeddings) != len(chunks):
            logger.warning(f"Final embedding mismatch for {file.filename}: chunks={len(chunks)}, embeddings={len(all_embeddings)}")
            min_length = min(len(chunks), len(all_embeddings))
            chunks = chunks[:min_length]
            all_embeddings = all_embeddings[:min_length]
        
        # 7. Save to Qdrant asynchronously
        save_success = await save_embeddings_async(
            collection_name=collection_name,
            file_id=upload_result["file_id"],
            filename=file.filename,
            chunks=chunks,
            embeddings=all_embeddings,
            target_path=target_path,
            target_project=target_project,
            blob_metadata=upload_result,
            auto_generated=False,
            source_template_id=None
        )
        
        if not save_success:
            # Clean up Azure blob if Qdrant save failed
            try:
                await blob_service.delete_file(str(CUSTOMER_ID), upload_result["blob_name"])
            except:
                pass
            return {"error": "Failed to save embeddings to database"}
        
        return {
            "fileId": upload_result["file_id"],
            "filename": file.filename,
            "chunks": len(chunks),
            "path": target_path,
            "project": target_project,
            "blobUrl": upload_result["blob_url"],
            "container": upload_result["container"],
            "blobPath": upload_result["blob_name"],
            "size": upload_result["size"],
            "contentType": file.content_type
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing {file.filename}: {e}", exc_info=True)
        return {"error": str(e)}



@router.post("/query")
async def query_docs(request: QueryRequest):
    collection_name = f"customer_{CUSTOMER_ID}_documents"
    try:
        query_emb = get_embeddings_from_llama([request.query], request.model)[0]
        results = search_similar(collection_name, query_emb, request.fileIds, request.top_k)
        top_chunks = [r.payload["text"] for r in results]
        context = "\n\n".join(top_chunks)
        prompt = f"Answer the question based on the following text:\n\n{context}\n\nQuestion: {request.query}\nAnswer:"
        logger.info(f"Query: {request.query}, Found {len(results)} relevant chunks")
        answer = await get_llama_chat_completion(prompt, request.model)
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
        conversation_id = None
        
        try:
            print(f"🎯 STARTING QUERY STREAM: '{request.query}'")
            
            # Get conversation ID from request or generate
            conversation_id = getattr(request, 'conversation_id', None)
            user_id = getattr(request, 'user_id', 'anonymous')
            
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
                # Notify client of new conversation
                yield f"data: {json.dumps({
                    'type': 'conversation_created',
                    'conversation_id': conversation_id
                })}\n\n"
            
            # Get conversation history
            conv_service = await get_conversation_service()
       
            # Add user message
            await conv_service.add_message(
                conversation_id=conversation_id,
                role="user",
                content=request.query,
                metadata={
                    "project": request.project_name,
                    "files": request.fileIds,
                    "model": request.model
                }
            )
            # Get recent history for context
            history_messages = await conv_service.get_recent_history(conversation_id, max_messages=3)
            conversation_history = "\n".join(history_messages) if history_messages else ""
         
            adaptive_top_k = calculate_adaptive_top_k(request.query)
            effective_top_k = max(request.top_k, adaptive_top_k)
            
            query_emb = get_embeddings_from_llama([request.query], request.model)[0]
            results = search_similar(collection_name, query_emb, request.fileIds, request.project_name, effective_top_k)
       
            top_chunks = [r.payload["text"] for r in results]
            context = "\n\n".join(top_chunks)
            
            local_llama = LocalLlamaService(model=request.model)
            full_answer = ""
            
            # Stream response with conversation_id in each chunk
            async for chunk in local_llama.get_llama_stream_completion(
                question=request.query,
                context=context,
                conversation_history=conversation_history
            ):
                full_answer += chunk
                yield f"data: {json.dumps({
                    'type': 'content', 
                    'content': chunk,
                    'conversation_id': conversation_id  # Include in content chunks
                })}\n\n"

            # Add assistant response
            await conv_service.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=full_answer,
                metadata={
                    "sources": [
                        {
                            "file_id": r.payload.get("file_id"),
                            "filename": r.payload.get("filename"),
                            "score": float(r.score)
                        }
                        for r in results[:3]
                    ]
                }
            )

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

            # Send sources with conversation_id
            sources_data = {
                "type": "sources", 
                "sources": formatted_results,
                "conversation_id": conversation_id
            }
            yield f"data: {json.dumps(sources_data)}\n\n"
            
            # Send completion with conversation info
            conversation = await conv_service.get_conversation(conversation_id)
            yield f"data: {json.dumps({
                'type': 'done', 
                'content': full_answer,
                'conversation_id': conversation_id,
                'history_length': len(conversation.get("messages", []))
            })}\n\n"
            
        except Exception as e:
            error_msg = f"Error in stream processing: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"

    # FIXED: Provide default value for conversation_id in headers
    # We'll get it from the request or generate a placeholder
    request_conversation_id = getattr(request, 'conversation_id', None)
    headers_conversation_id = request_conversation_id or str(uuid.uuid4())
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",
            "X-Conversation-ID": headers_conversation_id  # Always a string
        },
    )



@router.get("/download/{collection_type}/{file_id}")
async def download_file(
    collection_type: Literal["documents", "templates"],
    file_id: str
):
    """
    Download a file by its file_id from Azure Blob Storage
    """
    try:
        # Get the full collection name
        collection_name = get_collection_name(collection_type)
        
        print(f"🔍 Searching for file {file_id} in collection: {collection_name}")
        
        payload = await find_file_payload(collection_name, file_id)
        
        if not payload:
            raise HTTPException(
                status_code=404,
                detail=f"File {file_id} not found in collection {collection_type}"
            )
        
        # 1. Find the full_file_path from Qdrant
        full_file_path = payload.get("full_file_path") or payload.get("full_path") or payload.get("blob_name")
        
        if not full_file_path:
            raise HTTPException(
                status_code=404,
                detail=f"No blob path found for file {file_id}"
            )
        
        # The full_file_path IS the blob name in Azure!
        blob_name = full_file_path
        
        print(f"✅ Found blob path: {blob_name}")
        
        # 2. Extract original filename for download (with fallback)
        original_filename = payload.get("filename")
        
        # 3. Get blob service
        blob_service = await get_async_blob_service(collection_type)
        
        # 4. Download file from Azure Blob
        try:
            file_content, blob_info = await blob_service.download_file(
                customer_id=CUSTOMER_ID,
                blob_name=blob_name
            )
            
            # Use the original filename for download with safe fallback
            download_filename = original_filename or blob_info.get("filename") or f"file_{file_id}"
            
            # FIX: Ensure download_filename is a string before quoting
            if isinstance(download_filename, bytes):
                download_filename = download_filename.decode('utf-8', errors='ignore')
            
            # FIX: Only quote if we have a valid string
            if download_filename and isinstance(download_filename, str):
                safe_filename = urllib.parse.quote(download_filename)
            else:
                safe_filename = f"file_{file_id}"
            
            print(f"✅ Downloaded: {download_filename}, size: {len(file_content)} bytes")
            
            # 5. Return file as StreamingResponse
            return StreamingResponse(
                content=io.BytesIO(file_content),
                media_type=blob_info.get("content_type", "application/octet-stream"),
                headers={
                    "Content-Disposition": f"attachment; filename=\"{safe_filename}\"",
                    "Content-Length": str(len(file_content)),
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                    "X-File-Id": file_id,
                    "X-Blob-Path": blob_name
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Error downloading blob {blob_name}: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download file from storage: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error downloading file {file_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download file: {str(e)}"
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
