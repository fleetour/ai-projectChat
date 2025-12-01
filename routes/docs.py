from fastapi import APIRouter, Form, UploadFile, File, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
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
from services.file_service import build_file_structure_tree, normalize_target_path
from db.qdrant_service import ensure_collection, get_qdrant_client, search_similar, test_exact_vector_search
from services.local_llma_service import LocalLlamaService
from routes.schemas import QueryRequest
from config import FILES_DIR, CUSTOMER_ID, VECTOR_SIZE
from services.utils import extract_text_from_file

router = APIRouter(tags=["documents"])
logger = logging.getLogger(__name__)

local_llama = LocalLlamaService(model="llama3:8b")


@router.post("/upload")
async def upload_files(
    target_project: str = Form(""),
    target_path: str = Form(""),
    files: list[UploadFile] = File(...),
):
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
        try:
            file_id = str(uuid.uuid4())
            file_path = os.path.join(full_target_dir, f"{file_id}_{file.filename}")
            relative_file_path = os.path.join(target_path, f"{file_id}_{file.filename}")
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            text = extract_text_from_file(file_path)
            if not text or len(text.strip()) < 10:
                errors.append(f"File {file.filename}: No readable text content found")
                os.remove(file_path)
                continue

            chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
            chunks = [chunk for chunk in chunks if len(chunk.strip()) > 10]
            if not chunks:
                errors.append(f"File {file.filename}: No valid text chunks created")
                os.remove(file_path)
                continue

            logger.info(f"Processing {file.filename}: {len(chunks)} chunks, path: {target_path}")
            embeddings = get_embeddings_from_llama(chunks)
            if not embeddings or len(embeddings) != len(chunks):
                errors.append(f"File {file.filename}: Embedding count mismatch")
                os.remove(file_path)
                continue

            ensure_cosine_collection(qdrant_client, collection_name, vector_size=4096)

            save_embeddings_with_path(
                qdrant_client=qdrant_client,
                collection_name=collection_name,
                file_id=file_id,
                filename=file.filename,
                chunks=chunks,
                embeddings=embeddings,
                target_path=target_path,
                target_project=target_project,
            )

            upload_results.append({
                "fileId": file_id,
                "filename": file.filename,
                "chunks": len(chunks),
                "path": target_path,
                "fullPath": relative_file_path,
            })

        except Exception as e:
            errors.append(f"File {file.filename}: Processing failed - {str(e)}")
            logger.error(f"Error processing {file.filename}: {e}")
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)

    response_content = {"uploaded": upload_results, "targetPath": target_path, "fullTargetDir": full_target_dir}
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
            query_emb = get_embeddings_from_llama([request.query])[0]
            results = search_similar(collection_name, query_emb, request.fileIds, request.top_k)
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
