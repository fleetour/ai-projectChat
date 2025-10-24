from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os
import uuid
import json
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

from db.qdrant_service import ensure_collection, save_embeddings, search_similar
from services.embeddings_service import get_embeddings_from_mistral, extract_text_from_file, get_mistral_chat_completion

# === CONFIGURATION ===
FILES_DIR = "uploaded_files"
CUSTOMER_ID = 1
VECTOR_SIZE = 1024  # depends on your Mistral model

os.makedirs(FILES_DIR, exist_ok=True)

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


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    collection_name = f"customer_{CUSTOMER_ID}_documents"
    ensure_collection(collection_name, vector_size=VECTOR_SIZE)

    results = []

    for file in files:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(FILES_DIR, f"{file_id}_{file.filename}")

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        text = extract_text_from_file(file_path) 
        # Split text into chunks
        chunks = [text[i:i + 500] for i in range(0, len(text), 500)]

        # Get embeddings from Mistral
        embeddings = get_embeddings_from_mistral(chunks)

        # Save to Qdrant
        save_embeddings(collection_name, file_id, file.filename, chunks, embeddings)

        results.append({"fileId": file_id, "filename": file.filename})

    return JSONResponse(content={"uploaded": results})


@app.post("/query")
async def query_docs(request: QueryRequest):
    collection_name = f"customer_{CUSTOMER_ID}_documents"

    query_emb = get_embeddings_from_mistral([request.query])[0]

    # Step 2: Search for similar chunks in Qdrant using the query embedding
    results = search_similar(collection_name, query_emb, request.fileIds, request.top_k)

    # Step 3: Extract the actual TEXT from top results to form context
    top_chunks = [r.payload["text"] for r in results]
    context = "\n\n".join(top_chunks)

     # Step 4: Create prompt and send to Mistral chat model
    prompt = f"Answer the question based on the following text:\n\n{context}\n\nQuestion: {request.query}\nAnswer:"

    print("prompt:", prompt)

    try:
        answer = await get_mistral_chat_completion(prompt)
    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to get AI response: {str(e)}"}, 
            status_code=500
        )

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

@app.get("/files")
async def get_files():
    """Get the file structure from uploaded files and Qdrant"""
    try:
        collection_name = f"customer_{CUSTOMER_ID}_documents"
        
        # Get files from both the file system and Qdrant
        file_structure = await build_file_structure(collection_name)
        
        return file_structure
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving files: {str(e)}")

async def build_file_structure(collection_name: str) -> List[Dict[str, Any]]:
    """Build the file structure from uploaded files and Qdrant collections"""
    
    # Get all files from the uploaded_files directory
    uploaded_files = []
    if os.path.exists(FILES_DIR):
        for filename in os.listdir(FILES_DIR):
            if not filename.startswith('.'):  # Skip hidden files
                file_path = os.path.join(FILES_DIR, filename)
                if os.path.isfile(file_path):
                    # Extract file_id from filename (format: {file_id}_{original_filename})
                    parts = filename.split('_', 1)
                    if len(parts) == 2:
                        file_id, original_filename = parts
                        uploaded_files.append({
                            "file_id": file_id,
                            "filename": original_filename,
                            "uploaded_name": filename
                        })
    
    # Get file information from Qdrant
    qdrant_files = await get_files_from_qdrant(collection_name)
    
    # Merge the data
    file_info_map = {}
    for qfile in qdrant_files:
        file_info_map[qfile["file_id"]] = qfile
    
    for ufile in uploaded_files:
        if ufile["file_id"] not in file_info_map:
            file_info_map[ufile["file_id"]] = {
                "file_id": ufile["file_id"],
                "filename": ufile["filename"],
                "chunks": 0
            }
    
    # Organize by file type
    return organize_files_by_type(list(file_info_map.values()))

async def get_files_from_qdrant(collection_name: str) -> List[Dict[str, Any]]:
    """Get file information from Qdrant collection"""
    try:
        client = get_qdrant_client()
        
        # Get all points in the collection to extract file information
        # Note: This might be inefficient for large collections - consider storing file metadata separately
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=1000,  # Adjust based on your needs
            with_payload=True,
            with_vectors=False
        )
        
        files = {}
        for point in scroll_result[0]:
            payload = point.payload
            if payload and "file_id" in payload:
                file_id = payload["file_id"]
                if file_id not in files:
                    files[file_id] = {
                        "file_id": file_id,
                        "filename": payload.get("filename", "Unknown"),
                        "chunks": 0
                    }
                files[file_id]["chunks"] += 1
        
        return list(files.values())
        
    except Exception as e:
        print(f"Error getting files from Qdrant: {e}")
        return []

def organize_files_by_type(files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Organize files by their type/extension"""
    
    # Group files by extension
    pdf_files = [f for f in files if f["filename"].lower().endswith('.pdf')]
    docx_files = [f for f in files if f["filename"].lower().endswith('.docx')]
    excel_files = [f for f in files if any(f["filename"].lower().endswith(ext) for ext in ['.xlsx', '.xls'])]
    xml_files = [f for f in files if f["filename"].lower().endswith('.xml')]
    other_files = [f for f in files if f not in pdf_files + docx_files + excel_files + xml_files]
    
    structure = []
    
    if pdf_files:
        structure.append({
            "name": "PDF Documents",
            "type": "folder",
            "children": [
                {
                    "name": file["filename"],
                    "type": "file",
                    "fileId": file["file_id"],
                    "filename": file["filename"],
                    "extension": "pdf",
                    "chunks": file.get("chunks", 0)
                }
                for file in pdf_files
            ]
        })
    
    if docx_files:
        structure.append({
            "name": "Word Documents", 
            "type": "folder",
            "children": [
                {
                    "name": file["filename"],
                    "type": "file", 
                    "fileId": file["file_id"],
                    "filename": file["filename"],
                    "extension": "docx",
                    "chunks": file.get("chunks", 0)
                }
                for file in docx_files
            ]
        })
    
    if excel_files:
        structure.append({
            "name": "Excel Files",
            "type": "folder", 
            "children": [
                {
                    "name": file["filename"],
                    "type": "file",
                    "fileId": file["file_id"], 
                    "filename": file["filename"],
                    "extension": "xlsx",
                    "chunks": file.get("chunks", 0)
                }
                for file in excel_files
            ]
        })
    
    if xml_files:
        structure.append({
            "name": "XML Files",
            "type": "folder",
            "children": [
                {
                    "name": file["filename"],
                    "type": "file",
                    "fileId": file["file_id"],
                    "filename": file["filename"], 
                    "extension": "xml",
                    "chunks": file.get("chunks", 0)
                }
                for file in xml_files
            ]
        })
    
    if other_files:
        structure.append({
            "name": "Other Files",
            "type": "folder",
            "children": [
                {
                    "name": file["filename"],
                    "type": "file",
                    "fileId": file["file_id"],
                    "filename": file["filename"],
                    "extension": file["filename"].split('.')[-1] if '.' in file["filename"] else "file",
                    "chunks": file.get("chunks", 0)
                }
                for file in other_files
            ]
        })
    
    # If no files found, return empty structure
    if not structure:
        structure = [{
            "name": "Documents",
            "type": "folder", 
            "children": []
        }]
    
    return structure