import asyncio
from datetime import datetime
import os
import uuid
from fastapi import UploadFile, HTTPException
from typing import Any, Dict, List, Optional
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from db.qdrant_service import get_qdrant_client, get_qdrant_client_async
from services.azure_blob_service_async import get_async_blob_service
from services.templates_utilitis import classify_document_type, count_placeholders, extract_key_topics, find_completion_points, identify_sections
from services.utils import extract_text_from_bytes, extract_text_from_file, validate_file_type

logger = logging.getLogger(__name__)

# Configuration
TEMPLATES_BASE_DIR = "Templates"
CUSTOMER_ID = 1  # Your customer ID

# async def upload_template_files(
#     files: List[UploadFile],
#     target_path: str = "",
#     category: str = ""
# ) -> dict:
#     """
#     Upload template files and save metadata in Qdrant
#     """
#     upload_results = []
#     errors = []
    
#     # Normalize paths
#     target_path = normalize_template_path(target_path)
#     category = normalize_template_path(category)
    
#     # Create full target directory with category
#     if category:
#         full_target_dir = os.path.join(TEMPLATES_BASE_DIR, category, target_path)
#     else:
#         full_target_dir = os.path.join(TEMPLATES_BASE_DIR, target_path)
    
#     os.makedirs(full_target_dir, exist_ok=True)
    
#     # Get Qdrant client
#     qdrant_client = get_qdrant_client()
#     collection_name = f"customer_{CUSTOMER_ID}_templates"
    
#     # Ensure collection exists
#     await ensure_templates_collection(qdrant_client, collection_name)
    
#     for file in files:
#         try:
#             validate_file_type(file.filename)
#             file_id = str(uuid.uuid4())
#             safe_filename = f"{file_id}_{file.filename}"
            
#             # Create file path with category
#             if category:
#                 file_path = os.path.join(TEMPLATES_BASE_DIR, category, target_path, safe_filename)
#                 relative_file_path = os.path.join(category, target_path, safe_filename)
#             else:
#                 file_path = os.path.join(TEMPLATES_BASE_DIR, target_path, safe_filename)
#                 relative_file_path = os.path.join(target_path, safe_filename)
            
#             # Save the file to filesystem
#             with open(file_path, "wb") as f:
#                 content = await file.read()
#                 f.write(content)
            
#             # Verify file was saved
#             if not os.path.exists(file_path):
#                 errors.append(f"File {file.filename}: Failed to save")
#                 continue
            
#             # Get file info
#             file_size = os.path.getsize(file_path)
#             created_time = datetime.now()
            
#             # Analyze template content
#             template_analysis = await analyze_template_content(file_path, file.filename)
            
#             # Save metadata to Qdrant
#             metadata_id = await save_template_metadata_to_qdrant(
#                 qdrant_client=qdrant_client,
#                 collection_name=collection_name,
#                 file_id=file_id,
#                 original_filename=file.filename,
#                 safe_filename=safe_filename,
#                 file_path=file_path,
#                 relative_path=relative_file_path,
#                 target_path=target_path,
#                 category=category,  # NEW: Add category
#                 file_size=file_size,
#                 content_type=file.content_type,
#                 created_time=created_time,
#                 analysis=template_analysis
#             )
            
#             upload_results.append({
#                 "fileId": file_id,
#                 "metadataId": metadata_id,
#                 "filename": file.filename,
#                 "savedAs": safe_filename,
#                 "path": target_path,
#                 "category": category,  # NEW: Add category to response
#                 "fullPath": relative_file_path,
#                 "size": file_size,
#                 "fileType": file.content_type,
#                 "created": created_time.isoformat(),
#                 "analysis": template_analysis,
#                 "metadataSaved": metadata_id is not None
#             })
            
#             print(f"✅ Template uploaded: {file.filename}")
#             print(f"   📁 Category: {category}")
#             print(f"   📁 Path: {target_path}")
#             print(f"   💾 Saved to: {relative_file_path}")
#             print(f"   🗄️  Metadata ID: {metadata_id}")
            
#         except Exception as e:
#             errors.append(f"File {file.filename}: Upload failed - {str(e)}")
#             print(f"❌ Error uploading {file.filename}: {e}")
    
#     response_content = {
#         "uploaded": upload_results,
#         "targetPath": target_path,
#         "category": category,  # NEW: Add category to response
#         "fullTargetDir": full_target_dir,
#         "totalUploaded": len(upload_results)
#     }
    
#     if errors:
#         response_content["errors"] = errors
    
#     return response_content

async def upload_template_files(
    files: List[UploadFile],
    target_path: str = "",
    category: str = "",
    customer_id: str = "1"
) -> dict:
    """
    Upload template files to Azure Blob Storage
    """
    
    
    upload_results = []
    errors = []
    
    # Normalize paths
    target_path = normalize_template_path(target_path)
    category = normalize_template_path(category)
    
    # Get Qdrant client
    qdrant_client = get_qdrant_client()
    collection_name = f"customer_{customer_id}_templates"
    
    # Ensure collection exists
    await ensure_templates_collection(qdrant_client, collection_name)
    
    # Get blob service WITHOUT async context manager
    blob_service = await get_async_blob_service(base_folder="Templates")
    
    # Process files sequentially for now to avoid thread pool issues
    for file in files:
        try:
            result = await process_single_template_safe(
                blob_service=blob_service,
                qdrant_client=qdrant_client,
                file=file,
                target_path=target_path,
                category=category,
                collection_name=collection_name,
                customer_id=customer_id
            )
            
            if "error" in result:
                errors.append(f"File {file.filename}: {result['error']}")
            else:
                upload_results.append(result)
                logger.info(f"✅ Template uploaded: {file.filename}")
                
        except Exception as e:
            errors.append(f"File {file.filename}: Upload failed - {str(e)}")
            logger.error(f"❌ Error uploading {file.filename}: {e}")
    
    # Don't close blob_service here - let it be reused
    
    response_content = {
        "uploaded": upload_results,
        "targetPath": target_path,
        "category": category,
        "totalUploaded": len(upload_results),
        "totalProcessed": len(files)
    }
    
    if errors:
        response_content["errors"] = errors
    
    return response_content


async def process_single_template_safe(
    blob_service,
    qdrant_client,
    file: UploadFile,
    target_path: str,
    category: str,
    collection_name: str,
    customer_id: str
) -> Dict[str, Any]:
    """
    Process a single template file safely
    """
    try:
        # Validate file type
        validate_file_type(file.filename)
        
        # Read file content
        content = await file.read()
        
        # Upload to Azure Blob Storage
        upload_result = await blob_service.upload_file(
            customer_id=customer_id,
            file_content=content,
            filename=file.filename,
            category=category,
            target_path=target_path,
            content_type=file.content_type,
            metadata={
                "category": category,
                "type": "template",
                "filename": file.filename
            }
        )
        
        # Analyze template content
        template_analysis = await analyze_template_content_simple(content, file.filename)
        
        # Save metadata to Qdrant
        metadata_id = await save_template_metadata_async(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            file_id=upload_result["file_id"],
            filename=file.filename,
            blob_metadata=upload_result,
            target_path=target_path,
            category=category,
            analysis=template_analysis
        )
        
        return {
            "fileId": upload_result["file_id"],
            "metadataId": metadata_id,
            "filename": file.filename,
            "path": target_path,
            "category": category,
            "blobUrl": upload_result["blob_url"],
            "blobName": upload_result["blob_name"],
            "size": upload_result["size"],
            "fileType": file.content_type,
            "analysis": template_analysis,
            "metadataSaved": metadata_id is not None
        }
        
    except Exception as e:
        logger.error(f"❌ Processing error: {e}", exc_info=True)
        return {"error": str(e)}


async def save_template_metadata_async(
    qdrant_client,
    collection_name: str,
    file_id: str,
    filename: str,
    blob_metadata: Dict[str, Any],
    target_path: str,
    category: str,
    analysis: Dict[str, Any]
) -> str:
    """
    Save template metadata to Qdrant asynchronously
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        save_template_metadata_sync,
        qdrant_client,
        collection_name,
        file_id,
        filename,
        blob_metadata,
        target_path,
        category,
        analysis
    )


def save_template_metadata_sync(
    qdrant_client,
    collection_name: str,
    file_id: str,
    filename: str,
    blob_metadata: Dict[str, Any],
    target_path: str,
    category: str,
    analysis: Dict[str, Any]
) -> str:
    """
    Save template metadata to Qdrant using NAMED vectors
    """
    try:
        from qdrant_client.models import PointStruct
        
        metadata_id = str(uuid.uuid4())
        created_time = datetime.now()
        
        # Create payload
        payload = {
            "id": metadata_id,
            "file_id": file_id,
            "filename": filename,
            "safe_filename": f"{file_id}_{filename}",
            "full_file_path": blob_metadata.get("full_file_path", ""),
            "relative_path": blob_metadata.get("blob_name", ""),
            "target_path": target_path,
            "category": category,
            "file_size": blob_metadata.get("size", 0),
            "content_type": blob_metadata.get("content_type", ""),
            "created_time": created_time.isoformat(),
            "type": "template",
            "analysis": analysis,
            "storage_type": "azure_blob",
            "blob_name": blob_metadata.get("blob_name"),
            "blob_url": blob_metadata.get("blob_url"),
            "container": blob_metadata.get("container"),
            "base_folder": blob_metadata.get("base_folder", "Templates")
        }
        
        # Create a simple embedding
        import numpy as np
        text_for_embedding = f"{filename} {category} {target_path}"
        simple_embedding = create_simple_embedding(text_for_embedding, vector_size=384)
        
        # Normalize
        norm = np.linalg.norm(simple_embedding)
        if norm > 0:
            simple_embedding = (simple_embedding / norm).tolist()
        
        # **FIXED: Use named vector format**
        point = PointStruct(
            id=metadata_id,
            vector={
                "metadata": simple_embedding  # Named vector
            },
            payload=payload
        )
        
        # Save to Qdrant
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[point],
            wait=True
        )
        
        logger.info(f"✅ Template metadata saved with named vector: {filename}")
        return metadata_id
        
    except Exception as e:
        logger.error(f"❌ Failed to save template metadata: {e}", exc_info=True)
        return None

async def analyze_template_content_simple(content: bytes, filename: str) -> Dict[str, Any]:
    """
    Simple async template content analysis
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        analyze_template_content_simple_sync,
        content,
        filename
    )


def analyze_template_content_simple_sync(content: bytes, filename: str) -> Dict[str, Any]:
    """
    Simple sync template analysis
    """
    try:
        # Extract text
        text = extract_text_from_bytes(content, filename)
        
        return {
            "char_count": len(text),
            "word_count": len(text.split()),
            "language": "unknown",
            "has_tables": False,
            "has_images": False,
            "sections": [],
            "keywords": [],
            "estimated_pages": len(text) // 1500 + 1,
            "text_preview": text[:200] + "..." if len(text) > 200 else text
        }
    except Exception as e:
        logger.error(f"❌ Analysis error: {e}")
        return {
            "char_count": 0,
            "word_count": 0,
            "language": "unknown",
            "has_tables": False,
            "has_images": False,
            "sections": [],
            "keywords": [],
            "estimated_pages": 0,
            "text_preview": ""
        }


def create_simple_embedding(text: str, vector_size: int = 384) -> List[float]:
    """
    Create simple embedding for testing
    """
    import hashlib
    import struct
    
    if not text:
        return [0.0] * vector_size
    
    text_hash = hashlib.sha256(text.encode()).digest()
    floats = []
    
    for i in range(0, min(len(text_hash), vector_size * 4), 4):
        if i + 4 <= len(text_hash):
            val = struct.unpack('I', text_hash[i:i+4])[0]
            normalized = (val / 2**32) * 2 - 1
            floats.append(normalized)
    
    if len(floats) < vector_size:
        floats.extend([0.0] * (vector_size - len(floats)))
    
    return floats[:vector_size]


# Utility functions
def normalize_template_path(path: str) -> str:
    if not path:
        return ""
    path = path.strip("/").strip("\\")
    path = path.replace("\\", "/")
    while "//" in path:
        path = path.replace("//", "/")
    return path

async def ensure_templates_collection(qdrant_client: QdrantClient, collection_name: str):
    """
    Ensure the templates collection exists in Qdrant with proper vector configuration
    """
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        existing_collections = [col.name for col in collections.collections]
        
        if collection_name not in existing_collections:
            # Create collection with named vector configuration
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "metadata": VectorParams(size=384, distance=Distance.COSINE)
                }
            )
            print(f"✅ Created Qdrant collection: {collection_name}")
        else:
            print(f"✅ Using existing Qdrant collection: {collection_name}")
            
    except Exception as e:
        print(f"❌ Error ensuring collection {collection_name}: {e}")
        raise

async def save_template_metadata_to_qdrant(
    qdrant_client: QdrantClient,
    collection_name: str,
    file_id: str,
    original_filename: str,
    safe_filename: str,
    file_path: str,
    relative_path: str,
    target_path: str,
    category: str, 
    file_size: int,
    content_type: str,
    created_time: datetime,
    analysis: Dict[str, Any]
) -> Optional[str]:
    """
    Save template metadata to Qdrant as NoSQL data
    """
    try:
        print(f"🔍 Starting metadata save for: {file_id}")
        print(f"🔍 saving for category : {category}")
        
        # Create comprehensive metadata payload
        metadata_payload = {
            "file_id": file_id,
            "filename": original_filename,
            "safe_filename": safe_filename,
            "file_path": file_path,
            "full_file_path": file_path,
            "relative_path": relative_path,
            "target_path": target_path,
            "category": category,  # NEW: Add category to payload
            "file_size": file_size,
            "content_type": content_type,
            "created_time": created_time.isoformat(),
            "modified_time": created_time.isoformat(),
            "analysis": {
                "document_type": analysis.get("documentType", "unknown"),
                "total_length": analysis.get("totalLength", 0),
                "word_count": analysis.get("wordCount", 0) or len(analysis.get("textContent", "").split()),
                "section_count": len(analysis.get("sections", [])),
                "completion_points_count": len(analysis.get("completionPoints", [])),
                "placeholder_analysis": analysis.get("estimatedPlaceholders", {}),
                "key_topics": analysis.get("keyTopics", []),
                "estimated_complexity": analysis.get("estimatedComplexity", "unknown"),
                "has_analysis_errors": "error" in analysis
            },
            "status": "active",
            "usage_count": 0,
            "last_used": None,
            "tags": generate_template_tags(analysis, original_filename),
            # REMOVED: customer_id (since it's in collection name)
        }
        
        print(f"📦 Metadata payload created: {metadata_payload['original_filename']}")
        
        # Create a point with named vector
        point = PointStruct(
            id=file_id,
            vector={
                "metadata": [0.0]  # Named vector matching collection config
            },
            payload=metadata_payload
        )
        
        print(f"📍 Point created, attempting upsert to collection: {collection_name}")
        
        # Upsert to Qdrant
        operation_info = qdrant_client.upsert(
            collection_name=collection_name,
            points=[point],
            wait=True
        )
        
        print(f"✅ Metadata saved to Qdrant: {file_id}")
        return file_id
        
    except Exception as e:
        print(f"❌ Failed to save metadata to Qdrant: {e}")
        import traceback
        print(f"🔍 Stack trace: {traceback.format_exc()}")
        return None
    
async def get_template_metadata(file_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve template metadata from Qdrant by file ID asynchronously
    """
    try:
        qdrant_client = await get_qdrant_client_async()  # Use async version
        collection_name = f"customer_{CUSTOMER_ID}_templates"
        
        # Run synchronous Qdrant scroll in thread pool
        loop = asyncio.get_event_loop()
        points = await loop.run_in_executor(
            None,
            lambda: qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "file_id",
                            "match": {
                                "value": file_id
                            }
                        }
                    ]
                },
                limit=1
            )
        )
        
        if points and len(points[0]) > 0:  # Note: scroll returns (points, next_page_offset)
            return points[0][0].payload
        else:
            return None
            
    except Exception as e:
        print(f"❌ Error loading metadata for {file_id}: {e}")
        return None

async def update_template_usage(file_id: str):
    """
    Update template usage statistics in Qdrant
    """
    try:
        qdrant_client = get_qdrant_client()
        collection_name = f"customer_{CUSTOMER_ID}_templates"
        
        # Get current metadata
        current_metadata = await get_template_metadata(file_id)
        if not current_metadata:
            return False
        
        # Update usage count and last used time
        current_metadata["usage_count"] = current_metadata.get("usage_count", 0) + 1
        current_metadata["last_used"] = datetime.now().isoformat()
        current_metadata["modified_time"] = datetime.now().isoformat()
        
        # Update in Qdrant
        point = PointStruct(
            id=file_id,
            vector=[],  # Empty vector
            payload=current_metadata
        )
        
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[point],
            wait=True
        )
        
        print(f"✅ Updated usage for template: {file_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating usage for {file_id}: {e}")
        return False

async def list_all_templates_metadata(
    category: str = "",
    document_type: str = "",
    complexity: str = ""
) -> List[Dict[str, Any]]:
    """
    List all templates with their metadata from Qdrant with filtering
    """
    try:
        qdrant_client = get_qdrant_client()
        collection_name = f"customer_{CUSTOMER_ID}_templates"
        
        # Scroll through all points in the collection
        points, next_page = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1000,  # Adjust based on your needs
            with_payload=True,
            with_vectors=False  # We don't need vectors
        )
        
        all_metadata = []
        
        for point in points:
            metadata = point.payload
            
            # Apply filters
            if category and metadata.get("target_path") != category:
                continue
                
            if document_type and metadata.get("analysis", {}).get("document_type") != document_type:
                continue
                
            if complexity and metadata.get("analysis", {}).get("estimated_complexity") != complexity:
                continue
            
            all_metadata.append(metadata)
        
        # Sort by creation time (newest first)
        all_metadata.sort(key=lambda x: x.get("created_time", ""), reverse=True)
        
        return all_metadata
        
    except Exception as e:
        print(f"❌ Error listing templates metadata: {e}")
        return []

async def search_templates(
    query: str = "",
    tags: List[str] = None,
    document_type: str = ""
) -> List[Dict[str, Any]]:
    """
    Search templates by various criteria
    """
    try:
        qdrant_client = get_qdrant_client()
        collection_name = f"customer_{CUSTOMER_ID}_templates"
        
        # Get all templates first (since we're not using vectors for search)
        all_templates = await list_all_templates_metadata()
        
        filtered_templates = []
        
        for template in all_templates:
            # Text search in filename and analysis
            if query:
                query_lower = query.lower()
                filename_match = query_lower in template.get("original_filename", "").lower()
                doc_type_match = query_lower in template.get("analysis", {}).get("document_type", "").lower()
                topics_match = any(query_lower in topic.lower() for topic in template.get("analysis", {}).get("key_topics", []))
                
                if not (filename_match or doc_type_match or topics_match):
                    continue
            
            # Tag filtering
            if tags:
                template_tags = template.get("tags", [])
                if not all(tag in template_tags for tag in tags):
                    continue
            
            # Document type filtering
            if document_type and template.get("analysis", {}).get("document_type") != document_type:
                continue
            
            filtered_templates.append(template)
        
        return filtered_templates
        
    except Exception as e:
        print(f"❌ Error searching templates: {e}")
        return []

async def analyze_template_content(file_path: str, original_filename: str) -> dict:
    """
    Analyze template file to understand its structure and content needs
    """
    try:
        # Extract text from Word document
        text_content = extract_text_from_file(file_path)  # Reuse your existing function
        
        # Analyze template structure
        analysis = {
            "textContent": text_content,
            "totalLength": len(text_content),
            "sections": identify_sections(text_content),
            "completionPoints": find_completion_points(text_content),
            "documentType": classify_document_type(text_content, original_filename),
            "estimatedPlaceholders": count_placeholders(text_content),
            "keyTopics": extract_key_topics(text_content)
        }
        
        print(f"📊 Template Analysis: {original_filename}")
        print(f"   - Length: {analysis['totalLength']} chars")
        print(f"   - Sections: {len(analysis['sections'])}")
        print(f"   - Completion points: {len(analysis['completionPoints'])}")
        print(f"   - Document type: {analysis['documentType']}")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Template analysis failed for {file_path}: {e}")
        return {"error": str(e)}

def normalize_template_path(path: str) -> str:
    """
    Normalize and sanitize template target path
    
    Args:
        path: Raw path input from user
    
    Returns:
        Sanitized, safe path string
    """
    if not path or path.strip() == "":
        return ""
    
    # Clean the path
    path = path.strip()
    
    # Remove any leading/trailing slashes and normalize
    path = path.strip('/').strip('\\')
    
    # Replace any suspicious characters
    path = path.replace('..', '')  # Prevent directory traversal
    path = path.replace('~', '')   # Remove home directory references
    
    # Split and rejoin to normalize
    path_parts = [part for part in path.split('/') if part and part.strip()]
    
    return '/'.join(path_parts)

async def get_template_categories() -> List[str]:
    """
    Get list of existing template categories from Qdrant
    
    Returns:
        List of category names
    """
    try:
        qdrant_client = get_qdrant_client()
        collection_name = f"customer_{CUSTOMER_ID}_templates"
        
        # Scroll through all points to get unique categories
        points, next_page = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        
        categories = set()
        for point in points:
            payload = point.payload
            category = payload.get("category", "")
            if category:  # Only add non-empty categories
                categories.add(category)
        
        # Also check filesystem for backward compatibility
        if os.path.exists(TEMPLATES_BASE_DIR):
            for item in os.listdir(TEMPLATES_BASE_DIR):
                item_path = os.path.join(TEMPLATES_BASE_DIR, item)
                if os.path.isdir(item_path):
                    categories.add(item)
        
        return sorted(list(categories))
        
    except Exception as e:
        print(f"❌ Error getting categories from Qdrant: {e}")
        # Fallback to filesystem
        if not os.path.exists(TEMPLATES_BASE_DIR):
            return []
        
        categories = []
        for item in os.listdir(TEMPLATES_BASE_DIR):
            item_path = os.path.join(TEMPLATES_BASE_DIR, item)
            if os.path.isdir(item_path):
                categories.append(item)
        
        return sorted(categories)

async def list_templates(category: str = "", target_path: str = "") -> List[dict]:
    """
    List all templates from Qdrant with optional filtering
    
    Args:
        category: Optional category to filter by
        target_path: Optional sub-path to filter by
    
    Returns:
        List of template file information
    """
    try:
        qdrant_client = get_qdrant_client()
        collection_name = f"customer_{CUSTOMER_ID}_templates"
        
        # Scroll through all points
        points, next_page = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        
        templates = []
        for point in points:
            payload = point.payload
            
            # Apply filters
            if category and payload.get("category") != category:
                continue
                
            if target_path and payload.get("target_path") != target_path:
                continue
            
            # Convert payload to template info
            template_info = {
                "fileId": payload.get("file_id"),
                "filename": payload.get("original_filename"),
                "savedName": payload.get("safe_filename"),
                "path": payload.get("target_path"),
                "category": payload.get("category"),
                "fullPath": payload.get("relative_path"),
                "size": payload.get("file_size"),
                "fileType": payload.get("content_type"),
                "created": payload.get("created_time"),
                "modified": payload.get("modified_time"),
                "analysis": payload.get("analysis", {})
            }
            
            templates.append(template_info)
        
        # Sort by filename
        templates.sort(key=lambda x: x["filename"])
        return templates
        
    except Exception as e:
        print(f"❌ Error listing templates from Qdrant: {e}")
        # Fallback to filesystem for backward compatibility
        return await list_templates_filesystem(category, target_path)

async def list_templates_filesystem(category: str = "", target_path: str = "") -> List[dict]:
    """
    Fallback function to list templates from filesystem
    """
    try:
        # Build target directory path
        if category and target_path:
            target_dir = os.path.join(TEMPLATES_BASE_DIR, category, target_path)
        elif category:
            target_dir = os.path.join(TEMPLATES_BASE_DIR, category)
        elif target_path:
            target_dir = os.path.join(TEMPLATES_BASE_DIR, target_path)
        else:
            target_dir = TEMPLATES_BASE_DIR
        
        if not os.path.exists(target_dir):
            return []
        
        templates = []
        for filename in os.listdir(target_dir):
            file_path = os.path.join(target_dir, filename)
            if os.path.isfile(file_path):
                # Extract original filename from UUID-prefixed name
                original_name = '_'.join(filename.split('_')[1:]) if '_' in filename else filename
                
                # Determine category and path from file structure
                rel_path = os.path.relpath(file_path, TEMPLATES_BASE_DIR)
                path_parts = rel_path.split(os.sep)
                
                if len(path_parts) > 1:
                    file_category = path_parts[0]
                    file_path_dir = os.sep.join(path_parts[1:-1])
                else:
                    file_category = ""
                    file_path_dir = ""
                
                templates.append({
                    "filename": original_name,
                    "savedName": filename,
                    "path": file_path_dir,
                    "category": file_category,
                    "fullPath": rel_path,
                    "size": os.path.getsize(file_path),
                    "modified": os.path.getmtime(file_path)
                })
        
        return sorted(templates, key=lambda x: x["filename"])
        
    except Exception as e:
        print(f"❌ Error listing templates from filesystem: {e}")
        return []

### Help functions...
def generate_template_tags(analysis: Dict[str, Any], filename: str) -> List[str]:
    """
    Generate search tags based on template analysis for easy filtering
    """
    tags = []
    
    # Document type tags
    doc_type = analysis.get("documentType", "").lower()
    if doc_type:
        tags.append(f"type:{doc_type}")
        # Add common synonyms
        if doc_type == "proposal":
            tags.extend(["category:business", "purpose:sales"])
        elif doc_type == "contract":
            tags.extend(["category:legal", "purpose:agreement"])
        elif doc_type == "report":
            tags.extend(["category:analysis", "purpose:summary"])
        elif doc_type == "manual":
            tags.extend(["category:instruction", "purpose:guide"])
    
    # Complexity tags
    complexity = analysis.get("estimatedComplexity", "").lower()
    if complexity:
        tags.append(f"complexity:{complexity}")
    
    # Key topics as tags (limit to top 5)
    key_topics = analysis.get("keyTopics", [])
    for topic in key_topics[:5]:
        clean_topic = topic.lower().replace(' ', '_')
        tags.append(f"topic:{clean_topic}")
    
    # Placeholder count categories
    placeholder_analysis = analysis.get("estimatedPlaceholders", {})
    total_placeholders = placeholder_analysis.get("total", 0)
    
    if total_placeholders == 0:
        tags.extend(["placeholders:none", "completion:ready"])
    elif total_placeholders < 3:
        tags.extend(["placeholders:few", "completion:easy"])
    elif total_placeholders < 8:
        tags.extend(["placeholders:some", "completion:medium"])
    elif total_placeholders < 15:
        tags.extend(["placeholders:many", "completion:hard"])
    else:
        tags.extend(["placeholders:extensive", "completion:complex"])
    
    # Specific placeholder types
    for placeholder_type, count in placeholder_analysis.items():
        if count > 0 and placeholder_type != "total":
            tags.append(f"placeholder_type:{placeholder_type}")
    
    # File format tags
    file_ext = os.path.splitext(filename)[1].lower().replace('.', '')
    if file_ext:
        tags.append(f"format:{file_ext}")
        if file_ext in ['docx', 'doc']:
            tags.append("format:word")
        elif file_ext in ['txt', 'md']:
            tags.append("format:text")
    
    # Content length categories
    word_count = analysis.get("wordCount", 0)
    if word_count == 0:
        tags.append("length:empty")
    elif word_count < 100:
        tags.append("length:short")
    elif word_count < 500:
        tags.append("length:medium")
    elif word_count < 1000:
        tags.append("length:long")
    else:
        tags.append("length:extensive")
    
    # Section count categories
    section_count = len(analysis.get("sections", []))
    if section_count == 0:
        tags.append("sections:none")
    elif section_count < 3:
        tags.append("sections:few")
    elif section_count < 6:
        tags.append("sections:some")
    else:
        tags.append("sections:many")
    
    # Completion urgency
    completion_points = len(analysis.get("completionPoints", []))
    if completion_points > 10:
        tags.append("status:needs_work")
    elif completion_points > 0:
        tags.append("status:needs_completion")
    else:
        tags.append("status:complete")
    
    # Quality indicators
    if "error" in analysis:
        tags.append("quality:error")
    elif analysis.get("has_analysis_errors", False):
        tags.append("quality:warning")
    else:
        tags.append("quality:good")
    
    # Remove any potential duplicates and return
    return list(set(tags))



# Optional: Function to get all unique tags across templates
async def get_all_template_tags(category: str = "") -> Dict[str, List[str]]:
    """
    Get all unique tags organized by tag type
    """
    all_templates = await list_all_templates_metadata(category)
    
    tag_categories = {
        "type": set(),
        "complexity": set(),
        "topic": set(),
        "placeholders": set(),
        "completion": set(),
        "format": set(),
        "length": set(),
        "sections": set(),
        "status": set(),
        "quality": set(),
        "placeholder_type": set(),
        "category": set(),
        "purpose": set()
    }
    
    for template in all_templates:
        tags = template.get("tags", [])
        for tag in tags:
            if ':' in tag:
                category, value = tag.split(':', 1)
                if category in tag_categories:
                    tag_categories[category].add(value)
    
    # Convert sets to sorted lists
    return {category: sorted(values) for category, values in tag_categories.items()}