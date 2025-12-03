import os
from typing import List, Dict, Any, Optional

from fastapi import HTTPException
from qdrant_client import QdrantClient
from config import FILES_DIR
from db.qdrant_service import get_qdrant_client
import logging


logger = logging.getLogger(__name__)

def normalize_target_path(target_path: str) -> str:
    """
    Normalize and validate target path
    
    Args:
        target_path: User-provided target path
        
    Returns:
        Normalized, safe relative path
    """
    if not target_path or target_path.strip() == "":
        return ""
    
    # Normalize path (remove leading/trailing slashes, normalize separators)
    target_path = target_path.strip().replace('\\', '/')
    target_path = target_path.strip('/')
    
    # Security: Prevent path traversal attacks
    if '..' in target_path or target_path.startswith('/'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid path: Path traversal not allowed"
        )
    
    # Clean up multiple slashes and normalize
    target_path = '/'.join(part for part in target_path.split('/') if part and part != '.')
    
    return target_path


async def find_file_path_by_id(
    collection_name: str, 
    file_id: str
) -> str:
    """
    Find file path by file_id in Qdrant collection
    
    Args:
        collection_name: Name of the Qdrant collection
        file_id: ID of the file to find
        
    Returns:
        Full file path if found
        
    Raises:
        HTTPException: If file not found in collection
    """
    try:
        client = get_qdrant_client()
        
        # Search for points with the given file_id
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
        
        # Try different possible field names for file path
        file_path = payload.get("full_file_path") or payload.get("full_path") or payload.get("file_path")
        
        if file_path and os.path.exists(file_path):
            return file_path
        else:
            raise HTTPException(
                status_code=404,
                detail=f"File Path not found"
            )
        
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error finding file: {str(e)}"
        )

def get_file_extension(filename: str) -> str:
     """Extract file extension from filename"""
     return os.path.splitext(filename)[1].lower().replace('.', '') if '.' in filename else ""


# def flatten_tree_structure_simple(tree_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # """Simple working version of flatten_tree_structure"""
    # flat_list = []
    
    # def add_all_nodes(node):
    #     # Add current node
    #     node_copy = {**node}
    #     children = node_copy.pop("children", [])
    #     node_copy["hasChildren"] = len(children) > 0
    #     flat_list.append(node_copy)
        
    #     # Add all children recursively
    #     for child in children:
    #         add_all_nodes(child)
    
    # for node in tree_nodes:
    #     add_all_nodes(node)
    
    # print(f"✅ Final node count: {len(flat_list)}")
    # for node in flat_list:
    #     if node["fileType"] == "folder":
    #         print(f"   Folder: {node['name']} (ID: {node['id']})")
    
    # return flat_list


async def build_file_structure_tree(
    collection_name: str,
    project_name: Optional[str] = None,
    category_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Build the file structure as a tree for TreeList components using ONLY Qdrant as source
    Can filter by project, category, or both
    """
    
    # Get files from Qdrant (primary and only source for structure)
    qdrant_files = await get_files_from_qdrant(collection_name)
    
    print(f"Total files in Qdrant collection '{collection_name}': {len(qdrant_files)}")
    
    # Filter files based on criteria
    filtered_files = []
    for file in qdrant_files:
        matches_criteria = True
        
        # Filter by project if specified
        if project_name is not None:
            file_project = file.get("project", "")
            if file_project != project_name:
                matches_criteria = False
        
        # Filter by category if specified
        if category_name is not None:
            print(f"Checking file '{file.get('filename')}' for category '{category_name}'")
            file_category = file.get("category", "")
            print(f"  File category: '{file_category}'")
            if file_category != category_name:
                matches_criteria = False
        
        if matches_criteria:
            filtered_files.append(file)
    
    # Build filter description for logging
    filter_desc = []
    if project_name is not None:
        filter_desc.append(f"project='{project_name}'")
    if category_name is not None:
        filter_desc.append(f"category='{category_name}'")
    
    filter_str = " and ".join(filter_desc) if filter_desc else "no filter"
    print(f"Files with {filter_str}: {len(filtered_files)}")
    
    # Debug: Show filtered files
    for file in filtered_files:
        print(f"  - {file['filename']}: upload_path='{file.get('upload_path')}'")
    
    # Build tree structure using ONLY Qdrant data
    result = build_file_tree_from_qdrant(filtered_files)
    
    return result



async def get_files_from_qdrant(collection_name: str) -> List[Dict[str, Any]]:
    """Get file information from Qdrant collection with all metadata"""
    try:
        client = get_qdrant_client()
        
        # Get all points in the collection to extract file information
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        
        files = {}
        for point in scroll_result[0]:
            payload = point.payload
            if payload and "file_id" in payload:
                file_id = payload["file_id"]
                if file_id not in files:
                    # First time seeing this file - create entry
                    files[file_id] = {
                        "file_id": file_id,
                        "filename": payload.get("filename") or payload.get("original_filename", "unknown"),
                        "chunks": 0,
                        "upload_path": payload.get("upload_path", ""),
                        "full_file_path": payload.get("full_file_path", ""),
                        "upload_date": payload.get("upload_time"),
                        "project": payload.get("project", ""),
                        # Extract category - check multiple possible field names
                        "category": payload.get("category", ""),
                        "auto_generated": payload.get("auto_generated", False),
                        "source_template_id": payload.get("source_template_id"),
                    }
                
                # Increment chunk count for this file
                files[file_id]["chunks"] += 1
        
        result = list(files.values())
        print(f"Retrieved {len(result)} unique files from Qdrant collection '{collection_name}'")
        
        # Debug: Print payload structure for first file
        if result and len(scroll_result[0]) > 0:
            first_point = scroll_result[0][0]
            print(f"DEBUG: First point payload keys: {list(first_point.payload.keys())}")
            print(f"DEBUG: First point payload: {first_point.payload}")
        
        # Debug: Show categories for all files
        print("DEBUG: File categories found:")
        for file in result:
            print(f"  - File: {file['filename']}")
            print(f"    Category: '{file.get('category')}'")
            print(f"    All data: {file}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting files from Qdrant: {e}")
        return []


def build_file_tree_from_qdrant(files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a tree structure directly from Qdrant data using upload_path"""
    
    root_node = {
        "id": "root",
        "name": "Files",
        "fileType": "folder",
        "parentId": None,
        "relativePath": "",
        "children": [],
        "isExpanded": True
    }
    
    folders = {"": root_node}
    
    for file_info in files:
        # Use upload_path from Qdrant (this is the relative path within the project)
        upload_path = file_info.get("upload_path", "")
        filename = file_info.get("filename", "")
        file_id = file_info.get("file_id", "")
        
        print(f"Processing file from Qdrant: {filename}, path: '{upload_path}'")
        
        # Handle empty or None upload_path
        if upload_path is None:
            upload_path = ""
        
        # Ensure all parent folders exist in the tree
        current_path = ""
        # Split by both forward and backward slashes to handle different OS paths
        path_parts = []
        if upload_path:
            # Normalize path separators and split
            normalized_path = upload_path.replace('\\', '/')
            path_parts = [part for part in normalized_path.split('/') if part.strip()]
        
        print(f"  Path parts: {path_parts}")
        
        for i, part in enumerate(path_parts):
            parent_path = '/'.join(path_parts[:i]) if i > 0 else ""
            current_path = '/'.join(path_parts[:i+1])
            
            if current_path not in folders:
                folder_id = f"folder_{current_path}"
                parent_id = f"folder_{parent_path}" if parent_path else "root"
                
                folders[current_path] = {
                    "id": folder_id,
                    "name": part,
                    "fileType": "folder",
                    "parentId": parent_id,
                    "relativePath": current_path,
                    "children": [],
                    "isExpanded": False,
                    "data_source": "qdrant"
                }
                print(f"  Created folder: '{current_path}' with ID: {folder_id}, Parent: {parent_id}")
                
                # Add the folder to its parent
                if parent_path in folders:
                    if "children" not in folders[parent_path]:
                        folders[parent_path]["children"] = []
                    folders[parent_path]["children"].append(folders[current_path])
                    print(f"  Added folder '{current_path}' to parent '{parent_path}'")
                elif parent_path == "":  # Root level folder
                    root_node["children"].append(folders[current_path])
                    print(f"  Added folder '{current_path}' to root")
        
        # Create file node
        parent_id = f"folder_{upload_path}" if upload_path else "root"
        
        # Get file extension for display
        file_extension = get_file_extension(filename)
        
        file_node = {
            "id": file_id,
            "name": filename,
            "fileType": file_extension,
            "parentId": parent_id,
            "fileSize": 0,  # Not stored in Qdrant
            "chunks": file_info.get("chunks", 0),
            "uploadDate": file_info.get("upload_date"),
            "filePath": file_info.get("full_file_path", ""),
            "relativePath": upload_path,
            "isSelected": False,
            "hasChildren": False,
            "is_missing": file_info.get("is_missing", False),
            "metadata": {
                "originalFilename": filename,
                "fileId": file_id,
                "chunkCount": file_info.get("chunks", 0),
                "storagePath": upload_path,
                "project": file_info.get("project", ""),
                "autoGenerated": file_info.get("auto_generated", False),
                "sourceTemplateId": file_info.get("source_template_id"),
                "data_source": "qdrant"
            }
        }
        
        # Add file to its parent folder
        if upload_path in folders:
            if "children" not in folders[upload_path]:
                folders[upload_path]["children"] = []
            folders[upload_path]["children"].append(file_node)
            print(f"  Added file '{filename}' to folder '{upload_path}'")
        else:
            # Add to root if no folder path
            root_node["children"].append(file_node)
            print(f"  Added file '{filename}' to root (path: '{upload_path}')")
    
    # Flatten the structure
    result = flatten_tree_structure_simple([root_node])
    
    # Statistics
    folder_count = len([node for node in result if node["fileType"] == "folder"])
    file_count = len([node for node in result if node["fileType"] != "folder"])
    missing_count = len([node for node in result if node.get("is_missing", False)])
    
    print(f"🎉 Tree built: {len(result)} total nodes ({folder_count} folders, {file_count} files)")
    if missing_count > 0:
        print(f"⚠️  {missing_count} files marked as missing from filesystem")
    
    return result


def flatten_tree_structure_simple(tree_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simple working version of flatten_tree_structure"""
    flat_list = []
    
    def add_all_nodes(node):
        # Add current node
        node_copy = {**node}
        children = node_copy.pop("children", [])
        node_copy["hasChildren"] = len(children) > 0
        flat_list.append(node_copy)
        
        # Add all children recursively
        for child in children:
            add_all_nodes(child)
    
    for node in tree_nodes:
        add_all_nodes(node)
    
    return flat_list



