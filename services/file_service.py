import os
from typing import List, Dict, Any

from fastapi import HTTPException
from qdrant_client import QdrantClient
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



async def build_file_structure_tree(FILES_DIR: str, collection_name: str) -> List[Dict[str, Any]]:
    """Build the file structure as a tree for TreeList components with folder support"""
    
    # Get all files from the uploaded_files directory (recursively)
    uploaded_files = []
    if os.path.exists(FILES_DIR):
        for root, dirs, files in os.walk(FILES_DIR):
            for filename in files:
                if not filename.startswith('.'):  # Skip hidden files
                    file_path = os.path.join(root, filename)
                    if os.path.isfile(file_path):
                        # Extract file_id from filename (format: {file_id}_{original_filename})
                        parts = filename.split('_', 1)
                        if len(parts) == 2:
                            file_id, original_filename = parts
                            # Calculate relative path from FILES_DIR
                            relative_path = os.path.relpath(root, FILES_DIR)
                            if relative_path == '.':
                                relative_path = ""
                            
                            uploaded_files.append({
                                "file_id": file_id,
                                "filename": original_filename,
                                "uploaded_name": filename,
                                "file_path": file_path,
                                "relative_path": relative_path
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
                "chunks": 0,
                "file_path": ufile["file_path"],
                "relative_path": ufile["relative_path"]
            }
        else:
            # Update with path information
            file_info_map[ufile["file_id"]]["file_path"] = ufile["file_path"]
            file_info_map[ufile["file_id"]]["relative_path"] = ufile["relative_path"]
    
    # Build tree structure with folder support
    return build_file_tree_with_folders(list(file_info_map.values()))

def build_file_tree_with_folders(files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a tree structure that preserves folder hierarchy"""
    
    root_node = {
        "id": "root",
        "name": "Files",
        "fileType": "folder",
        "parentId": None,
        "relativePath": "",
        "children": [],
        "isExpanded": True
    }
    
    # Dictionary to track all folders
    folders = {"": root_node}
    
    for file_info in files:
        relative_path = file_info.get("relative_path", "")
        filename = file_info.get("filename", "")
        
        # Ensure all parent folders exist in the tree
        current_path = ""
        path_parts = relative_path.split('/') if relative_path else []
        
        for i, part in enumerate(path_parts):
            parent_path = '/'.join(path_parts[:i]) if i > 0 else ""
            current_path = '/'.join(path_parts[:i+1])
            
            if current_path not in folders:
                folder_id = f"folder_{current_path}" if current_path else "root"
                parent_id = f"folder_{parent_path}" if parent_path else "root"
                
                folders[current_path] = {
                    "id": folder_id,
                    "name": part,
                    "fileType": "folder",
                    "parentId": parent_id,
                    "relativePath": current_path,
                    "children": [],
                    "isExpanded": False
                }
        
        # Create file node
        parent_folder_path = relative_path
        parent_id = f"folder_{parent_folder_path}" if parent_folder_path else "root"
        
        file_node = create_file_node_with_path(file_info, parent_id)
        
        # Add file to its parent folder
        if parent_folder_path in folders:
            folders[parent_folder_path].setdefault("children", []).append(file_node)
        else:
            # If folder doesn't exist in our structure, add to root
            root_node.setdefault("children", []).append(file_node)
    
    # Convert folders dictionary to flat list
    all_nodes = []
    
    def add_node_and_children(node):
        # Add current node
        node_copy = node.copy()
        if "children" in node_copy:
            node_copy["hasChildren"] = len(node_copy["children"]) > 0
            # Remove children for flat structure
            children = node_copy.pop("children", [])
        else:
            node_copy["hasChildren"] = False
            children = []
        
        all_nodes.append(node_copy)
        
        # Add children recursively
        for child in children:
            add_node_and_children(child)
    
    add_node_and_children(root_node)
    return all_nodes

def create_file_node_with_path(file_info: Dict[str, Any], parent_id: str) -> Dict[str, Any]:
    """Create a file node with path information"""
    filename = file_info.get("filename", "")
    file_extension = get_file_extension(filename)
    relative_path = file_info.get("relative_path", "")
    
    return {
        "id": file_info["file_id"],
        "name": filename,
        "fileType": file_extension,
        "parentId": parent_id,
        "fileSize": file_info.get("file_size", 0),
        "chunks": file_info.get("chunks", 0),
        "uploadDate": file_info.get("upload_date"),
        "filePath": file_info.get("file_path"),
        "relativePath": relative_path,
        "isSelected": False,
        "hasChildren": False,
        "metadata": {
            "originalFilename": filename,
            "fileId": file_info["file_id"],
            "chunkCount": file_info.get("chunks", 0),
            "storagePath": relative_path
        }
    }

def build_file_tree(files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a tree structure from files for TreeList component"""
    
    # Root node for the entire file structure
    root_node = {
        "id": "root",
        "name": "Files",
        "fileType": "folder",
        "parentId": None,
        "children": [],
        "isExpanded": True
    }
    
    # Organize files by their file type (folder structure)
    file_type_nodes = {}
    
    for file_info in files:
        filename = file_info.get("filename", "")
        file_extension = get_file_extension(filename).lower()
        file_type = get_file_type_category(file_extension)
        
        # Create file type folder if it doesn't exist
        if file_type not in file_type_nodes:
            file_type_nodes[file_type] = {
                "id": f"folder_{file_type}",
                "name": file_type,
                "fileType": "folder",
                "parentId": "root",
                "children": [],
                "isExpanded": True
            }
        
        # Create file node
        file_node = {
            "id": file_info["file_id"],
            "name": filename,
            "fileType": file_extension,
            "parentId": file_type_nodes[file_type]["id"],
            "fileSize": file_info.get("file_size", 0),
            "chunks": file_info.get("chunks", 0),
            "uploadDate": file_info.get("upload_date"),
            "filePath": file_info.get("file_path"),
            "isSelected": False,
            "hasChildren": False
        }
        
        file_type_nodes[file_type]["children"].append(file_node)
    
    # Add all file type folders to root
    root_node["children"] = list(file_type_nodes.values())
    
    # Return as flat array (required by most TreeList components)
    return flatten_tree_structure([root_node])

def flatten_tree_structure(tree_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert tree structure to flat array with parentId references"""
    flat_list = []
    
    def traverse_nodes(nodes: List[Dict[str, Any]], parent_id: str = None):
        for node in nodes:
            # Add current node
            flat_node = {**node}
            if "children" in flat_node:
                flat_node["hasChildren"] = len(flat_node["children"]) > 0
                # Don't include children in flat structure
                flat_node.pop("children", None)
            
            flat_list.append(flat_node)
            
            # Recursively traverse children
            if "children" in node and node["children"]:
                traverse_nodes(node["children"], node["id"])
    
    traverse_nodes(tree_nodes)
    return flat_list

def get_file_type_category(file_extension: str) -> str:
    """Categorize files by type"""
    file_type_categories = {
        # Documents
        "pdf": "Documents",
        "doc": "Documents",
        "docx": "Documents",
        "txt": "Documents",
        "rtf": "Documents",
        
        # Spreadsheets
        "xls": "Spreadsheets",
        "xlsx": "Spreadsheets",
        "csv": "Spreadsheets",
        
        # Presentations
        "ppt": "Presentations",
        "pptx": "Presentations",
        
        # Images
        "jpg": "Images",
        "jpeg": "Images",
        "png": "Images",
        "gif": "Images",
        "bmp": "Images",
        "svg": "Images",
        
        # Archives
        "zip": "Archives",
        "rar": "Archives",
        "7z": "Archives",
        "tar": "Archives",
        
        # Code
        "py": "Code Files",
        "js": "Code Files",
        "html": "Code Files",
        "css": "Code Files",
        "json": "Code Files",
        "xml": "Code Files"
    }
    
    return file_type_categories.get(file_extension, "Other Files")

def get_file_extension(filename: str) -> str:
    """Extract file extension from filename"""
    return os.path.splitext(filename)[1].lower().replace('.', '') if '.' in filename else ""




async def get_files_from_qdrant(collection_name: str) -> List[Dict[str, Any]]:
    """Get file information from Qdrant collection"""
    try:
        client = get_qdrant_client()
        
        # Get all points in the collection to extract file information
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=1000,
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
        logger.error(f"Error getting files from Qdrant: {e}")
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