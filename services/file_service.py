import os
from typing import List, Dict, Any, Optional

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



# async def build_file_structure_tree(project_name: str, FILES_DIR: str, collection_name: str) -> List[Dict[str, Any]]:
#     """Build the file structure as a tree for TreeList components with folder support"""
    
#     # Get all files from the uploaded_files directory (recursively)
#     uploaded_files = []
#     full_target_dir = os.path.join(FILES_DIR, project_name)

#     if os.path.exists(full_target_dir):
#         for root, dirs, files in os.walk(full_target_dir):
           
#             for filename in files:
#                 if not filename.startswith('.'):  # Skip hidden files
#                     file_path = os.path.join(root, filename)
#                     if os.path.isfile(file_path):
#                         # Extract file_id from filename (format: {file_id}_{original_filename})
#                         parts = filename.split('_', 1)
#                         if len(parts) == 2:
#                             file_id, original_filename = parts
#                             # Calculate relative path from FILES_DIR
#                             relative_path = os.path.relpath(root, FILES_DIR)
#                             if relative_path == '.':
#                                 relative_path = ""
                            
#                             uploaded_files.append({
#                                 "file_id": file_id,
#                                 "filename": original_filename,
#                                 "uploaded_name": filename,
#                                 "file_path": file_path,
#                                 "relative_path": relative_path,
#                                 "project": project_name
#                             })
                            
#         print(f"Total files found in filesystem: {len(uploaded_files)}")
#     else:
#         print(f"ERROR: Directory {FILES_DIR} does not exist!")
    
#     # Get file information from Qdrant
    
#     qdrant_files = await get_files_from_qdrant(collection_name)
    
    
#     # Debug: Print Qdrant files
      
#     # Merge the data
#     file_info_map = {}
    
#     # Add files from Qdrant first
#     for qfile in qdrant_files:
#         file_info_map[qfile["file_id"]] = qfile
       
    
#     # Add/update with files from filesystem
#     for ufile in uploaded_files:
#         if ufile["file_id"] in file_info_map:
        
#             file_info_map[ufile["file_id"]].update({
#                 "file_path": ufile["file_path"],
#                 "uploaded_name": ufile["uploaded_name"]
#             })
#             # Only use filesystem path if Qdrant doesn't have one
#             if not file_info_map[ufile["file_id"]].get("relative_path"):
#                 file_info_map[ufile["file_id"]]["relative_path"] = ufile["relative_path"]
#         else:
#             file_info_map[ufile["file_id"]] = {
#                 "file_id": ufile["file_id"],
#                 "filename": ufile["filename"],
#                 "chunks": 0,
#                 "file_path": ufile["file_path"],
#                 "relative_path": ufile["relative_path"],
#                 "uploaded_name": ufile["uploaded_name"],
#                 "project": ufile["project"]
#             }
    
   
    
#     # Debug: Print final file list
#     for file_id, file_info in file_info_map.items():
#         print(f"Final file: {file_id} -> {file_info.get('filename')} -> Path: {file_info.get('relative_path')}")
    
#     # Build tree structure
#     result = build_file_tree_with_folders_fixed(list(file_info_map.values()))

    
#     return result

# def build_file_tree_with_folders(files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """Build a tree structure that preserves folder hierarchy"""
    
#     root_node = {
#         "id": "root",
#         "name": "Files",
#         "fileType": "folder",
#         "parentId": None,
#         "relativePath": "",
#         "children": [],
#         "isExpanded": True
#     }
    
#     # Dictionary to track all folders
#     folders = {"": root_node}
    
#     for file_info in files:
#         relative_path = file_info.get("relative_path", "")
#         filename = file_info.get("filename", "")
        
#         print(f"Processing file: {filename}, path: {relative_path}")  # Debug output
        
#         # Ensure all parent folders exist in the tree
#         current_path = ""
#         path_parts = relative_path.split('/') if relative_path else []
        
#         for i, part in enumerate(path_parts):
#             parent_path = '/'.join(path_parts[:i]) if i > 0 else ""
#             current_path = '/'.join(path_parts[:i+1])
            
#             if current_path not in folders:
#                 folder_id = f"folder_{current_path}" if current_path else "root"
#                 parent_id = f"folder_{parent_path}" if parent_path else "root"
                
#                 folders[current_path] = {
#                     "id": folder_id,
#                     "name": part,
#                     "fileType": "folder",
#                     "parentId": parent_id,
#                     "relativePath": current_path,
#                     "children": [],
#                     "isExpanded": False
#                 }
#                 print(f"Created folder: {current_path} with ID: {folder_id}")
        
#         # Create file node - WICHTIG: Verwende den korrekten parent_id
#         parent_folder_path = relative_path
#         parent_id = f"folder_{parent_folder_path}" if parent_folder_path else "root"
        
#         file_node = create_file_node_with_path(file_info, parent_id)
#         print(f"Created file node: {filename} with parent: {parent_id}")
        
#         # Add file to its parent folder
#         if parent_folder_path in folders:
#             if "children" not in folders[parent_folder_path]:
#                 folders[parent_folder_path]["children"] = []
#             folders[parent_folder_path]["children"].append(file_node)
#             print(f"Added file {filename} to folder {parent_folder_path}")
#         else:
#             # If folder doesn't exist in our structure, add to root
#             root_node["children"].append(file_node)
#             print(f"Added file {filename} to root (folder {parent_folder_path} not found)")
    
#     # Debug: Print complete folder structure
#     print("=== COMPLETE FOLDER STRUCTURE ===")
#     for path, folder in folders.items():
#         child_count = len(folder.get("children", []))
#         child_names = [child["name"] for child in folder.get("children", [])]
#         print(f"Folder '{path}' ({folder['id']}): {child_count} children -> {child_names}")
    
#     # Convert folders dictionary to flat list
#     all_nodes = []
    
#     def add_node_and_children(node):
#         # Add current node
#         node_copy = node.copy()
#         children = node_copy.pop("children", [])
#         node_copy["hasChildren"] = len(children) > 0
        
#         all_nodes.append(node_copy)
        
#         # Add children recursively
#         for child in children:
#             add_node_and_children(child)
    
#     add_node_and_children(root_node)
    
#     print(f"=== FINAL RESULT ===")
#     print(f"Total nodes in tree: {len(all_nodes)}")
#     for node in all_nodes:
#         if node["fileType"] == "folder":
#             print(f"Folder: {node['name']} (ID: {node['id']}) -> Parent: {node['parentId']}")
#         else:
#             print(f"File: {node['name']} -> Parent: {node['parentId']}")
    
#     return all_nodes

# def create_file_node_with_path(file_info: Dict[str, Any], parent_id: str) -> Dict[str, Any]:
#     """Create a file node with path information"""
#     filename = file_info.get("filename", "")
#     file_extension = get_file_extension(filename)
#     relative_path = file_info.get("relative_path", "")
    
#     return {
#         "id": file_info["file_id"],
#         "name": filename,
#         "fileType": file_extension,
#         "parentId": parent_id,
#         "fileSize": file_info.get("file_size", 0),
#         "chunks": file_info.get("chunks", 0),
#         "uploadDate": file_info.get("upload_date"),
#         "filePath": file_info.get("file_path"),
#         "relativePath": relative_path,
#         "isSelected": False,
#         "hasChildren": False,
#         "metadata": {
#             "originalFilename": filename,
#             "fileId": file_info["file_id"],
#             "chunkCount": file_info.get("chunks", 0),
#             "storagePath": relative_path
#         }
#     }

# def build_file_tree(files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """Build a tree structure from files for TreeList component"""
    
#     # Root node for the entire file structure
#     root_node = {
#         "id": "root",
#         "name": "Files",
#         "fileType": "folder",
#         "parentId": None,
#         "children": [],
#         "isExpanded": True
#     }
    
#     # Organize files by their file type (folder structure)
#     file_type_nodes = {}
    
#     for file_info in files:
#         filename = file_info.get("filename", "")
#         file_extension = get_file_extension(filename).lower()
#         file_type = get_file_type_category(file_extension)
        
#         # Create file type folder if it doesn't exist
#         if file_type not in file_type_nodes:
#             file_type_nodes[file_type] = {
#                 "id": f"folder_{file_type}",
#                 "name": file_type,
#                 "fileType": "folder",
#                 "parentId": "root",
#                 "children": [],
#                 "isExpanded": True
#             }
        
#         # Create file node
#         file_node = {
#             "id": file_info["file_id"],
#             "name": filename,
#             "fileType": file_extension,
#             "parentId": file_type_nodes[file_type]["id"],
#             "fileSize": file_info.get("file_size", 0),
#             "chunks": file_info.get("chunks", 0),
#             "uploadDate": file_info.get("upload_date"),
#             "filePath": file_info.get("file_path"),
#             "isSelected": False,
#             "hasChildren": False
#         }
        
#         file_type_nodes[file_type]["children"].append(file_node)
    
#     # Add all file type folders to root
#     root_node["children"] = list(file_type_nodes.values())
    
#     # Return as flat array (required by most TreeList components)
#     return flatten_tree_structure_simple([root_node])


# def build_file_tree_with_folders_fixed(files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """Fixed version that properly adds folders to root"""
    
#     root_node = {
#         "id": "root",
#         "name": "Files",
#         "fileType": "folder",
#         "parentId": None,
#         "relativePath": "",
#         "children": [],
#         "isExpanded": True
#     }
    
#     folders = {"": root_node}
    
#     for file_info in files:
#         relative_path = file_info.get("relative_path", "")
#         filename = file_info.get("filename", "")
        
#         print(f"Processing file: {filename}, path: '{relative_path}'")
        
#         # Ensure all parent folders exist in the tree
#         current_path = ""
#         path_parts = relative_path.split('/') if relative_path else []
        
#         for i, part in enumerate(path_parts):
#             parent_path = '/'.join(path_parts[:i]) if i > 0 else ""
#             current_path = '/'.join(path_parts[:i+1])
            
#             if current_path not in folders:
#                 folder_id = f"folder_{current_path}"
#                 parent_id = f"folder_{parent_path}" if parent_path else "root"
                
#                 folders[current_path] = {
#                     "id": folder_id,
#                     "name": part,
#                     "fileType": "folder",
#                     "parentId": parent_id,
#                     "relativePath": current_path,
#                     "children": [],
#                     "isExpanded": False
#                 }
#                 print(f"Created folder: '{current_path}' with ID: {folder_id}, Parent: {parent_id}")
                
#                 # WICHTIG: Füge den neuen Ordner zum Parent hinzu
#                 if parent_path in folders:
#                     if "children" not in folders[parent_path]:
#                         folders[parent_path]["children"] = []
#                     folders[parent_path]["children"].append(folders[current_path])
#                     print(f"Added folder '{current_path}' to parent '{parent_path}'")
#                 elif parent_path == "":  # Wenn Parent root ist
#                     root_node["children"].append(folders[current_path])
#                     print(f"Added folder '{current_path}' to root")
        
#         # Create file node
#         parent_id = f"folder_{relative_path}" if relative_path else "root"
#         file_node = create_file_node_with_path(file_info, parent_id)
#         print(f"Created file node: {filename} with parent: {parent_id}")
        
#         # Add file to its parent folder
#         if relative_path in folders:
#             if "children" not in folders[relative_path]:
#                 folders[relative_path]["children"] = []
#             folders[relative_path]["children"].append(file_node)
#             print(f"Added file {filename} to folder '{relative_path}'")
#         else:
#             # If folder doesn't exist in our structure, add to root
#             root_node["children"].append(file_node)
#             print(f"Added file {filename} to root (folder '{relative_path}' not found)")
    
#     # Debug: Print complete structure before flattening
#     print("=== COMPLETE STRUCTURE BEFORE FLATTENING ===")
#     def print_structure(node, level=0):
#         indent = "  " * level
#         children_count = len(node.get("children", []))
#         print(f"{indent}{node['name']} ({node['id']}) - {children_count} children")
#         for child in node.get("children", []):
#             print_structure(child, level + 1)
    
#     print_structure(root_node)
    
#     # Flatten the structure
#     result = flatten_tree_structure_simple([root_node])
    
#     # Final verification
#     folder_count = len([node for node in result if node["fileType"] == "folder"])
#     file_count = len([node for node in result if node["fileType"] != "folder"])
#     print(f"🎉 FINAL RESULT: {len(result)} total nodes ({folder_count} folders, {file_count} files)")
    
#     # Check if new_folder is in the result
#     new_folder_nodes = [node for node in result if node["name"] == "new_folder"]
#     print(f"📁 new_folder found: {len(new_folder_nodes)} times")
    
#     return result

# def flatten_tree_structure(tree_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """Convert tree structure to flat array with parentId references - FIXED VERSION"""
#     flat_list = []
    
#     def traverse_nodes(nodes: List[Dict[str, Any]]):
#         for node in nodes:
#             # Create a copy of the node without children
#             node_copy = node.copy()
#             children = node_copy.pop("children", [])
#             node_copy["hasChildren"] = len(children) > 0
            
#             # Add the node to flat list
#             flat_list.append(node_copy)
            
#             # Recursively traverse children
#             if children:
#                 traverse_nodes(children)
    
#     traverse_nodes(tree_nodes)
    
#     # Debug: Check if all nodes are included
#     folder_nodes = [node for node in flat_list if node["fileType"] == "folder"]
#     file_nodes = [node for node in flat_list if node["fileType"] != "folder"]
#     print(f"✅ Flattened structure: {len(flat_list)} total nodes ({len(folder_nodes)} folders, {len(file_nodes)} files)")
    
#     return flat_list

# def get_file_type_category(file_extension: str) -> str:
#     """Categorize files by type"""
#     file_type_categories = {
#         # Documents
#         "pdf": "Documents",
#         "doc": "Documents",
#         "docx": "Documents",
#         "txt": "Documents",
#         "rtf": "Documents",
        
#         # Spreadsheets
#         "xls": "Spreadsheets",
#         "xlsx": "Spreadsheets",
#         "csv": "Spreadsheets",
        
#         # Presentations
#         "ppt": "Presentations",
#         "pptx": "Presentations",
        
#         # Images
#         "jpg": "Images",
#         "jpeg": "Images",
#         "png": "Images",
#         "gif": "Images",
#         "bmp": "Images",
#         "svg": "Images",
        
#         # Archives
#         "zip": "Archives",
#         "rar": "Archives",
#         "7z": "Archives",
#         "tar": "Archives",
        
#         # Code
#         "py": "Code Files",
#         "js": "Code Files",
#         "html": "Code Files",
#         "css": "Code Files",
#         "json": "Code Files",
#         "xml": "Code Files"
#     }
    
#     return file_type_categories.get(file_extension, "Other Files")

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



