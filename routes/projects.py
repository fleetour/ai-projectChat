from datetime import datetime
import os
from fastapi import APIRouter, Depends, HTTPException
from pathlib import Path
from services.dependencies import get_customer_id
from services.file_service import build_file_structure_tree
from services.projects_handler import projects_handler
import logging

from services.utils import get_collection_name

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["projects"])



@router.get("/{project_name}/files")
async def get_project_files(project_name: str, customer_id: str = Depends(get_customer_id)):
    """Get all files in a specific project"""
    try:
        
        collection_name = get_collection_name("documents",customer_id)
 
        result = await build_file_structure_tree(project_name=project_name, collection_name=collection_name )
        
        return result
    except Exception as e:
        logger.error(f"Error retrieving files for project '{project_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving project files: {str(e)}")