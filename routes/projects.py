from datetime import datetime
import os
from fastapi import APIRouter, HTTPException
from pathlib import Path
from config import CUSTOMER_ID, FILES_DIR
from services.file_service import build_file_structure_tree
from services.projects_handler import projects_handler
from routes.schemas import (
    ProjectInfo,
    ProjectResponse,
    CreateProjectRequest,
    CreateProjectResponse,
    DeleteProjectResponse,
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("/", response_model=ProjectResponse)
async def get_all_projects():
    try:
        projects_base_path = Path("Projects")
        projects = []
        projects_base_path.mkdir(exist_ok=True)
        for item in projects_base_path.iterdir():
            if item.is_dir():
                project = ProjectInfo(name=item.name, path=str(item.absolute()))
                projects.append(project)
        return ProjectResponse(projects=projects)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get projects: {str(e)}")


@router.post("/", response_model=CreateProjectResponse)
async def create_project(request: CreateProjectRequest):
    try:
        projects_base_path = Path("Projects")
        projects_base_path.mkdir(exist_ok=True)
        project_path = projects_base_path / request.name
        if project_path.exists():
            return CreateProjectResponse(success=False, message="Project already exists")
        project_path.mkdir()
        project = ProjectInfo(name=request.name, path=str(project_path.absolute()))
        return CreateProjectResponse(success=True, project=project, message="Project created successfully")
    except Exception as e:
        return CreateProjectResponse(success=False, message=f"Failed to create project: {str(e)}")


@router.get("/{project_name}")
async def get_project(project_name: str):
    try:
        project_path = Path("Projects") / project_name
        if not project_path.exists() or not project_path.is_dir():
            raise HTTPException(status_code=404, detail="Project not found")
        return ProjectInfo(name=project_name, path=str(project_path.absolute()))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get project: {str(e)}")


@router.delete("/{project_name}", response_model=DeleteProjectResponse)
async def delete_project(project_name: str):
    try:
        success = projects_handler.delete_project(project_name)
        if success:
            return DeleteProjectResponse(success=True, message="Project deleted successfully")
        else:
            return DeleteProjectResponse(success=False, message="Project not found or could not be deleted")
    except Exception as e:
        return DeleteProjectResponse(success=False, message=f"Failed to delete project: {str(e)}")


@router.get("/{project_name}/files")
async def get_project_files(project_name: str):
    """Get all files in a specific project"""
    try:
        
        collection_name = f"customer_{CUSTOMER_ID}_documents"
 
        result = await build_file_structure_tree(project_name=project_name, collection_name=collection_name )
        
        return result
    except Exception as e:
        logger.error(f"Error retrieving files for project '{project_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving project files: {str(e)}")