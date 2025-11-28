import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ProjectInfo(BaseModel):
    name: str
    path: str
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    file_count: int = 0
    size: int = 0

class ProjectsHandler:
    def __init__(self, projects_base_path: str = "Projects"):
        self.projects_base_path = Path(projects_base_path)
        self._ensure_projects_directory()
    
    def _ensure_projects_directory(self) -> None:
        """Ensure the projects directory exists"""
        try:
            self.projects_base_path.mkdir(exist_ok=True)
            logger.info(f"Projects directory ensured at: {self.projects_base_path.absolute()}")
        except Exception as e:
            logger.error(f"Failed to create projects directory: {e}")
            raise
    
    def _get_file_info(self, folder_path: Path) -> Dict[str, any]:
        """Get file count and total size for a folder"""
        try:
            file_count = 0
            total_size = 0
            
            for file_path in folder_path.rglob('*'):
                if file_path.is_file():
                    file_count += 1
                    total_size += file_path.stat().st_size
            
            return {
                "file_count": file_count,
                "size": total_size
            }
        except Exception as e:
            logger.warning(f"Could not read file info for {folder_path}: {e}")
            return {"file_count": 0, "size": 0}
    
    def _get_folder_timestamps(self, folder_path: Path) -> Dict[str, Optional[str]]:
        """Get creation and modification timestamps for a folder"""
        try:
            stat = folder_path.stat()
            return {
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
        except Exception as e:
            logger.warning(f"Could not get timestamps for {folder_path}: {e}")
            return {"created_at": None, "modified_at": None}
    
    def get_all_projects(self) -> List[ProjectInfo]:
        """Get all projects from the Projects directory"""
        try:
            projects = []
            
            if not self.projects_base_path.exists():
                logger.warning(f"Projects directory does not exist: {self.projects_base_path}")
                return []
            
            # List all directories in the Projects folder
            for item in self.projects_base_path.iterdir():
                if item.is_dir():
                    try:
                        # Get basic folder info
                        file_info = self._get_file_info(item)
                        timestamps = self._get_folder_timestamps(item)
                        
                        project = ProjectInfo(
                            name=item.name,
                            path=str(item.absolute()),
                            created_at=timestamps["created_at"],
                            modified_at=timestamps["modified_at"],
                            file_count=file_info["file_count"],
                            size=file_info["size"]
                        )
                        projects.append(project)
                        
                    except Exception as e:
                        logger.error(f"Error processing project {item.name}: {e}")
                        # Still add basic project info even if details fail
                        projects.append(ProjectInfo(
                            name=item.name,
                            path=str(item.absolute())
                        ))
            
            # Sort projects by modification date (newest first)
            projects.sort(key=lambda x: x.modified_at or "", reverse=True)
            
            logger.info(f"Found {len(projects)} projects")
            return projects
            
        except Exception as e:
            logger.error(f"Error getting projects list: {e}")
            return []
    
    def get_project_by_name(self, project_name: str) -> Optional[ProjectInfo]:
        """Get specific project by name"""
        try:
            project_path = self.projects_base_path / project_name
            
            if not project_path.exists() or not project_path.is_dir():
                logger.warning(f"Project not found: {project_name}")
                return None
            
            file_info = self._get_file_info(project_path)
            timestamps = self._get_folder_timestamps(project_path)
            
            return ProjectInfo(
                name=project_path.name,
                path=str(project_path.absolute()),
                created_at=timestamps["created_at"],
                modified_at=timestamps["modified_at"],
                file_count=file_info["file_count"],
                size=file_info["size"]
            )
            
        except Exception as e:
            logger.error(f"Error getting project {project_name}: {e}")
            return None
    
    def create_project(self, project_name: str) -> Optional[ProjectInfo]:
        """Create a new project folder"""
        try:
            # Sanitize project name
            sanitized_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            if not sanitized_name:
                raise ValueError("Invalid project name")
            
            project_path = self.projects_base_path / sanitized_name
            
            if project_path.exists():
                logger.warning(f"Project already exists: {sanitized_name}")
                return None
            
            project_path.mkdir(parents=True, exist_ok=False)
            logger.info(f"Created new project: {sanitized_name}")
            
            return self.get_project_by_name(sanitized_name)
            
        except Exception as e:
            logger.error(f"Error creating project {project_name}: {e}")
            return None
    
    def delete_project(self, project_name: str) -> bool:
        """Delete a project folder"""
        try:
            project_path = self.projects_base_path / project_name
            
            if not project_path.exists():
                logger.warning(f"Project not found for deletion: {project_name}")
                return False
            
            # Use shutil to remove directory tree
            import shutil
            shutil.rmtree(project_path)
            
            logger.info(f"Deleted project: {project_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting project {project_name}: {e}")
            return False

# Global instance
projects_handler = ProjectsHandler()