from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    query: str
    fileIds: Optional[List[str]] = None
    project_name: Optional[str] = None
    top_k: int = 5
    model: str

class UploadRequest(BaseModel):
    target_path: Optional[str] = ""


class ProjectInfo(BaseModel):
    name: str
    path: str
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    file_count: int = 0
    size: int = 0


class ProjectResponse(BaseModel):
    projects: List[ProjectInfo]


class CreateProjectRequest(BaseModel):
    name: str


class CreateProjectResponse(BaseModel):
    success: bool
    project: Optional[ProjectInfo] = None
    message: str = ""


class DeleteProjectResponse(BaseModel):
    success: bool
    message: str = ""
