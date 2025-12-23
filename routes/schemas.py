from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    query: str
    fileIds: Optional[List[str]] = None
    project_name: Optional[str] = None
    top_k: int = 5
    model: str
    conversation_id: Optional[str] = None  

class UploadRequest(BaseModel):
    target_path: Optional[str] = ""


# Pydantic model for title update
class UpdateTitleRequest(BaseModel):
    new_title: str

class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    last_message: str

class ConversationsListResponse(BaseModel):
    conversations: List[ConversationResponse]
    count: int