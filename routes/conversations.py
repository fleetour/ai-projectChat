

import logging
from fastapi import APIRouter
from pydantic import BaseModel

from db.qdrant_service import get_qdrant_client
from routes.schemas import UpdateTitleRequest
from services.conversation_service import AsyncConversationService, ConversationService
from services.local_llma_service import LocalLlamaService


router = APIRouter(tags=["conversations"])
logger = logging.getLogger(__name__)


@router.get("/conversations")
async def get_conversations(user_id: str):
    """Get user's conversations with titles."""
    conv_service = AsyncConversationService(get_qdrant_client())
    conversations = await conv_service.get_conversation_list(user_id)
    
    return {
        "conversations": conversations,
        "count": len(conversations)
    }


@router.put("/conversation/{conversation_id}/title")
async def update_title(conversation_id: str, title: UpdateTitleRequest):
    """Update conversation title."""
    conv_service = AsyncConversationService(get_qdrant_client())
    
    await conv_service.update_conversation_title(
        conversation_id=conversation_id,
        title=title.new_title,
        user_id=title.user_id
    )
    
    return {"status": "updated", "title": title.new_title}


@router.get("/conversation/{conversation_id}")
async def get_conversation_details(conversation_id: str, user_id: str):
    """Get full conversation with title and messages."""
    conv_service = AsyncConversationService(get_qdrant_client())
    conversation = await conv_service.get_conversation(conversation_id, user_id)
    
    return {
        "id": conversation["conversation_id"],
        "title": conversation.get("title", "New Conversation"),
        "created_at": conversation["created_at"],
        "updated_at": conversation["updated_at"],
        "message_count": len(conversation["messages"]),
        "messages": conversation["messages"][-20:]  # Last 20 messages
    }





@router.post("/conversation/summarize/{conversation_id}")
async def summarize_conversation(conversation_id: str):
    """Create a summary of a long conversation."""
    conv_service = ConversationService(get_qdrant_client())
    conversation = await conv_service.get_conversation(conversation_id)
    
    if len(conversation["messages"]) < 10:
        return {"message": "Conversation too short to summarize"}
    
    # Create summary prompt
    messages_text = "\n".join([
        f"{msg['role']}: {msg['content'][:200]}..."
        for msg in conversation["messages"][-15:]  # Last 15 messages
    ])
    
    summary_prompt = f"""Please summarize this conversation in 2-3 sentences:

{messages_text}

Summary:"""
    
    # Get summary from LLM
    local_llama = LocalLlamaService()
    summary = ""
    async for chunk in local_llama.get_llama_stream_completion(summary_prompt, context=""):
        summary += chunk
    
    # Add summary to conversation metadata
    conversation["summary"] = summary.strip()
    
    return {
        "conversation_id": conversation_id,
        "summary": summary,
        "message_count": len(conversation["messages"])
    }