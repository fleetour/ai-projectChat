

import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from db.qdrant_service import get_qdrant_client
from routes.schemas import ConversationsListResponse, UpdateTitleRequest
from services.conversation_service import AsyncConversationService, get_conversation_service
from services.dependencies import get_current_user_id, get_customer_id
from services.local_llma_service import LocalLlamaService


router = APIRouter(tags=["conversations"])
logger = logging.getLogger(__name__)



@router.get("/conversations", response_model=ConversationsListResponse)
async def get_conversations(
    user_id: str = Depends(get_current_user_id),
    customer_id: str = Depends(get_customer_id),
    limit: int = 50
):
    """Get user's conversations with titles."""
    conv_service = await get_conversation_service()  # 👈 Use the async factory
    conversations = await conv_service.get_conversation_list(
        user_id=user_id,
        customer_id=customer_id,
        limit=limit
    )
    
    return {
        "conversations": conversations,
        "count": len(conversations)
    }

@router.put("/conversations/{conversation_id}/title")
async def update_title(
    conversation_id: str, 
    title_request: UpdateTitleRequest,
    user_id: str = Depends(get_current_user_id),
    customer_id: str = Depends(get_customer_id)
):
    """Update conversation title."""
    conv_service = await get_conversation_service()
    
    success = await conv_service.update_conversation_title(
        conversation_id=conversation_id,
        title=title_request.new_title,
        user_id=user_id,
        customer_id=customer_id
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied")
    
    return {"status": "updated", "title": title_request.new_title}

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user_id: str = Depends(get_current_user_id),
    customer_id: str = Depends(get_customer_id)
):
    """Delete a conversation."""
    conv_service = await get_conversation_service()
    
    success = await conv_service.delete_conversation(
        conversation_id=conversation_id,
        user_id=user_id,
        customer_id=customer_id
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied")
    
    return {"status": "deleted", "conversation_id": conversation_id}

@router.delete("/conversations")
async def clear_all_conversations(
    user_id: str = Depends(get_current_user_id),
    customer_id: str = Depends(get_customer_id)
):
    """Clear all conversations for the current user."""
    conv_service = await get_conversation_service()
    
    success = await conv_service.clear_user_conversations(
        user_id=user_id,
        customer_id=customer_id
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to clear conversations")
    
    return {"status": "cleared", "message": "All conversations cleared"}


@router.get("/conversations/{conversation_id}")
async def get_conversation_details(conversation_id: str,
    customer_id: str = Depends(get_customer_id),
    user_id: str = Depends(get_current_user_id)):
    """Get full conversation with title and messages."""
    conv_service = AsyncConversationService(get_qdrant_client())
    conversation = await conv_service.get_conversation(conversation_id, user_id, customer_id)
    
    return {
        "id": conversation["conversation_id"],
        "title": conversation.get("title", "New Conversation"),
        "created_at": conversation["created_at"],
        "updated_at": conversation["updated_at"],
        "message_count": len(conversation["messages"]),
        "messages": conversation["messages"][-20:]  # Last 20 messages
    }





@router.post("/conversations/summarize/{conversation_id}")
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