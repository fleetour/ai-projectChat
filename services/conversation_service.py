# services/conversation_service.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, List, Optional
from datetime import datetime
import uuid
import logging
from qdrant_client import QdrantClient
from qdrant_client import models

from db.qdrant_service import get_qdrant_client, get_qdrant_client_async
from services.embeddings_service import get_embeddings_from_llama
from services.utils import get_collection_name

logger = logging.getLogger(__name__)

class AsyncConversationService:
    """Thread-safe conversation service for multiple concurrent users."""
    
    _conversation_cache: Dict[str, Dict] = {}
    _cache_lock = asyncio.Lock()
    _collection_initialized = False
    _collection_lock = asyncio.Lock()
    
    def __init__(self, qdrant_client: QdrantClient = None):
        self.qdrant_client = qdrant_client
        self.user_locks: Dict[str, asyncio.Lock] = {}
        # CHANGED: Use async-aware executor with proper cleanup
        self._qdrant_executor = ThreadPoolExecutor(
            max_workers=3,
            thread_name_prefix="qdrant_conversation_"
        )
        # CHANGED: Add shutdown flag
        self._shutdown = False
    
    # CHANGED: Added async method for proper cleanup
    async def shutdown(self):
        """Cleanup resources."""
        self._shutdown = True
        if hasattr(self, '_qdrant_executor'):
            self._qdrant_executor.shutdown(wait=True)
    
    async def _ensure_conversation_collection_lazy(self, customer_id: str):
        """Lazy initialization - only check/create collection once."""
        if self._collection_initialized or not self.qdrant_client:
            # CHANGED: Use logger instead of print
            logger.debug("Collection already initialized or no Qdrant client.")
            return
        
        async with self._collection_lock:
            # CHANGED: Double-check pattern
            if not self._collection_initialized:
                await self._ensure_conversation_collection(customer_id)
                self._collection_initialized = True
                logger.info("Conversation collection initialized")
    
    async def get_user_lock(self, user_id: str) -> asyncio.Lock:
        """Get or create user-specific lock."""
        async with self._cache_lock:
            if user_id not in self.user_locks:
                self.user_locks[user_id] = asyncio.Lock()
        return self.user_locks[user_id]
    
    async def get_conversation(self, conversation_id: str, user_id: str, customer_id: str) -> Dict:
        """Get conversation with caching."""
        async with self._cache_lock:
            if conversation_id in self._conversation_cache:
                return self._conversation_cache[conversation_id].copy()
        
        # Load from database
        if self.qdrant_client:
            # CHANGED: Don't ensure collection here - it will be created on save
            conversation = await self._load_from_qdrant(conversation_id, customer_id)
            if conversation:
                async with self._cache_lock:
                    self._conversation_cache[conversation_id] = conversation
                return conversation.copy()
        
        # Create new conversation
        new_conv = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "customer_id": customer_id,
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        async with self._cache_lock:
            self._conversation_cache[conversation_id] = new_conv
        
        return new_conv.copy()
    
    async def _async_qdrant_scroll(self, **kwargs):
        """Run Qdrant scroll operation in thread pool asynchronously."""
        if self._shutdown:
            raise RuntimeError("Service is shutting down")
            
        loop = asyncio.get_event_loop()
        
        # CHANGED: Handle potential exceptions
        try:
            scroll_func = partial(self.qdrant_client.scroll, **kwargs)
            return await loop.run_in_executor(
                self._qdrant_executor,
                scroll_func
            )
        except Exception as e:
            logger.error(f"Error in async Qdrant scroll: {e}")
            raise
    
    async def _load_from_qdrant(self, conversation_id: str, customer_id: str) -> Optional[Dict]:
        """Load conversation from Qdrant asynchronously."""
        if not self.qdrant_client:
            return None
        
        try:
            logger.debug(f"Loading conversation {conversation_id} from Qdrant")
            collection_name = get_collection_name("conversations", customer_id)
            # CHANGED: Use try-except for collection not found
            try:
                search_result = await self._async_qdrant_scroll(
                    collection_name=collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="conversation_id",
                                match=models.MatchValue(value=conversation_id)
                            )
                        ]
                    ),
                    limit=1,
                    with_payload=True
                )
                
                points, _ = search_result
                if points and len(points) > 0:
                    logger.debug(f"Found conversation {conversation_id} in Qdrant")
                    return points[0].payload
                else:
                    logger.debug(f"Conversation {conversation_id} not found in Qdrant")
                    
            except Exception as e:
                # Collection might not exist - that's OK
                logger.debug(f"Could not load conversation (collection might not exist): {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading conversation from Qdrant: {e}")
        
        return None
    
    async def _ensure_conversation_collection(self, customer_id: str):
        """Ensure conversations collection exists in Qdrant asynchronously."""
        if not self.qdrant_client:
            return
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._qdrant_executor,
                self._ensure_conversation_collection_sync,
                customer_id
            )
        except Exception as e:
            logger.error(f"Error ensuring conversation collection: {e}")
            raise
    
    def _ensure_conversation_collection_sync(self, customer_id: str):
        """Create collection if it doesn't exist, otherwise do nothing."""
        collection_name = get_collection_name("conversations", customer_id)
        
        try:
            # Try to get the collection - if successful, it exists
            try:
                self.qdrant_client.get_collection(collection_name)
                logger.debug(f"✅ Collection '{collection_name}' already exists")
                return
            except Exception as e:
                # Collection doesn't exist or other error
                if "not found" in str(e).lower() or "doesn't exist" in str(e).lower():
                    # Collection doesn't exist - create it
                    logger.info(f"Collection '{collection_name}' not found. Creating it...")
                else:
                    # Some other error, but still try to create
                    logger.warning(f"Error checking collection, will try to create: {e}")
            
            # Create collection with correct vector size
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=1024,  # Your embedding dimension
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"✅ Created collection '{collection_name}' with 1024-dim vectors")
            
        except Exception as e:
            logger.error(f"Error ensuring conversation collection: {e}")
            raise
    
    async def add_message(self, conversation_id: str, role: str, content: str, 
                         user_id: str, customer_id: str, **kwargs) -> Dict:
        """Add message to conversation with proper async handling."""
        # CHANGED: Get user lock properly
        user_lock = await self.get_user_lock(user_id) if user_id else asyncio.Lock()
        
        async with user_lock:
            logger.debug(f"Adding {role} message to conversation {conversation_id}")
            
            # CHANGED: Get conversation within the user lock to prevent race conditions
            # Check cache first
            async with self._cache_lock:
                if conversation_id in self._conversation_cache:
                    conversation = self._conversation_cache[conversation_id].copy()
                else:
                    # Try to load from Qdrant
                    conversation = await self._load_from_qdrant(conversation_id, customer_id)
                    if not conversation:
                        # Create new conversation
                        conversation = {
                            "conversation_id": conversation_id,
                            "user_id": user_id,
                            "title": None,
                            "messages": [],
                            "created_at": datetime.now().isoformat(),
                            "updated_at": datetime.now().isoformat()
                        }
            
            # Create message
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                **kwargs
            }
            
            # Update conversation
            conversation["messages"].append(message)
            conversation["updated_at"] = datetime.now().isoformat()
            
            # Auto-generate title from first user message
            if (role == "user" and not conversation.get("title") and 
                len(conversation["messages"]) == 1):
                conversation["title"] = await self.generate_conversation_title(content)
            
            # Update cache
            async with self._cache_lock:
                self._conversation_cache[conversation_id] = conversation
            
            # CHANGED: Fire-and-forget save with error handling
            asyncio.create_task(
                self._save_to_qdrant_async(conversation, customer_id)
            ).add_done_callback(
                lambda task: logger.error(f"Save task failed: {task.exception()}") 
                if task.exception() else None
            )
            
            return conversation
    
    async def _save_to_qdrant_async(self, conversation: Dict, customer_id: str):
        """Async save with lazy collection creation and proper error handling."""
        if not self.qdrant_client or self._shutdown:
            return
        
        try:
            # Ensure collection exists (only for first save)
            await self._ensure_conversation_collection_lazy(customer_id)
            
            # Save to Qdrant
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._qdrant_executor,
                partial(self._save_to_qdrant_sync, conversation, customer_id)
            )
            
            logger.debug(f"Saved conversation {conversation['conversation_id']} to Qdrant")
            
        except Exception as e:
            logger.error(f"Failed to save conversation to Qdrant: {e}")
            # CHANGED: Don't re-raise to prevent breaking the async flow
    
    def _save_to_qdrant_sync(self, conversation: Dict, customer_id: str):
        """Synchronous save to Qdrant (runs in thread pool)."""
        try:
            collection_name = get_collection_name("conversations", customer_id)
            
            logger.debug(f"Saving conversation {conversation['conversation_id']} to Qdrant...")
            
            # Create embedding from conversation text
            all_text = " ".join([
                msg["content"] for msg in conversation.get("messages", [])[-5:]
            ])
            
            if not all_text.strip():
                all_text = "Empty conversation"
                
            # Get embeddings - this returns a LIST of embeddings
            embeddings_list = get_embeddings_from_llama(all_text)
            
            logger.debug(f"Received {len(embeddings_list)} embeddings for conversation")
            
            # FIX: Handle the list of embeddings properly
            # We need to use just ONE embedding for the conversation
            simple_embedding = None
            
            if isinstance(embeddings_list, list) and len(embeddings_list) > 0:
                # Check if first element is also a list (nested structure)
                if isinstance(embeddings_list[0], list):
                    # We have multiple embeddings like [[emb1], [emb2], ...]
                    # Take the FIRST embedding from the list
                    simple_embedding = embeddings_list[0]
                    logger.debug(f"Using first embedding from list of {len(embeddings_list)} embeddings")
                else:
                    # We have a single flat list like [emb1]
                    simple_embedding = embeddings_list
                    logger.debug("Using single flat embedding list")
            else:
                # Fallback: create dummy embedding
                simple_embedding = [0.0] * 1024
                logger.warning("No embeddings received, using dummy embedding")
            
            # Ensure it's exactly 1024 dimensions
            if len(simple_embedding) != 1024:
                logger.warning(f"Embedding length is {len(simple_embedding)}, expected 1024. Truncating/padding.")
                if len(simple_embedding) > 1024:
                    simple_embedding = simple_embedding[:1024]
                else:
                    simple_embedding = simple_embedding + [0.0] * (1024 - len(simple_embedding))
            
            logger.debug(f"Final embedding ready: length={len(simple_embedding)}")
            
            # Generate point ID
            try:
                # Try to use conversation_id as UUID
                point_id = uuid.UUID(conversation["conversation_id"])
            except (ValueError, AttributeError):
                # Generate deterministic UUID from conversation_id
                point_id = uuid.uuid5(uuid.NAMESPACE_DNS, conversation["conversation_id"])
            
            logger.debug(f"Generated point ID: {point_id}")
            
            point = models.PointStruct(
                id=str(point_id),  # Convert to string for UUID
                vector=simple_embedding,
                payload=conversation
            )
            
            # Save with wait=True for reliability
            logger.debug(f"Saving point to collection {collection_name}...")
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=[point],
                wait=True
            )
            
            logger.info(f"✅ Successfully saved conversation {conversation['conversation_id']} to Qdrant")
            
            # Optional: Verify the save
            try:
                retrieved = self.qdrant_client.retrieve(
                    collection_name=collection_name,
                    ids=[str(point_id)],
                    with_payload=False
                )
                if retrieved:
                    logger.debug(f"✅ Verified: Point saved successfully")
            except Exception as verify_error:
                logger.debug(f"Could not verify save: {verify_error}")
            
        except Exception as e:
            logger.error(f"Error in sync Qdrant save: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    # CHANGED: Removed duplicate _save_to_qdrant method since we're using _save_to_qdrant_async
    
    async def generate_conversation_title(self, first_message: str, max_length: int = 50) -> str:
        """Generate conversation title."""
        # Simple rule-based title extraction
        title = self._extract_title_rules(first_message)
        
        if not title or len(title) > max_length:
            # If rule-based fails or too long, truncate
            words = first_message.split()[:8]
            title = " ".join(words)
            if len(title) > max_length:
                title = title[:max_length-3] + "..."
        
        return title or "New Conversation"
    
    def _extract_title_rules(self, message: str) -> str:
        """Rule-based title extraction."""
        message = message.strip()
        if not message:
            return ""
            
        sentences = message.replace('?', '.').replace('!', '.').split('.')
        first_sentence = sentences[0].strip()
        
        # Remove common prefixes
        prefixes = [
            "Can you", "Could you", "Please", "I need", "I want", 
            "What is", "How to", "Explain", "Tell me", "Show me"
        ]
        
        for prefix in prefixes:
            if first_sentence.lower().startswith(prefix.lower()):
                first_sentence = first_sentence[len(prefix):].strip()
        
        title = first_sentence.strip('?!. ')
        if len(title) > 50:
            title = title[:47] + "..."
        
        return title if title and len(title) > 5 else ""
    
    async def get_recent_history(self, conversation_id: str, max_messages: int = 5) -> List[str]:
        """Get recent conversation history for context."""
        conversation = await self.get_conversation(conversation_id)
        
        if not conversation["messages"]:
            return []
        
        # Format last N messages
        recent = conversation["messages"][-max_messages:]
        formatted = []
        
        for msg in recent:
            if msg["role"] == "user":
                formatted.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                formatted.append(f"Assistant: {msg['content']}")
        
        return formatted
    
    async def enhance_query_with_history(self, conversation_id: str, query: str) -> str:
        """Enhance query with conversation history."""
        history = await self.get_recent_history(conversation_id, max_messages=3)
        
        if not history:
            return query
        
        history_context = "\n".join(history)
        enhanced = f"""Based on our previous conversation:
{history_context}

Current question: {query}

Please consider the conversation history when answering."""
        
        return enhanced
    
     
    async def get_conversation_list(self, user_id: str, customer_id: str, limit: int = 50) -> List[Dict]:
        """
        Get all conversations for a user, sorted by updated_at (newest first).
        Returns conversations with minimal data (id, title, date, message count).
        """
        if not self.qdrant_client:
            logger.warning("No Qdrant client, returning empty conversation list")
            return []
        
        try:
            collection_name = get_collection_name("conversations", customer_id)
            
            # Use scroll to get all conversations for this user
            all_points = []
            next_page_offset = None
            
            while True:
                try:
                    scroll_result = await self._async_qdrant_scroll(
                        collection_name=collection_name,
                        scroll_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="user_id",
                                    match=models.MatchValue(value=user_id)
                                )
                            ]
                        ),
                        limit=100,
                        offset=next_page_offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    points, next_page_offset = scroll_result
                    all_points.extend(points)
                    
                    if not next_page_offset or len(all_points) >= limit:
                        break
                        
                except Exception as e:
                    if "not found" in str(e).lower():
                        # Collection doesn't exist yet - no conversations
                        logger.debug(f"No conversation collection for customer {customer_id}")
                        return []
                    raise
            
            # Format conversations for frontend
            conversations = []
            for point in all_points[:limit]:  # Apply limit
                payload = point.payload
                conversations.append({
                    "id": payload.get("conversation_id"),
                    "title": payload.get("title") or "Untitled Conversation",
                    "created_at": payload.get("created_at"),
                    "updated_at": payload.get("updated_at"),
                    "message_count": len(payload.get("messages", [])),
                    "last_message": self._get_last_message_preview(payload.get("messages", []))
                })
            
            # Sort by updated_at (newest first)
            conversations.sort(key=lambda x: x["updated_at"], reverse=True)
            
            return conversations
            
        except Exception as e:
            logger.error(f"Error getting conversation list: {e}")
            return []

    def _get_last_message_preview(self, messages: List[Dict]) -> str:
        """Get a preview of the last message."""
        if not messages:
            return ""
        
        # Get last message
        last_msg = messages[-1]
        content = last_msg.get("content", "")
        
        # Truncate if too long
        if len(content) > 50:
            return content[:47] + "..."
        
        return content

    async def update_conversation_title(self, conversation_id: str, title: str, 
                                    user_id: str, customer_id: str) -> bool:
        """
        Update conversation title.
        Returns True if successful, False if conversation not found.
        """
        user_lock = await self.get_user_lock(user_id)
        
        async with user_lock:
            try:
                # First try cache
                conversation = None
                async with self._cache_lock:
                    if conversation_id in self._conversation_cache:
                        conversation = self._conversation_cache[conversation_id].copy()
                
                # If not in cache, load from Qdrant
                if not conversation:
                    conversation = await self._load_from_qdrant(conversation_id, customer_id)
                
                if not conversation:
                    logger.warning(f"Conversation {conversation_id} not found for user {user_id}")
                    return False
                
                # Check ownership
                if conversation.get("user_id") != user_id:
                    logger.warning(f"User {user_id} trying to update conversation owned by {conversation.get('user_id')}")
                    return False
                
                # Update title
                conversation["title"] = title
                conversation["updated_at"] = datetime.now().isoformat()
                
                # Update cache
                async with self._cache_lock:
                    self._conversation_cache[conversation_id] = conversation
                
                # Save to Qdrant
                asyncio.create_task(self._save_to_qdrant_async(conversation, customer_id))
                
                logger.info(f"Updated title for conversation {conversation_id}: {title}")
                return True
                
            except Exception as e:
                logger.error(f"Error updating conversation title: {e}")
                return False

    async def delete_conversation(self, conversation_id: str, user_id: str, customer_id: str) -> bool:
        """
        Delete a conversation.
        Returns True if successful, False if not found or error.
        """
        user_lock = await self.get_user_lock(user_id)
        
        async with user_lock:
            try:
                collection_name = f"customer_{customer_id}_conversations"
                
                # Get the point ID first
                search_result = await self._async_qdrant_scroll(
                    collection_name=collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="conversation_id",
                                match=models.MatchValue(value=conversation_id)
                            ),
                            models.FieldCondition(
                                key="user_id",
                                match=models.MatchValue(value=user_id)
                            )
                        ]
                    ),
                    limit=1,
                    with_payload=False
                )
                
                points, _ = search_result
                if not points:
                    logger.warning(f"Conversation {conversation_id} not found for deletion")
                    return False
                
                # Delete from Qdrant
                point_id = points[0].id
                
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self._qdrant_executor,
                    lambda: self.qdrant_client.delete(
                        collection_name=collection_name,
                        points_selector=models.PointIdsList(
                            points=[point_id]
                        )
                    )
                )
                
                # Remove from cache
                async with self._cache_lock:
                    if conversation_id in self._conversation_cache:
                        del self._conversation_cache[conversation_id]
                
                logger.info(f"Deleted conversation {conversation_id} for user {user_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error deleting conversation: {e}")
                return False

    async def clear_user_conversations(self, user_id: str, customer_id: str) -> bool:
        """
        Clear all conversations for a user.
        Returns True if successful.
        """
        try:
            collection_name = get_collection_name("conversations", customer_id)
            
            # Find all conversation IDs for this user
            all_points = []
            next_page_offset = None
            
            while True:
                scroll_result = await self._async_qdrant_scroll(
                    collection_name=collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="user_id",
                                match=models.MatchValue(value=user_id)
                            )
                        ]
                    ),
                    limit=100,
                    offset=next_page_offset,
                    with_payload=False
                )
                
                points, next_page_offset = scroll_result
                all_points.extend(points)
                
                if not next_page_offset:
                    break
            
            if not all_points:
                logger.info(f"No conversations to clear for user {user_id}")
                return True
            
            # Delete all points
            point_ids = [point.id for point in all_points]
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._qdrant_executor,
                lambda: self.qdrant_client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(
                        points=point_ids
                    )
                )
            )
            
            # Clear user's conversations from cache
            async with self._cache_lock:
                # Remove all cached conversations for this user
                to_remove = [
                    conv_id for conv_id, conv in self._conversation_cache.items()
                    if conv.get("user_id") == user_id
                ]
                for conv_id in to_remove:
                    del self._conversation_cache[conv_id]
            
            logger.info(f"Cleared {len(point_ids)} conversations for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing user conversations: {e}")
            return False

# CHANGED: Fixed duplicate global variable declaration
_conversation_service_instance: Optional[AsyncConversationService] = None
_conversation_service_lock = asyncio.Lock()

async def get_conversation_service() -> AsyncConversationService:
    """Get or create conversation service instance."""
    global _conversation_service_instance
    
    if _conversation_service_instance is not None:
        return _conversation_service_instance
    
    async with _conversation_service_lock:
        if _conversation_service_instance is None:
            logger.info("Creating conversation service instance...")
            
            logger.debug("Getting Qdrant client...")
            qdrant_client = await get_qdrant_client_async()
            logger.debug("Got Qdrant client")
            
            logger.debug("Creating AsyncConversationService...")
            _conversation_service_instance = AsyncConversationService(qdrant_client)
            logger.info("Created AsyncConversationService")
    
    return _conversation_service_instance

# CHANGED: Added cleanup function
async def cleanup_conversation_service():
    """Cleanup conversation service resources."""
    global _conversation_service_instance
    if _conversation_service_instance:
        await _conversation_service_instance.shutdown()
        _conversation_service_instance = None