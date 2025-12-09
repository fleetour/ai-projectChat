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

from config import CUSTOMER_ID
from db.qdrant_service import get_qdrant_client, get_qdrant_client_async
from services.embeddings_service import get_embeddings_from_llama

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
        # Initialize thread pool for async Qdrant operations
        self._qdrant_executor = ThreadPoolExecutor(
            max_workers=3,
            thread_name_prefix="qdrant_conversation_"
        )
    
    async def _ensure_conversation_collection_lazy(self):
        """Lazy initialization - only check/create collection once."""
        if self._collection_initialized or not self.qdrant_client:
            return
        
        async with self._collection_lock:
            if not self._collection_initialized:
                await self._ensure_conversation_collection()
                self._collection_initialized = True
    
    
    async def get_user_lock(self, user_id: str) -> asyncio.Lock:
        async with self._cache_lock:
            if user_id not in self.user_locks:
                self.user_locks[user_id] = asyncio.Lock()
        return self.user_locks[user_id]
    
    async def get_conversation(self, conversation_id: str, user_id: str = None) -> Dict:
        """Get conversation with caching."""
        async with self._cache_lock:
            if conversation_id in self._conversation_cache:
                return self._conversation_cache[conversation_id].copy()
        
        # Load from database
        if self.qdrant_client:
            # Don't ensure collection here for new conversations
            # Only ensure when actually saving
            conversation = await self._load_from_qdrant(conversation_id)
            if conversation:
                async with self._cache_lock:
                    self._conversation_cache[conversation_id] = conversation
                return conversation.copy()
        
        # Create new conversation (no Qdrant calls needed here!)
        new_conv = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "title": None,
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        async with self._cache_lock:
            self._conversation_cache[conversation_id] = new_conv
        
        return new_conv.copy()
    
    async def _async_qdrant_scroll(self, **kwargs):
        """Run Qdrant scroll operation in thread pool asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Create a partial function with the scroll call
        scroll_func = partial(self.qdrant_client.scroll, **kwargs)
        
        # Run in thread pool to avoid blocking
        return await loop.run_in_executor(
            self._qdrant_executor,
            scroll_func
        )
    
    async def _load_from_qdrant(self, conversation_id: str) -> Optional[Dict]:
        """Load conversation from Qdrant asynchronously."""
        if not self.qdrant_client:
            return None
        
        try:
            print(f"🔍 Loading conversation {conversation_id} from Qdrant")
            
            # FIX: Skip collection check for loading
            # Collection should exist if there's data, but if not, just return None
            
            # Use async scroll
            search_result = await self._async_qdrant_scroll(
                collection_name=f"customer_{CUSTOMER_ID}_conversations",
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
                print(f"✅ Found conversation {conversation_id} in Qdrant")
                return points[0].payload
            else:
                print(f"📭 Conversation {conversation_id} not found in Qdrant")
                
        except Exception as e:
            # Collection might not exist - that's OK for new conversations
            logger.debug(f"Could not load conversation (collection might not exist): {e}")
        
        return None
    
    async def _ensure_conversation_collection(self):
        """Ensure conversations collection exists in Qdrant asynchronously."""
        if not self.qdrant_client:
            return
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._qdrant_executor,
                self._ensure_conversation_collection_sync
            )
        except Exception as e:
            logger.error(f"Error ensuring conversation collection: {e}")
    
    def _ensure_conversation_collection_sync(self):
        """Synchronous version of collection check."""
        collection_name = f"customer_{CUSTOMER_ID}_conversations"
        
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if collection_name not in collection_names:
                # Create collection
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=384,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created conversations collection: {collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring conversation collection: {e}")
    
    async def add_message(self, conversation_id: str, role: str, content: str, 
                     user_id: str = None, **kwargs) -> Dict:
        """Optimized: Fast path for new conversations."""
        user_lock = await self.get_user_lock(user_id) if user_id else self._cache_lock
        
        async with user_lock:
            print(f"💬 Adding {role} message to conversation {conversation_id}")
        
        # Check cache first without any Qdrant calls
        async with self._cache_lock:
            if conversation_id in self._conversation_cache:
                conversation = self._conversation_cache[conversation_id].copy()
            else:
                # New conversation - create minimal object
                conversation = {
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "title": None,
                    "messages": [],
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        conversation["messages"].append(message)
        conversation["updated_at"] = datetime.now().isoformat()
        
        # Auto-generate title from first user message
        if role == "user" and not conversation.get("title") and len(conversation["messages"]) == 1:
            conversation["title"] = await self.generate_conversation_title(content)
        
        # Save to cache
        async with self._cache_lock:
            self._conversation_cache[conversation_id] = conversation
        
        # Fire-and-forget Qdrant save - don't wait for it
        # This prevents blocking the stream response
        asyncio.create_task(self._save_to_qdrant_async(conversation))
        
        return conversation
    
    async def _save_to_qdrant_async(self, conversation: Dict):
        """Async save with lazy collection creation."""
        if not self.qdrant_client:
            return
        
        try:
            # Ensure collection exists (only for first save)
            await self._ensure_conversation_collection_lazy()
            
            # Save to Qdrant
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._qdrant_executor,
                partial(self._save_to_qdrant_sync, conversation)
            )
            
        except Exception as e:
            logger.error(f"Failed to save conversation to Qdrant: {e}")

    
    

    
    async def _save_to_qdrant(self, conversation: Dict):
        """Save conversation to Qdrant asynchronously."""
        if not self.qdrant_client:
            return
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._qdrant_executor,
                partial(self._save_to_qdrant_sync, conversation)
            )
        except Exception as e:
            logger.error(f"Failed to save conversation to Qdrant: {e}")
    
    def _save_to_qdrant_sync(self, conversation: Dict):
        """Synchronous save to Qdrant (runs in thread pool)."""
        try:
            collection_name = f"customer_{CUSTOMER_ID}_conversations"
            
            # Create a simple embedding from conversation text
            all_text = " ".join([
                msg["content"] for msg in conversation.get("messages", [])[-5:]  # Last 5 messages
            ])
            simple_embedding = get_embeddings_from_llama(all_text)
            
            point = models.PointStruct(
                id=conversation["conversation_id"],
                vector=simple_embedding,
                payload=conversation
            )
            
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=[point],
                wait=False  # Don't wait for confirmation
            )
            
            logger.debug(f"Saved conversation to Qdrant: {conversation['conversation_id']}")
            
        except Exception as e:
            logger.error(f"Error in sync Qdrant save: {e}")
    
    
    async def generate_conversation_title(self, first_message: str, max_length: int = 50) -> str:
        """Generate conversation title."""
        # Simple rule-based title extraction
        title = self._extract_title_rules(first_message)
        
        if not title or len(title) > max_length:
            # If rule-based fails or too long, truncate
            words = first_message.split()[:8]  # First 8 words
            title = " ".join(words)
            if len(title) > max_length:
                title = title[:max_length-3] + "..."
        
        return title or "New Conversation"
    
    def _extract_title_rules(self, message: str) -> str:
        """Rule-based title extraction."""
        message = message.strip()
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

_conversation_service_instance: Optional[AsyncConversationService] = None
_conversation_service_lock = asyncio.Lock()


_conversation_service_instance: Optional[AsyncConversationService] = None
_conversation_service_lock = asyncio.Lock()

async def get_conversation_service() -> AsyncConversationService:
    """Get or create conversation service instance."""
    global _conversation_service_instance
    
    if _conversation_service_instance is not None:
        return _conversation_service_instance
    
    async with _conversation_service_lock:
        if _conversation_service_instance is None:
            print("🔄 Creating conversation service instance...")
            
            print("📡 Getting Qdrant client...")
            qdrant_client = await get_qdrant_client_async()
            print("✅ Got Qdrant client")
            
            print("🏗️ Creating AsyncConversationService...")
            _conversation_service_instance = AsyncConversationService(qdrant_client)
            print("✅ Created AsyncConversationService")
    
    return _conversation_service_instance