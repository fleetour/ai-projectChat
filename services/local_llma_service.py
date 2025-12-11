import subprocess
import json
import asyncio
from typing import List, Optional, AsyncGenerator
import logging
import os

import numpy as np

from services.utils import normalize_vector


logger = logging.getLogger(__name__)

class LocalLlamaService:
    def __init__(self, model: str = "mistral:7b"): #llama3:8b
        self.model = model
        self.ollama_path = self._find_ollama_path()
        self._test_connection()
    
    def _find_ollama_path(self) -> str:
        """Find Ollama executable path"""
        try:
            result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
            if result.returncode == 0:
                ollama_path = result.stdout.strip()
                logger.info(f"Found Ollama via which: {ollama_path}")
                return ollama_path
        except Exception:
            pass
            
        return "ollama"  # Hope it's in PATH
    
    def _test_connection(self):
        """Test connection to Ollama"""
        try:
            # Simple test - just try to list models
            result = subprocess.run([self.ollama_path, "list"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"✅ Ollama connection test passed. Using model: {self.model}")
            else:
                logger.warning(f"Ollama list failed: {result.stderr}")
        except Exception as e:
            logger.warning(f"Could not verify Ollama connection: {e}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using local Llama model - FIXED VERSION"""
        embeddings = []
        
        for text in texts:
            try:
                # Clean the text
                cleaned_text = self._clean_text(text)
                if not cleaned_text:
                    embeddings.append(self._create_fallback_embedding())
                    continue
                
                # Use API for embeddings
                import requests
                response = requests.post(
                    "http://localhost:11434/api/embeddings",
                    json={
                        "model": self.model,
                        "prompt": cleaned_text
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    embedding = response.json().get('embedding', [])
                    
                    # Debug: Check what we're getting
                    print(f"📊 Raw embedding: {len(embedding)} dimensions")
                    if embedding:
                        raw_norm = np.linalg.norm(embedding)
                        print(f"   Raw norm: {raw_norm:.6f}")
                    
                    # Handle size mismatch properly
                    if len(embedding) != 1024:
                        print(f"⚠️  Embedding size mismatch: {len(embedding)} != 1024")
                        if len(embedding) < 1024:
                            # Pad with random values instead of zeros
                            padding = [np.random.normal(0, 0.1) for _ in range(1024 - len(embedding))]
                            embedding = embedding + padding
                            print(f"   Padded to 1024 dimensions")
                        else:
                            embedding = embedding[:1024]
                            print(f"   Truncated to 1024 dimensions")
                    
                    # NORMALIZE for cosine similarity
                    normalized_embedding = normalize_vector(embedding)
                    final_norm = np.linalg.norm(normalized_embedding)
                    print(f"   Final normalized norm: {final_norm:.6f}")
                    
                    embeddings.append(normalized_embedding)
                    
                else:
                    print(f"❌ Embedding API error: {response.status_code}")
                    embeddings.append(self._create_fallback_embedding())
                    
            except Exception as e:
                print(f"❌ Error getting embedding: {e}")
                embeddings.append(self._create_fallback_embedding())
        
        # Final verification
        print(f"🎯 Generated {len(embeddings)} embeddings")
        for i, emb in enumerate(embeddings[:2]):  # Check first 2
            norm = np.linalg.norm(emb)
            zero_count = sum(1 for x in emb if x == 0)
            print(f"   Embedding {i}: norm={norm:.6f}, zeros={zero_count}/1024")
        
        return embeddings

    def _create_fallback_embedding(self, dimension: int = 1024) -> List[float]:
        """Create a non-zero fallback embedding"""
        import random
        # Create random embedding with normal distribution
        embedding = [random.gauss(0, 1) for _ in range(dimension)]
        # Normalize it
        return normalize_vector(embedding)

    def _clean_text(self, text: str) -> str:
        """Clean text for embedding generation"""
        if not text or not text.strip():
            return ""
        
        # Remove excessive whitespace but keep the content
        cleaned = ' '.join(text.split())
        
        # Ensure minimum length
        if len(cleaned) < 10:
            return ""
        
        return cleaned

    async def get_chat_completion_with_fullprompt(self, full_prompt: str) -> str:
        """Get chat completion using local Llama asynchronously"""
       
        try:
            # Use the API instead of command line - much more reliable
            import requests
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "top_p": 0.9,
                            "num_predict": 500  # Limit response length
                        }
                    },
                    timeout=60  # 60 second timeout
                )
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                logger.info(f"✅ Llama response received: {len(answer)} characters")
                return answer
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return "I'm sorry, I encountered an error processing your request."
                
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return "I'm sorry, I couldn't process your request at the moment."
            
    async def get_chat_completion(self, prompt: str, context: str = "") -> str:
        """Get chat completion using local Llama asynchronously"""
        full_prompt = self._build_prompt(prompt, context)
        
        try:
            # Use the API instead of command line - much more reliable
            import requests
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "top_p": 0.9,
                            "num_predict": 500  # Limit response length
                        }
                    },
                    timeout=60  # 60 second timeout
                )
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                logger.info(f"✅ Llama response received: {len(answer)} characters")
                return answer
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return "I'm sorry, I encountered an error processing your request."
                
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return "I'm sorry, I couldn't process your request at the moment."
    
    async def get_llama_stream_completion(
        self, 
        question: str, 
        context: str = "", 
        conversation_history: str = "",
        context_description: str = ""
    ) -> AsyncGenerator[str, None]:
        """Stream completion with conversation history support."""
        full_prompt = self._build_prompt_with_history(
            question=question,
            context=context,
            conversation_history=conversation_history,
            context_description=context_description
        )
        
        try:
            import aiohttp
            import json
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": True,
                        "options": {
                            "temperature": 0.3,
                            "top_p": 0.9,
                            "num_predict": 1000
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Stream API error: {response.status} - {error_text}")
                        yield "I'm sorry, I encountered an error processing your request."
                        return
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line:
                            try:
                                data = json.loads(line)
                                if 'response' in data:
                                    chunk = data['response']
                                    yield chunk
                                
                                if data.get('done', False):
                                    logger.info("✅ Stream completion finished")
                                    break
                                    
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse JSON: {line}")
                                continue
                            except Exception as e:
                                logger.error(f"Error processing stream chunk: {e}")
                                continue
                
        except asyncio.TimeoutError:
            logger.error("Stream request timed out")
            yield "I'm sorry, the request timed out."
        except Exception as e:
            logger.error(f"Error in stream completion: {e}")
            yield "I'm sorry, I couldn't process your request at the moment."
    
    def _build_prompt_with_history(
        self, 
        question: str, 
        context: str = "", 
        conversation_history: str = "",
        context_description: str = ""
    ) -> str:
        """Build prompt with conversation history and context awareness."""
        
        # Base system message with context awareness
        system_message = f"""You are a helpful assistant for project documents.

**CRITICAL LANGUAGE RULES:**
1. **ALWAYS respond in the SAME LANGUAGE as the user's question**
2. If question is in German → respond in German
3. If question is in English → respond in English  
4. If question is in French → respond in French
5. If question is in Arabic → respond in Arabic
6. If question mixes languages → use the main language of the question
7. **NEVER** translate the question or switch languages unless explicitly asked

{context_description}

IMPORTANT: Format your response using Markdown for better readability:
- Use **bold** for important terms
- Use *italic* for emphasis  
- Use `code` for technical terms
- Use lists with - or * for multiple items
- Use headings with ## when appropriate
- Reference specific documents when relevant

Consider the conversation history when answering."""
        
        # Build the full prompt
        prompt_parts = []
        
        # 1. System message
        prompt_parts.append(system_message)
        
        # 2. Conversation history (if any)
        if conversation_history:
            prompt_parts.append(f"\nConversation History:\n{conversation_history}")
        
        # 3. Document context
        if context:
            prompt_parts.append(f"\nRelevant Document Context:\n{context}")
        
        # 4. Current question
        prompt_parts.append(f"\nCurrent Question: {question}")
        
        # 5. Instruction
        prompt_parts.append("\nPlease answer based on the document context and conversation history. If the context doesn't contain enough information, please say so.\n\nAnswer in Markdown format:")
        
        return "\n".join(prompt_parts)
