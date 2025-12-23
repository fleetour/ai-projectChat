import subprocess
import json
import asyncio
from typing import List, Optional, AsyncGenerator, Dict, Any
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from services.utils import normalize_vector


logger = logging.getLogger(__name__)

class LlmProvider(Enum):
    LOCAL_OLLAMA = "local_ollama"
    MISTRAL_AI = "mistral_ai"


class BaseLlmService(ABC):
    """Abstract base class for LLM services"""
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @abstractmethod
    async def get_chat_completion(self, prompt: str, context: str = "") -> str:
        pass
    
    @abstractmethod
    async def get_chat_completion_with_fullprompt(self, full_prompt: str) -> str:
        pass
    
    @abstractmethod
    async def get_llama_stream_completion(
        self, 
        question: str, 
        context: str = "", 
        conversation_history: str = "",
        context_description: str = ""
    ) -> AsyncGenerator[str, None]:
        pass
    
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
    
    def _create_fallback_embedding(self, dimension: int = 1024) -> List[float]:
        """Create a non-zero fallback embedding"""
        import random
        # Create random embedding with normal distribution
        embedding = [random.gauss(0, 1) for _ in range(dimension)]
        # Normalize it
        return normalize_vector(embedding)


class LocalOllamaService(BaseLlmService):
    def __init__(self, model: str = "mistral:7b"):
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
        """Get embeddings using local Ollama model"""
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

    async def get_chat_completion_with_fullprompt(self, full_prompt: str) -> str:
        """Get chat completion using local Ollama asynchronously"""
        try:
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
                            "num_predict": 500
                        }
                    },
                    timeout=60
                )
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                logger.info(f"✅ Ollama response received: {len(answer)} characters")
                return answer
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return "I'm sorry, I encountered an error processing your request."
                
        except Exception as e:
            logger.error(f"Error in Ollama chat completion: {e}")
            return "I'm sorry, I couldn't process your request at the moment."
            
    async def get_chat_completion(self, prompt: str, context: str = "") -> str:
        """Get chat completion using local Ollama asynchronously"""
        full_prompt = self._build_prompt_with_history(prompt, context)
        return await self.get_chat_completion_with_fullprompt(full_prompt)
    
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
                        logger.error(f"Ollama stream API error: {response.status} - {error_text}")
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
                                    logger.info("✅ Ollama stream completion finished")
                                    break
                                    
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse JSON: {line}")
                                continue
                            except Exception as e:
                                logger.error(f"Error processing stream chunk: {e}")
                                continue
                
        except asyncio.TimeoutError:
            logger.error("Ollama stream request timed out")
            yield "I'm sorry, the request timed out."
        except Exception as e:
            logger.error(f"Error in Ollama stream completion: {e}")
            yield "I'm sorry, I couldn't process your request at the moment."


class MistralAiService(BaseLlmService):
    def __init__(self, model: str = "mistral-medium"):
        """
        Initialize Mistral AI service.
        
        Args:
            api_key: Mistral AI API key (defaults to MISTRAL_API_KEY env var)
            model: Model to use (mistral-tiny, mistral-small, mistral-medium, mistral-large-latest)
        """
        self.api_key = os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral AI API key is required. Set MISTRAL_API_KEY environment variable.")
        
        self.model = model
        self.base_url = "https://api.mistral.ai/v1"
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Mistral AI API"""
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Try to list models
            response = requests.get(f"{self.base_url}/models", headers=headers, timeout=10)
            if response.status_code == 200:
                logger.info(f"✅ Mistral AI connection test passed. Using model: {self.model}")
            else:
                logger.warning(f"Mistral AI connection test failed: {response.status_code} - {response.text}")
        except Exception as e:
            logger.warning(f"Could not verify Mistral AI connection: {e}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Mistral AI API"""
        embeddings = []
        
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Mistral AI embedding API
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json={
                    "model": "mistral-embed",
                    "input": texts
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                for item in data.get("data", []):
                    embedding = item.get("embedding", [])
                    
                    # Mistral embeddings are already 1024-dimensional and normalized
                    if len(embedding) != 1024:
                        logger.warning(f"Mistral embedding size mismatch: {len(embedding)} != 1024")
                        # Still normalize it
                        embedding = normalize_vector(embedding)
                    
                    embeddings.append(embedding)
                    
                logger.info(f"✅ Generated {len(embeddings)} embeddings via Mistral AI")
                
            else:
                logger.error(f"Mistral AI embedding API error: {response.status_code} - {response.text}")
                # Create fallback embeddings
                for _ in texts:
                    embeddings.append(self._create_fallback_embedding())
                    
        except Exception as e:
            logger.error(f"❌ Error getting Mistral AI embeddings: {e}")
            # Create fallback embeddings
            for _ in texts:
                embeddings.append(self._create_fallback_embedding())
        
        return embeddings

    async def get_chat_completion_with_fullprompt(self, full_prompt: str) -> str:
        """Get chat completion using Mistral AI asynchronously"""
        try:
            import requests
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": full_prompt
                            }
                        ],
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "max_tokens": 500
                    },
                    timeout=60
                )
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                logger.info(f"✅ Mistral AI response received: {len(answer)} characters")
                return answer
            else:
                logger.error(f"Mistral AI API error: {response.status_code} - {response.text}")
                return "I'm sorry, I encountered an error processing your request."
                
        except Exception as e:
            logger.error(f"Error in Mistral AI chat completion: {e}")
            return "I'm sorry, I couldn't process your request at the moment."
    
    async def get_chat_completion(self, prompt: str, context: str = "") -> str:
        """Get chat completion using Mistral AI asynchronously"""
        full_prompt = self._build_prompt_with_history(prompt, context)
        return await self.get_chat_completion_with_fullprompt(full_prompt)
    
    async def get_llama_stream_completion(
        self, 
        question: str, 
        context: str = "", 
        conversation_history: str = "",
        context_description: str = ""
    ) -> AsyncGenerator[str, None]:
        """Stream completion with conversation history support using Mistral AI."""
        full_prompt = self._build_prompt_with_history(
            question=question,
            context=context,
            conversation_history=conversation_history,
            context_description=context_description
        )
        
        try:
            import aiohttp
            import json
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 1000,
                "stream": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Mistral AI stream API error: {response.status} - {error_text}")
                        yield "I'm sorry, I encountered an error processing your request."
                        return
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data_str = line[6:]  # Remove 'data: ' prefix
                            
                            if data_str == '[DONE]':
                                logger.info("✅ Mistral AI stream completion finished")
                                break
                            
                            try:
                                data = json.loads(data_str)
                                choices = data.get('choices', [])
                                if choices:
                                    delta = choices[0].get('delta', {})
                                    chunk = delta.get('content', '')
                                    if chunk:
                                        yield chunk
                                        
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse Mistral AI JSON: {data_str}")
                                continue
                            except Exception as e:
                                logger.error(f"Error processing Mistral AI stream chunk: {e}")
                                continue
                
        except asyncio.TimeoutError:
            logger.error("Mistral AI stream request timed out")
            yield "I'm sorry, the request timed out."
        except Exception as e:
            logger.error(f"Error in Mistral AI stream completion: {e}")
            yield "I'm sorry, I couldn't process your request at the moment."


class LlmService:
    """Unified LLM service that can work with different providers"""
    
    def __init__(
        self, 
        provider: Optional[LlmProvider] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize LLM service with configuration from environment or parameters.
        
        Priority: Parameters > Environment Variables > Defaults
        
        Args:
            provider: Which LLM provider to use (overrides env)
            model: Model name to use (overrides env)
            api_key: API key for cloud providers (overrides env)
            **kwargs: Additional provider-specific arguments
        """
        # Determine provider from parameter or environment
        self.provider = self._determine_provider(provider)
        
        # Determine model from parameter or environment
        self.model = self._determine_model(model)
        
        # Determine API key (for cloud providers)
        self.api_key = self._determine_api_key(api_key)
        
        # Validate configuration
        self._validate_configuration()
        
        # Initialize the appropriate service
        self.service = self._initialize_service()
        
        logger.info(f"✅ Initialized LLM service with provider: {self.provider.value}, model: {self.model}")
    
    def _determine_provider(self, provider_param: Optional[LlmProvider]) -> LlmProvider:
        """Determine provider from parameter or environment"""
        if provider_param:
            return provider_param
        
        # Read from environment variable
        env_provider = os.environ.get("LLM_PROVIDER", "local_ollama").lower()
        
        # Map string to enum
        provider_map = {
            "local_ollama": LlmProvider.LOCAL_OLLAMA,
            "local": LlmProvider.LOCAL_OLLAMA,
            "ollama": LlmProvider.LOCAL_OLLAMA,
            "mistral_ai": LlmProvider.MISTRAL_AI,
            "mistral": LlmProvider.MISTRAL_AI,
            "mistralai": LlmProvider.MISTRAL_AI,
        }
        
        if env_provider in provider_map:
            return provider_map[env_provider]
        
        logger.warning(f"Unknown LLM_PROVIDER in env: {env_provider}, defaulting to LOCAL_OLLAMA")
        return LlmProvider.LOCAL_OLLAMA
    
    def _determine_model(self, model_param: Optional[str]) -> str:
        """Determine model from parameter or environment"""
        if model_param:
            return model_param
        
        # Read from environment variable with provider-specific defaults
        if self.provider == LlmProvider.LOCAL_OLLAMA:
            default_model = "mistral:7b"
        elif self.provider == LlmProvider.MISTRAL_AI:
            default_model = "mistral-medium"
        else:
            default_model = "mistral:7b"
        
        return os.environ.get("LLM_MODEL", default_model)
    
    def _determine_api_key(self, api_key_param: Optional[str]) -> Optional[str]:
        """Determine API key from parameter or environment"""
        if api_key_param:
            return api_key_param
        
        # Only needed for cloud providers
        if self.provider == LlmProvider.MISTRAL_AI:
            return os.environ.get("MISTRAL_API_KEY")
        
        return None
    
    def _validate_configuration(self) -> None:
        """Validate the configuration before initializing service"""
        # Validate API key for cloud providers
        if self.provider == LlmProvider.MISTRAL_AI and not self.api_key:
            raise ValueError(
                "Mistral AI API key is required. "
                "Set MISTRAL_API_KEY environment variable or pass api_key parameter."
            )
        
        # Log warnings for potential issues
        if self.provider == LlmProvider.LOCAL_OLLAMA:
            # Check if Ollama is likely running
            if not self._is_ollama_available():
                logger.warning(
                    "Local Ollama provider selected, but Ollama may not be running. "
                    "Make sure Ollama is installed and running on http://localhost:11434"
                )
    
    def _is_ollama_available(self) -> bool:
        """Check if Ollama service is available"""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _initialize_service(self):
        """Initialize the appropriate service based on provider"""
        if self.provider == LlmProvider.LOCAL_OLLAMA:
            return LocalOllamaService(model=self.model)
            
        elif self.provider == LlmProvider.MISTRAL_AI:
            return MistralAiService(model=self.model)
            
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from the active provider"""
        return self.service.get_embeddings(texts)
    
    async def get_chat_completion(self, prompt: str, context: str = "") -> str:
        """Get chat completion from the active provider"""
        return await self.service.get_chat_completion(prompt, context)
    
    async def get_chat_completion_with_fullprompt(self, full_prompt: str) -> str:
        """Get chat completion with full prompt from the active provider"""
        return await self.service.get_chat_completion_with_fullprompt(full_prompt)
    
    async def get_llama_stream_completion(
        self, 
        question: str, 
        context: str = "", 
        conversation_history: str = "",
        context_description: str = ""
    ) -> AsyncGenerator[str, None]:
        """Stream completion from the active provider"""
        async for chunk in self.service.get_llama_stream_completion(
            question, context, conversation_history, context_description
        ):
            yield chunk

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from the active provider"""
        return self.service.get_embeddings(texts)
    
    async def get_chat_completion(self, prompt: str, context: str = "") -> str:
        """Get chat completion from the active provider"""
        return await self.service.get_chat_completion(prompt, context)
    
    async def get_chat_completion_with_fullprompt(self, full_prompt: str) -> str:
        """Get chat completion with full prompt from the active provider"""
        return await self.service.get_chat_completion_with_fullprompt(full_prompt)
    
    async def get_llama_stream_completion(
        self, 
        question: str, 
        context: str = "", 
        conversation_history: str = "",
        context_description: str = ""
    ) -> AsyncGenerator[str, None]:
        """Stream completion from the active provider"""
        async for chunk in self.service.get_llama_stream_completion(
            question, context, conversation_history, context_description
        ):
            yield chunk


# Backward compatibility alias
class LocalLlamaService(LocalOllamaService):
    """Deprecated: Use LlmService with provider=LOCAL_OLLAMA instead"""
    def __init__(self, model: str = "mistral:7b"):
        import warnings
        warnings.warn(
            "LocalLlamaService is deprecated. Use LlmService(provider=LlmProvider.LOCAL_OLLAMA, model=...) instead.",
            DeprecationWarning
        )
        super().__init__(model=model)