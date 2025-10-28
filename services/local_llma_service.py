import subprocess
import json
import asyncio
from typing import List, Optional, AsyncGenerator
import logging
import os

logger = logging.getLogger(__name__)

class LocalLlamaService:
    def __init__(self, model: str = "llama3:8b"):
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
        """Get embeddings using local Llama model"""
        embeddings = []
        
        for text in texts:
            try:
                # Clean the text
                cleaned_text = self._clean_text(text)
                if not cleaned_text:
                    embeddings.append([0.0] * 1024)
                    continue
                
                # Use API for embeddings (more reliable)
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
                    # Ensure correct size
                    if len(embedding) != 1024:
                        if len(embedding) < 1024:
                            embedding = embedding + [0.0] * (1024 - len(embedding))
                        else:
                            embedding = embedding[:1024]
                    embeddings.append(embedding)
                else:
                    logger.error(f"Embedding API error: {response.status_code}")
                    embeddings.append([0.0] * 1024)
                    
            except Exception as e:
                logger.error(f"Error getting embedding: {e}")
                embeddings.append([0.0] * 1024)
        
        return embeddings
    
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
    
    async def get_llama_stream_completion(self, prompt: str, context: str = "") -> AsyncGenerator[str, None]:
        """Stream completion using local Llama model"""
        full_prompt = self._build_prompt(prompt, context)
        
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
                                
                                # Check if this is the final response
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
    
    # In deiner local_llama_service.py - _build_prompt Methode anpassen
    def _build_prompt(self, question: str, context: str = "") -> str:
        """Build the prompt for the model with markdown formatting"""
        if context:
            return f"""Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question, please say so.

    IMPORTANT: Format your response using Markdown for better readability:
    - Use **bold** for important terms
    - Use *italic* for emphasis  
    - Use `code` for technical terms
    - Use lists with - or * for multiple items
    - Use headings with ## when appropriate

    Context:
    {context}

    Question: {question}

    Answer in Markdown format:"""
        else:
            return f"""Please answer the following question using Markdown formatting for better readability:

    Question: {question}

    Answer in Markdown format:"""

    # Global instance
local_llama = LocalLlamaService(model="llama3:8b")