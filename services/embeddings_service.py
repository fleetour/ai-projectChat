import os
from docx import Document
import requests
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/embeddings"

# Initialize Mistral client
try:
    client = Mistral(api_key=MISTRAL_API_KEY)  # Make sure to set your API key
except ImportError:
    print("Warning: mistralai package not installed")
    client = None


def get_embeddings_from_mistral(texts: list):
    """Get embeddings from Mistral Cloud API for a list of text chunks."""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        MISTRAL_API_URL,
        json={"model": "mistral-embed", "input": texts},
        headers=headers
    )

    response.raise_for_status()
    data = response.json()

    print("Mistral embeddings response:", data)  # Debugging line
    # Return embeddings in same order
    return [item["embedding"] for item in data["data"]]



def extract_text_from_file(file_path: str) -> str:
    """Extract text from .docx, .pdf, or .txt files."""
    ext = file_path.lower()
    if ext.endswith(".pdf"):
        from pdfminer.high_level import extract_text
        return extract_text(file_path)
    elif ext.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext.endswith(".txt"):
        with open(file_path, "r") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
async def get_mistral_chat_completion(prompt: str, model: str = "mistral-medium-latest") -> str:
    """
    Get chat completion from Mistral API asynchronously
    """
    if client is None:
        raise Exception("Mistral client not configured. Please install mistralai package and set MISTRAL_API_KEY")
    
    try:
        response = await client.chat.complete_async(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        raise Exception(f"Failed to get Mistral chat completion: {str(e)}")