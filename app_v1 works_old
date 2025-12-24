import os
import tempfile
import logging
from typing import List, Dict
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from docx import Document
from pydantic import BaseModel
import uuid
from mistralai.models import UserMessage

# Replace this with your actual Mistral client
from mistralai import Mistral

model = "mistral-medium-latest"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Mistral RAG Demo")

# In-memory storage
file_store: Dict[str, List[Dict]] = {}  # file_id -> list of {"text": chunk, "embedding": [...]}

class QueryRequest(BaseModel):
    file_id: str
    query: str

class QueryResponse(BaseModel):
    answer: str


UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Mistral client (make sure you have MISTRAL_API_KEY in your env)
mistral = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# In-memory document store
DOCUMENT_STORE = []  # Each item: {"text": chunk_text, "embedding": np.array([...]), "source": filename}


# ------------------ Utility Functions ------------------

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



def get_embedding(text: str) -> np.ndarray:
    """Generate embedding vector from text using Mistral."""
    response = mistral.embeddings.create(
        model="mistral-embed",
        inputs=[text]   # <- must be a list
    )
    return np.array(response.data[0].embedding)


# ------------------ Upload Endpoint ------------------

# Upload endpoint
@app.post("/upload")
async def upload(file: UploadFile):
    # Save file locally
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Extract text and embeddings if needed
    text = extract_text_from_file(file_path)
    chunks = chunk_text(text)
    for chunk in chunks:
        embedding = get_embedding(chunk)
        # store embedding somewhere (DB, file, or in-memory)

    return {"message": f"{file.filename} uploaded and processed.", "path": file_path}


# ------------------ Query Endpoint ------------------

# Query endpoint
@app.post("/query")
async def query_file(file_name: str, request: QueryRequest):
    file_path = os.path.join(UPLOAD_DIR, file_name)
    if not os.path.exists(file_path):
        return {"error": "File not found. Please upload it first."}

    text = extract_text_from_file(file_path)
    top_chunk = select_top_chunk(request.query, text)  # your helper
    answer = await call_model_completion(request.query, top_chunk)
    return {"answer": answer}

# ------------------------------
# Helpers (implement yourself)
def extract_text(content: bytes, filename: str) -> str:
    # handle .txt, .docx, .pdf
    return "extracted text from file"

def split_text(text: str, chunk_size: int = 500) -> List[str]:
    # split by words for better semantic chunks
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


async def call_model_completion(question: str, context: str) -> str:
    prompt = f"""
You are an assistant that extracts information from documents.
Context:
{context}

Question: {question}
Please answer with only the value, no extra text.
"""
    print("Prompt for completion:", prompt)
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    # Make the API call using inputs instead of messages
 

    chat_response = await client.chat.complete_async(
        model=model,
        messages=[UserMessage(content=prompt)],
    )

    print("Chat response:", chat_response)
    # Extract the generated text
    return chat_response


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def select_top_chunk(query, chunks):
    """
    Return the chunk that contains the most words from the query.
    """
    query_words = set(query.lower().split())
    best_chunk = None
    max_matches = 0

    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        matches = len(query_words & chunk_words)
        if matches > max_matches:
            max_matches = matches
            best_chunk = chunk

    return best_chunk if best_chunk else chunks[0]