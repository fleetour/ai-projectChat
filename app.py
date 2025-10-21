import os
import json
import math
from typing import List
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from mistralai import Mistral
from io import BytesIO
from docx import Document

app = FastAPI(title="Document Q&A with Mistral")

# ---------- CONFIG ----------
CHUNK_SIZE = 500  # number of words per chunk
STORAGE_DIR = "stored_files"  # directory to store embeddings
os.makedirs(STORAGE_DIR, exist_ok=True)

# Mistral client
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# ---------- HELPERS ----------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=[text]
    )
    return response.data[0].embedding

def save_embeddings(file_name: str, embeddings: list):
    path = os.path.join(STORAGE_DIR, f"{file_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)

def load_embeddings(file_name: str) -> list:
    path = os.path.join(STORAGE_DIR, f"{file_name}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    dot = sum(a*b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a*a for a in vec1))
    norm2 = math.sqrt(sum(b*b for b in vec2))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0

def select_top_chunks(question: str, chunks_with_embeddings: list, top_n=3):
    question_embedding = get_embedding(question)
    sorted_chunks = sorted(
        chunks_with_embeddings,
        key=lambda item: cosine_similarity(question_embedding, item["embedding"]),
        reverse=True
    )
    return [item["text"] for item in sorted_chunks[:top_n]]

async def extract_text_from_upload(file: UploadFile) -> str:
    content = await file.read()
    if file.filename.endswith(".docx"):
        doc = Document(BytesIO(content))
        text = "\n".join([p.text for p in doc.paragraphs])
        return text
    else:
        # fallback: try decoding as utf-8
        return content.decode("utf-8", errors="ignore")

# ---------- ENDPOINTS ----------

@app.post("/upload")
async def upload_file(file: UploadFile):
    text = await extract_text_from_upload(file)
    
    chunks = chunk_text(text)
    embeddings = [{"text": chunk, "embedding": get_embedding(chunk)} for chunk in chunks]
    
    save_embeddings(file.filename, embeddings)
    return {"status": "success", "file": file.filename, "chunks": len(chunks)}

@app.post("/query")
async def query_file(file_name: str = Form(...), question: str = Form(...)):
    embeddings = load_embeddings(file_name)
    if not embeddings:
        return JSONResponse({"error": "File not found or not processed."}, status_code=404)
    
    #top_chunk = select_top_chunk(question, embeddings)
    top_chunks = select_top_chunks(question, embeddings, top_n=3)
    context = "\n\n".join(top_chunks)
    
    prompt = f"Answer the question based on the following text:\n\n{context}\n\nQuestion: {question}\nAnswer:"

    print("prompt:", prompt)
    # Use Mistral chat completion
    response = await client.chat.complete_async(
        model="mistral-medium-latest",
        messages=[{"role": "user", "content": prompt}]
    )
  #  print("response:", response)
   # return response
    answer = response.choices[0].message.content
    return {"answer": answer, "file": file_name}

# ---------- LIST AVAILABLE FILES ----------
@app.get("/files")
def list_files():
    files = [f.replace(".json","") for f in os.listdir(STORAGE_DIR) if f.endswith(".json")]
    return {"files": files}
