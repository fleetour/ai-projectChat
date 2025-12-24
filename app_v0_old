# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os, io, math, json
import numpy as np
import faiss
import requests
from typing import List
# pip install python-docx pdfminer.six pytesseract pillow faiss-cpu fastapi uvicorn requests

# --- utils for text extraction (very simple) ---
def extract_text_from_pdf(file_bytes: bytes) -> str:
    from io import BytesIO
    from pdfminer.high_level import extract_text_to_fp
    out = BytesIO()
    extract_text_to_fp(BytesIO(file_bytes), out)
    return out.getvalue().decode(errors="ignore")

def extract_text_from_docx(file_bytes: bytes) -> str:
    import docx
    from io import BytesIO
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_image(file_bytes: bytes) -> str:
    from PIL import Image
    import pytesseract
    img = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(img)

def chunk_text(text, max_words=1000):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]


# --- Mistral API helpers ---
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise RuntimeError("Set MISTRAL_API_KEY env var")

MISTRAL_BASE = "https://api.mistral.ai/v1"  # docs show base endpoints (check console) - adapt if needed

def mistral_embed_batch(texts: List[str], model="mistral-embed-1"):
    # POST to embeddings endpoint (docs: embeddings usage)
    url = f"{MISTRAL_BASE}/embeddings"
    payload = {"model": model, "input": texts}
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    # assume data["data"] is list of embeddings objects
    embeddings = [item["embedding"] for item in data["data"]]
    return embeddings

def mistral_chat_completion(prompt: str, model="mistral-7b-instruct-v0.3"):
    url = f"{MISTRAL_BASE}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role":"system","content":"You are a helpful assistant."},
                     {"role":"user","content": prompt}],
        "max_tokens": 800
    }
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

# --- FAISS store (simple) ---
DIM = 1536  # adjust to embedding dim returned by the model (update after first response)
index = None
metadatas = []  # parallel list of metadata dicts

def ensure_index(dim):
    global index
    if index is None:
        index = faiss.IndexFlatL2(dim)

def add_embeddings_to_index(embeddings: List[List[float]], metas: List[dict]):
    ensure_index(len(embeddings[0]))
    vecs = np.array(embeddings).astype("float32")
    index.add(vecs)
    metadatas.extend(metas)

def search_index(query_emb, k=5):
    if index is None or index.ntotal == 0:
        return []
    q = np.array([query_emb]).astype("float32")
    D, I = index.search(q, k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        results.append({"score": float(dist), "metadata": metadatas[idx]})
    return results


# --- FastAPI app ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

class QueryReq(BaseModel):
    question: str
    top_k: int = 5

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    print("Uploading file...")
    content = await file.read()
    text = ""
    name = file.filename.lower()
    print("Uploading file:", file.filename)
    try:
        if name.endswith(".pdf"):
            text = extract_text_from_pdf(content)
        elif name.endswith(".docx") or name.endswith(".doc"):
            text = extract_text_from_docx(content)
        elif name.endswith((".png", ".jpg", ".jpeg", ".tiff")):
            text = extract_text_from_image(content)
        else:
            text = content.decode(errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"extract failed: {e}")

    if not text.strip():
        return {"error": "Uploaded file contains no readable text."}
    
    
    chunks = chunk_text(text)
    # embed chunks in batches
    batch_size = 16
    all_embeds = []
    metas = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        embeds = mistral_embed_batch(batch)
        all_embeds.extend(embeds)
        for j, chunk in enumerate(batch):
            metas.append({"filename": file.filename, "chunk_index": i+j, "text_snippet": chunk[:400]})
    # ensure DIM matches
    global DIM
    DIM = len(all_embeds[0])
    add_embeddings_to_index(all_embeds, metas)
    return {"status":"ok", "chunks_added": len(all_embeds)}

@app.post("/query")
def query(req: QueryReq):
    q_emb = mistral_embed_batch([req.question])[0]
    hits = search_index(q_emb, k=req.top_k)
    # assemble context
    context_texts = [h["metadata"]["text_snippet"] for h in hits]
    prompt = "Answer the question below using only the context. If the answer is not present, say 'I don't know'.\n\nContext:\n"
    for i,c in enumerate(context_texts):
        prompt += f"\n[{i}] {c}\n"
    prompt += f"\n\nQuestion: {req.question}\n\nAnswer:"
    resp = mistral_chat_completion(prompt)
    # extract assistant content (adapt if response schema differs)
    assistant = resp.get("choices",[{}])[0].get("message",{}).get("content") or resp.get("choices",[{}])[0].get("text")
    return {"answer": assistant, "sources": hits}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
