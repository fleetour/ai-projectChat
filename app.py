from fastapi import APIRouter, FastAPI, Form, HTTPException, Response, UploadFile, File
import os
from fastapi.middleware.cors import CORSMiddleware
import logging


# Create router

# === CONFIGURATION ===
FILES_DIR = "Projects"
CUSTOMER_ID = 1
VECTOR_SIZE = 1024  # depends on your Mistral model

router = APIRouter(prefix="/templates", tags=["templates"])

os.makedirs(FILES_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from config import FILES_DIR

# Import routers
from routes.templates import router as templates_router
from routes.projects import router as projects_router
from routes.docs import router as docs_router
from routes.conversations import router as conversation_router


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="Local Qdrant RAG with Cloud Embeddings")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", 
                   "https://projectgpt-ed262.web.app",
                   "https://projectour-49ace.web.app",
                   "https://dev.projectsgpt.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Ensure Projects dir exists (kept for compatibility)
import os
os.makedirs(FILES_DIR, exist_ok=True)

# Register routers
app.include_router(templates_router)
app.include_router(projects_router)
app.include_router(docs_router)
app.include_router(conversation_router)
