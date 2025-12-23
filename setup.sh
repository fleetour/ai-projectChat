#!/bin/bash
# setup.sh — initial setup for mistral-rag project

echo "🔧 Setting up Mistral RAG environment..."

# create venv if not exists
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

# activate venv
source venv/bin/activate

# upgrade pip
pip install --upgrade pip

# install dependencies
#pip install fastapi uvicorn requests faiss-cpu python-docx pdfminer.six pillow pytesseract

# Install requirements
pip install -r requirements.txt

echo "✅ Setup complete. You can now run ./run.sh to start the server."

