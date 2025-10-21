#!/bin/bash
# run.sh — start the Mistral RAG API

# load API key if saved
if [ -f ".env" ]; then
  source .env
fi

# check API key
if [ -z "$MISTRAL_API_KEY" ]; then
  echo "❌ No MISTRAL_API_KEY found. Run ./set_mistral_key.sh first."
  exit 1
fi

# activate virtual environment
if [ ! -d "venv" ]; then
  echo "❌ No virtual environment found. Run ./setup.sh first."
  exit 1
fi

source venv/bin/activate

# run FastAPI
echo "🚀 Starting Mistral RAG server..."
uvicorn app:app --reload
