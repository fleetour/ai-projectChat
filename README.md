# 📘 AI Project Chat

This project provides a **FastAPI-based document question-answering service** using **Mistral AI**.  
You can upload files (PDF, DOCX, TXT, etc.), process them into text chunks, embed those chunks,  
and then query the model to answer questions based on the document’s content.

---

## 🚀 Features

- Upload PDF, DOCX, or TXT documents  
- Automatically extract and chunk document text  
- Generate embeddings for semantic search  
- Use Mistral AI for question answering  
- Simple REST API endpoints  
- Works locally or in a production environment  

---

## 🧠 Prerequisites

Before starting, make sure you have:

- **Python 3.10+** installed  
- **Mistral API key** (you can get one from [https://mistral.ai](https://mistral.ai))

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/AI-ProjectChat.git
cd AI-ProjectChat
```
## ⚙️ Quick Setup (Recommended)

You can set up everything automatically using the included scripts.

### 1️⃣ Run setup
```bash
./setup.sh
```
This script:

Creates a Python virtual environment

Installs all dependencies

Sets up your environment

### 2️⃣ Set your Mistral API key
```./set-mistralkey.sh```

This script:

Prompts you to enter your MISTRAL_API_KEY

Stores it safely in the .env file

### 3️⃣ Run the application
```./run.sh```


This script starts the FastAPI server with Uvicorn.
After running it, open your browser at:

👉 http://127.0.0.1:8000

👉 Swagger Docs: http://127.0.0.1:8000/docs

### 🧰 Manual Setup (Optional)

If you prefer manual setup, you can still do it step-by-step:

```python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# create a .env file
```echo "MISTRAL_API_KEY=your_mistral_api_key_here" > .env```

# start the server
```uvicorn app:app --reload```

📤 Upload a File

Send a POST request to:

```POST /upload```


Body (form-data):

file: your document (PDF, DOCX, or TXT)

Response:

```
{
  "message": "File uploaded successfully",
  "chunks_count": 42
}
```

❓ Ask a Question

Send a POST request to:

```POST /ask```


Body (JSON):
```
{
  "question": "What is the title of the file?"
}
```


Response:
```
{
  "answer": "AI in Smart Energy Management"
}
```

🧩 Project Structure
AI-ProjectChat/
│
├── app.py                    # Main FastAPI app
├── embeddings.py             # Embedding generation logic
├── utils.py                  # Helper functions
├── setup.sh                  # Script to install dependencies
├── set-mistralkey.sh         # Script to set your Mistral API key
├── run.sh                    # Script to start the FastAPI server
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (ignored in Git)
└── README.md                 # This file

🧾 License

This project is licensed under the MIT License.

👨‍💻 Author

Mohmad Ahmad
Founder & CTO — Cloudpioneer Solutions
🌐 cloudpioneer.de


---
