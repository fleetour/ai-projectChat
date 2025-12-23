import os

# Basic configuration constants used across routers
FILES_DIR = "Projects"
CUSTOMER_ID = 1
VECTOR_SIZE = 1024

os.makedirs(FILES_DIR, exist_ok=True)

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-default-secret-key-change-this")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
TOKEN_EXPIRE_MINUTES = int(os.getenv("TOKEN_EXPIRE_MINUTES", 30))
