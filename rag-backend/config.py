import os
from dotenv import load_dotenv

# Loads variables from a .env file in the project root into the process
# environment. Safe to call even if .env doesn't exist (e.g. in production
# where real env vars are set another way).
load_dotenv()

class Settings:

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "documents")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


settings = Settings()