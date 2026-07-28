"""
Central application configuration.
Reads environment variables (with sane defaults) using pydantic-settings.
"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- General ---
    APP_NAME: str = "DocuMind - AI Research & Knowledge Assistant"
    APP_VERSION: str = "1.0.0"
    ENV: str = "development"

    # --- Storage ---
    RAW_DOCS_DIR: str = "./data/raw_documents"
    VECTOR_DB_DIR: str = "./data/vector_db"
    DATASET_DIR: str = "./data/dataset"
    SQLITE_DB_PATH: str = "./data/app.db"

    # --- ML / Classifier ---
    MODEL_PATH: str = "./models/tf_classifier.h5"
    TOKENIZER_PATH: str = "./models/tokenizer.pickle"
    LABELS_PATH: str = "./models/labels.json"

    # --- Embeddings ---
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- LLM ---
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"
    LLM_MODEL: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.0

    # --- Chunking ---
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # --- Retrieval ---
    TOP_K: int = 4

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Ensure required directories exist at import time
for _dir in [settings.RAW_DOCS_DIR, settings.VECTOR_DB_DIR, settings.DATASET_DIR, "./models"]:
    os.makedirs(_dir, exist_ok=True)
