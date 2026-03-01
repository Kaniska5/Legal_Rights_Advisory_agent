"""
ChromaDB vector store setup - persisted locally.
Uses app embeddings for consistency.
"""

import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma

from app.embeddings import get_embedding_model

# Persist under project root / chroma_db
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_db"))
COLLECTION_NAME = "indian_legal_laws"


def get_chroma_client() -> chromadb.PersistentClient:
    """Persistent ChromaDB client (local disk)."""
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def get_vectorstore() -> Chroma:
    """LangChain Chroma vectorstore using local embeddings."""
    client = get_chroma_client()
    embeddings = get_embedding_model()
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
