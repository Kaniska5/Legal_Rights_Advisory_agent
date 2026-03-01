"""
Local embedding model loader - all-MiniLM-L6-v2 (HuggingFace).
Uses langchain_community + sentence-transformers; no paid APIs.
"""

from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Singleton HuggingFace embeddings (all-MiniLM-L6-v2)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
