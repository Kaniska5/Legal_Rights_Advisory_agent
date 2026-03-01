"""
ChromaDB vector retrieval tool - semantic search over Indian law text.
"""

from app.vectorstore import get_vectorstore


def vector_retrieval_search(query: str, law_category: str | None = None, k: int = 5) -> str:
    """
    Semantic search over Indian legal documents (IPC, CrPC, Consumer Protection Act).
    Use for finding contextually relevant provisions based on the user's situation.

    Args:
        query: Natural language legal query or situation description.
        law_category: Optional filter - 'criminal_law' or 'consumer_protection'. If None, search all.
        k: Number of chunks to retrieve (default 5).

    Returns:
        Concatenated text of retrieved relevant passages with metadata.
    """
    vs = get_vectorstore()
    # Optional: filter by metadata if we store law_type in Chroma
    filter_dict = None
    if law_category:
        filter_dict = {"law_type": law_category}

    try:
        if filter_dict:
            docs = vs.similarity_search(query, k=k, filter=filter_dict)
        else:
            docs = vs.similarity_search(query, k=k)
    except Exception as e:
        return f"Vector search failed: {e}. Ensure ChromaDB is populated (run: python -m app.scripts.build_vectorstore)."

    if not docs:
        return "No relevant provisions found in the knowledge base. Use structured law search for key sections."

    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        source = meta.get("source", meta.get("act", "Unknown"))
        parts.append(f"[{i}] (Source: {source})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
