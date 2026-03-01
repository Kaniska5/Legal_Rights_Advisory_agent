"""
Build ChromaDB vector store from law text in data/.
Run once (or after updating data): python -m app.scripts.build_vectorstore
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.embeddings import get_embedding_model
from app.vectorstore.chroma_setup import (
    COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    get_chroma_client,
)


def load_law_documents():
    """Load law text from data/law_documents.json or data/*.txt with metadata."""
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    json_path = data_dir / "law_documents.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("documents", [])

    # Fallback: simple chunks from structured act names (so Chroma has something)
    return [
        {
            "text": "Indian Penal Code Section 506: Criminal intimidation. Whoever commits criminal intimidation shall be punished with imprisonment up to 2 years or fine or both. If threat is to cause death or grievous hurt, imprisonment up to 7 years or fine or both.",
            "metadata": {"law_type": "criminal_law", "source": "IPC", "section": "IPC 506"},
        },
        {
            "text": "Indian Penal Code Section 323: Voluntarily causing hurt. Whoever voluntarily causes hurt shall be punished with imprisonment up to 1 year or fine up to Rs 1000 or both.",
            "metadata": {"law_type": "criminal_law", "source": "IPC", "section": "IPC 323"},
        },
        {
            "text": "Code of Criminal Procedure Section 154: FIR. Every information relating to commission of a cognizable offence shall be recorded by the officer in charge of a police station. Refusal to record FIR is illegal.",
            "metadata": {"law_type": "criminal_law", "source": "CrPC", "section": "CrPC 154"},
        },
        {
            "text": "Consumer Protection Act 2019: Consumer means any person who buys any goods or avails any service for consideration. Deficiency means any fault or shortcoming in the quality or performance of a service.",
            "metadata": {"law_type": "consumer_protection", "source": "CPA 2019"},
        },
    ]


def main():
    print("Loading embedding model...")
    embeddings = get_embedding_model()
    print("Loading law documents...")
    documents = load_law_documents()
    if not documents:
        print("No documents in data/law_documents.json. Add entries with 'text' and 'metadata' (law_type, source).")
        return

    texts = [d["text"] for d in documents]
    metadatas = [d.get("metadata", {}) for d in documents]

    print(f"Adding {len(texts)} chunks to ChromaDB at {CHROMA_PERSIST_DIR}...")
    client = get_chroma_client()

    # Delete existing collection so we can rebuild
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Chroma expects metadata values to be str, int, float or bool
    safe_metadatas = []
    for m in metadatas:
        safe = {}
        for k, v in m.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                safe[k] = v
            else:
                safe[k] = str(v)
        safe_metadatas.append(safe)

    # Embed in batches
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_metas = safe_metadatas[i : i + batch_size]
        emb = embeddings.embed_documents(batch_texts)
        ids = [f"doc_{i + j}" for j in range(len(batch_texts))]
        collection.add(ids=ids, embeddings=emb, documents=batch_texts, metadatas=batch_metas)

    print("ChromaDB build complete.")


if __name__ == "__main__":
    main()
