# Legal Rights Advisory System (India)

An **agentic AI backend** for Indian citizens covering **Criminal Law** and **Consumer Protection**. Uses a LangChain tool-calling agent with hybrid retrieval (structured law DB + ChromaDB) and returns structured JSON advice.

## Jurisdiction & Domain

- **Jurisdiction:** India  
- **Laws:** Indian Penal Code (IPC), Code of Criminal Procedure (CrPC), Consumer Protection Act 2019  

## Tech Stack

- **Python 3.10+**
- **FastAPI** – REST API
- **LangChain** – Tool-calling agent
- **ChromaDB** – Vector store (persisted locally)
- **HuggingFace sentence-transformers** – `all-MiniLM-L6-v2` (local embeddings)
- **Fully local** – No paid APIs (Ollama or local HuggingFace model for LLM)

## Architecture

```
User → FastAPI → LangChain Agent → Tool Selection
                                        ├── Structured Law DB Tool
                                        └── ChromaDB Vector Retrieval Tool
                                    → LLM → Structured JSON
                                    → FastAPI → Formatted Response
```

## Setup & Run

Run all commands from the **project root** (`legal_ai_agent/`).

### 1. Create virtual environment

```bash
cd legal_ai_agent
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama (for local LLM)

Download from [ollama.ai](https://ollama.ai), install, then:

```bash
ollama pull llama3.2
```

(Or `ollama pull mistral` / another model. Set `OLLAMA_MODEL` if you use a different name.)

### 4. Populate ChromaDB (first run)

Builds the vector index from `data/law_documents.json`:

```bash
python -m app.scripts.build_vectorstore
```

### 5. Start the API server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Open the frontend

In a browser open: **http://localhost:8000**

Enter a legal question or situation and click **Get legal guidance** to see structured advice.

## API

- **GET /** – Serves the frontend (HTML).
- **POST /api/query** – Submit a legal query.

Request body:

```json
{ "query": "Someone threatened me with a knife. What are my rights?" }
```

Response: structured JSON with `law_category`, `relevant_sections`, `legal_explanation`, `citizen_actions`, etc.

## Response Schema

```json
{
  "is_crime": "yes/no",
  "law_category": "criminal_law | consumer_protection",
  "relevant_sections": ["IPC 506", "CrPC 154"],
  "legal_explanation": "...",
  "citizen_actions": ["..."],
  "possible_punishment": "...",
  "escalation_authority": "...",
  "disclaimer": "..."
}
```

## Project Structure

```
legal_ai_agent/
├── app/
│   ├── main.py           # FastAPI app + routes
│   ├── agent.py          # LangChain agent
│   ├── tools/            # Agent tools
│   ├── vectorstore/      # ChromaDB setup
│   ├── structured_db/    # Law metadata JSON
│   ├── embeddings/       # Embedding loader
│   ├── utils/            # Helpers
│   └── scripts/          # build_vectorstore, etc.
├── data/                 # Law text for indexing
├── chroma_db/            # Persisted ChromaDB (created on first run)
├── requirements.txt
└── README.md
```


## Disclaimer

This system is for **informational purposes only** and does not constitute legal advice. Users should consult a qualified lawyer for their specific situation.
