"""
FastAPI application for Legal Rights Advisory System.
Serves API and frontend; delegates queries to LangChain agent.
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent import run_query
from app.utils import LegalAdviceResponse, parse_agent_output_to_response

app = FastAPI(
    title="Legal Rights Advisory API",
    description="AI-powered legal rights advisory for Indian citizens (Criminal Law & Consumer Protection)",
    version="1.0.0",
)

# Allow frontend to call API from same origin or different port (e.g. during dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    success: bool = True
    data: LegalAdviceResponse
    raw_agent_output: str | None = None  # optional, for debugging


# Paths for static and index
BASE_DIR = Path(__file__).resolve().parent.parent
static_dir = BASE_DIR / "static"
index_path = static_dir / "index.html"

# Load index.html once at startup so root route always works
_INDEX_HTML: str | None = None


def _get_index_html() -> str:
    global _INDEX_HTML
    if _INDEX_HTML is not None:
        return _INDEX_HTML
    if index_path.exists():
        _INDEX_HTML = index_path.read_text(encoding="utf-8")
        return _INDEX_HTML
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Legal Rights Advisory</title></head><body>"
        "<h1>Legal Rights Advisory</h1><p>Frontend not found. Put index.html in <code>static/</code>.</p>"
        "<p>API: POST /api/query with body {\"query\": \"your question\"}</p></body></html>"
    )


@app.get("/")
async def root():
    """Serve the frontend."""
    return HTMLResponse(_get_index_html())


# Mount static after defining / so / is not shadowed
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.post("/api/query", response_model=QueryResponse)
async def api_query(request: QueryRequest):
    """
    Submit a legal query. Returns structured advice (law category, sections, actions, etc.).
    """
    query = (request.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    try:
        raw_output = run_query(query)
        response = parse_agent_output_to_response(raw_output)
        return QueryResponse(
            success=True,
            data=response,
            raw_agent_output=raw_output if os.environ.get("DEBUG_RESPONSE") else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "legal-advisory"}
