"""
Structured law metadata loader - key sections from IPC, CrPC, Consumer Protection Act.
"""

import json
from pathlib import Path
from typing import Any

STRUCTURED_DB_PATH = Path(__file__).resolve().parent / "laws_metadata.json"


def load_structured_laws() -> dict[str, Any]:
    """Load structured law metadata JSON."""
    with open(STRUCTURED_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_sections_by_category(category: str) -> list[dict[str, Any]]:
    """Get sections for a law category: criminal_law or consumer_protection."""
    data = load_structured_laws()
    return data.get("categories", {}).get(category, [])
