"""
Structured law database tool - returns key sections from IPC, CrPC, CPA by category.
"""

from app.structured_db import get_sections_by_category


def structured_law_search(law_category: str) -> str:
    """
    Search the structured law database for relevant sections.
    Use when you need exact section metadata: act name, section number, title, punishment, bailability.

    Args:
        law_category: One of 'criminal_law' or 'consumer_protection'

    Returns:
        JSON string of relevant sections with act, section, title, summary, punishment, bailable.
    """
    if law_category not in ("criminal_law", "consumer_protection"):
        return f"Invalid category. Use 'criminal_law' or 'consumer_protection'. Got: {law_category}"

    sections = get_sections_by_category(law_category)
    import json
    return json.dumps(sections, indent=2, ensure_ascii=False)
