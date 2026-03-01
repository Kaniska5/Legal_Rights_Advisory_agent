"""
Structured response schema for legal advice API.
"""

import json
import re
from typing import Any

from pydantic import BaseModel, Field


class LegalAdviceResponse(BaseModel):
    """Structured JSON response for legal query."""

    is_crime: str = Field(description="'yes' or 'no'")
    law_category: str = Field(description="'criminal_law' or 'consumer_protection'")
    relevant_sections: list[str] = Field(default_factory=list, description="e.g. ['IPC 506', 'CrPC 154']")
    legal_explanation: str = Field(default="", description="Plain-language explanation")
    citizen_actions: list[str] = Field(default_factory=list, description="Recommended steps for the citizen")
    possible_punishment: str = Field(default="", description="If criminal, possible punishment; else N/A or applicable remedy")
    escalation_authority: str = Field(default="", description="Where to complain / which authority")
    disclaimer: str = Field(
        default="This is general information only, not legal advice. Consult a qualified lawyer for your situation.",
        description="Standard disclaimer",
    )


def _dict_to_response(data: dict[str, Any], fallback_explanation: str = "") -> LegalAdviceResponse:
    default_disclaimer = "This is general information only, not legal advice. Consult a qualified lawyer for your situation."
    return LegalAdviceResponse(
        is_crime=data.get("is_crime", "unknown"),
        law_category=data.get("law_category", "criminal_law"),
        relevant_sections=data.get("relevant_sections", []),
        legal_explanation=data.get("legal_explanation", fallback_explanation),
        citizen_actions=data.get("citizen_actions", []),
        possible_punishment=data.get("possible_punishment", ""),
        escalation_authority=data.get("escalation_authority", ""),
        disclaimer=data.get("disclaimer", default_disclaimer),
    )


def parse_agent_output_to_response(raw: str) -> LegalAdviceResponse:
    """
    Parse agent's text output into LegalAdviceResponse.
    Tries to extract JSON block first; otherwise builds from defaults.
    """
    # Try to find JSON in the response (e.g. inside ```json ... ``` or raw {...})
    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return _dict_to_response(data)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Try raw JSON object
    obj_match = re.search(r"\{[\s\S]*\}", raw)
    if obj_match:
        try:
            data = json.loads(obj_match.group(0))
            return _dict_to_response(data, fallback_explanation=raw[:500])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Fallback: use raw text as explanation
    return LegalAdviceResponse(
        is_crime="unknown",
        law_category="criminal_law",
        relevant_sections=[],
        legal_explanation=raw[:800] if raw else "No response generated.",
        citizen_actions=["Consult a lawyer with your specific facts."],
        possible_punishment="",
        escalation_authority="",
        disclaimer="This is general information only, not legal advice. Consult a qualified lawyer for your situation.",
    )
