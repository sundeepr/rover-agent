"""
Parse and validate the raw text response from Claude into a structured
LLMNavigationResponse.  Claude may wrap JSON in markdown code fences;
this module strips them before parsing.
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from rover_agent.models import LLMNavigationResponse


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```", re.IGNORECASE)


def parse_navigation_response(raw_text: str) -> LLMNavigationResponse:
    """
    Extract JSON from *raw_text* (with or without markdown code fences) and
    validate it against the LLMNavigationResponse schema.

    Raises:
        ValueError: JSON cannot be decoded.
        pydantic.ValidationError: JSON is valid but doesn't match the schema.
    """
    # Strip code fences if present, otherwise use the raw text directly.
    match = _CODE_FENCE_RE.search(raw_text)
    json_str = match.group(1) if match else raw_text.strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON: {exc}\n"
            f"Raw response (first 500 chars): {raw_text[:500]!r}"
        ) from exc

    try:
        return LLMNavigationResponse.model_validate(data)
    except ValidationError:
        raise
