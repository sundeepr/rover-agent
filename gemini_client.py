"""
Simple Gemini vision client for rover navigation.
Reads GEMINI_API_KEY from the environment.
"""

import json
import logging
import os
import re

from google import genai
from google.genai import types

MODEL = "gemini-3.1-pro-preview"

log = logging.getLogger("rover.gemini")

_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Explicit schema so Gemini always produces complete, valid JSON
_WAYPOINT_SCHEMA = {
    "type": "object",
    "properties": {
        "rank":        {"type": "integer"},
        "x":           {"type": "integer"},
        "y":           {"type": "integer"},
        "description": {"type": "string"},
        "probability": {"type": "number"},
    },
    "required": ["rank", "x", "y", "description", "probability"],
}

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "phase":       {"type": "integer"},
        "goal_status": {"type": "string",
                        "enum": ["in_progress", "phase1_complete",
                                 "mission_complete", "no_path"]},
        "reasoning":   {"type": "string"},
        "waypoints":   {"type": "array", "items": _WAYPOINT_SCHEMA},
        "confidence":  {"type": "number"},
    },
    "required": ["phase", "goal_status", "reasoning", "waypoints", "confidence"],
}


def get_waypoint(image_frames: list[bytes], system_prompt: str, user_prompt: str) -> dict:
    """
    Send one or more images + prompts to Gemini and return the parsed JSON response dict.

    image_frames: JPEG bytes in chronological order, oldest first, newest last.
    """
    log.debug("Sending request to %s (%d frames, newest %d bytes)",
              MODEL, len(image_frames), len(image_frames[-1]))

    contents = [
        types.Part.from_bytes(data=frame, mime_type="image/jpeg")
        for frame in image_frames
    ]
    contents.append(types.Part.from_text(text=user_prompt))

    response = _client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=4096,
            temperature=0.3,
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
        ),
    )

    raw = response.text or ""
    log.debug("Raw response (%d chars): %s", len(raw), raw[:500])

    # Strip markdown fences if present
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if match:
        log.debug("Stripped markdown fences from response")
        raw = match.group(1)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        log.error("JSON parse failed: %s", e)
        log.error("Raw response was: %s", raw)
        raise
