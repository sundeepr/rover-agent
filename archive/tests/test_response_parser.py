import pytest
from pydantic import ValidationError

from rover_agent.llm.response_parser import parse_navigation_response
from rover_agent.models import GoalStatus


# ── Helpers ───────────────────────────────────────────────────────────────────

IN_PROGRESS_JSON = """{
    "goal_status": "in_progress",
    "reasoning": "I can see a red cone ahead to the right.",
    "waypoint": {
        "target_pixel_x": 400,
        "target_pixel_y": 350,
        "description": "red cone"
    },
    "obstacles_detected": [],
    "confidence": 0.85
}"""

ACHIEVED_JSON = """{
    "goal_status": "achieved",
    "reasoning": "The red cone is directly in front of the rover.",
    "waypoint": null,
    "obstacles_detected": [],
    "confidence": 0.95
}"""


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_parse_plain_json_in_progress():
    result = parse_navigation_response(IN_PROGRESS_JSON)
    assert result.goal_status == GoalStatus.IN_PROGRESS
    assert result.waypoint is not None
    assert result.waypoint.target_pixel_x == 400
    assert result.waypoint.target_pixel_y == 350
    assert result.waypoint.description == "red cone"
    assert result.confidence == pytest.approx(0.85)


def test_parse_plain_json_achieved():
    result = parse_navigation_response(ACHIEVED_JSON)
    assert result.goal_status == GoalStatus.ACHIEVED
    assert result.waypoint is None


def test_parse_json_in_markdown_fences():
    wrapped = f"```json\n{IN_PROGRESS_JSON}\n```"
    result = parse_navigation_response(wrapped)
    assert result.goal_status == GoalStatus.IN_PROGRESS


def test_parse_json_in_generic_fences():
    wrapped = f"```\n{IN_PROGRESS_JSON}\n```"
    result = parse_navigation_response(wrapped)
    assert result.waypoint is not None


def test_parse_with_leading_text():
    """Claude sometimes adds a sentence before the JSON block."""
    text = f"Here is my analysis:\n```json\n{IN_PROGRESS_JSON}\n```"
    result = parse_navigation_response(text)
    assert result.goal_status == GoalStatus.IN_PROGRESS


def test_parse_with_obstacles():
    json_str = """{
        "goal_status": "in_progress",
        "reasoning": "Cone ahead but rock in the way.",
        "waypoint": {"target_pixel_x": 300, "target_pixel_y": 400, "description": "gap"},
        "obstacles_detected": ["large rock on left", "puddle on right"],
        "confidence": 0.6
    }"""
    result = parse_navigation_response(json_str)
    assert len(result.obstacles_detected) == 2
    assert "large rock on left" in result.obstacles_detected


def test_parse_no_path():
    json_str = """{
        "goal_status": "no_path",
        "reasoning": "Wall blocking all routes.",
        "waypoint": null,
        "obstacles_detected": ["wall"],
        "confidence": 0.99
    }"""
    result = parse_navigation_response(json_str)
    assert result.goal_status == GoalStatus.NO_PATH
    assert result.waypoint is None


def test_invalid_json_raises_value_error():
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_navigation_response("this is not json at all")


def test_schema_mismatch_raises_validation_error():
    """goal_status with an unknown value should fail Pydantic validation."""
    bad_json = """{
        "goal_status": "unknown_status",
        "reasoning": "test",
        "waypoint": null,
        "obstacles_detected": [],
        "confidence": 0.5
    }"""
    with pytest.raises(ValidationError):
        parse_navigation_response(bad_json)


def test_confidence_out_of_range_raises_validation_error():
    bad_json = """{
        "goal_status": "in_progress",
        "reasoning": "test",
        "waypoint": {"target_pixel_x": 320, "target_pixel_y": 240, "description": "x"},
        "obstacles_detected": [],
        "confidence": 1.5
    }"""
    with pytest.raises(ValidationError):
        parse_navigation_response(bad_json)
