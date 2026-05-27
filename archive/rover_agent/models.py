from __future__ import annotations

import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class GoalStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    ACHIEVED = "achieved"
    NO_PATH = "no_path"
    FAILED = "failed"


class Waypoint(BaseModel):
    """A navigation target expressed as pixel coordinates in the captured image."""

    target_pixel_x: int = Field(description="Horizontal pixel (0 = left edge)")
    target_pixel_y: int = Field(description="Vertical pixel (0 = top edge)")
    description: str = Field(description="Human-readable description of the target landmark")


class LLMNavigationResponse(BaseModel):
    """Structured response from Claude for each navigation step."""

    goal_status: GoalStatus
    reasoning: str = Field(description="Claude's scene analysis and decision rationale")
    waypoint: Optional[Waypoint] = Field(
        default=None,
        description="Next waypoint target; None when goal_status is not IN_PROGRESS",
    )
    obstacles_detected: list[str] = Field(
        default_factory=list,
        description="Obstacles or hazards spotted in the image",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this decision")


class RoverPosition(BaseModel):
    """Accumulated dead-reckoning position relative to the start point."""

    x_meters: float = 0.0       # East is positive
    y_meters: float = 0.0       # North is positive
    heading_degrees: float = 0.0  # 0 = north, 90 = east


class NavigationState(BaseModel):
    """Complete state of a navigation session."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_goal: str
    goal_status: GoalStatus = GoalStatus.IN_PROGRESS
    current_position: RoverPosition = Field(default_factory=RoverPosition)
    waypoints_visited: list[Waypoint] = Field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 50
    llm_responses: list[LLMNavigationResponse] = Field(default_factory=list)
