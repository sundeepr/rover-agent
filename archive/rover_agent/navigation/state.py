"""
NavigationStateManager — owns and mutates the NavigationState throughout a session.

All position tracking uses simple dead-reckoning: the rover records each
rotation + forward drive and accumulates the estimated (x, y, heading).
This is not GPS-accurate but is good enough for short-range goal checking.
"""

from __future__ import annotations

import math

from rover_agent.models import GoalStatus, LLMNavigationResponse, NavigationState, Waypoint


class NavigationStateManager:
    def __init__(self, user_goal: str, max_iterations: int = 50) -> None:
        self.state = NavigationState(
            user_goal=user_goal,
            max_iterations=max_iterations,
        )

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def should_continue(self) -> bool:
        return (
            self.state.goal_status == GoalStatus.IN_PROGRESS
            and self.state.iteration_count < self.state.max_iterations
        )

    def get_history_summary(self) -> list[str]:
        """Return a human-readable list of waypoints visited so far."""
        return [
            f"Moved toward '{wp.description}' "
            f"(pixel {wp.target_pixel_x}, {wp.target_pixel_y})"
            for wp in self.state.waypoints_visited
        ]

    def get_position_summary(self) -> str:
        p = self.state.current_position
        return (
            f"{p.x_meters:+.1f}m east, {p.y_meters:+.1f}m north of start; "
            f"heading {p.heading_degrees:.0f}°"
        )

    # ── Mutations ─────────────────────────────────────────────────────────────

    def record_llm_response(self, response: LLMNavigationResponse) -> None:
        """Store the LLM response and advance the iteration counter."""
        self.state.llm_responses.append(response)
        self.state.goal_status = response.goal_status
        self.state.iteration_count += 1

    def record_waypoint_executed(
        self,
        waypoint: Waypoint,
        bearing_degrees: float,
        step_distance_meters: float,
    ) -> None:
        """
        Update dead-reckoning position after the rover executes a waypoint.

        Args:
            waypoint:             The waypoint that was navigated to.
            bearing_degrees:      Rotation applied (degrees, + = clockwise).
            step_distance_meters: Distance driven forward.
        """
        pos = self.state.current_position

        # Update heading first.
        pos.heading_degrees = (pos.heading_degrees + bearing_degrees) % 360

        # Project the forward drive onto the x-y plane.
        heading_rad = math.radians(pos.heading_degrees)
        pos.x_meters += step_distance_meters * math.sin(heading_rad)
        pos.y_meters += step_distance_meters * math.cos(heading_rad)

        self.state.waypoints_visited.append(waypoint)
