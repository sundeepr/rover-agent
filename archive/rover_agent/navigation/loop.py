"""
NavigationLoop — the main orchestrator.

Lifecycle:
  1. Capture image
  2. Ask LLM for next waypoint
  3. Convert pixel → bearing via pixel_to_motion
  4. Execute rotation + drive
  5. Update state
  6. Repeat until goal achieved, no_path, failed, or max_iterations reached
"""

from __future__ import annotations

import logging
import time

from rover_agent.config import CONFIG
from rover_agent.hardware.base import RoverHardware
from rover_agent.llm.base import NavigationLLMClient
from rover_agent.models import GoalStatus, NavigationState
from rover_agent.navigation.pixel_to_motion import pixel_to_bearing, should_rotate
from rover_agent.navigation.state import NavigationStateManager

logger = logging.getLogger(__name__)


class NavigationLoop:
    def __init__(
        self,
        hardware: RoverHardware,
        llm_client: NavigationLLMClient,
        user_goal: str,
        max_iterations: int = CONFIG.MAX_ITERATIONS,
        step_distance_meters: float = CONFIG.STEP_DISTANCE_METERS,
        min_rotation_degrees: float = CONFIG.MIN_ROTATION_DEGREES,
        camera_hfov_degrees: float = CONFIG.CAMERA_HFOV_DEGREES,
        image_width: int = CONFIG.IMAGE_WIDTH,
        step_delay_seconds: float = CONFIG.STEP_DELAY_SECONDS,
    ) -> None:
        self._hw = hardware
        self._llm = llm_client
        self._step_distance = step_distance_meters
        self._min_rotation = min_rotation_degrees
        self._hfov = camera_hfov_degrees
        self._image_width = image_width
        self._step_delay = step_delay_seconds
        self._state_mgr = NavigationStateManager(user_goal, max_iterations)

    def run(self) -> NavigationState:
        """Run the navigation loop until a terminal condition is reached."""
        goal = self._state_mgr.state.user_goal
        logger.info("Navigation started. Goal: %r", goal)

        try:
            while self._state_mgr.should_continue:
                self._step()
                if self._step_delay > 0:
                    time.sleep(self._step_delay)
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
            self._hw.motors.stop()
            self._state_mgr.state.goal_status = GoalStatus.FAILED
        except Exception:
            logger.exception("Unhandled error in navigation loop.")
            self._hw.motors.stop()
            raise

        self._log_summary()
        return self._state_mgr.state

    # ── Private ───────────────────────────────────────────────────────────────

    def _step(self) -> None:
        iteration = self._state_mgr.state.iteration_count + 1
        logger.info("─── Step %d ───", iteration)

        # 1. Capture image.
        image_bytes = self._hw.camera.capture_image()
        logger.debug("Image captured (%d bytes).", len(image_bytes))

        # 2. Query LLM.
        llm_response = self._llm.get_next_waypoint(
            image_bytes=image_bytes,
            user_goal=self._state_mgr.state.user_goal,
            navigation_history=self._state_mgr.get_history_summary(),
            position_summary=self._state_mgr.get_position_summary(),
        )
        logger.info("Status: %s | Confidence: %.0f%%", llm_response.goal_status, llm_response.confidence * 100)
        logger.info("Reasoning: %s", llm_response.reasoning)

        if llm_response.obstacles_detected:
            logger.warning("Obstacles: %s", ", ".join(llm_response.obstacles_detected))

        # 3. Record response (updates goal_status + iteration_count).
        self._state_mgr.record_llm_response(llm_response)

        # 4. Execute waypoint if still navigating.
        if llm_response.goal_status == GoalStatus.IN_PROGRESS and llm_response.waypoint:
            wp = llm_response.waypoint
            bearing = pixel_to_bearing(wp.target_pixel_x, self._image_width, self._hfov)
            logger.info(
                "Waypoint: %r @ pixel (%d, %d) → bearing %.1f°, drive %.2fm",
                wp.description, wp.target_pixel_x, wp.target_pixel_y,
                bearing, self._step_distance,
            )
            self._execute_waypoint(bearing)
            self._state_mgr.record_waypoint_executed(wp, bearing, self._step_distance)

        elif llm_response.goal_status == GoalStatus.ACHIEVED:
            logger.info("Goal ACHIEVED after %d steps.", self._state_mgr.state.iteration_count)

        elif llm_response.goal_status == GoalStatus.NO_PATH:
            logger.warning("LLM reports no valid path to goal.")

    def _execute_waypoint(self, bearing_degrees: float) -> None:
        """Rotate then drive forward one step."""
        if should_rotate(bearing_degrees, self._min_rotation):
            logger.debug("Rotating %.1f°", bearing_degrees)
            self._hw.motors.rotate(bearing_degrees)
        else:
            logger.debug("Bearing %.1f° below threshold — skipping rotation.", bearing_degrees)

        self._hw.motors.drive_forward(self._step_distance)

    def _log_summary(self) -> None:
        state = self._state_mgr.state
        logger.info(
            "Navigation complete. Status=%s | Steps=%d | Waypoints=%d | %s",
            state.goal_status,
            state.iteration_count,
            len(state.waypoints_visited),
            self._state_mgr.get_position_summary(),
        )
