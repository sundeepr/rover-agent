"""
GeminiStrategy — stateless Gemini vision strategy with a rolling frame buffer.

This is the "3-frame history" approach that was originally baked into
rover_agent.py._gemini_query(). Extracted here so alternative strategies
can be swapped in without touching the agent loop or web display.

Each call to run_query():
  1. Encodes the current frame to JPEG.
  2. Prepends up to frame_buffer_size prior frames for temporal context.
  3. Calls gemini_client.get_waypoint() with the structured prompts.
  4. Draws waypoint overlay and saves captures.
  5. Updates AgentState and drives the Roomba.
  6. Handles phase transitions (frame buffer clear + U-turn).
"""

import collections
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import gemini_client
import prompts
from navigation_strategy import AgentState, NavigationStrategy

log = logging.getLogger("rover.gemini_strategy")

# ── Overlay drawing ────────────────────────────────────────────────────────────

# Rank 1 = green (best), rank 2 = yellow, rank 3 = orange
_WP_COLORS = {1: (0, 255, 0), 2: (0, 220, 255), 3: (0, 140, 255)}


def draw_overlay(frame: np.ndarray, result: dict, step: int) -> np.ndarray:
    """Draw waypoint markers and mission info onto a copy of frame."""
    if not result:
        return frame

    out = frame.copy()

    for wp in result.get("waypoints", []):
        rank = wp.get("rank", 1)
        color = _WP_COLORS.get(rank, (255, 255, 255))
        x, y = int(wp["x"]), int(wp["y"])

        cv2.drawMarker(out, (x, y), color, cv2.MARKER_CROSS, markerSize=26, thickness=2)
        cv2.circle(out, (x, y), 14, color, 2)

        prob = wp.get("probability", 0)
        label = f"#{rank} {prob:.0%} {wp.get('description', '')[:30]}"
        cv2.putText(out, label, (x + 16, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    lines = [
        f"Step {step}  Phase {result.get('phase', '?')}  {result.get('goal_status', '')}",
        f"Confidence: {result.get('confidence', 0):.0%}",
    ]
    y_pos = 24
    for line in lines:
        cv2.putText(out, line, (11, y_pos + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, line, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y_pos += 22

    return out


# ── Strategy ───────────────────────────────────────────────────────────────────

class GeminiStrategy(NavigationStrategy):
    """
    Stateless Gemini vision strategy with a rolling frame buffer.

    Sends the last `frame_buffer_size` frames (oldest → newest) plus the
    current frame as temporal context to Gemini on every query.
    """

    def __init__(self, frame_buffer_size: int = 3):
        self._frame_buffer: collections.deque = collections.deque(maxlen=frame_buffer_size)
        self._buffer_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "gemini"

    def on_reset(self) -> None:
        with self._buffer_lock:
            self._frame_buffer.clear()
        log.info("GeminiStrategy frame buffer cleared")

    def run_query(
        self,
        state: AgentState,
        frame: np.ndarray,
        captures_dir: Path,
        roomba_ctrl,
    ) -> None:
        try:
            self._do_query(state, frame, captures_dir, roomba_ctrl)
        except Exception as e:
            with state.result_lock:
                state.llm_query_start = 0.0
            log.error("Strategy error: %s", e, exc_info=True)
        finally:
            state.query_in_flight.clear()

    def _do_query(
        self,
        state: AgentState,
        frame: np.ndarray,
        captures_dir: Path,
        roomba_ctrl,
    ) -> None:
        # 1. Encode current frame to JPEG
        _, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_bytes = jpeg_buf.tobytes()

        # 2. Build image list and update buffer atomically
        with self._buffer_lock:
            image_frames = list(self._frame_buffer) + [image_bytes]
            self._frame_buffer.append(image_bytes)

        # 3. Snapshot step / phase / trajectory under lock
        with state.result_lock:
            step = state.step
            phase = state.phase
            trajectory_snapshot = list(state.trajectory)

        user_prompt = prompts.build_user_prompt(phase, step, trajectory_snapshot)
        log.info("--- Step %d | Phase %d | %d image(s) sent ---", step, phase, len(image_frames))

        cv2.imwrite(str(captures_dir / f"step_{step:04d}_raw.jpg"), frame)

        t0 = time.time()
        with state.result_lock:
            state.llm_query_start = t0

        # 4. Query Gemini
        result = gemini_client.get_waypoint(
            image_frames, prompts.SYSTEM_PROMPT, user_prompt
        )
        elapsed = time.time() - t0

        # 5. Annotate frame and update llm_frame
        annotated = draw_overlay(frame, result, step)
        with state.llm_lock:
            state.llm_frame = annotated.copy()
        cv2.imwrite(str(captures_dir / f"step_{step:04d}_annotated.jpg"), annotated)

        # 6. Log result
        status     = result.get("goal_status", "unknown")
        nav_mode   = result.get("navigation_mode", "unknown")
        reasoning  = result.get("reasoning", "")
        confidence = result.get("confidence", 0)
        top = next((w for w in result.get("waypoints", []) if w.get("rank") == 1), None)

        log.info("Status     : %s", status)
        log.info("Nav mode   : %s", nav_mode)
        log.info("Confidence : %.0f%%", confidence * 100)
        log.info("Reasoning  : %s", reasoning)
        log.info("Response time: %.2fs", elapsed)
        for wp in result.get("waypoints", []):
            log.info("Waypoint #%d: (%d, %d) p=%.0f%% — %s",
                     wp.get("rank"), wp["x"], wp["y"],
                     wp.get("probability", 0) * 100,
                     wp.get("description", ""))

        # 7. Update shared state
        with state.result_lock:
            state.latest_result = result
            state.llm_query_start = 0.0
            state.llm_response_s = elapsed
            if top:
                state.trajectory.append({
                    "step": step, "phase": phase,
                    "x": top["x"], "y": top["y"],
                    "description": top.get("description", ""),
                })

        # 8. Drive Roomba
        if roomba_ctrl and status == "in_progress" and top:
            try:
                roomba_ctrl.navigate_to_waypoint(top, nav_mode)
            except Exception as e:
                log.error("Roomba drive error: %s", e, exc_info=True)

        # 9. Phase transitions
        if status == "phase1_complete":
            with self._buffer_lock:
                self._frame_buffer.clear()
            with state.result_lock:
                state.phase = 2
            log.info(">> Phase 1 complete — executing U-turn")
            if roomba_ctrl:
                try:
                    roomba_ctrl.uturn()
                except Exception as e:
                    log.error("U-turn error: %s", e, exc_info=True)
        elif status == "mission_complete":
            log.info(">> Mission complete after %d steps!", step)
