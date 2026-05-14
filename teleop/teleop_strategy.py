"""
TeleopStrategy — NavigationStrategy for human teleoperation data collection.

The operator clicks waypoints on the web UI camera feed; this strategy
steers the rover toward them while recording every frame to the dataset
format in teleop/dataset_recorder.py.

Joystick override (via control_server.py WebSocket) always takes priority
over waypoint following. The rover resumes waypoint following automatically
when the joystick is released.

Usage
─────
    python rover_agent.py --strategy teleop --rover atlas \\
        --atlas-port /dev/ttyACM0 --interval 0.1 \\
        --dataset-dir ./dataset \\
        --teleop-instruction "drive between the crop rows"
"""

import logging
import math
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from navigation_strategy import AgentState, NavigationStrategy
from teleop.dataset_recorder import DatasetRecorder

log = logging.getLogger("rover.teleop")

# Pixels per degree → used for bearing from normalised x
_IMAGE_WIDTH     = 640
_CAMERA_HFOV_DEG = 138.0

# Waypoint dead-band: if bearing < this, drive straight
_BEARING_DEAD_BAND_DEG = 5.0

# Forward speed while waypoint-following
_FOLLOW_VEL_MM_S = 80

# Spin speed used when turning to face a waypoint
_SPIN_VEL_MM_S   = 60


def _pixel_to_bearing(pixel_x: int) -> float:
    """Horizontal pixel offset → bearing degrees (positive = right)."""
    offset   = pixel_x - _IMAGE_WIDTH / 2
    fraction = offset / _IMAGE_WIDTH
    return fraction * _CAMERA_HFOV_DEG


# Minimum steps the rover must be aligned before advancing to next waypoint
_WAYPOINT_ADVANCE_STEPS = 5


class TeleopStrategy(NavigationStrategy):

    def __init__(
        self,
        dataset_dir: str  = "./dataset",
        instruction: str  = "",
        fps: int          = 10,
    ):
        self._dataset_dir   = dataset_dir
        self._instruction   = instruction
        self._fps           = fps
        self._recorder: Optional[DatasetRecorder] = None
        self._last_vel      = 0
        self._last_radius   = 0x8000
        self._executing     = False   # True while following waypoints
        self._aligned_steps = 0       # steps spent aligned to current waypoint
        self._active_wp_idx = -1      # index of current waypoint (for log dedup)

    @property
    def name(self) -> str:
        return "teleop"

    def on_reset(self) -> None:
        if self._recorder is not None:
            self._recorder.close()
            self._recorder = None
        self._executing = False

    def set_goal(self, goal: str) -> None:
        if goal:
            self._instruction = goal

    def run_query(
        self,
        state: AgentState,
        frame: np.ndarray,
        captures_dir: Path,
        rover_ctrl,
    ) -> None:
        t0 = time.time()
        try:
            # ── 1. Handle episode commands ─────────────────────────────────
            cmd  = state.teleop_episode_cmd
            meta = state.teleop_episode_meta
            if cmd:
                state.teleop_episode_cmd = ""
                if cmd == "start":
                    if self._recorder is not None:
                        self._recorder.close()
                    self._recorder = DatasetRecorder(self._dataset_dir, meta)
                    log.info("Episode started: %s", self._recorder.episode_id)
                elif cmd == "stop":
                    if self._recorder is not None:
                        path = self._recorder.close()
                        log.info("Episode saved: %s  (%d frames)",
                                 path, self._recorder.step)
                        self._recorder = None
                elif cmd == "discard":
                    if self._recorder is not None:
                        self._recorder.discard()
                        log.info("Episode discarded")
                        self._recorder = None
                elif cmd == "execute":
                    if state.teleop_waypoints:
                        self._executing     = True
                        self._aligned_steps = 0
                        self._active_wp_idx = 0
                        log.info("Executing %d waypoints", len(state.teleop_waypoints))
                    else:
                        log.warning("Execute pressed but no waypoints set")

            # ── 2. Determine drive command ─────────────────────────────────
            operator_active = (
                state.operator_control is not None
                and state.operator_until > time.time()
            )

            if not state.paused.is_set():
                if operator_active:
                    oc  = state.operator_control
                    fwd = oc.get("fwd", 0)
                    trn = oc.get("turn", 0)
                    vel, radius = _joy_to_drive(fwd, trn, _FOLLOW_VEL_MM_S)
                    if rover_ctrl:
                        rover_ctrl.drive_raw(vel, radius)
                    self._last_vel    = vel
                    self._last_radius = radius

                elif self._executing and state.teleop_waypoints:
                    nx, ny  = state.teleop_waypoints[0]
                    pixel_x = int(nx * _IMAGE_WIDTH)
                    bearing = _pixel_to_bearing(pixel_x)

                    if abs(bearing) > _BEARING_DEAD_BAND_DEG:
                        radius = -1 if bearing > 0 else 1
                        vel    = _SPIN_VEL_MM_S
                        self._aligned_steps = 0
                    else:
                        vel    = _FOLLOW_VEL_MM_S
                        radius = 0x8000
                        self._aligned_steps += 1

                    if rover_ctrl:
                        rover_ctrl.drive_raw(vel, radius)
                    self._last_vel    = vel
                    self._last_radius = radius

                    # Log only when waypoint changes
                    if self._active_wp_idx != id(state.teleop_waypoints[0]):
                        self._active_wp_idx = id(state.teleop_waypoints[0])
                        log.info("Waypoint [%.2f, %.2f]  bearing=%.1f°", nx, ny, bearing)

                    # Advance to next waypoint once sufficiently aligned
                    if self._aligned_steps >= _WAYPOINT_ADVANCE_STEPS:
                        state.teleop_waypoints = state.teleop_waypoints[1:]
                        self._aligned_steps = 0
                        if state.teleop_waypoints:
                            log.info("Advanced to next waypoint (%d remaining)",
                                     len(state.teleop_waypoints))
                        else:
                            # All waypoints done — stop and reset
                            self._executing = False
                            if rover_ctrl:
                                rover_ctrl.stop()
                            log.info("All waypoints executed — ready for next set")

                else:
                    if rover_ctrl:
                        rover_ctrl.stop()
                    self._last_vel    = 0
                    self._last_radius = 0x8000

            # ── 3. Recording loop at target FPS ────────────────────────────
            if self._recorder is not None:
                oc        = state.operator_control or {}
                joy_fwd   = oc.get("fwd", 0) / 100.0
                joy_turn  = oc.get("turn", 0) / 100.0
                instr     = state.teleop_episode_meta.get(
                    "instruction", self._instruction)
                frame_interval = 1.0 / self._fps
                elapsed        = time.time() - t0

                # Write first frame immediately, then sleep to hit target FPS
                self._recorder.write_frame(
                    frame         = frame,
                    instruction   = instr,
                    vel_mm_s      = self._last_vel,
                    radius_mm     = self._last_radius,
                    joy_fwd       = joy_fwd,
                    joy_turn      = joy_turn,
                    waypoints_norm= list(state.teleop_waypoints),
                )

                remaining = frame_interval - (time.time() - t0)
                if remaining > 0:
                    time.sleep(remaining)

            # ── 4. Annotate display frame ──────────────────────────────────
            display = _annotate(frame, state.teleop_waypoints,
                                self._recorder, self._last_vel,
                                self._last_radius)
            with state.llm_lock:
                state.llm_frame = display

            with state.result_lock:
                state.llm_response_s  = time.time() - t0
                state.llm_query_start = 0.0
                state.latest_result   = {
                    "strategy":   self.name,
                    "recording":  self._recorder is not None,
                    "episode_id": self._recorder.episode_id if self._recorder else None,
                    "frames":     self._recorder.step if self._recorder else 0,
                    "vel_mm_s":   self._last_vel,
                }

        except Exception:
            log.exception("TeleopStrategy error")
        finally:
            state.query_in_flight.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _joy_to_drive(fwd: int, turn: int, max_vel: int) -> tuple[int, int]:
    """Mirror of control_server._joy_to_drive."""
    MAX_ANG_RAD_S = 0.5
    vel_mm_s = fwd * max_vel // 100
    if turn == 0:
        return vel_mm_s, 0x8000
    ang_rad_s = (turn / 100.0) * MAX_ANG_RAD_S
    if vel_mm_s == 0:
        spin_vel = int(abs(ang_rad_s) / MAX_ANG_RAD_S * max_vel)
        radius   = -1 if turn > 0 else 1
        return spin_vel, radius
    radius = int(math.copysign(
        min(32767, abs(vel_mm_s / ang_rad_s)), -ang_rad_s))
    return vel_mm_s, radius


def _annotate(
    frame: np.ndarray,
    waypoints: list,
    recorder: Optional[DatasetRecorder],
    vel: int,
    radius: int,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Draw waypoints
    pts = []
    for i, (nx, ny) in enumerate(waypoints):
        px = int(nx * w)
        py = int(ny * h)
        pts.append((px, py))
        colour = (0, 255, 100) if i == 0 else (0, 180, 60)
        cv2.circle(out, (px, py), 10, colour, -1)
        cv2.putText(out, str(i + 1), (px + 12, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

    for a, b in zip(pts, pts[1:]):
        cv2.line(out, a, b, (0, 160, 50), 2)

    # HUD
    r_str  = "straight" if radius == 0x8000 else f"r={radius}mm"
    status = "● REC" if recorder else "○ idle"
    frames = recorder.step if recorder else 0
    hud = [
        f"teleop  {status}  {frames} frames",
        f"vel={vel}mm/s  {r_str}",
        f"waypoints: {len(waypoints)}",
    ]
    for i, txt in enumerate(hud):
        colour = (0, 60, 255) if recorder else (0, 230, 255)
        cv2.putText(out, txt, (12, 36 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2, cv2.LINE_AA)

    return out
