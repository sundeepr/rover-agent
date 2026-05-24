"""
CloudOmniVLAStrategy — rover-side WebSocket client for cloud OmniVLA inference.

Sends one camera frame per query step to omnivla_cloud_server.py running on
a cloud GPU.  Receives 8 waypoints back and converts them to a drive command
locally.  No ML inference happens on the rover.

Architecture
────────────
  rover_agent (1 Hz) → run_query() → JPEG over WebSocket → cloud server
                     ←   waypoints (8×4)  ←

  A dedicated asyncio event loop runs on a daemon thread and owns the
  WebSocket connection.  run_query() submits send/receive coroutines to that
  loop via asyncio.run_coroutine_threadsafe(), keeping the rover's main
  threads free of async machinery.

  Reconnection is automatic with exponential back-off (3 s → 30 s cap).
  While disconnected the rover stops (run_query returns without driving).

State machine
─────────────
  CONNECTING   — WebSocket not yet established (or reconnecting after drop)
  WAITING_GOAL — connected to server but no goal text set yet
  NAVIGATING   — goal set, sending frames, applying waypoints to rover

Usage
─────
    python rover_agent.py --strategy cloud_omnivla \\
        --cloud-server ws://192.168.1.100:8765 \\
        --goal "Follow the crop row" --interval 1.0 \\
        --rover atlas --atlas-port /dev/ttyACM0
"""

import asyncio
import base64
import io
import json
import logging
import threading
import time
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

from navigation_strategy import AgentState, NavigationStrategy
from omnivla_strategy import (
    METRIC_SPACING, WAYPOINT_IDX,
    _waypoint_to_drive, _annotate,
)

log = logging.getLogger("rover.cloud_omnivla")

# ── Guard rail detection (red HSV, same logic as camera_calibrate.py) ────────
_RAIL_H_LO1, _RAIL_H_HI1 =   0,  12   # lower red hue range
_RAIL_H_LO2, _RAIL_H_HI2 = 165, 180   # upper red hue range (wraps at 180)
_RAIL_S_LO                =  80        # minimum saturation
_RAIL_V_LO                =  60        # minimum value
_RAIL_MIN_WIDTH           =  10        # px — blobs narrower than this are noise
_RAIL_MIN_AREA            =  50        # px²

_CORRECTION_VEL_MM_S   = 50    # spin speed during 5° correction
_CORRECTION_DURATION_S = 0.5   # approximate time to spin ~5° in place
_MAX_RAIL_CORRECTIONS  =  5    # give up after this many correction attempts

_rail_kernel = None   # lazily initialised (avoids import-time OpenCV call)


def _detect_rails_hsv(frame: np.ndarray) -> tuple[int | None, int | None]:
    """Return (left_cx, right_cx) for the two largest red blobs, or (None, None).

    Identical algorithm to camera_calibrate.py::detect_rails.
    """
    global _rail_kernel
    if _rail_kernel is None:
        _rail_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low1 = np.array([_RAIL_H_LO1, _RAIL_S_LO, _RAIL_V_LO])
    hi1  = np.array([_RAIL_H_HI1, 255,         255        ])
    low2 = np.array([_RAIL_H_LO2, _RAIL_S_LO, _RAIL_V_LO])
    hi2  = np.array([_RAIL_H_HI2, 255,         255        ])
    mask = cv2.bitwise_or(cv2.inRange(hsv, low1, hi1),
                          cv2.inRange(hsv, low2, hi2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _rail_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _rail_kernel)

    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, n):
        w_b = stats[i, cv2.CC_STAT_WIDTH]
        a_b = stats[i, cv2.CC_STAT_AREA]
        if w_b >= _RAIL_MIN_WIDTH and a_b >= _RAIL_MIN_AREA:
            blobs.append((int(centroids[i][0]), a_b))
    if len(blobs) < 2:
        return None, None
    blobs.sort(key=lambda b: b[1], reverse=True)   # two largest blobs
    xs = sorted([blobs[0][0], blobs[1][0]])
    return xs[0], xs[1]


def _waypoint4_pixel_x(waypoints: np.ndarray, frame_w: int, frame_h: int) -> int:
    """Pixel x of waypoint[WAYPOINT_IDX] projected onto the camera frame.

    Mirrors the projection used in omnivla_strategy._annotate():
        px = cx − dy * scale
    where dy is the lateral component (positive = left in image).
    """
    cx    = frame_w // 2
    scale = min(frame_h, frame_w) * 0.3
    dy    = float(waypoints[WAYPOINT_IDX][1]) * METRIC_SPACING
    return max(0, min(frame_w - 1, int(cx - dy * scale)))


# Downscale frames before sending to limit cloud bandwidth.
# The full OmniVLA model accepts any resolution — PrismaticProcessor handles it.
_SEND_W, _SEND_H   = 640, 480
_JPEG_QUALITY      = 85


def _letterbox(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize preserving aspect ratio, pad with black to exact w×h."""
    fh, fw = frame.shape[:2]
    scale  = min(w / fw, h / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[(h - nh) // 2:(h - nh) // 2 + nh,
        (w - nw) // 2:(w - nw) // 2 + nw] = resized
    return out
_RECONNECT_BASE    = 3.0    # initial reconnect delay (seconds)
_RECONNECT_MAX     = 30.0   # max reconnect delay


class _NavState(Enum):
    CONNECTING   = auto()
    WAITING_GOAL = auto()
    NAVIGATING   = auto()


class CloudOmniVLAStrategy(NavigationStrategy):
    """
    Rover-side strategy that delegates all inference to a cloud OmniVLA server.

    Parameters
    ----------
    server_url : str
        WebSocket URL of omnivla_cloud_server.py, e.g. "ws://1.2.3.4:8765".
    goal : str
        Initial navigation goal.  Can be updated live via set_goal().
    """

    def __init__(self, server_url: str, goal: str = "", max_lin_mm_s: int = 25):
        self._server_url    = server_url
        self._goal          = goal
        self._max_lin_mm_s  = max_lin_mm_s

        self._nav_state  = _NavState.CONNECTING
        self._state_lock = threading.Lock()

        # Asyncio event loop in a dedicated daemon thread
        self._loop = asyncio.new_event_loop()
        self._ws   = None   # websockets connection (set from asyncio thread)

        # Request / response synchronisation
        self._response_event    = threading.Event()
        self._pending_response: dict | None = None

        threading.Thread(target=self._start_loop, daemon=True,
                         name="cloud-ws").start()

        if goal:
            with self._state_lock:
                self._nav_state = _NavState.WAITING_GOAL

    # ── NavigationStrategy interface ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "cloud_omnivla"

    def on_reset(self) -> None:
        with self._state_lock:
            if self._nav_state == _NavState.NAVIGATING:
                self._nav_state = _NavState.WAITING_GOAL
        log.info("CloudOmniVLAStrategy reset")

    def set_goal(self, goal: str) -> None:
        self._goal = goal
        # Forward goal to cloud server immediately (best-effort)
        asyncio.run_coroutine_threadsafe(
            self._ws_send_goal(goal), self._loop
        )
        with self._state_lock:
            if self._nav_state != _NavState.CONNECTING:
                self._nav_state = _NavState.NAVIGATING
        log.info("Goal set: '%s'", goal)

    # ── WebSocket connection management ───────────────────────────────────────

    def _start_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_loop())

    async def _connect_loop(self) -> None:
        import websockets

        delay = _RECONNECT_BASE
        while True:
            try:
                log.info("Connecting to cloud server: %s", self._server_url)
                async with websockets.connect(
                    self._server_url,
                    ping_interval=20,
                    ping_timeout=30,
                ) as ws:
                    self._ws = ws
                    delay = _RECONNECT_BASE   # reset back-off on success
                    await self._receive_loop(ws)
            except Exception as e:
                log.warning("Cloud server disconnected (%s) — retry in %.0fs", e, delay)
            finally:
                self._ws = None
                with self._state_lock:
                    self._nav_state = _NavState.CONNECTING
                # Unblock any run_query waiting for a response
                self._pending_response = None
                self._response_event.set()

            await asyncio.sleep(delay)
            delay = min(delay * 1.5, _RECONNECT_MAX)

    async def _receive_loop(self, ws) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("Bad JSON from cloud server: %r", raw[:80])
                continue

            mtype = msg.get("type")
            if mtype == "ready":
                log.info("Cloud server ready — model loaded")
                # Resend goal in case server restarted since last connection
                if self._goal:
                    await self._ws_send_goal(self._goal)
                with self._state_lock:
                    self._nav_state = (
                        _NavState.NAVIGATING if self._goal
                        else _NavState.WAITING_GOAL
                    )
            elif mtype in ("waypoints", "error"):
                self._pending_response = msg
                self._response_event.set()
            else:
                log.debug("Unknown server message type: %s", mtype)

    async def _ws_send_goal(self, goal: str) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.send(json.dumps({"type": "goal", "goal": goal}))
            log.debug("Goal sent to cloud server: '%s'", goal)
        except Exception as e:
            log.debug("Goal send failed: %s", e)

    async def _ws_send_infer(self, frame_b64: str, goal: str) -> None:
        ws = self._ws
        if ws is None:
            raise ConnectionError("Not connected to cloud server")
        await ws.send(json.dumps({
            "type":      "infer",
            "goal":      goal,
            "frame_b64": frame_b64,
        }))

    # ── Query (called by rover_agent at 1 Hz) ─────────────────────────────────

    def run_query(
        self,
        state: AgentState,
        frame: np.ndarray,
        captures_dir: Path,
        rover_ctrl,
    ) -> None:
        try:
            self._do_query(state, frame, rover_ctrl)
        except Exception as e:
            log.error("CloudOmniVLA error: %s", e, exc_info=True)
            with state.result_lock:
                state.llm_query_start = 0.0
        finally:
            state.query_in_flight.clear()

    def _do_query(self, state: AgentState, frame: np.ndarray, rover_ctrl) -> None:
        t0 = time.time()
        with state.result_lock:
            step  = state.step
            phase = state.phase
            state.llm_query_start = t0

        with self._state_lock:
            nav = self._nav_state

        if nav == _NavState.CONNECTING:
            log.info("Step %d | waiting for cloud server connection…", step)
            self._write_result(state, step, phase, None, 0, 0x8000,
                               "connecting", time.time() - t0)
            return

        if not self._goal:
            log.info("Step %d | no goal yet — send one via the web UI", step)
            self._write_result(state, step, phase, None, 0, 0x8000,
                               "waiting_goal", time.time() - t0)
            return

        # ── Encode frame (letterbox to preserve aspect ratio) ─────────────────
        send_frame = _letterbox(frame, _SEND_W, _SEND_H)
        _, buf = cv2.imencode(".jpg", send_frame,
                              [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        frame_b64 = base64.b64encode(buf.tobytes()).decode()

        # ── Send infer request ────────────────────────────────────────────────
        self._response_event.clear()
        self._pending_response = None
        try:
            asyncio.run_coroutine_threadsafe(
                self._ws_send_infer(frame_b64, self._goal), self._loop
            ).result(timeout=5.0)
        except Exception as e:
            log.warning("Step %d | send failed: %s", step, e)
            self._write_result(state, step, phase, None, 0, 0x8000,
                               "send_error", time.time() - t0)
            return

        # ── Wait for waypoints — allow up to 30s for full OmniVLA inference ────
        budget = max(30.0, state.query_interval - (time.time() - t0))
        log.info("Step %d | waiting for cloud response (budget=%.1fs)…", step, budget)
        if not self._response_event.wait(timeout=budget):
            log.warning("Step %d | cloud server timed out after %.1fs", step, budget)
            self._write_result(state, step, phase, None, 0, 0x8000,
                               "timeout", time.time() - t0)
            return

        resp = self._pending_response
        log.info("Step %d | raw response: %s", step,
                 str(resp)[:200] if resp else "None")
        if resp is None or resp.get("type") != "waypoints":
            msg = resp.get("message", "unknown") if resp else "disconnected"
            log.warning("Step %d | no waypoints: %s", step, msg)
            self._write_result(state, step, phase, None, 0, 0x8000,
                               "no_waypoints", time.time() - t0)
            return

        # ── Waypoint → drive command (all local, no ML) ───────────────────────
        waypoints = np.array(resp["waypoints"])   # [8, 4]
        cloud_s   = resp.get("elapsed", 0.0)

        # ── Guard rail safety: correct if waypoint[4] is outside red rails ────
        for _corr in range(_MAX_RAIL_CORRECTIONS + 1):
            # Grab the freshest available frame (updated if we just spun)
            _check_frame = frame
            with state.raw_lock:
                if state.raw_frame is not None:
                    _check_frame = state.raw_frame.copy()

            _fh, _fw = _check_frame.shape[:2]
            _left_cx, _right_cx = _detect_rails_hsv(_check_frame)

            if _left_cx is None or _right_cx is None:
                # Rails not visible — can't enforce; proceed normally
                log.debug("Step %d | guard rails not detected, skipping check", step)
                break

            _wp_px = _waypoint4_pixel_x(waypoints, _fw, _fh)

            if _left_cx <= _wp_px <= _right_cx:
                # Waypoint is safely within the rails
                if _corr > 0:
                    log.info("Step %d | waypoint within rails after %d correction(s)",
                             step, _corr)
                break

            if _corr == _MAX_RAIL_CORRECTIONS:
                log.warning("Step %d | max corrections (%d) reached — proceeding anyway",
                            step, _MAX_RAIL_CORRECTIONS)
                break

            # Waypoint outside rails — stop and spin opposite direction by ~5°
            # spin RIGHT (radius=-1) when waypoint is left of left rail
            # spin LEFT  (radius=+1) when waypoint is right of right rail
            _spin_dir = 1 if _wp_px > _right_cx else -1
            _side     = "left of left" if _wp_px < _left_cx else "right of right"
            log.info(
                "Step %d | correction %d: wp_px=%d outside rails [%d, %d] (%s rail) — "
                "stopping, spinning %s ~5°",
                step, _corr + 1, _wp_px, _left_cx, _right_cx, _side,
                "left" if _spin_dir == 1 else "right",
            )

            _op_active = (state.operator_control is not None
                          and state.operator_until > time.time())
            if rover_ctrl and not state.paused.is_set() and not _op_active:
                rover_ctrl.stop()
                time.sleep(0.1)
                rover_ctrl.drive_raw(_CORRECTION_VEL_MM_S, _spin_dir)
                time.sleep(_CORRECTION_DURATION_S)
                rover_ctrl.stop()
                time.sleep(0.1)

            # Re-encode latest frame and request a fresh inference
            with state.raw_lock:
                if state.raw_frame is not None:
                    frame = state.raw_frame.copy()
            _send = _letterbox(frame, _SEND_W, _SEND_H)
            _, _buf = cv2.imencode(".jpg", _send,
                                   [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
            _b64 = base64.b64encode(_buf.tobytes()).decode()

            self._response_event.clear()
            self._pending_response = None
            try:
                asyncio.run_coroutine_threadsafe(
                    self._ws_send_infer(_b64, self._goal), self._loop
                ).result(timeout=5.0)
            except Exception as _e:
                log.warning("Step %d | correction re-infer send failed: %s", step, _e)
                break

            log.info("Step %d | waiting for corrected inference…", step)
            if not self._response_event.wait(timeout=30.0):
                log.warning("Step %d | correction re-infer timed out", step)
                break

            _resp2 = self._pending_response
            if _resp2 is None or _resp2.get("type") != "waypoints":
                log.warning("Step %d | correction re-infer returned no waypoints", step)
                break

            waypoints = np.array(_resp2["waypoints"])
            cloud_s   = _resp2.get("elapsed", 0.0)
            log.info("Step %d | re-inference complete; checking rails again…", step)

        vel, radius = _waypoint_to_drive(waypoints, self._max_lin_mm_s)
        elapsed = time.time() - t0

        r_str = "straight" if radius == 0x8000 else f"r={radius}mm"
        log.info("Step %d | vel=%d  %s  cloud=%.2fs  total=%.2fs",
                 step, vel, r_str, cloud_s, elapsed)

        operator_active = (state.operator_control is not None
                           and state.operator_until > time.time())
        if rover_ctrl and not state.paused.is_set() and not operator_active:
            rover_ctrl.drive_raw(vel, radius)
        elif operator_active:
            log.info("Step %d | operator override — skipping drive", step)

        # ── Annotate display frame ────────────────────────────────────────────
        display = _annotate(frame, waypoints, vel, radius, self._goal)
        with state.llm_lock:
            state.llm_frame = display

        with self._state_lock:
            self._nav_state = _NavState.NAVIGATING

        self._write_result(state, step, phase, waypoints, vel, radius,
                           "navigating", elapsed)

    # ── Result writer ─────────────────────────────────────────────────────────

    def _write_result(
        self, state, step, phase, waypoints, vel, radius, goal_status, elapsed
    ) -> None:
        h, w = 480, 640
        ui_waypoints = []
        if waypoints is not None:
            with state.raw_lock:
                if state.raw_frame is not None:
                    h, w = state.raw_frame.shape[:2]
            cx, cy = w // 2, h
            scale  = min(h, w) * 0.3
            for i, wp_i in enumerate(waypoints[:3]):
                px = int(cx - float(wp_i[1]) * METRIC_SPACING * scale)
                py = int(cy - float(wp_i[0]) * METRIC_SPACING * scale)
                ui_waypoints.append({
                    "rank":        i + 1,
                    "x":           max(0, min(w - 1, px)),
                    "y":           max(0, min(h - 1, py)),
                    "description": f"wp[{i}] +{float(wp_i[0]) * METRIC_SPACING:.2f}m",
                    "probability": round(1.0 - i * 0.1, 1),
                })

        r_str  = "straight" if radius == 0x8000 else f"r={radius}mm"
        result = {
            "phase":           phase,
            "navigation_mode": "cloud_omnivla",
            "goal_status":     goal_status,
            "reasoning":       (
                f"cloud_omnivla  vel={vel}mm/s {r_str} | goal='{self._goal}'"
            ),
            "waypoints":       ui_waypoints,
            "confidence":      1.0 if goal_status == "navigating" else 0.0,
        }

        with state.result_lock:
            state.latest_result   = result
            state.llm_query_start = 0.0
            state.llm_response_s  = elapsed
            if ui_waypoints:
                top = ui_waypoints[0]
                state.trajectory.append({
                    "step": step, "phase": phase,
                    "x": top["x"], "y": top["y"],
                    "description": top["description"],
                })

        if state.recorder:
            state.recorder.write_decision({
                "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
                "step":        step,
                "phase":       phase,
                "elapsed_s":   round(elapsed, 3),
                "strategy":    self.name,
                "goal_status": goal_status,
                "vel_mm_s":    vel,
                "radius_mm":   radius if radius != 0x8000 else None,
                "result":      result,
            })
