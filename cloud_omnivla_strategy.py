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

    def __init__(self, server_url: str, goal: str = "",
                 max_lin_mm_s: int = 150, icr_offset_m: float = 0.480):
        self._server_url    = server_url
        self._goal          = goal
        self._max_lin_mm_s  = max_lin_mm_s
        self._icr_offset_m  = icr_offset_m

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
        if resp and resp.get("type") == "waypoints":
            import math as _math
            wps = resp["waypoints"]
            log.info(
                "Step %d | waypoints (8×4)  [fwd, lat, sin_yaw, cos_yaw → yaw_deg]:\n%s",
                step,
                "\n".join(
                    f"  [{i}] fwd={w[0]:.4f}  lat={w[1]:.4f}  "
                    f"sin={w[2]:.4f}  cos={w[3]:.4f}  "
                    f"yaw={_math.degrees(_math.atan2(w[2], w[3])):+.1f}°"
                    + (" ← used" if i == 4 else "")
                    for i, w in enumerate(wps)
                ),
            )
        else:
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
        vel, radius = _waypoint_to_drive(waypoints, self._max_lin_mm_s,
                                         self._icr_offset_m)
        elapsed = time.time() - t0

        cloud_s = resp.get("elapsed", 0.0)
        r_str   = "straight" if radius == 0x8000 else f"r={radius}mm"
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
