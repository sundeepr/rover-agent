"""
cloud_cosmos_strategy.py — Cosmos3-Edge reasoning strategies (Options 1 & 2).

Two strategy classes share one WebSocket connection to cosmos_cloud_server.py
running in reasoning_supervisor or reasoning_driver mode.

── CosmosReasoningSupervisorStrategy (Option 1) ────────────────────────────────
  A fast local strategy (crop-row centroid) handles steering at ~5 Hz.
  Cosmos reasoning runs on the cloud every SUPERVISION_INTERVAL seconds and
  returns:
    drift      : "left" | "right" | "center"
    drift_mm   : estimated lateral offset
    row_end    : bool
    observation: one-sentence scene description

  The supervisor correction is blended into the local strategy's steering:
    - drift left/right → bias the steering radius for the next N local steps
    - row_end=True     → rover stops and waits for operator input (or can be
                          wired to trigger a row-change strategy)

── CosmosReasoningDriverStrategy (Option 2) ─────────────────────────────────────
  Cosmos drives the robot directly.  Every query cycle the rover sends a frame
  and receives {"velocity": int, "radius": int, "reasoning": str}.
  drive_raw() is called with those values.  Between cloud responses the rover
  holds its last command (coasts) or stops if paused.

Both strategies share the same WebSocket reconnect/backoff logic from
cloud_omnivla_strategy.py.

Usage
─────
  # Option 1 — Cosmos supervises crop-row centroid
  python rover_agent.py --strategy cosmos_supervisor \\
      --cosmos-server ws://<cloud-ip>:8767 \\
      --goal "Follow the crop row" --interval 5.0

  # Option 2 — Cosmos drives directly
  python rover_agent.py --strategy cosmos_driver \\
      --cosmos-server ws://<cloud-ip>:8767 \\
      --goal "Navigate to the next room" --interval 3.0
"""

import asyncio
import base64
import io
import json
import logging
import math
import threading
import time
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

from navigation_strategy import AgentState, NavigationStrategy

log = logging.getLogger("rover.cloud_cosmos")

# ── Shared constants ──────────────────────────────────────────────────────────

_SEND_W, _SEND_H   = 640, 480
_JPEG_QUALITY      = 85
_RECONNECT_BASE    = 3.0
_RECONNECT_MAX     = 30.0

# How many local steps to hold a supervisor correction before re-centering
_CORRECTION_HOLD_STEPS = 10

# Minimum turn radius — prevents spin-in-place for small angles
_MIN_TURN_RADIUS = 150   # mm


def _steering_angle_to_radius(angle_deg: float) -> int:
    """
    Convert a steering angle (degrees) to a Roomba OI radius (mm).

    angle_deg:  0 = straight, positive = left, negative = right
                clamped to [-45, 45]
    Returns Roomba radius: 32767=straight, positive=left, negative=right
    """
    if abs(angle_deg) < 0.5:
        return _STEER_STRAIGHT
    angle_rad = math.radians(abs(angle_deg))
    # radius = wheelbase / (2 * sin(angle)) — simplified for differential drive
    # Use 235mm as Roomba wheelbase
    radius = int(235.0 / (2.0 * math.sin(angle_rad)))
    radius = max(_MIN_TURN_RADIUS, radius)
    return radius if angle_deg > 0 else -radius

# Steering radii for supervisor corrections (Roomba OI, mm)
_STEER_LEFT_HARD   =  600
_STEER_LEFT_SOFT   = 1200
_STEER_RIGHT_SOFT  = -1200
_STEER_RIGHT_HARD  =  -600
_STEER_STRAIGHT    = 0x8000   # 32767

# Default forward velocity (mm/s)
_DEFAULT_VEL       = 120
_MIN_VEL           = 100   # minimum velocity for supervisor local steer


class _ConnState(Enum):
    CONNECTING   = auto()
    WAITING_GOAL = auto()
    READY        = auto()


def _letterbox(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    fh, fw = frame.shape[:2]
    scale  = min(w / fw, h / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[(h - nh) // 2:(h - nh) // 2 + nh,
        (w - nw) // 2:(w - nw) // 2 + nw] = resized
    return out


def _draw_steering_arrow(out: np.ndarray, vel: int, radius: int) -> None:
    """
    Draw a steering direction arrow at the bottom-center of the frame.

    Straight ahead  → arrow points straight up
    Turning left/right → arrow deflects in that direction
    Arrow length scales with velocity.
    """
    h, w = out.shape[:2]

    ox, oy  = w // 2, h - 30          # arrow origin (bottom-center)
    max_len = h * 0.30                 # max arrow length in pixels
    length  = max_len * min(vel, 200) / 200.0

    if radius == _STEER_STRAIGHT or radius == 0:
        angle = -math.pi / 2           # straight up
    else:
        # Positive radius = left turn, negative = right turn
        # Map radius to a heading offset: tighter turn → bigger angle
        # Clamp to ±75° so the arrow stays on screen
        max_deflect = math.radians(75)
        deflect = max_deflect * min(abs(radius), 2000) / 2000.0
        if radius > 0:
            angle = -math.pi / 2 - deflect   # left
        else:
            angle = -math.pi / 2 + deflect   # right

    tx = int(ox + length * math.cos(angle))
    ty = int(oy + length * math.sin(angle))

    color = (0, 255, 255)   # yellow
    # Shaft
    cv2.line(out, (ox, oy), (tx, ty), (0, 0, 0), 5, cv2.LINE_AA)
    cv2.line(out, (ox, oy), (tx, ty), color,      3, cv2.LINE_AA)
    # Arrowhead
    cv2.arrowedLine(out, (ox, oy), (tx, ty), (0, 0, 0), 5,
                    cv2.LINE_AA, tipLength=0.3)
    cv2.arrowedLine(out, (ox, oy), (tx, ty), color,      3,
                    cv2.LINE_AA, tipLength=0.3)
    # Robot dot
    cv2.circle(out, (ox, oy), 8, (0, 0, 0),   -1)
    cv2.circle(out, (ox, oy), 6, (0, 255, 0), -1)


def _annotate_frame(frame: np.ndarray, strategy_name: str, goal: str,
                    vel: int, radius: int, status: str,
                    lines: list[str] | None = None) -> np.ndarray:
    """
    Draw a HUD overlay and steering arrow on a copy of frame and return it.
    Written to state.llm_frame so the web server shows it in the LLM panel.
    """
    out  = frame.copy()
    h, w = out.shape[:2]

    # Steering arrow (drawn first, HUD text on top)
    if vel > 0:
        _draw_steering_arrow(out, vel, radius)

    r_str = "straight" if radius == _STEER_STRAIGHT else f"r={radius}mm"

    overlay_lines = [
        f"{strategy_name}  [{status}]",
        f"goal: {goal[:60]}",
        f"vel={vel}mm/s  {r_str}",
    ] + (lines or [])

    y = 28
    for txt in overlay_lines:
        cv2.putText(out, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0),   3, cv2.LINE_AA)
        cv2.putText(out, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        y += 24
    return out


# ── Shared WebSocket mixin ─────────────────────────────────────────────────────

class _CosmosWebSocketMixin:
    """
    Manages a persistent WebSocket connection to cosmos_cloud_server.py.
    Subclasses provide _on_message(msg) to handle incoming JSON.
    """

    def _ws_init(self, server_url: str) -> None:
        self._server_url     = server_url
        self._conn_state     = _ConnState.CONNECTING
        self._conn_lock      = threading.Lock()
        self._loop           = asyncio.new_event_loop()
        self._ws             = None
        self._response_event = threading.Event()
        self._pending_resp   = None
        threading.Thread(target=self._run_loop, daemon=True,
                         name="cosmos-ws").start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_loop())

    async def _connect_loop(self) -> None:
        import websockets
        delay = _RECONNECT_BASE
        while True:
            try:
                log.info("Connecting to Cosmos server: %s", self._server_url)
                async with websockets.connect(
                    self._server_url, ping_interval=20, ping_timeout=30
                ) as ws:
                    self._ws = ws
                    delay = _RECONNECT_BASE
                    await self._recv_loop(ws)
            except Exception as e:
                log.warning("Cosmos server disconnected (%s) — retry in %.0fs", e, delay)
            finally:
                self._ws = None
                with self._conn_lock:
                    self._conn_state = _ConnState.CONNECTING
                self._pending_resp = None
                self._response_event.set()
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, _RECONNECT_MAX)

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("Bad JSON from Cosmos server: %r", raw[:80])
                continue
            mtype = msg.get("type")
            if mtype == "ready":
                log.info("Cosmos server ready (mode=%s)", msg.get("mode", "?"))
                if self._goal:
                    await self._ws_send_goal(self._goal)
                with self._conn_lock:
                    self._conn_state = (
                        _ConnState.READY if self._goal else _ConnState.WAITING_GOAL
                    )
            elif mtype in ("supervision", "drive", "error"):
                self._pending_resp = msg
                self._response_event.set()
            else:
                log.debug("Unknown server message: %s", mtype)

    async def _ws_send_goal(self, goal: str) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.send(json.dumps({"type": "goal", "goal": goal}))
        except Exception as e:
            log.debug("Goal send failed: %s", e)

    async def _ws_send_feedback(self, message: str) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.send(json.dumps({"type": "feedback", "message": message}))
        except Exception as e:
            log.debug("Feedback send failed: %s", e)

    def _send_feedback(self, message: str) -> None:
        asyncio.run_coroutine_threadsafe(
            self._ws_send_feedback(message), self._loop)

    async def _ws_send_infer(self, frame_b64: str, goal: str,
                             history: list | None = None) -> None:
        ws = self._ws
        if ws is None:
            raise ConnectionError("Not connected to Cosmos server")
        msg = {"type": "infer", "goal": goal, "frame_b64": frame_b64}
        if history:
            msg["history"] = history
        await ws.send(json.dumps(msg))

    def _encode_frame(self, frame: np.ndarray) -> str:
        send = _letterbox(frame, _SEND_W, _SEND_H)
        _, buf = cv2.imencode(".jpg", send,
                              [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        return base64.b64encode(buf.tobytes()).decode()

    def _send_infer_sync(self, frame_b64: str, goal: str,
                         history: list | None = None,
                         timeout: float = 5.0) -> None:
        self._response_event.clear()
        self._pending_resp = None
        asyncio.run_coroutine_threadsafe(
            self._ws_send_infer(frame_b64, goal, history), self._loop
        ).result(timeout=timeout)

    def set_goal(self, goal: str) -> None:
        self._goal = goal
        asyncio.run_coroutine_threadsafe(
            self._ws_send_goal(goal), self._loop)
        with self._conn_lock:
            if self._conn_state != _ConnState.CONNECTING:
                self._conn_state = _ConnState.READY
        log.info("Goal set: '%s'", goal)


# ── Option 1: Cosmos Reasoning Supervisor ─────────────────────────────────────

class CosmosReasoningSupervisorStrategy(_CosmosWebSocketMixin, NavigationStrategy):
    """
    Option 1: Cosmos watches the scene periodically and feeds drift corrections
    to a local crop-row centroid steering loop.

    The local steering is a simple centroid-based approach:
      - Find the brightest (path) region in the bottom half of the frame
      - Steer toward the centre of that region
      - Cosmos correction biases the steering radius for _CORRECTION_HOLD_STEPS steps
    """

    def __init__(self, server_url: str, goal: str = "",
                 max_lin_mm_s: int = _DEFAULT_VEL,
                 supervision_interval: float = 5.0):
        self._goal                  = goal
        self._max_lin_mm_s          = max_lin_mm_s
        self._supervision_interval  = supervision_interval

        # Supervisor state
        self._last_supervision_t    = 0.0
        self._correction_radius     = _STEER_STRAIGHT
        self._correction_steps_left = 0
        self._row_end               = False
        self._last_observation      = ""
        self._supervision_lock      = threading.Lock()

        self._ws_init(server_url)
        if goal:
            with self._conn_lock:
                self._conn_state = _ConnState.WAITING_GOAL

    @property
    def name(self) -> str:
        return "cosmos_supervisor"

    def on_reset(self) -> None:
        with self._supervision_lock:
            self._correction_radius     = _STEER_STRAIGHT
            self._correction_steps_left = 0
            self._row_end               = False
        log.info("CosmosReasoningSupervisorStrategy reset")

    def run_query(self, state: AgentState, frame: np.ndarray,
                  captures_dir: Path, rover_ctrl) -> None:
        try:
            self._do_query(state, frame, rover_ctrl)
        except Exception as e:
            log.error("CosmosSupervior error: %s", e, exc_info=True)
        finally:
            state.query_in_flight.clear()

    def _do_query(self, state: AgentState, frame: np.ndarray, rover_ctrl) -> None:
        t0 = time.time()
        with state.result_lock:
            step  = state.step
            phase = state.phase
            state.llm_query_start = t0

        with self._conn_lock:
            conn = self._conn_state

        def _publish_status(status: str) -> None:
            ann = _annotate_frame(frame, self.name, self._goal, 0, _STEER_STRAIGHT, status)
            with state.llm_lock:
                state.llm_frame = ann

        if conn == _ConnState.CONNECTING:
            log.info("Step %d | waiting for Cosmos server…", step)
            _publish_status("connecting")
            self._write_result(state, step, phase, "connecting", 0,
                               _STEER_STRAIGHT, t0)
            return

        if not self._goal:
            log.info("Step %d | no goal yet", step)
            _publish_status("waiting_goal")
            self._write_result(state, step, phase, "waiting_goal", 0,
                               _STEER_STRAIGHT, t0)
            return

        # ── Check if row ended ────────────────────────────────────────────────
        with self._supervision_lock:
            if self._row_end:
                log.info("Step %d | row end detected by Cosmos — stopping", step)
                if rover_ctrl:
                    rover_ctrl.drive_raw(0, _STEER_STRAIGHT)
                _publish_status("row_end")
                self._write_result(state, step, phase, "row_end", 0,
                                   _STEER_STRAIGHT, t0)
                return

        # ── Local steering: path centroid in bottom half of frame ─────────────
        vel, local_radius = self._local_steer(frame)

        # ── Apply Cosmos correction if active ─────────────────────────────────
        with self._supervision_lock:
            if self._correction_steps_left > 0:
                radius = self._correction_radius
                self._correction_steps_left -= 1
                log.debug("Step %d | applying Cosmos correction r=%d (%d steps left)",
                          step, radius, self._correction_steps_left)
            else:
                radius = local_radius

        # ── Drive ─────────────────────────────────────────────────────────────
        operator_active = (state.operator_control is not None
                           and state.operator_until > time.time())
        if rover_ctrl and not state.paused.is_set() and not operator_active:
            rover_ctrl.drive_raw(vel, radius)

        # ── Fire async Cosmos supervision if interval elapsed ─────────────────
        now = time.time()
        if now - self._last_supervision_t >= self._supervision_interval:
            self._last_supervision_t = now
            threading.Thread(
                target=self._run_supervision,
                args=(frame.copy(), self._goal),
                daemon=True,
                name="cosmos-supervision",
            ).start()

        elapsed = time.time() - t0
        with self._supervision_lock:
            obs  = self._last_observation
            corr = self._correction_steps_left

        log.info("Step %d | local vel=%d r=%d | cosmos obs='%s'",
                 step, vel, radius, obs[:60] if obs else "none yet")

        # ── Publish annotated frame to web UI ─────────────────────────────────
        annotated = _annotate_frame(
            frame, self.name, self._goal, vel, radius, "navigating",
            lines=[
                f"cosmos: {obs[:70]}" if obs else "cosmos: (no supervision yet)",
                f"correction_steps={corr}",
            ],
        )
        with state.llm_lock:
            state.llm_frame = annotated

        self._write_result(state, step, phase, "navigating", vel, radius, t0, obs)

    def _local_steer(self, frame: np.ndarray) -> tuple[int, int]:
        """
        Simple centroid steer: find the centre of the brightest region in
        the bottom half of the frame and steer toward it.
        Returns (velocity, radius).
        """
        h, w   = frame.shape[:2]
        bottom = frame[h // 2:, :]
        gray   = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        moments = cv2.moments(thr)
        if moments["m00"] > 100:
            cx = int(moments["m10"] / moments["m00"])
        else:
            cx = w // 2  # no path found — go straight

        offset = cx - w // 2  # positive = path is to the right

        if abs(offset) < w * 0.05:
            radius = _STEER_STRAIGHT
        elif offset > 0:
            radius = _STEER_RIGHT_SOFT if offset < w * 0.15 else _STEER_RIGHT_HARD
        else:
            radius = _STEER_LEFT_SOFT if abs(offset) < w * 0.15 else _STEER_LEFT_HARD

        vel = max(_MIN_VEL, self._max_lin_mm_s)
        return vel, radius

    def _run_supervision(self, frame: np.ndarray, goal: str) -> None:
        """Background thread: send frame to Cosmos, apply correction."""
        try:
            log.info("Supervision: encoding frame and sending to cloud…")
            frame_b64 = self._encode_frame(frame)
            self._send_infer_sync(frame_b64, goal, timeout=10.0)
            log.info("Supervision: frame sent, waiting for response (up to 120s)…")

            budget = 120.0   # first inference can be slow (model warm-up)
            if not self._response_event.wait(timeout=budget):
                log.warning("Cosmos supervision timed out after %.0fs — "
                            "check cloud server logs", budget)
                return

            resp = self._pending_resp
            log.info("Supervision: got response type=%s", resp.get("type") if resp else "None")
            if resp is None or resp.get("type") != "supervision":
                log.warning("Unexpected supervision response: %s", resp)
                return

            drift    = resp.get("drift", "center")
            drift_mm = int(resp.get("drift_mm", 0))
            row_end  = bool(resp.get("row_end", False))
            obs      = resp.get("observation", "")
            elapsed  = resp.get("elapsed", 0.0)

            log.info("Cosmos supervision: drift=%s drift_mm=%d row_end=%s "
                     "obs='%s'  elapsed=%.2fs",
                     drift, drift_mm, row_end, obs[:80], elapsed)

            with self._supervision_lock:
                self._last_observation = obs
                self._row_end          = row_end

                if drift == "left":
                    self._correction_radius = (
                        _STEER_LEFT_HARD if abs(drift_mm) > 150
                        else _STEER_LEFT_SOFT
                    )
                    self._correction_steps_left = _CORRECTION_HOLD_STEPS
                elif drift == "right":
                    self._correction_radius = (
                        _STEER_RIGHT_HARD if abs(drift_mm) > 150
                        else _STEER_RIGHT_SOFT
                    )
                    self._correction_steps_left = _CORRECTION_HOLD_STEPS
                else:
                    self._correction_radius     = _STEER_STRAIGHT
                    self._correction_steps_left = 0

        except Exception as e:
            log.error("Supervision thread error: %s", e, exc_info=True)

    def _write_result(self, state, step, phase, status, vel, radius, t0,
                      observation: str = "") -> None:
        elapsed = time.time() - t0
        r_str   = "straight" if radius == _STEER_STRAIGHT else f"r={radius}mm"
        with self._supervision_lock:
            obs = observation or self._last_observation

        result = {
            "phase":           phase,
            "navigation_mode": self.name,
            "goal_status":     status,
            "reasoning":       f"local_steer vel={vel} {r_str} | cosmos='{obs[:80]}'",
            "waypoints":       [],
            "confidence":      1.0 if status == "navigating" else 0.0,
        }
        with state.result_lock:
            state.latest_result   = result
            state.llm_query_start = 0.0
            state.llm_response_s  = elapsed

        if state.recorder:
            state.recorder.write_decision({
                "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
                "step":        step,
                "phase":       phase,
                "elapsed_s":   round(elapsed, 3),
                "strategy":    self.name,
                "goal_status": status,
                "vel_mm_s":    vel,
                "radius_mm":   radius if radius != _STEER_STRAIGHT else None,
                "result":      result,
            })


# ── Option 2: Cosmos Reasoning Driver ─────────────────────────────────────────

class CosmosReasoningDriverStrategy(_CosmosWebSocketMixin, NavigationStrategy):
    """
    Option 2: Cosmos drives the robot directly via natural language reasoning.
    Sends a frame every query cycle, receives {velocity, radius, reasoning},
    calls drive_raw(). Between cloud responses the last command is held.
    """

    cycle_interval = 0.3   # same as cosmos_av — agent loop calls us every 300ms

    def __init__(self, server_url: str, goal: str = "",
                 max_lin_mm_s: int = _DEFAULT_VEL,
                 response_timeout: float = 60.0):
        self._goal             = goal
        self._max_lin_mm_s     = max_lin_mm_s
        self._response_timeout = response_timeout

        self._last_vel        = 0
        self._last_radius     = _STEER_STRAIGHT
        self._last_reasoning  = ""
        self._goal_reached    = False
        self._infer_in_flight = False
        # Rolling history of last N responses fed back into each prompt
        self._history: list   = []
        self._history_maxlen  = 5

        self._ws_init(server_url)
        if goal:
            with self._conn_lock:
                self._conn_state = _ConnState.WAITING_GOAL

    @property
    def name(self) -> str:
        return "cosmos_driver"

    def on_reset(self) -> None:
        self._last_vel        = 0
        self._last_radius     = _STEER_STRAIGHT
        self._goal_reached    = False
        self._infer_in_flight = False
        self._history         = []
        log.info("CosmosReasoningDriverStrategy reset")

    def run_query(self, state: AgentState, frame: np.ndarray,
                  captures_dir: Path, rover_ctrl) -> None:
        try:
            self._do_query(state, frame, rover_ctrl)
        except Exception as e:
            log.error("CosmosDriver error: %s", e, exc_info=True)
        finally:
            state.query_in_flight.clear()

    def _do_query(self, state: AgentState, frame: np.ndarray, rover_ctrl) -> None:
        t0 = time.time()
        with state.result_lock:
            step  = state.step
            phase = state.phase
            state.llm_query_start = t0

        with self._conn_lock:
            conn = self._conn_state

        def _pub(status: str, vel: int = 0, radius: int = _STEER_STRAIGHT,
                 lines: list | None = None) -> None:
            ann = _annotate_frame(frame, self.name, self._goal, vel, radius,
                                  status, lines)
            with state.llm_lock:
                state.llm_frame = ann

        _pub("running")

        if conn == _ConnState.CONNECTING:
            _pub("connecting")
            self._write_result(state, step, phase, "connecting", 0, _STEER_STRAIGHT, "", t0)
            return

        if not self._goal:
            _pub("waiting_goal")
            self._write_result(state, step, phase, "waiting_goal", 0, _STEER_STRAIGHT, "", t0)
            return

        if self._goal_reached:
            _pub("goal_achieved", lines=[f"cosmos: {self._last_reasoning[:80]}"])
            self._write_result(state, step, phase, "goal_achieved", 0,
                               _STEER_STRAIGHT, self._last_reasoning, t0)
            return

        # ── Fire inference request (non-blocking) ─────────────────────────────
        # Only send the request once. cycle_interval=0.3s means the agent loop
        # calls us again in 300ms — each call sends a keepalive and checks if
        # the response has arrived. Never block here waiting for the cloud.
        if not self._infer_in_flight:
            self._infer_in_flight = True
            self._response_event.clear()
            self._pending_resp = None
            frame_b64 = self._encode_frame(frame)
            history = list(self._history) if self._history else None
            try:
                self._send_infer_sync(frame_b64, self._goal,
                                      history=history, timeout=5.0)
                log.info("Step %d | inference request sent (history=%d steps)…",
                         step, len(self._history))
            except Exception as e:
                log.warning("Step %d | send failed: %s", step, e)
                self._infer_in_flight = False

        # ── Send keepalive while waiting ───────────────────────────────────────
        operator_active = (state.operator_control is not None
                           and state.operator_until > time.time())
        if rover_ctrl and not state.paused.is_set() and not operator_active:
            rover_ctrl.drive_raw(0, _STEER_STRAIGHT)
        _pub("waiting_response", lines=["waiting for cosmos response…"])
        self._write_result(state, step, phase, "waiting_response",
                           0, _STEER_STRAIGHT, self._last_reasoning, t0)

        # ── Check if response has arrived ──────────────────────────────────────
        if not self._response_event.is_set():
            return   # not yet — come back next cycle (300ms)

        self._infer_in_flight = False
        resp = self._pending_resp
        if resp is None or resp.get("type") != "drive":
            msg = resp.get("message", "unknown") if resp else "disconnected"
            log.warning("Step %d | unexpected response: %s", step, msg)
            self._write_result(state, step, phase, "error",
                               self._last_vel, self._last_radius,
                               self._last_reasoning, t0)
            return

        vel           = int(max(0, min(self._max_lin_mm_s, resp.get("velocity", 0))))
        angle         = float(resp.get("steering_angle", 0.0))
        radius        = _steering_angle_to_radius(angle)
        reasoning     = resp.get("reasoning", "")
        goal_achieved = bool(resp.get("goal_achieved", False))
        cloud_s       = resp.get("elapsed", 0.0)
        elapsed       = time.time() - t0

        # ── Client-side validation of goal_achieved ────────────────────────────
        # 1. progress < 90 → override regardless of goal_achieved flag
        # 2. If line positions reported, verify midpoint independently
        # In both cases send feedback to server so next prompt knows why
        progress  = int(resp.get("progress", 0))
        left_pct  = resp.get("left_line_pct")
        right_pct = resp.get("right_line_pct")

        if goal_achieved:
            override_reason = None

            if progress < 90:
                override_reason = (f"goal_achieved rejected: progress={progress}% "
                                   f"is below the required 90% threshold. "
                                   f"Keep working toward the goal.")

            if left_pct is not None and right_pct is not None:
                left_pct  = int(left_pct)
                right_pct = int(right_pct)
                midpoint  = (left_pct + right_pct) / 2.0
                spread    = right_pct - left_pct
                if not (45.0 <= midpoint <= 55.0 and spread >= 20):
                    override_reason = (f"goal_achieved rejected: left={left_pct}% "
                                       f"right={right_pct}% midpoint={midpoint:.1f}% "
                                       f"— robot is not centered between the lines.")

            if override_reason:
                log.warning("Step %d | goal_achieved OVERRIDDEN — %s", step, override_reason)
                goal_achieved = False
                reasoning     = f"[client override] {override_reason}"
                self._send_feedback(override_reason)
            else:
                log.info("Step %d | goal_achieved validated (progress=%d%%)", step, progress)

        self._last_vel       = vel
        self._last_radius    = radius
        self._last_reasoning = reasoning

        # Append to history — log angle, progress and line positions for model feedback
        history_entry = {
            "velocity":       vel,
            "steering_angle": angle,
            "progress":       progress,
            "reasoning":      reasoning,
        }
        if left_pct is not None and right_pct is not None:
            history_entry["left_line_pct"]  = left_pct
            history_entry["right_line_pct"] = right_pct
        self._history.append(history_entry)
        if len(self._history) > self._history_maxlen:
            self._history.pop(0)

        r_str = "straight" if radius == _STEER_STRAIGHT else f"r={radius}mm"
        log.info("Step %d | vel=%d angle=%.1f° %s | goal_achieved=%s | history=%d | cloud=%.2fs",
                 step, vel, angle, r_str, goal_achieved, len(self._history), cloud_s)
        log.info("Step %d | full response: %s", step, json.dumps(resp))

        # ── Goal achieved ──────────────────────────────────────────────────────
        if goal_achieved:
            self._goal_reached = True
            log.info("Step %d | goal_achieved — stopping", step)
            if rover_ctrl and not state.paused.is_set() and not operator_active:
                rover_ctrl.drive_raw(0, _STEER_STRAIGHT)
            _pub("goal_achieved", 0, _STEER_STRAIGHT,
                 [f"cosmos: {reasoning[:80]}"])
            self._write_result(state, step, phase, "goal_achieved",
                               0, _STEER_STRAIGHT, reasoning, t0)
            return

        # ── Execute drive command ──────────────────────────────────────────────
        # No immediate stop needed — next keepalive cycle (300ms) sends
        # drive_raw(0) naturally, giving the Roomba time to actually move.
        if rover_ctrl and not state.paused.is_set() and not operator_active:
            rover_ctrl.drive_raw(vel, radius)

        _pub("navigating", vel, radius, [f"cosmos: {reasoning[:80]}"])
        self._write_result(state, step, phase, "navigating", vel, radius, reasoning, t0)

    def _write_result(self, state, step, phase, status,
                      vel, radius, reasoning, t0) -> None:
        elapsed = time.time() - t0
        r_str   = "straight" if radius == _STEER_STRAIGHT else f"r={radius}mm"
        result  = {
            "phase":           phase,
            "navigation_mode": self.name,
            "goal_status":     status,
            "reasoning":       f"cosmos_driver vel={vel} {r_str} | {reasoning[:120]}",
            "waypoints":       [],
            "confidence":      1.0 if status == "navigating" else 0.0,
        }
        with state.result_lock:
            state.latest_result   = result
            state.llm_query_start = 0.0
            state.llm_response_s  = elapsed

        if state.recorder:
            state.recorder.write_decision({
                "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
                "step":        step,
                "phase":       phase,
                "elapsed_s":   round(elapsed, 3),
                "strategy":    self.name,
                "goal_status": status,
                "vel_mm_s":    vel,
                "radius_mm":   radius if radius != _STEER_STRAIGHT else None,
                "result":      result,
            })
