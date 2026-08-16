"""
cosmos_av_strategy.py — Cosmos3-Edge AV policy strategy (Option 4).

Uses CosmosActionCondition(mode="policy", domain_name="av") on the cloud
to predict a 16-step, 9D action chunk from a rolling buffer of camera frames.

The AV domain was trained on autonomous vehicle dashcam data.  The 9D action
space likely encodes ego-motion as a pose delta + velocity vector.  We extract
the forward (dim 0) and lateral (dim 1) components and map them to Roomba
drive_raw(velocity, radius) commands.

Action mapping
──────────────
  AV action[0]  → forward displacement (normalised)   → velocity mm/s
  AV action[1]  → lateral displacement (normalised)   → steering radius mm
  remaining 7D  → rotation / other pose components (ignored for Roomba)

The mapping is intentionally simple and will need calibration once you see
what the model actually outputs on your scene.  Adjust _AV_VEL_SCALE and
_AV_LAT_SCALE to taste.

Architecture
────────────
  rover_agent (query loop)
    → buffer last N frames
    → send to cosmos_cloud_server (av_policy mode) over WebSocket
    ← receive 16 × 9D action chunk
    → execute actions at _ACTION_HZ for _CHUNK_DURATION seconds
    → send next frame batch when chunk is exhausted

Usage
─────
  python rover_agent.py --strategy cosmos_av \\
      --cosmos-server ws://<cloud-ip>:8767 \\
      --goal "Follow the crop row straight ahead" --interval 0.5
"""

import asyncio
import base64
import collections
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

log = logging.getLogger("rover.cosmos_av")

# ── Constants ──────────────────────────────────────────────────────────────────

_SEND_W, _SEND_H  = 640, 480
_JPEG_QUALITY     = 85
_RECONNECT_BASE   = 3.0
_RECONNECT_MAX    = 30.0

_FRAME_BUFFER_LEN = 5       # number of frames kept for each policy call
_ACTION_HZ        = 5       # rate at which buffered actions are executed
_CHUNK_SIZE       = 16      # must match --chunk-size on the server

# Action → drive mapping scales (tune after first experiment)
_AV_VEL_SCALE     = 150.0   # normalised forward → mm/s (dim 0)
_AV_LAT_SCALE     = 2000.0  # normalised lateral  → radius mm (dim 1)
_MAX_VEL          = 200     # mm/s hard cap
_MIN_VEL          = 80      # mm/s minimum when actions are received (never stop mid-chunk)
_MIN_RADIUS       = 200     # mm minimum arc radius (prevent spin-in-place)
_STRAIGHT         = 0x8000  # Roomba "straight" sentinel

# Lateral deadband: treat |lat| < this as straight
_LAT_DEADBAND     = 0.03


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


def _annotate_frame(frame: np.ndarray, strategy_name: str, goal: str,
                    vel: int, radius: int, status: str,
                    lines: list[str] | None = None) -> np.ndarray:
    """Draw HUD overlay and return annotated copy for state.llm_frame."""
    out   = frame.copy()
    r_str = "straight" if radius == _STRAIGHT else f"r={radius}mm"
    overlay = [
        f"{strategy_name}  [{status}]",
        f"goal: {goal[:60]}",
        f"vel={vel}mm/s  {r_str}",
    ] + (lines or [])
    y = 28
    for txt in overlay:
        cv2.putText(out, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0),   3, cv2.LINE_AA)
        cv2.putText(out, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        y += 24
    return out


def _av_action_to_drive(action: list, min_vel: int = _MIN_VEL) -> tuple[int, int]:
    """
    Map one 9D AV action vector to (velocity mm/s, radius mm).

    The AV domain was trained on car dashcam data so action values are
    normalised pose deltas — typically very small (0.001–0.1 range).
    We log the raw values so you can tune _AV_VEL_SCALE and _AV_LAT_SCALE.

    Assumed dims: [fwd, lat, z, qx, qy, qz, qw, vel, steer]
    """
    fwd = float(action[0]) if len(action) > 0 else 0.0
    lat = float(action[1]) if len(action) > 1 else 0.0

    log.debug("action raw: fwd=%.4f lat=%.4f  all=%s",
              fwd, lat, [f"{x:.3f}" for x in action])

    # Scale forward component → velocity, with minimum so rover keeps moving
    vel_raw = fwd * _AV_VEL_SCALE
    if vel_raw > 10:
        # Model predicts meaningful forward motion
        vel = int(min(_MAX_VEL, vel_raw))
    elif vel_raw > 0:
        # Very small positive value — clamp to minimum so we don't stop
        vel = min_vel
    else:
        # Negative or zero — stop
        vel = 0

    if abs(lat) < _LAT_DEADBAND:
        radius = _STRAIGHT
    else:
        # Positive lat = left in typical ego-frame conventions
        raw_r = int(_AV_LAT_SCALE / abs(lat))
        raw_r = max(_MIN_RADIUS, raw_r)
        radius = raw_r if lat > 0 else -raw_r

    return vel, radius


class CosmosAvPolicyStrategy(NavigationStrategy):
    """
    Option 4: AV domain policy — cloud inference, local action execution.
    """

    def __init__(self, server_url: str, goal: str = "",
                 max_lin_mm_s: int = _MAX_VEL,
                 response_timeout: float = 60.0):
        self._server_url       = server_url
        self._goal             = goal
        self._max_lin_mm_s     = max_lin_mm_s
        self._response_timeout = response_timeout

        self._conn_state  = _ConnState.CONNECTING
        self._conn_lock   = threading.Lock()
        self._loop        = asyncio.new_event_loop()
        self._ws          = None
        self._resp_event  = threading.Event()
        self._pending     = None

        # Rolling frame buffer
        self._frame_buf: collections.deque = collections.deque(
            maxlen=_FRAME_BUFFER_LEN)
        self._frame_lock = threading.Lock()

        # Pending action chunk to be executed between cloud calls
        self._action_chunk: list = []
        self._chunk_lock   = threading.Lock()
        self._chunk_idx    = 0

        # Last drive command — repeated as keepalive while waiting for cloud
        self._last_vel    = _MIN_VEL   # start moving forward immediately
        self._last_radius = _STRAIGHT

        # True while an inference request is in flight to the cloud
        self._infer_in_flight = False

        threading.Thread(target=self._run_loop, daemon=True,
                         name="cosmos-av-ws").start()

    # Fast cycle so the agent loop sends keepalive drives while waiting for cloud
    cycle_interval = 0.3   # seconds between run_query() calls

    @property
    def name(self) -> str:
        return "cosmos_av"

    def on_reset(self) -> None:
        with self._chunk_lock:
            self._action_chunk = []
            self._chunk_idx    = 0
        log.info("CosmosAvPolicyStrategy reset")

    # ── WebSocket management ───────────────────────────────────────────────────

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
                    delay    = _RECONNECT_BASE
                    await self._recv_loop(ws)
            except Exception as e:
                log.warning("Cosmos server disconnected (%s) — retry in %.0fs", e, delay)
            finally:
                self._ws = None
                with self._conn_lock:
                    self._conn_state = _ConnState.CONNECTING
                self._pending = None
                self._resp_event.set()
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, _RECONNECT_MAX)

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            mtype = msg.get("type")
            if mtype == "ready":
                log.info("Cosmos AV server ready")
                if self._goal:
                    await self._send_goal_async(self._goal)
                with self._conn_lock:
                    self._conn_state = (
                        _ConnState.READY if self._goal else _ConnState.WAITING_GOAL)
            elif mtype in ("actions", "error"):
                self._pending = msg
                self._resp_event.set()

    async def _send_goal_async(self, goal: str) -> None:
        ws = self._ws
        if ws:
            try:
                await ws.send(json.dumps({"type": "goal", "goal": goal}))
            except Exception as e:
                log.debug("Goal send failed: %s", e)

    async def _send_infer_async(self, frames_b64: list, goal: str) -> None:
        ws = self._ws
        if ws is None:
            raise ConnectionError("Not connected")
        await ws.send(json.dumps({
            "type":       "infer",
            "goal":       goal,
            "frames_b64": frames_b64,
        }))

    def set_goal(self, goal: str) -> None:
        self._goal = goal
        asyncio.run_coroutine_threadsafe(
            self._send_goal_async(goal), self._loop)
        with self._conn_lock:
            if self._conn_state != _ConnState.CONNECTING:
                self._conn_state = _ConnState.READY
        log.info("Goal set: '%s'", goal)

    # ── Query ──────────────────────────────────────────────────────────────────

    def run_query(self, state: AgentState, frame: np.ndarray,
                  captures_dir: Path, rover_ctrl) -> None:
        try:
            self._do_query(state, frame, rover_ctrl)
        except Exception as e:
            log.error("CosmosAV error: %s", e, exc_info=True)
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

        with self._conn_lock:
            conn = self._conn_state

        def _pub(status: str, vel: int = 0, radius: int = _STRAIGHT,
                 lines: list | None = None) -> None:
            ann = _annotate_frame(frame, self.name, self._goal, vel, radius,
                                  status, lines)
            with state.llm_lock:
                state.llm_frame = ann

        # Always publish the current frame immediately so the web UI stays live
        _pub("running")

        if conn == _ConnState.CONNECTING:
            log.info("Step %d | waiting for Cosmos server…", step)
            _pub("connecting")
            self._write_result(state, step, phase, "connecting", 0, _STRAIGHT, [], t0)
            return

        if not self._goal:
            log.info("Step %d | no goal yet", step)
            _pub("waiting_goal")
            self._write_result(state, step, phase, "waiting_goal", 0, _STRAIGHT, [], t0)
            return

        # ── Buffer this frame ──────────────────────────────────────────────────
        send = _letterbox(frame, _SEND_W, _SEND_H)
        _, buf = cv2.imencode(".jpg", send, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        frame_b64 = base64.b64encode(buf.tobytes()).decode()
        with self._frame_lock:
            self._frame_buf.append(frame_b64)
            frames_b64 = list(self._frame_buf)

        # ── If we still have actions in the chunk, execute the next one ────────
        with self._chunk_lock:
            if self._chunk_idx < len(self._action_chunk):
                action = self._action_chunk[self._chunk_idx]
                self._chunk_idx += 1
                vel, radius = _av_action_to_drive(action)
                vel = min(vel, self._max_lin_mm_s)
                log.info("Step %d | executing chunk action %d/%d  vel=%d r=%s",
                         step, self._chunk_idx, len(self._action_chunk),
                         vel, "straight" if radius == _STRAIGHT else f"{radius}mm")
                operator_active = (state.operator_control is not None
                                   and state.operator_until > time.time())
                if rover_ctrl and not state.paused.is_set() and not operator_active:
                    rover_ctrl.drive_raw(vel, radius)
                _pub("executing_chunk", vel, radius,
                     [f"action {self._chunk_idx}/{len(self._action_chunk)}"])
                self._write_result(state, step, phase, "executing_chunk",
                                   vel, radius, [action], t0)
                return

        # ── Chunk exhausted — fire inference request async and return immediately ─
        # cycle_interval=0.3s means this function is called again in 0.3s.
        # Each call sends a keepalive drive and checks if the response arrived.
        # This keeps the Roomba watchdog satisfied without a blocking wait.
        if not self._infer_in_flight:
            self._infer_in_flight = True
            self._resp_event.clear()
            self._pending = None
            try:
                asyncio.run_coroutine_threadsafe(
                    self._send_infer_async(frames_b64, self._goal), self._loop
                ).result(timeout=5.0)
                log.info("Step %d | inference request sent, waiting for chunk…", step)
            except Exception as e:
                log.warning("Step %d | send failed: %s", step, e)
                self._infer_in_flight = False

        # ── Send keepalive drive while waiting for cloud response ─────────────
        operator_active = (state.operator_control is not None
                           and state.operator_until > time.time())
        if rover_ctrl and not state.paused.is_set() and not operator_active:
            rover_ctrl.drive_raw(self._last_vel, self._last_radius)
        _pub("waiting_chunk", self._last_vel, self._last_radius,
             ["waiting for cosmos chunk…"])
        self._write_result(state, step, phase, "waiting_chunk",
                           self._last_vel, self._last_radius, [], t0)

        # ── Check if response has arrived ──────────────────────────────────────
        if not self._resp_event.is_set():
            return   # not yet — come back next cycle

        self._infer_in_flight = False
        resp = self._pending
        if resp is None or resp.get("type") != "actions":
            msg = resp.get("message", "unknown") if resp else "disconnected"
            log.warning("Step %d | unexpected response: %s", step, msg)
            self._write_result(state, step, phase, "error",
                               self._last_vel, self._last_radius, [], t0)
            return

        actions = resp.get("actions", [])
        cloud_s = resp.get("elapsed", 0.0)
        elapsed = time.time() - t0

        log.info("Step %d | received %d actions  cloud=%.2fs  total=%.2fs",
                 step, len(actions), cloud_s, elapsed)

        # Log raw values — needed for tuning _AV_VEL_SCALE / _AV_LAT_SCALE
        for i, a in enumerate(actions):
            vel_i, r_i = _av_action_to_drive(a)
            log.info("  action[%02d]: raw=%s → vel=%d r=%s",
                     i, [f"{x:.4f}" for x in a],
                     vel_i, "straight" if r_i == _STRAIGHT else str(r_i))

        with self._chunk_lock:
            self._action_chunk = actions
            self._chunk_idx    = 0

        if actions:
            first_vel, first_radius = _av_action_to_drive(actions[0])
            first_vel = min(first_vel, self._max_lin_mm_s)
            self._last_vel    = first_vel
            self._last_radius = first_radius
            with self._chunk_lock:
                self._chunk_idx = 1
            if rover_ctrl and not state.paused.is_set() and not operator_active:
                rover_ctrl.drive_raw(first_vel, first_radius)
            _pub("navigating", first_vel, first_radius,
                 [f"{len(actions)} actions  cloud={cloud_s:.1f}s"])
            self._write_result(state, step, phase, "navigating",
                               first_vel, first_radius, actions, t0)
        else:
            _pub("no_actions", self._last_vel, self._last_radius)
            self._write_result(state, step, phase, "no_actions",
                               self._last_vel, self._last_radius, [], t0)

    def _write_result(self, state, step, phase, status, vel, radius, actions, t0) -> None:
        elapsed = time.time() - t0
        r_str   = "straight" if radius == _STRAIGHT else f"r={radius}mm"

        # Build UI waypoints from action chunk (project onto frame)
        ui_waypoints = []
        h, w = 480, 640
        with state.raw_lock:
            if state.raw_frame is not None:
                h, w = state.raw_frame.shape[:2]
        cx, cy = w // 2, h
        for i, a in enumerate(actions[:5]):
            fwd = float(a[0]) if a else 0.0
            lat = float(a[1]) if len(a) > 1 else 0.0
            scale = min(h, w) * 0.4
            px = int(cx - lat * scale)
            py = int(cy - fwd * scale)
            ui_waypoints.append({
                "rank":        i + 1,
                "x":           max(0, min(w - 1, px)),
                "y":           max(0, min(h - 1, py)),
                "description": f"av[{i}] fwd={fwd:.3f} lat={lat:.3f}",
                "probability": round(1.0 - i * 0.08, 2),
            })

        result = {
            "phase":           phase,
            "navigation_mode": self.name,
            "goal_status":     status,
            "reasoning":       f"cosmos_av vel={vel} {r_str} | {len(actions)} actions",
            "waypoints":       ui_waypoints,
            "confidence":      1.0 if status in ("navigating", "executing_chunk") else 0.0,
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
                "goal_status": status,
                "vel_mm_s":    vel,
                "radius_mm":   radius if radius != _STRAIGHT else None,
                "result":      result,
            })
