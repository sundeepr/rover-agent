"""
cosmos_trajectory_strategy.py — Cosmos3-Edge goal-conditioned trajectory ranking (Option 6).

Sends a camera frame + open-ended goal to cosmos_cloud_server.py running in
trajectory_ranking mode.  The server samples the Cosmos3 policy N times,
scores each sample, and returns all trajectories ranked best-first.

The rover executes the highest-ranked trajectory's first action immediately,
then holds subsequent actions from that chunk until the next cloud response.

The full ranked trajectory list is written to AgentState so the web UI can
display all candidate paths with their scores — visually similar to how
OmniVLA shows waypoint confidence.

This is the most flexible option: the goal can be anything —
  "navigate through the next room"
  "follow the red line on the floor"
  "find the blue box and move toward it"
  "move along the wall on the left"

Architecture
────────────
  rover_agent → JPEG + goal text → cosmos_cloud_server (trajectory_ranking)
              ←  N ranked trajectories (each: rank, score, 16×9D actions, description)
              → execute rank-1 actions + display all on UI

Usage
─────
  python rover_agent.py --strategy cosmos_trajectory \\
      --cosmos-server ws://<cloud-ip>:8767 \\
      --goal "Follow the red line on the floor" --interval 1.0
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

log = logging.getLogger("rover.cosmos_trajectory")

# ── Constants ──────────────────────────────────────────────────────────────────

_SEND_W, _SEND_H  = 640, 480
_JPEG_QUALITY     = 85
_RECONNECT_BASE   = 3.0
_RECONNECT_MAX    = 30.0

_STRAIGHT         = 0x8000
_MAX_VEL          = 200
_MIN_RADIUS       = 200

# Action → drive mapping (same as cosmos_av_strategy — tune empirically)
_AV_VEL_SCALE     = 150.0
_AV_LAT_SCALE     = 2000.0
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


def _action_to_drive(action: list, max_vel: int) -> tuple[int, int]:
    fwd = float(action[0]) if len(action) > 0 else 0.0
    lat = float(action[1]) if len(action) > 1 else 0.0
    vel = int(min(max_vel, max(0, fwd * _AV_VEL_SCALE)))
    if abs(lat) < _LAT_DEADBAND:
        radius = _STRAIGHT
    else:
        raw_r  = int(_AV_LAT_SCALE / abs(lat))
        raw_r  = max(_MIN_RADIUS, raw_r)
        radius = raw_r if lat > 0 else -raw_r
    return vel, radius


class CosmosTrajectoryStrategy(NavigationStrategy):
    """
    Option 6: goal-conditioned trajectory ranking via Cosmos3-Edge policy.
    """

    def __init__(self, server_url: str, goal: str = "",
                 max_lin_mm_s: int = _MAX_VEL,
                 response_timeout: float = 120.0):
        self._server_url       = server_url
        self._goal             = goal
        self._max_lin_mm_s     = max_lin_mm_s
        self._response_timeout = response_timeout

        self._conn_state = _ConnState.CONNECTING
        self._conn_lock  = threading.Lock()
        self._loop       = asyncio.new_event_loop()
        self._ws         = None
        self._resp_event = threading.Event()
        self._pending    = None

        # Active trajectory chunk (best-ranked)
        self._active_actions: list = []
        self._active_chunk_idx     = 0
        self._chunk_lock           = threading.Lock()

        # All ranked trajectories from last cloud response (for UI)
        self._all_trajectories: list = []
        self._traj_lock              = threading.Lock()

        threading.Thread(target=self._run_loop, daemon=True,
                         name="cosmos-traj-ws").start()

    @property
    def name(self) -> str:
        return "cosmos_trajectory"

    def on_reset(self) -> None:
        with self._chunk_lock:
            self._active_actions   = []
            self._active_chunk_idx = 0
        with self._traj_lock:
            self._all_trajectories = []
        log.info("CosmosTrajectoryStrategy reset")

    # ── WebSocket ──────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_loop())

    async def _connect_loop(self) -> None:
        import websockets
        delay = _RECONNECT_BASE
        while True:
            try:
                log.info("Connecting to Cosmos trajectory server: %s", self._server_url)
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
                log.info("Cosmos trajectory server ready")
                if self._goal:
                    await self._send_goal_async(self._goal)
                with self._conn_lock:
                    self._conn_state = (
                        _ConnState.READY if self._goal else _ConnState.WAITING_GOAL)
            elif mtype in ("trajectories", "error"):
                self._pending = msg
                self._resp_event.set()

    async def _send_goal_async(self, goal: str) -> None:
        ws = self._ws
        if ws:
            try:
                await ws.send(json.dumps({"type": "goal", "goal": goal}))
            except Exception as e:
                log.debug("Goal send failed: %s", e)

    async def _send_infer_async(self, frame_b64: str, goal: str) -> None:
        ws = self._ws
        if ws is None:
            raise ConnectionError("Not connected")
        await ws.send(json.dumps({
            "type":      "infer",
            "goal":      goal,
            "frame_b64": frame_b64,
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
            log.error("CosmosTrajectory error: %s", e, exc_info=True)
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

        if conn == _ConnState.CONNECTING:
            log.info("Step %d | waiting for Cosmos server…", step)
            self._write_result(state, step, phase, "connecting", 0, _STRAIGHT, [], t0)
            return

        if not self._goal:
            log.info("Step %d | no goal yet", step)
            self._write_result(state, step, phase, "waiting_goal", 0, _STRAIGHT, [], t0)
            return

        # ── Execute buffered actions from previous cloud response ──────────────
        with self._chunk_lock:
            if self._active_chunk_idx < len(self._active_actions):
                action = self._active_actions[self._active_chunk_idx]
                self._active_chunk_idx += 1
                vel, radius = _action_to_drive(action, self._max_lin_mm_s)
                log.info("Step %d | chunk action %d/%d  vel=%d r=%s",
                         step, self._active_chunk_idx,
                         len(self._active_actions),
                         vel, "str" if radius == _STRAIGHT else str(radius))
                operator_active = (state.operator_control is not None
                                   and state.operator_until > time.time())
                if rover_ctrl and not state.paused.is_set() and not operator_active:
                    rover_ctrl.drive_raw(vel, radius)
                with self._traj_lock:
                    trajs = list(self._all_trajectories)
                self._write_result(state, step, phase, "executing_chunk",
                                   vel, radius, trajs, t0)
                return

        # ── Encode frame and request new trajectory set ────────────────────────
        send = _letterbox(frame, _SEND_W, _SEND_H)
        _, buf = cv2.imencode(".jpg", send, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        frame_b64 = base64.b64encode(buf.tobytes()).decode()

        self._resp_event.clear()
        self._pending = None
        try:
            asyncio.run_coroutine_threadsafe(
                self._send_infer_async(frame_b64, self._goal), self._loop
            ).result(timeout=5.0)
        except Exception as e:
            log.warning("Step %d | send failed: %s", step, e)
            self._write_result(state, step, phase, "send_error", 0, _STRAIGHT, [], t0)
            return

        budget = max(self._response_timeout,
                     state.query_interval - (time.time() - t0))
        log.info("Step %d | waiting for trajectory ranking (budget=%.1fs)…",
                 step, budget)
        if not self._resp_event.wait(timeout=budget):
            log.warning("Step %d | trajectory ranking timed out", step)
            self._write_result(state, step, phase, "timeout", 0, _STRAIGHT, [], t0)
            return

        resp = self._pending
        if resp is None or resp.get("type") != "trajectories":
            msg = resp.get("message", "unknown") if resp else "disconnected"
            log.warning("Step %d | unexpected response: %s", step, msg)
            self._write_result(state, step, phase, "error", 0, _STRAIGHT, [], t0)
            return

        trajectories = resp.get("trajectories", [])
        cloud_s      = resp.get("elapsed", 0.0)
        elapsed      = time.time() - t0

        # ── Log all ranked trajectories ────────────────────────────────────────
        log.info("Step %d | %d trajectories received  cloud=%.2fs  total=%.2fs",
                 step, len(trajectories), cloud_s, elapsed)
        for t_info in trajectories:
            log.info("  rank=%d  score=%.4f  '%s'",
                     t_info.get("rank", "?"),
                     t_info.get("score", 0.0),
                     t_info.get("description", ""))

        with self._traj_lock:
            self._all_trajectories = trajectories

        # ── Load best trajectory into action buffer ────────────────────────────
        best = trajectories[0] if trajectories else {}
        best_actions = best.get("actions", [])

        with self._chunk_lock:
            self._active_actions   = best_actions
            self._active_chunk_idx = 0

        # Execute first action immediately
        if best_actions:
            vel, radius = _action_to_drive(best_actions[0], self._max_lin_mm_s)
            with self._chunk_lock:
                self._active_chunk_idx = 1
            operator_active = (state.operator_control is not None
                               and state.operator_until > time.time())
            if rover_ctrl and not state.paused.is_set() and not operator_active:
                rover_ctrl.drive_raw(vel, radius)
            self._write_result(state, step, phase, "navigating",
                               vel, radius, trajectories, t0)
        else:
            self._write_result(state, step, phase, "no_actions",
                               0, _STRAIGHT, trajectories, t0)

    def _write_result(self, state, step, phase, status,
                      vel, radius, trajectories: list, t0) -> None:
        elapsed = time.time() - t0
        r_str   = "straight" if radius == _STRAIGHT else f"r={radius}mm"

        # Build UI waypoints — one entry per ranked trajectory,
        # projected onto the frame from the first action of each.
        h, w = 480, 640
        with state.raw_lock:
            if state.raw_frame is not None:
                h, w = state.raw_frame.shape[:2]
        cx, cy = w // 2, h
        scale  = min(h, w) * 0.4

        ui_waypoints = []
        for t_info in trajectories:
            actions = t_info.get("actions", [])
            score   = float(t_info.get("score", 0.0))
            rank    = int(t_info.get("rank", 0))
            desc    = t_info.get("description", f"rank {rank}")
            if actions:
                fwd = float(actions[0][0]) if actions[0] else 0.0
                lat = float(actions[0][1]) if len(actions[0]) > 1 else 0.0
                px  = int(cx - lat * scale)
                py  = int(cy - fwd * scale)
            else:
                px, py = cx, cy
            ui_waypoints.append({
                "rank":        rank,
                "x":           max(0, min(w - 1, px)),
                "y":           max(0, min(h - 1, py)),
                "description": desc,
                "probability": round(max(0.0, min(1.0, score)), 4),
            })

        best_desc = trajectories[0].get("description", "") if trajectories else ""
        result = {
            "phase":           phase,
            "navigation_mode": self.name,
            "goal_status":     status,
            "reasoning":       (
                f"cosmos_traj vel={vel} {r_str} | "
                f"{len(trajectories)} candidates | best='{best_desc}'"
            ),
            "waypoints":       ui_waypoints,
            "confidence":      trajectories[0].get("score", 0.0) if trajectories else 0.0,
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
                "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
                "step":         step,
                "phase":        phase,
                "elapsed_s":    round(elapsed, 3),
                "strategy":     self.name,
                "goal_status":  status,
                "vel_mm_s":     vel,
                "radius_mm":    radius if radius != _STRAIGHT else None,
                "trajectories": [
                    {"rank": t.get("rank"), "score": t.get("score"),
                     "description": t.get("description")}
                    for t in trajectories
                ],
                "result":       result,
            })
