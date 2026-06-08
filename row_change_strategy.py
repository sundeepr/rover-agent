#!/usr/bin/env python3
"""
row_change_strategy.py — End-of-row detection and row-change maneuver via Qwen2.5-VL.

State machine
─────────────
  FOLLOWING    Drive forward continuously; ask Qwen every N seconds whether the
               rover has reached the end of the row.  Two consecutive YES answers
               trigger the maneuver.

  EXITING_ROW  Drive forward a fixed distance to fully clear the row end before
               turning.

  TURNING_1    Tank-turn 90° left so the rover faces across the rows (toward the
               next row).

  FINDING_ROW  Nudge forward in small steps.  After each step ask Qwen whether the
               rover is positioned over a crop row.  Stop when YES.

  TURNING_2    Tank-turn 90° left again so the rover faces back down the rows
               (opposite direction to the previous pass — boustrophedon pattern).

  ALIGNING     Nudge forward or backward in small steps until Qwen says the rover
               is centred on the row and ready to go.

  Then loop back to FOLLOWING.

Qwen server
───────────
  Requires qwen_cloud_server.py running on a machine with GPU.
  Protocol: {"type": "infer", "instruction": "...", "frame_b64": "..."}
            {"type": "response", "text": "..."}

Usage
─────
  python rover_agent.py --strategy row_change \\
      --qwen-server ws://192.168.1.100:8766 \\
      --rover atlas --atlas-port /dev/ttyACM0 \\
      --device /dev/cam-front
"""

import asyncio
import base64
import json
import logging
import threading
import time
from enum import Enum, auto

import cv2
import numpy as np

from navigation_strategy import AgentState, NavigationStrategy
from crop_guard_strategy import _process_wheel_frame
from frame_source import open_frame_source, FrameSource

log = logging.getLogger("rover.row_change")

_JPEG_QUALITY = 80

# ── Qwen prompts ──────────────────────────────────────────────────────────────

_PROMPT_END_OF_ROW = (
    "Look at this camera image from an agricultural rover driving along a crop row. "
    "The crop row the rover is currently following appears in the BOTTOM HALF of the image. "
    "Ignore any crops or fields visible in the top half of the image — those are distant "
    "fields and not the row being followed. "
    "Has the rover reached the END of the crop row it is currently on? "
    "Signs of row end: the bottom half shows bare soil or open space with no more plants, "
    "the planted area in the bottom half terminates, a headland or turning area is visible. "
    "Do not write code. Reply with only the single word YES or NO."
)

_PROMPT_OVER_ROW = (
    "Look at this camera image from an agricultural rover. "
    "Is the rover currently positioned over or immediately next to a crop row? "
    "A crop row means plants or vegetation forming a line visible in the image. "
    "Do not write code. Reply with only the single word YES or NO."
)

_PROMPT_ALIGNED = (
    "Look at this camera image from an agricultural rover. "
    "Is the rover centred and aligned along a crop row, ready to drive straight down it? "
    "Do not write code. Reply with exactly one of these words: "
    "ALIGNED (rover is on the row), "
    "FORWARD (the row is further ahead of the rover), "
    "BACKWARD (the row is behind the rover)."
)


# ── Lightweight blocking Qwen WebSocket client ────────────────────────────────

class _QwenClient:
    """
    Persistent WebSocket connection to qwen_cloud_server.py.

    query() is blocking and safe to call from any thread.
    The asyncio event loop runs on a dedicated daemon thread.
    """

    _RECONNECT_DELAY = 3.0

    def __init__(self, url: str):
        self._url   = url
        self._ws    = None
        self._ready = False

        self._response_event = threading.Event()
        self._pending: dict | None = None
        self._lock = threading.Lock()

        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._run_loop, daemon=True,
                         name="qwen-ws").start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_loop())

    async def _connect_loop(self) -> None:
        import websockets
        delay = self._RECONNECT_DELAY
        while True:
            try:
                log.info("Connecting to Qwen server: %s", self._url)
                async with websockets.connect(self._url,
                                              ping_interval=30,
                                              ping_timeout=120) as ws:
                    self._ws    = ws
                    self._ready = True
                    delay       = self._RECONNECT_DELAY
                    log.info("Qwen server connected")
                    await self._recv_loop(ws)
            except Exception as e:
                log.warning("Qwen server disconnected (%s) — retry in %.0fs", e, delay)
            finally:
                self._ws    = None
                self._ready = False
                with self._lock:
                    self._pending = None
                self._response_event.set()   # unblock any waiting query()
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 30.0)

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") in ("response", "error", "ready"):
                with self._lock:
                    self._pending = msg
                self._response_event.set()

    def query(self, frame_jpeg: bytes, instruction: str, timeout: float = 120.0) -> str | None:
        """
        Send one frame + instruction, block until the model replies.

        Returns the response text, or None on timeout / error / disconnect.
        """
        if not self._ready or self._ws is None:
            log.warning("Qwen server not connected — skipping query")
            return None

        frame_b64 = base64.b64encode(frame_jpeg).decode()

        self._response_event.clear()
        with self._lock:
            self._pending = None

        try:
            asyncio.run_coroutine_threadsafe(
                self._ws.send(json.dumps({
                    "type":        "infer",
                    "instruction": instruction,
                    "frame_b64":   frame_b64,
                })),
                self._loop,
            ).result(timeout=5.0)
        except Exception as e:
            log.warning("Qwen send failed: %s", e)
            return None

        if not self._response_event.wait(timeout=timeout):
            log.warning("Qwen query timed out after %.0fs", timeout)
            return None

        with self._lock:
            resp = self._pending

        if resp is None:
            return None
        if resp.get("type") == "error":
            log.warning("Qwen error: %s", resp.get("message"))
            return None
        return resp.get("text", "")


# ── Strategy ──────────────────────────────────────────────────────────────────

class _Phase(Enum):
    FOLLOWING   = auto()
    EXITING_ROW = auto()
    TURNING_1   = auto()
    FINDING_ROW = auto()
    TURNING_2   = auto()
    ALIGNING    = auto()


class RowChangeStrategy(NavigationStrategy):
    """
    End-of-row detection + row-change maneuver using Qwen2.5-VL.

    Parameters
    ----------
    qwen_server_url : str
        WebSocket URL of qwen_cloud_server.py, e.g. "ws://192.168.1.100:8766".
    forward_vel : int
        Forward speed during row following (mm/s).
    qwen_interval_s : float
        Seconds between Qwen end-of-row checks during FOLLOWING.
    end_confirmations : int
        Consecutive YES answers required before triggering the maneuver.
    exit_distance_mm : int
        Distance (mm) to drive forward after row end confirmed, to clear the
        row before turning.
    turn_90_duration_s : float
        Time (s) for a 90° tank-turn.  Calibrate on the actual hardware.
    nudge_mm : int
        Step size (mm) for each lateral nudge while FINDING_ROW.
    nudge_vel : int
        Speed (mm/s) for nudge moves.
    max_find_nudges : int
        Max lateral nudges before giving up and trying to align anyway.
    max_align_nudges : int
        Max forward/backward nudges before giving up on fine alignment.
    """

    requires_goal  = False
    cycle_interval = 0.05   # kept short so agent_loop calls run_query quickly;
                             # the real pacing is inside _main_loop

    def __init__(
        self,
        qwen_server_url:    str,
        forward_vel:        int   = 50,
        qwen_interval_s:    float = 3.0,
        end_confirmations:  int   = 2,
        exit_distance_mm:   int   = 600,
        turn_90_duration_s: float = 4.5,
        nudge_mm:           int   = 150,
        nudge_vel:          int   = 30,
        max_find_nudges:    int   = 20,
        max_align_nudges:   int   = 10,
    ):
        self._qwen              = _QwenClient(qwen_server_url)
        self._forward_vel       = forward_vel
        self._qwen_interval_s   = qwen_interval_s
        self._end_confirmations = end_confirmations
        self._exit_distance_mm  = exit_distance_mm
        self._turn_90_s         = turn_90_duration_s
        self._nudge_mm          = nudge_mm
        self._nudge_vel         = nudge_vel
        self._max_find          = max_find_nudges
        self._max_align         = max_align_nudges

        self._phase          = _Phase.FOLLOWING
        self._confirmations  = 0
        self._started        = False   # ensures _main_loop runs only once
        self._started_lock   = threading.Lock()

    @property
    def name(self) -> str:
        return "row_change"

    def on_reset(self) -> None:
        self._phase         = _Phase.FOLLOWING
        self._confirmations = 0
        log.info("RowChangeStrategy reset")

    # ── NavigationStrategy interface ──────────────────────────────────────────

    def run_query(
        self,
        state: AgentState,
        frame: np.ndarray,
        captures_dir,
        rover_ctrl,
    ) -> None:
        # Only one main loop should ever run.  agent_loop fires run_query on a
        # daemon thread every cycle_interval; we gate on _started so the second
        # call just clears query_in_flight and exits immediately.
        with self._started_lock:
            if self._started:
                state.query_in_flight.clear()
                return
            self._started = True

        try:
            self._main_loop(state, rover_ctrl)
        except Exception as e:
            log.error("RowChangeStrategy fatal error: %s", e, exc_info=True)
        finally:
            state.query_in_flight.clear()

    # ── Main control loop ─────────────────────────────────────────────────────

    def _main_loop(self, state: AgentState, rover_ctrl) -> None:
        last_qwen_t = 0.0

        while True:
            if state.paused.is_set():
                if rover_ctrl:
                    rover_ctrl.stop()
                time.sleep(0.1)
                continue

            frame = self._latest_frame(state)
            if frame is None:
                time.sleep(0.05)
                continue

            # ── FOLLOWING ─────────────────────────────────────────────────────
            if self._phase == _Phase.FOLLOWING:
                if rover_ctrl:
                    rover_ctrl.drive_raw(self._forward_vel, 0x8000)

                now = time.time()
                if now - last_qwen_t >= self._qwen_interval_s:
                    last_qwen_t = now
                    answer = self._qwen_query(frame, _PROMPT_END_OF_ROW)
                    log.info("[FOLLOWING] Qwen: %r  confirmations=%d",
                             answer, self._confirmations)
                    if self._is_yes(answer):
                        self._confirmations += 1
                        if self._confirmations >= self._end_confirmations:
                            log.info("End of row confirmed — starting maneuver")
                            self._confirmations = 0
                            self._run_maneuver(state, rover_ctrl)
                            last_qwen_t = 0.0
                    else:
                        self._confirmations = 0

                time.sleep(0.05)

    # ── Row-change maneuver ───────────────────────────────────────────────────

    def _run_maneuver(self, state: AgentState, rover_ctrl) -> None:
        # ── Phase 1: exit the row ─────────────────────────────────────────────
        self._phase = _Phase.EXITING_ROW
        log.info("[EXITING_ROW] driving %.0f mm forward to clear row end",
                 self._exit_distance_mm)
        self._drive_distance(rover_ctrl, self._exit_distance_mm, self._forward_vel)

        # ── Phase 2: turn 90° left ────────────────────────────────────────────
        self._phase = _Phase.TURNING_1
        log.info("[TURNING_1] 90° left turn (%.1fs)", self._turn_90_s)
        self._turn_left_90(rover_ctrl)

        # ── Phase 3: nudge forward until Qwen sees a crop row ─────────────────
        self._phase = _Phase.FINDING_ROW
        log.info("[FINDING_ROW] nudging forward to find next row")
        found = False
        for i in range(self._max_find):
            if state.paused.is_set():
                break
            self._drive_distance(rover_ctrl, self._nudge_mm, self._nudge_vel)
            time.sleep(0.3)   # settle before grabbing frame
            frame = self._latest_frame(state)
            if frame is not None:
                answer = self._qwen_query(frame, _PROMPT_OVER_ROW)
                log.info("[FINDING_ROW] nudge %d/%d  Qwen: %r",
                         i + 1, self._max_find, answer)
                if self._is_yes(answer):
                    found = True
                    break
        if not found:
            log.warning("[FINDING_ROW] row not found after %d nudges — continuing anyway",
                        self._max_find)

        # ── Phase 4: turn 90° left again ─────────────────────────────────────
        self._phase = _Phase.TURNING_2
        log.info("[TURNING_2] 90° left turn (%.1fs)", self._turn_90_s)
        self._turn_left_90(rover_ctrl)

        # ── Phase 5: fine alignment ───────────────────────────────────────────
        self._phase = _Phase.ALIGNING
        log.info("[ALIGNING] fine-aligning on row")
        for i in range(self._max_align):
            if state.paused.is_set():
                break
            time.sleep(0.3)
            frame = self._latest_frame(state)
            if frame is None:
                continue
            answer = self._qwen_query(frame, _PROMPT_ALIGNED)
            log.info("[ALIGNING] step %d/%d  Qwen: %r", i + 1, self._max_align, answer)
            if not answer:
                continue
            upper = answer.upper()
            if "ALIGNED" in upper:
                log.info("[ALIGNING] aligned — resuming row following")
                break
            elif "FORWARD" in upper:
                self._drive_distance(rover_ctrl, self._nudge_mm, self._nudge_vel)
            elif "BACKWARD" in upper:
                self._drive_distance(rover_ctrl, self._nudge_mm, -self._nudge_vel)
        else:
            log.warning("[ALIGNING] could not align after %d nudges — resuming anyway",
                        self._max_align)

        # ── Back to following ─────────────────────────────────────────────────
        self._phase = _Phase.FOLLOWING
        log.info("Row change complete — resuming FOLLOWING")

    # ── Motion helpers ────────────────────────────────────────────────────────

    def _drive_distance(self, rover_ctrl, distance_mm: int, vel_mm_s: int) -> None:
        """Drive forward (vel>0) or backward (vel<0) for the time to cover distance_mm."""
        if rover_ctrl is None or distance_mm <= 0:
            return
        duration = abs(distance_mm) / abs(vel_mm_s)
        rover_ctrl.drive_raw(vel_mm_s, 0x8000)
        time.sleep(duration)
        rover_ctrl.stop()
        time.sleep(0.1)

    def _turn_left_90(self, rover_ctrl) -> None:
        """Tank-turn 90° left in place."""
        if rover_ctrl is None:
            return
        rover_ctrl.drive_raw(1, -1)   # spin left: vel=1 (non-zero), radius=-1
        time.sleep(self._turn_90_s)
        rover_ctrl.stop()
        time.sleep(0.2)

    # ── Qwen helpers ──────────────────────────────────────────────────────────

    def _qwen_query(self, frame: np.ndarray, prompt: str) -> str | None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        if not ok:
            return None
        return self._qwen.query(buf.tobytes(), prompt)

    @staticmethod
    def _is_yes(answer: str | None) -> bool:
        if not answer:
            return False
        return "YES" in answer.upper()

    @staticmethod
    def _latest_frame(state: AgentState) -> np.ndarray | None:
        with state.raw_lock:
            return state.raw_frame.copy() if state.raw_frame is not None else None
