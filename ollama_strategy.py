"""
OllamaStrategy — rover navigation using a local Ollama vision model.

Sends a rolling buffer of recent frames plus their previous predictions
to Ollama, which infers the rover's motion direction and returns the next
navigation point and path centerline.

The next_point [x, y] (normalized 0-1) is converted to a drive command:
  - x < 0.45  → steer left
  - x > 0.55  → steer right
  - else       → straight

Usage:
    python rover_agent.py --strategy ollama \\
        --ollama-server http://<host>:11434 \\
        --ollama-model qwen2.5vl \\
        --rover atlas --atlas-port /dev/ttyACM0
"""

import base64
import json
import logging
import re
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from navigation_strategy import AgentState, NavigationStrategy

log = logging.getLogger("rover.ollama")

_HISTORY_SIZE  = 5
_JPEG_QUALITY  = 85
_SEND_W, _SEND_H = 640, 480

# Drive parameters
_FWD_VEL       = 60     # mm/s forward speed
_TURN_RADIUS   = 400    # mm turning radius magnitude
_DEAD_BAND     = 0.05   # ± around x=0.5 treated as straight


# ── Prompt ────────────────────────────────────────────────────────────────────

def _build_prompt(history: deque) -> str:
    n = len(history)
    past_lines = []
    for i, (_, result) in enumerate(list(history)[:-1]):
        if result and result.get("next_point"):
            p = result["next_point"]
            past_lines.append(
                f"  Frame {i+1}: next_point=[{p[0]:.2f},{p[1]:.2f}]  "
                f"direction={result.get('motion_direction','?')}"
            )
        else:
            past_lines.append(f"  Frame {i+1}: next_point=unknown")

    history_text = "\n".join(past_lines) if past_lines else "  (no prior frames)"

    return f"""\
You are a vision system for a farm rover driving between crop rows.

You are given {n} image(s) in chronological order (oldest first, newest last).
Previous predicted navigation points:
{history_text}

Using this motion history, analyse the LAST image (frame {n}) and predict \
where the rover should navigate next.

Return ONLY valid JSON:
{{
  "motion_direction": "left|right|straight",
  "next_point": [x, y],
  "path_points": [[x, y], ...],
  "path_visible": true/false,
  "confidence": 0.0-1.0,
  "reason": "one sentence"
}}

Coordinate rules (ALL values normalized 0.0-1.0):
  x=0.0=left edge, x=1.0=right edge, x=0.5=center.
  y=0.0=top edge,  y=1.0=bottom edge.

next_point: single best point to steer toward in the CURRENT (last) image.
  Must be on open soil between crop rows, not on any plant.

path_points: 4-6 points tracing the gap center in the CURRENT image, \
bottom (y≈0.9) to top (y≈0.2).
  x MUST change across points to follow the actual gap curvature.
  A constant x column is WRONG.

motion_direction: inferred from how next_point x shifted across past frames.

path_visible: false if no clear open soil gap is visible.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    clean = re.sub(r'```(?:json)?', '', text).strip()
    for candidate in [clean, text]:
        try:
            d = json.loads(candidate)
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            pass
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            pass
    return None


def _annotate(frame: np.ndarray, result: dict) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    def px(nx, ny):
        return int(nx * w), int(ny * h)

    pts    = result.get("path_points", [])
    px_pts = [px(x, y) for x, y in pts]
    for i, pt in enumerate(px_pts):
        cv2.circle(out, pt, 5, (0, 200, 0), -1)
        if i > 0:
            cv2.line(out, px_pts[i - 1], pt, (0, 200, 0), 2)

    np_ = result.get("next_point")
    if np_:
        nx, ny = px(np_[0], np_[1])
        cv2.circle(out, (nx, ny), 12, (0, 255, 0), -1)
        cv2.circle(out, (nx, ny), 14, (255, 255, 255), 2)
        cv2.putText(out, "NEXT", (nx + 16, ny + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    direction = result.get("motion_direction", "")
    conf      = result.get("confidence", 0.0)
    visible   = result.get("path_visible", False)
    vis_color = (0, 255, 100) if visible else (0, 0, 255)
    cv2.putText(out, f"{'PATH OK' if visible else 'NO PATH'}  conf:{conf:.2f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, vis_color, 2)
    cv2.putText(out, f"direction: {direction}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(out, result.get("reason", "")[:90], (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 220, 100), 1)
    return out


def _next_point_to_drive(next_point: list) -> tuple[int, int]:
    """Convert normalized next_point [x, y] to (vel_mm_s, radius_mm)."""
    x = float(next_point[0])
    error = x - 0.5   # positive = gap is right of centre → steer right
    if abs(error) <= _DEAD_BAND:
        return _FWD_VEL, 0x8000   # straight
    radius = -int(_TURN_RADIUS / (error * 10))
    radius = max(-2000, min(2000, radius))
    return _FWD_VEL, radius


# ── Strategy ──────────────────────────────────────────────────────────────────

class OllamaStrategy(NavigationStrategy):
    """
    Navigation strategy using a local Ollama vision model with frame history.

    Parameters
    ----------
    ollama_url : str
        Base URL of the Ollama API, e.g. "http://localhost:11434".
    model : str
        Ollama model name, e.g. "qwen2.5vl".
    history_size : int
        Number of frames to keep in the rolling history buffer.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen2.5vl",
        history_size: int = _HISTORY_SIZE,
    ):
        self._url     = ollama_url.rstrip("/")
        self._model   = model
        self._history: deque = deque(maxlen=history_size)
        log.info("OllamaStrategy: model=%s  server=%s  history=%d",
                 model, ollama_url, history_size)

    @property
    def name(self) -> str:
        return "ollama"

    def on_reset(self) -> None:
        self._history.clear()
        log.info("OllamaStrategy reset — history cleared")

    # ── run_query ─────────────────────────────────────────────────────────────

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
            log.error("OllamaStrategy error: %s", e, exc_info=True)
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

        # Resize for network efficiency
        h, w = frame.shape[:2]
        send = (cv2.resize(frame, (_SEND_W, _SEND_H))
                if (w != _SEND_W or h != _SEND_H) else frame)

        # Add current frame to history (result filled in after inference)
        self._history.append((send, None))

        # Build images list + prompt
        images = []
        for f, _ in self._history:
            _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
            images.append(base64.b64encode(buf.tobytes()).decode())

        prompt  = _build_prompt(self._history)
        payload = json.dumps({
            "model":  self._model,
            "prompt": prompt,
            "images": images,
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{self._url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = json.loads(resp.read()).get("response", "").strip()
        except Exception as e:
            log.warning("Step %d | Ollama request failed: %s", step, e)
            self._write_result(state, step, phase, None, 0, 0x8000,
                               "request_error", time.time() - t0)
            return

        log.info("Step %d | Ollama raw: %r", step, raw[:120])
        result = _extract_json(raw)
        if result is None:
            log.warning("Step %d | JSON parse failed", step)
            self._write_result(state, step, phase, None, 0, 0x8000,
                               "parse_error", time.time() - t0)
            return

        # Update history entry with result
        self._history[-1] = (send, result)

        next_point = result.get("next_point")
        path_visible = result.get("path_visible", False)
        direction    = result.get("motion_direction", "?")
        elapsed      = time.time() - t0

        log.info("Step %d | visible=%s  direction=%s  next=%s  elapsed=%.2fs",
                 step, path_visible, direction, next_point, elapsed)

        if not path_visible or not next_point:
            self._write_result(state, step, phase, result, 0, 0x8000,
                               "no_path", elapsed)
            return

        vel, radius = _next_point_to_drive(next_point)
        r_str = "straight" if radius == 0x8000 else f"r={radius}mm"
        log.info("Step %d | vel=%d  %s", step, vel, r_str)

        operator_active = (state.operator_control is not None
                           and state.operator_until > time.time())
        if rover_ctrl and not state.paused.is_set() and not operator_active:
            rover_ctrl.drive_raw(vel, radius)
        elif operator_active:
            log.info("Step %d | operator override — skipping drive", step)

        display = _annotate(frame, result)
        with state.llm_lock:
            state.llm_frame = display

        self._write_result(state, step, phase, result, vel, radius,
                           "navigating", elapsed)

    # ── Result writer ─────────────────────────────────────────────────────────

    def _write_result(self, state, step, phase, result, vel, radius,
                      status, elapsed) -> None:
        h, w = 480, 640
        ui_waypoints = []

        if result and result.get("path_points"):
            with state.raw_lock:
                if state.raw_frame is not None:
                    h, w = state.raw_frame.shape[:2]
            for i, pt in enumerate(result["path_points"][:3]):
                px = int(float(pt[0]) * w)
                py = int(float(pt[1]) * h)
                ui_waypoints.append({
                    "rank":        i + 1,
                    "x":           max(0, min(w - 1, px)),
                    "y":           max(0, min(h - 1, py)),
                    "description": f"path[{i}]",
                    "probability": round(1.0 - i * 0.1, 1),
                })

        r_str  = "straight" if radius == 0x8000 else f"r={radius}mm"
        reason = result.get("reason", "") if result else ""
        direction = result.get("motion_direction", "") if result else ""
        res = {
            "phase":           phase,
            "navigation_mode": "ollama",
            "goal_status":     status,
            "reasoning":       (
                f"ollama  vel={vel}mm/s {r_str}  dir={direction} | {reason}"
            ),
            "waypoints":       ui_waypoints,
            "confidence":      result.get("confidence", 0.0) if result else 0.0,
        }

        with state.result_lock:
            state.latest_result   = res
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
                "radius_mm":   radius if radius != 0x8000 else None,
                "result":      res,
            })
