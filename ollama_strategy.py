"""
OllamaStrategy — rover navigation using a local Ollama vision model.

Sends a rolling buffer of recent forward-camera frames plus their previous
predictions to Ollama. If a downward-facing camera is available it is also
sent as an extra context image showing the wheels and surrounding plants,
so the model can detect when the rover is drifting over crops.

The next_point [x, y] (normalized 0-1) is converted to a drive command:
  - x < 0.45  → steer left
  - x > 0.55  → steer right
  - else       → straight

Usage (with down camera):
    python rover_agent.py --strategy ollama \\
        --ollama-server http://<host>:11434 \\
        --ollama-model qwen2.5vl \\
        --down-device 1 \\
        --rover atlas --atlas-port /dev/ttyACM0
"""

import base64
import json
import logging
import re
import threading
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


def _letterbox(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize preserving aspect ratio, pad with black to exact w×h."""
    fh, fw = frame.shape[:2]
    scale  = min(w / fw, h / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    y0  = (h - nh) // 2
    x0  = (w - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out

# Drive parameters
_FWD_VEL       = 60     # mm/s forward speed
_TURN_RADIUS   = 400    # mm turning radius magnitude
_DEAD_BAND     = 0.05   # ± around x=0.5 treated as straight


# ── Prompt ────────────────────────────────────────────────────────────────────

def _build_prompt(history: deque, has_down_cam: bool, rover_type: str = "atlas") -> str:
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

    down_section = ""
    if has_down_cam:
        down_section = f"""
The LAST image (image {n + 1}) is from the DOWNWARD-facing camera mounted \
below the rover, showing the wheels and immediately surrounding ground.
Use it to detect if any plants are currently under or very close to the wheels.
If plants are visible under the wheels, set wheel_on_crop=true and steer \
away from them (opposite direction to the plant contact side).
"""

    down_field = (
        '\n  "wheel_on_crop": true/false,\n'
        '  "crop_contact_side": "left|right|none",'
        if has_down_cam else ""
    )

    if rover_type == "roomba":
        scene_understanding = """\
You are the navigation system of a floor-cleaning robot (Roomba-like).

========================
OBJECTIVE
========================
Follow the BROWN CARPET visible in the forward camera.

- The brown carpet is the ONLY valid path.
- The robot must stay centered on the carpet at all times.
- If the carpet curves, follow the curve smoothly.
- If near the edge, steer back toward the center.

========================
VISUAL RULES
========================
Identify the brown carpet based on:
- Continuous brown region on the floor
- Consistent texture (fabric-like, not reflective)
- Distinct from surrounding floor (tile/wood/other colors)

Ignore:
- Walls, furniture, shadows
- Other rugs or non-brown surfaces
- Reflections or lighting artifacts

========================
BEHAVIOR RULES
========================
1. CENTERING:
   - Always bias steering toward the center of the carpet.

2. CURVATURE:
   - If the carpet bends, adjust path_points to follow the curve.
   - Do NOT output a straight line unless the carpet is straight.

3. EDGE CORRECTION:
   - If the robot is close to one edge, steer toward the opposite side.

4. LOSS OF PATH:
   - If the carpet is partially visible → continue toward visible segment.
   - If not visible → set path_visible=false and confidence < 0.3.

========================
INPUT
========================
You are given ONE forward-facing camera image.

========================
OUTPUT (STRICT JSON ONLY)
========================
{
  "motion_direction": "left | right | straight",
  "next_point": [x, y],
  "path_points": [[x, y], ...],
  "path_visible": true/false,
  "confidence": 0.0-1.0,
  "reason": "one short sentence"
}

========================
COORDINATE SYSTEM
========================
All values normalized [0.0, 1.0]:

- x: 0.0 = left edge, 1.0 = right edge
- y: 0.0 = top (far), 1.0 = bottom (near robot)

========================
CONSTRAINTS
========================
next_point:
- Must lie on the CENTER of the carpet
- Must be in lower-middle region (0.5 ≤ y ≤ 0.9)

path_points:
- 4 to 6 points
- Ordered from near → far (y decreases)
- Must follow carpet centerline
- x MUST change if carpet curves
- DO NOT output constant x unless carpet is perfectly straight

motion_direction:
- "left" if next_point.x < 0.45
- "right" if next_point.x > 0.55
- "straight" otherwise

confidence:
- High (0.8–1.0): carpet clearly visible and continuous
- Medium (0.5–0.8): partial visibility
- Low (<0.5): unclear or missing

========================
FAILURE CONDITIONS (AVOID)
========================
- Do NOT output points off the carpet
- Do NOT output straight path for curved carpet
- Do NOT hallucinate carpet if none visible
- Do NOT include obstacles in path

========================
GOAL
========================
Produce a smooth, centered trajectory that keeps the robot fully on the brown carpet.
"""
    else:
        scene_understanding = """\
SCENE UNDERSTANDING:
- Crop rows: uniform, evenly-spaced plants arranged in straight or gently \
curved lines. These are the TARGET rows the rover must drive BETWEEN.
- Weeds: scattered randomly, irregular shapes, NOT in rows. \
The rover may drive over weeds — they are NOT obstacles.
- The rover must stay in the open soil corridor between the two nearest crop rows.
- NEVER steer the rover onto a crop row plant."""

    path_label = "carpet centre" if rover_type == "roomba" else "soil corridor centre"
    path_visible_desc = (
        "false only if no brown carpet is visible in the forward camera"
        if rover_type == "roomba"
        else "false only if no clear soil corridor between crop rows is visible"
    )

    return f"""\
You are the navigation system of a rover.

{scene_understanding}

You are given {n} forward-camera image(s) in chronological order \
(oldest first, newest last).{down_section}

Previous predicted navigation points:
{history_text}

Analyse the CURRENT forward-camera frame (image {n}) and return ONLY valid JSON:
{{{down_field}
  "motion_direction": "left|right|straight",
  "next_point": [x, y],
  "path_points": [[x, y], ...],
  "path_visible": true/false,
  "confidence": 0.0-1.0,
  "reason": "one sentence"
}}

Coordinate rules (ALL values normalized 0.0-1.0):
  x=0.0=left edge, x=1.0=right edge, x=0.5=center.
  y=0.0=top,  y=1.0=bottom.

next_point:
  - The single point the rover should steer toward in the CURRENT frame.
  - Must be on the {path_label}.
  - If wheel_on_crop=true, steer immediately away from crop_contact_side.

path_points:
  - 4-6 points tracing the {path_label}, bottom (y≈0.9) to top (y≈0.2).
  - x MUST vary to follow the path curvature. A constant x is WRONG.

motion_direction: inferred from how next_point x shifted across past frames.

path_visible: {path_visible_desc}.
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

    direction     = result.get("motion_direction", "")
    conf          = result.get("confidence", 0.0)
    visible       = result.get("path_visible", False)
    wheel_on_crop = result.get("wheel_on_crop", False)
    contact_side  = result.get("crop_contact_side", "none")

    vis_color = (0, 255, 100) if visible else (0, 0, 255)
    cv2.putText(out, f"{'PATH OK' if visible else 'NO PATH'}  conf:{conf:.2f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, vis_color, 2)
    cv2.putText(out, f"direction: {direction}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if wheel_on_crop:
        cv2.putText(out, f"!! WHEEL ON CROP ({contact_side}) !!", (10, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
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
        rover_type: str = "atlas",
    ):
        self._url        = ollama_url.rstrip("/")
        self._model      = model
        self._rover_type = rover_type
        self._history: deque = deque(maxlen=history_size)
        self._down_frame: np.ndarray | None = None
        self._down_lock  = threading.Lock()
        self._response_times: deque = deque(maxlen=5)   # last 5 elapsed seconds
        self._last_vel    = 0
        self._last_radius = 0x8000
        self._drive_lock  = threading.Lock()
        log.info("OllamaStrategy: model=%s  server=%s  history=%d  rover=%s",
                 model, ollama_url, history_size, rover_type)

    @property
    def name(self) -> str:
        return "ollama"

    def on_reset(self) -> None:
        self._history.clear()
        log.info("OllamaStrategy reset — history cleared")

    def update_down_frame(self, frame: np.ndarray) -> None:
        """Called by rover_agent's down-camera loop with each new frame."""
        with self._down_lock:
            self._down_frame = frame

    def _get_down_frame(self) -> np.ndarray | None:
        """Called by agent_publisher to push down frame to web UI."""
        with self._down_lock:
            return self._down_frame.copy() if self._down_frame is not None else None

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

        # Stop rover while inference is running
        if rover_ctrl and not state.paused.is_set():
            rover_ctrl.stop()

        # Letterbox to preserve aspect ratio
        send = _letterbox(frame, _SEND_W, _SEND_H)

        # Snapshot down frame (may be None if no down camera)
        with self._down_lock:
            down = self._down_frame.copy() if self._down_frame is not None else None

        # Add current frame to history (result filled in after inference)
        self._history.append((send, None))

        # Build images: forward history frames, then down frame last (if available)
        images = []
        for f, _ in self._history:
            _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
            images.append(base64.b64encode(buf.tobytes()).decode())

        if down is not None:
            down_small = _letterbox(down, _SEND_W, _SEND_H)
            _, buf = cv2.imencode(".jpg", down_small,
                                  [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
            images.append(base64.b64encode(buf.tobytes()).decode())
            log.debug("Down camera frame included in Ollama request")

        prompt  = _build_prompt(self._history, has_down_cam=down is not None,
                               rover_type=self._rover_type)
        log.info("Step %d | Prompt:\n%s", step, prompt)
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

        wheel_on_crop    = result.get("wheel_on_crop", False)
        crop_contact_side = result.get("crop_contact_side", "none")

        # Track response time
        self._response_times.append(round(elapsed, 2))
        log.info("Step %d | visible=%s  direction=%s  next=%s  wheel_on_crop=%s(%s)  elapsed=%.2fs",
                 step, path_visible, direction, next_point,
                 wheel_on_crop, crop_contact_side, elapsed)

        if not path_visible or not next_point:
            self._write_result(state, step, phase, result, 0, 0x8000,
                               "no_path", elapsed)
            return

        vel, radius = _next_point_to_drive(next_point)
        r_str = "straight" if radius == 0x8000 else f"r={radius}mm"
        log.info("Step %d | vel=%d  %s  → executing for 1s", step, vel, r_str)

        with self._drive_lock:
            self._last_vel    = vel
            self._last_radius = radius

        display = _annotate(frame, result)
        with state.llm_lock:
            state.llm_frame = display

        self._write_result(state, step, phase, result, vel, radius,
                           "navigating", elapsed)

        # Drive for 1 second then stop before next inference
        if rover_ctrl and not state.paused.is_set():
            operator_active = (state.operator_control is not None
                               and state.operator_until > time.time())
            if not operator_active:
                rover_ctrl.drive_raw(vel, radius)
                time.sleep(1.0)
                rover_ctrl.stop()
                log.info("Step %d | drive complete — stopping rover", step)

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
            "response_times":  list(self._response_times),
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
