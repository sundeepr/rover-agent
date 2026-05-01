#!/usr/bin/env python3
"""
ollama_waypoint_viewer.py

Sends the last N frames (history buffer) plus their predictions to a local
Ollama vision model. The model infers the rover's motion direction from the
sequence and predicts where to navigate in the current frame.

Usage:
    python experimental/ollama_waypoint_viewer.py --image frame.jpg
    python experimental/ollama_waypoint_viewer.py --video clip.mp4
    python experimental/ollama_waypoint_viewer.py --camera 0
    python experimental/ollama_waypoint_viewer.py --video clip.mp4 --model qwen2.5vl:7b --history 5

Keyboard: Space = run inference   q/Esc = quit
"""

import argparse
import base64
import json
import re
import sys
from collections import deque

import cv2
import numpy as np

OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5vl"
HISTORY_SIZE  = 5


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(history: deque) -> str:
    """Build a temporal prompt from the frame history."""
    n = len(history)

    # Summarise past predictions (all frames except the last)
    past_lines = []
    for i, (_, result) in enumerate(list(history)[:-1]):
        if result and result.get("next_point"):
            np_ = result["next_point"]
            past_lines.append(f"  Frame {i+1}: next_point = [{np_[0]:.2f}, {np_[1]:.2f}]")
        else:
            past_lines.append(f"  Frame {i+1}: next_point = unknown")

    history_text = "\n".join(past_lines) if past_lines else "  (no prior frames)"

    return f"""\
You are the navigation system of a farm rover driving between crop rows.

SCENE UNDERSTANDING:
- Crop rows: uniform, evenly-spaced plants in straight or gently curved lines. \
The rover must drive BETWEEN these rows and NEVER onto them.
- Weeds: scattered randomly with irregular shapes, NOT in rows. \
The rover may drive over weeds — they are NOT obstacles.
- Stay in the open soil corridor between the two nearest crop rows.

You are given {n} image(s) in chronological order (oldest first, newest last).
Previous predicted navigation points:
{history_text}

Analyse the LAST image (frame {n}) and return ONLY valid JSON:
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
  y=0.0=top,  y=1.0=bottom.

next_point:
  - Where the rover should steer in the CURRENT (last) image.
  - Must be in the open soil gap between the two nearest crop rows.
  - Weeds are NOT obstacles — only avoid crop row plants.

path_points:
  - 4-6 points tracing the soil corridor centre, bottom (y≈0.9) to top (y≈0.2).
  - x MUST vary across points to follow corridor curvature. Constant x is WRONG.

motion_direction: inferred from how next_point x shifted across past frames.
  left=x decreasing, right=x increasing, straight=x stable.

path_visible: false only if no clear soil corridor between crop rows is visible.
"""


# ── Ollama call ───────────────────────────────────────────────────────────────

def query_ollama(model: str, history: deque) -> str:
    images = []
    for frame, _ in history:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        images.append(base64.b64encode(buf.tobytes()).decode())

    prompt  = _build_prompt(history)
    payload = json.dumps({
        "model": model, "prompt": prompt, "images": images, "stream": False
    }).encode()

    import urllib.request
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read()).get("response", "").strip()


# ── JSON extraction ───────────────────────────────────────────────────────────

def extract_json(text: str) -> dict | None:
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


# ── Drawing ───────────────────────────────────────────────────────────────────

def _px(nx, ny, w, h):
    return int(nx * w), int(ny * h)


def draw_result(frame: np.ndarray, result: dict) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    path_visible = result.get("path_visible", False)
    conf         = result.get("confidence", 0.0)
    direction    = result.get("motion_direction", "")

    # Green path dots + line
    pts    = result.get("path_points", [])
    px_pts = [_px(x, y, w, h) for x, y in pts]
    for i, pt in enumerate(px_pts):
        cv2.circle(out, pt, 6, (0, 200, 0), -1)
        if i > 0:
            cv2.line(out, px_pts[i - 1], pt, (0, 200, 0), 2)

    # Next point — large bright dot with white ring
    np_ = result.get("next_point")
    if np_:
        nx, ny = _px(np_[0], np_[1], w, h)
        cv2.circle(out, (nx, ny), 12, (0, 255, 0), -1)
        cv2.circle(out, (nx, ny), 14, (255, 255, 255), 2)
        cv2.putText(out, "NEXT", (nx + 16, ny + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # HUD
    vis_text  = "PATH OK" if path_visible else "NO PATH"
    vis_color = (0, 255, 100) if path_visible else (0, 0, 255)
    dir_color = {"left": (0, 200, 255), "right": (0, 200, 255),
                 "straight": (100, 255, 100)}.get(direction, (200, 200, 200))
    cv2.putText(out, f"{vis_text}  conf:{conf:.2f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, vis_color, 2)
    cv2.putText(out, f"direction: {direction}", (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, dir_color, 1)
    cv2.putText(out, result.get("reason", "")[:90], (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 100), 1)
    return out


# ── Inference ─────────────────────────────────────────────────────────────────

def process(frame: np.ndarray, model: str, history: deque) -> np.ndarray:
    # Add current frame with no result yet
    history.append((frame, None))

    print(f"\nQuerying {model} ({len(history)} frame(s) in history)...", flush=True)
    try:
        raw = query_ollama(model, history)
    except Exception as e:
        print(f"ERROR: {e}")
        return frame

    print(f"Response:\n{raw}\n")

    result = extract_json(raw)
    if result is None:
        print("Could not parse JSON from response.")
        return frame

    # Update last history entry with the result
    history[-1] = (frame, result)

    print(f"Path visible    : {result.get('path_visible')}")
    print(f"Motion direction: {result.get('motion_direction')}")
    print(f"Next point      : {result.get('next_point')}")
    print(f"Path points     : {result.get('path_points')}")
    print(f"Confidence      : {result.get('confidence')}")
    print(f"Reason          : {result.get('reason')}")

    return draw_result(frame, result)


# ── Input sources ─────────────────────────────────────────────────────────────

def run_image(path: str, model: str, history_size: int) -> None:
    frame = cv2.imread(path)
    if frame is None:
        sys.exit(f"Cannot read: {path}")
    history = deque(maxlen=history_size)
    out = process(frame, model, history)
    cv2.imshow("Rover Path", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(path: str, model: str, history_size: int) -> None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {path}")
    print("Space = run inference on current frame   q/Esc = quit")
    history  = deque(maxlen=history_size)
    annotated = None
    ret, frame = cap.read()
    while ret:
        cv2.imshow("Rover Path", annotated if annotated is not None else frame)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):
            break
        annotated = process(frame, model, history)
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()


def run_camera(index: int, model: str, history_size: int) -> None:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {index}")
    print("Space = capture & infer   q/Esc = quit")
    history   = deque(maxlen=history_size)
    annotated = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Rover Path", annotated if annotated is not None else frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord(' '):
            annotated = process(frame, model, history)
    cap.release()
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Farm rover path viewer with frame history context")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",   metavar="FILE")
    src.add_argument("--video",   metavar="FILE")
    src.add_argument("--camera",  metavar="IDX", type=int)
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--history", default=HISTORY_SIZE, type=int,
                        help=f"Number of frames to keep in history (default: {HISTORY_SIZE})")
    args = parser.parse_args()

    if args.image:
        run_image(args.image, args.model, args.history)
    elif args.video:
        run_video(args.video, args.model, args.history)
    else:
        run_camera(args.camera, args.model, args.history)


if __name__ == "__main__":
    main()
