#!/usr/bin/env python3
"""
test_moondream_detect_web.py — Moondream2 .detect() smoke test, viewed in
the same web UI the navigation strategies use: a standalone web_server.py
process + AgentPublisher pushing frames/status to it over HTTP. Same
mechanism rover_agent.py itself uses — not the (unused) web_display.py.

Same purpose as test_moondream_detect.py (verify .detect() exists and can
actually find a solar panel) but for a headless/SSH'd-in machine — no
display/X needed, just open a browser to the web_server.py UI.

  - "Live camera" feed  → raw camera feed, updated every frame.
  - "LLM" feed          → last annotated detect() result (green boxes +
    status banner), refreshed once every --detect-interval seconds —
    detect() is slow, so unlike the raw feed it is NOT run every frame
    (same reasoning as row_change_strategy's qwen_interval_s).
  - Pause button on the web UI pauses the detect loop (camera keeps
    streaming).

Usage
─────
    # 1. Start the standalone web server once (survives this script restarting):
    python web_server.py --port 5001

    # 2. Run this against a local camera or an RTSP stream:
    python test_moondream_detect_web.py --model-path /path/to/moondream2 \\
        --device 0 --object "solar panel" --web-server http://localhost:5001

    python test_moondream_detect_web.py --model-path /path/to/moondream2 \\
        --device rtsp://10.0.1.103:554/stream1 --object "solar panel" \\
        --web-server http://localhost:5001

Then open http://<web_server-host>:5001/ in a browser.
"""

import argparse
import logging
import sys
import threading
import time

import cv2

from agent_publisher import AgentPublisher
from frame_source import open_frame_source
from moondream_cloud_server import InferenceEngine
from navigation_strategy import AgentState

log = logging.getLogger("test_moondream_detect_web")


def _device(value: str):
    """Accept an int index, a /dev path, or a ws://|rtsp:// URL (same as rover_agent.py)."""
    try:
        return int(value)
    except ValueError:
        return value


_BOX_COLOR    = (0, 255, 255)   # bright yellow (BGR) — high contrast against sky/panel/vegetation
_CORNER_LEN   = 24
_BOX_THICKNESS = 4


def _annotate(frame, objects: list[dict], object_name: str, elapsed: float):
    out = frame.copy()
    h, w = out.shape[:2]

    for obj in objects:
        # Clamp well inside the frame — boxes that touch y=0/y=h exactly (common
        # when Moondream's box spans the full visible height) get their top/bottom
        # edges drawn right on the frame border, where they're easy to miss or get
        # clipped by downstream JPEG/video resizing. Inset by a couple px so every
        # edge is guaranteed to render on visible pixels.
        x1 = max(2, min(int(obj["x_min"]), w - 3))
        y1 = max(2, min(int(obj["y_min"]), h - 3))
        x2 = max(x1 + 1, min(int(obj["x_max"]), w - 2))
        y2 = max(y1 + 1, min(int(obj["y_max"]), h - 2))

        cv2.rectangle(out, (x1, y1), (x2, y2), _BOX_COLOR, _BOX_THICKNESS)
        # Corner markers in a contrasting outline so the box reads even if an
        # edge lands on a similarly-coloured background.
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                               (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(out, (cx, cy), (cx + dx * _CORNER_LEN, cy), (0, 0, 0), _BOX_THICKNESS + 2)
            cv2.line(out, (cx, cy), (cx, cy + dy * _CORNER_LEN), (0, 0, 0), _BOX_THICKNESS + 2)
            cv2.line(out, (cx, cy), (cx + dx * _CORNER_LEN, cy), _BOX_COLOR, _BOX_THICKNESS)
            cv2.line(out, (cx, cy), (cx, cy + dy * _CORNER_LEN), _BOX_COLOR, _BOX_THICKNESS)

        area = (x2 - x1) * (y2 - y1)
        label_y = y1 - 10 if y1 > 30 else y2 + 26
        cv2.putText(out, f"{object_name} ({area}px^2)", (x1 + 4, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, f"{object_name} ({area}px^2)", (x1 + 4, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _BOX_COLOR, 2, cv2.LINE_AA)

    status = (f"'{object_name}': {len(objects)} found  elapsed={elapsed:.2f}s"
              if objects else f"'{object_name}': NONE found  elapsed={elapsed:.2f}s")
    colour = (0, 220, 80) if objects else (0, 60, 220)
    cv2.rectangle(out, (0, 0), (w, 36), (30, 30, 30), -1)
    cv2.putText(out, status, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    return out


def _camera_thread(state: AgentState, device, running: threading.Event) -> None:
    """
    Continuously grab frames into state.raw_frame — same role agent_loop plays.

    Uses frame_source.open_frame_source() so device can be a local index/path
    ("/dev/video0"), a "ws://" stream, or an "rtsp://" URL (auto-reconnecting).
    """
    src = open_frame_source(device, "moondream-test")
    if src is None:
        log.error("Cannot open camera/stream %s", device)
        running.clear()
        return
    log.info("Camera/stream %s opened", device)

    while running.is_set():
        frame = src.read()
        if frame is None:
            time.sleep(0.1)
            continue
        with state.raw_lock:
            state.raw_frame = frame
        time.sleep(0.03)   # ~30fps cap on the raw feed

    src.release()


def _detect_loop(state: AgentState, engine: InferenceEngine, object_name: str,
                 interval_s: float, running: threading.Event) -> None:
    """Runs detect() on the latest camera frame every interval_s seconds."""
    while running.is_set():
        if state.paused.is_set():
            time.sleep(0.2)
            continue

        with state.raw_lock:
            frame = state.raw_frame.copy() if state.raw_frame is not None else None
        if frame is None:
            time.sleep(0.2)
            continue

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            time.sleep(interval_s)
            continue

        t0 = time.time()
        try:
            result = engine.detect(buf.tobytes(), object_name)
        except Exception as e:
            log.error("detect() raised: %s", e, exc_info=True)
            time.sleep(interval_s)
            continue

        elapsed = result.get("elapsed", time.time() - t0)
        objects = result.get("objects", [])
        log.info("detect('%s') -> %d objects  elapsed=%.2fs  raw=%s",
                 object_name, len(objects), elapsed, objects)

        annotated = _annotate(frame, objects, object_name, elapsed)
        with state.llm_lock:
            state.llm_frame = annotated

        with state.result_lock:
            state.latest_result = {
                "strategy":   "moondream_detect_smoketest",
                "step":       state.step,
                "object":     object_name,
                "found":      len(objects),
                "objects":    objects,
                "elapsed_s":  elapsed,
            }
            state.step += 1

        time.sleep(interval_s)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, metavar="DIR",
                        help="Path to downloaded Moondream2 directory")
    parser.add_argument("--device", type=_device, default=0, metavar="INDEX|PATH|RTSP_URL",
                        help="Camera device index, /dev path, or rtsp:// URL (default 0)")
    parser.add_argument("--object", type=str, default="solar panel",
                        help="Object name to ground/detect (default 'solar panel')")
    parser.add_argument("--detect-interval", type=float, default=2.0, metavar="SECS",
                        help="Seconds between detect() calls (default 2.0 — "
                             "detect is too slow to run every frame)")
    parser.add_argument("--device-map", default="auto",
                        help="HuggingFace device_map (default: auto)")
    parser.add_argument("--web-server", type=str, default="http://localhost:5001",
                        metavar="URL",
                        help="URL of the running web_server.py (default: "
                             "http://localhost:5001) — start it separately first: "
                             "python web_server.py --port 5001")
    args = parser.parse_args()

    log.info("Loading Moondream2 from %s (this can take a while)…", args.model_path)
    engine = InferenceEngine(model_path=args.model_path, device_map=args.device_map)
    engine.load()

    if not hasattr(engine._model, "detect"):
        sys.exit(f"FATAL: loaded model has no .detect() method — "
                 f"available attrs: {[a for a in dir(engine._model) if not a.startswith('_')]}")
    log.info(".detect() method found on model")

    state = AgentState()
    running = threading.Event()
    running.set()

    threading.Thread(target=_camera_thread, args=(state, args.device, running),
                     daemon=True, name="camera").start()
    threading.Thread(target=_detect_loop,
                     args=(state, engine, args.object, args.detect_interval, running),
                     daemon=True, name="detect").start()

    log.info("Publishing to %s — open that URL in a browser to view", args.web_server)
    publisher = AgentPublisher(args.web_server)
    try:
        publisher.run(state, rover_ctrl=None, strategy=None)
    except KeyboardInterrupt:
        pass
    finally:
        running.clear()


if __name__ == "__main__":
    main()
