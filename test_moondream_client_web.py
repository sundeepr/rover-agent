#!/usr/bin/env python3
"""
test_moondream_client_web.py — end-to-end test of the REAL client/server split:
this process has no torch/transformers/model weights at all, it only talks
over the network to a running moondream_cloud_server.py via MoondreamClient
(moondream_client.py), the same way a strategy eventually will.

Unlike test_moondream_detect.py / test_moondream_detect_web.py (which load
Moondream2 in-process and call InferenceEngine.detect() directly, bypassing
the wire protocol entirely), this script exercises the actual
WebSocket round trip: JPEG → base64 → {"type":"detect",...} → server →
{"type":"detections",...} → base64-decoded boxes back here.

Viewed in the browser via the same mechanism rover_agent.py uses:
AgentPublisher -> a standalone web_server.py process.

Usage
─────
    # 1. On the GPU box — start the model server once:
    python moondream_cloud_server.py --model-path /path/to/moondream2 \\
        --device-map cuda:0 --port 8767

    # 2. On the web UI box (can be a different, GPU-less machine) — start it once:
    python web_server.py --port 5001

    # 3. Run this anywhere on the network — camera + client, no GPU needed here:
    python test_moondream_client_web.py \\
        --moondream-server ws://<gpu-box>:8767 \\
        --device rtsp://10.0.1.103:554/video/live?channel=1\\&subtype=1 \\
        --object "solar panel" \\
        --web-server http://<web-ui-box>:5001

Then open http://<web-ui-box>:5001/ in a browser.
"""

import argparse
import logging
import threading
import time

import cv2

from agent_publisher import AgentPublisher
from frame_source import open_frame_source
from moondream_client import MoondreamClient
from navigation_strategy import AgentState

log = logging.getLogger("test_moondream_client_web")


def _device(value: str):
    """Accept an int index, a /dev path, or a ws://|rtsp:// URL (same as rover_agent.py)."""
    try:
        return int(value)
    except ValueError:
        return value


_BOX_COLOR     = (0, 255, 255)   # bright yellow (BGR) — high contrast against sky/panel/vegetation
_CORNER_LEN    = 24
_BOX_THICKNESS = 4


def _annotate(frame, objects: list[dict], object_name: str, elapsed: float, connected: bool):
    out = frame.copy()
    h, w = out.shape[:2]

    for obj in objects:
        # Clamp well inside the frame — boxes that touch y=0/y=h exactly get
        # their top/bottom edges drawn right on the frame border, easy to miss
        # or clipped by downstream resizing. See test_moondream_detect_web.py
        # for the full story on why this matters.
        x1 = max(2, min(int(obj["x_min"]), w - 3))
        y1 = max(2, min(int(obj["y_min"]), h - 3))
        x2 = max(x1 + 1, min(int(obj["x_max"]), w - 2))
        y2 = max(y1 + 1, min(int(obj["y_max"]), h - 2))

        cv2.rectangle(out, (x1, y1), (x2, y2), _BOX_COLOR, _BOX_THICKNESS)
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

    if not connected:
        status = "MOONDREAM SERVER NOT CONNECTED"
        colour = (0, 60, 220)
    elif objects:
        status = f"'{object_name}': {len(objects)} found  elapsed={elapsed:.2f}s"
        colour = (0, 220, 80)
    else:
        status = f"'{object_name}': NONE found  elapsed={elapsed:.2f}s"
        colour = (0, 60, 220)
    cv2.rectangle(out, (0, 0), (w, 36), (30, 30, 30), -1)
    cv2.putText(out, status, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    return out


def _camera_thread(state: AgentState, device, running: threading.Event) -> None:
    """Continuously grab frames into state.raw_frame — same role agent_loop plays."""
    src = open_frame_source(device, "moondream-client-test")
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


def _detect_loop(state: AgentState, client: MoondreamClient, object_name: str,
                 interval_s: float, running: threading.Event) -> None:
    """Sends the latest camera frame to moondream_cloud_server.py every interval_s seconds."""
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
        objects = client.detect(buf.tobytes(), object_name, timeout=30.0)
        elapsed = time.time() - t0
        connected = client.ready

        if objects is None:
            log.warning("detect('%s') -> no response  connected=%s  elapsed=%.2fs",
                       object_name, connected, elapsed)
            objects = []
        else:
            log.info("detect('%s') -> %d objects  elapsed=%.2fs  raw=%s",
                     object_name, len(objects), elapsed, objects)

        annotated = _annotate(frame, objects, object_name, elapsed, connected)
        with state.llm_lock:
            state.llm_frame = annotated

        with state.result_lock:
            state.latest_result = {
                "strategy":   "moondream_client_smoketest",
                "step":       state.step,
                "object":     object_name,
                "found":      len(objects),
                "objects":    objects,
                "elapsed_s":  elapsed,
                "connected":  connected,
            }
            state.step += 1

        time.sleep(interval_s)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moondream-server", type=str, required=True, metavar="URL",
                        help="WebSocket URL of moondream_cloud_server.py, "
                             "e.g. ws://192.168.1.100:8767")
    parser.add_argument("--device", type=_device, default=0, metavar="INDEX|PATH|RTSP_URL",
                        help="Camera device index, /dev path, or rtsp:// URL (default 0)")
    parser.add_argument("--object", type=str, default="solar panel",
                        help="Object name to ground/detect (default 'solar panel')")
    parser.add_argument("--detect-interval", type=float, default=2.0, metavar="SECS",
                        help="Seconds between detect() calls (default 2.0 — "
                             "detect is too slow to run every frame)")
    parser.add_argument("--web-server", type=str, default="http://localhost:5001",
                        metavar="URL",
                        help="URL of the running web_server.py (default: "
                             "http://localhost:5001) — start it separately first: "
                             "python web_server.py --port 5001")
    args = parser.parse_args()

    log.info("Connecting to Moondream server at %s (no local model load — "
             "this process never imports torch)", args.moondream_server)
    client = MoondreamClient(args.moondream_server)

    state = AgentState()
    running = threading.Event()
    running.set()

    threading.Thread(target=_camera_thread, args=(state, args.device, running),
                     daemon=True, name="camera").start()
    threading.Thread(target=_detect_loop,
                     args=(state, client, args.object, args.detect_interval, running),
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
