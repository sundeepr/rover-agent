#!/usr/bin/env python3
"""
record_all_cameras.py — simple hardcoded multi-camera recorder.

Opens the 3 RTSP cameras + 1 USB camera below, each on its own thread, and
writes each to its own MJPG .avi file (MJPG chosen to match session_recorder.py's
existing convention — each frame is independently decodable, so a crash or
Ctrl-C mid-recording doesn't corrupt the whole file the way some codecs can).

Everything is hardcoded per the ask — edit SOURCES below to change cameras.

Usage
─────
    python record_all_cameras.py
    # Ctrl-C to stop all recordings cleanly.

Output
──────
    recordings/<timestamp>/<name>.avi   — one file per camera
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

log = logging.getLogger("record_all_cameras")

# ── Hardcoded camera sources ─────────────────────────────────────────────────
SOURCES = {
    "cam_10_0_1_101": "rtsp://admin:Cam3ra_1234@10.0.1.101:554/video/live?channel=1&subtype=1",
    "cam_10_0_1_102": "rtsp://admin:Cam3ra_1234@10.0.1.102:554/video/live?channel=1&subtype=1",
    "cam_10_0_1_103": "rtsp://admin:Cam3ra_1234@10.0.1.103:554/video/live?channel=1&subtype=1",
    "usb_video0":     0,   # /dev/video0
}

OUTPUT_ROOT   = Path("recordings")
FALLBACK_FPS  = 15.0     # used when the source doesn't report a usable FPS
RECONNECT_S   = 3.0      # wait before retrying a dropped/failed camera


def _open_capture(source) -> "cv2.VideoCapture | None":
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    return cap


def _record_camera(name: str, source, out_path: Path, running: threading.Event) -> None:
    writer = None
    cap = None

    while running.is_set():
        if cap is None:
            log.info("[%s] connecting to %s …", name, source)
            cap = _open_capture(source)
            if cap is None:
                log.warning("[%s] failed to open — retrying in %.0fs", name, RECONNECT_S)
                time.sleep(RECONNECT_S)
                continue
            log.info("[%s] connected", name)

        ret, frame = cap.read()
        if not ret or frame is None:
            log.warning("[%s] read failed — reconnecting", name)
            cap.release()
            cap = None
            time.sleep(RECONNECT_S)
            continue

        if writer is None:
            h, w = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 1:
                fps = FALLBACK_FPS
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            log.info("[%s] recording %dx%d @ %.1ffps -> %s", name, w, h, fps, out_path)

        writer.write(frame)

    if cap is not None:
        cap.release()
    if writer is not None:
        writer.release()
    log.info("[%s] stopped, file closed", name)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    session_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    log.info("Recording to %s", session_dir.resolve())

    running = threading.Event()
    running.set()

    threads = []
    for name, source in SOURCES.items():
        out_path = session_dir / f"{name}.avi"
        t = threading.Thread(target=_record_camera, args=(name, source, out_path, running),
                             daemon=True, name=name)
        t.start()
        threads.append(t)

    log.info("Recording %d camera(s). Ctrl-C to stop.", len(threads))
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        log.info("Stopping…")
    finally:
        running.clear()
        for t in threads:
            t.join(timeout=5.0)
        log.info("All recordings closed in %s", session_dir.resolve())


if __name__ == "__main__":
    main()
