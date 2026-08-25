#!/usr/bin/env python3
"""
test_moondream_detect.py — standalone smoke test for Moondream2's .detect()
grounding API, no websocket server involved.

Purpose: verify BEFORE writing any strategy code that Moondream2's .detect()
exists on the installed checkpoint, returns the expected shape, and can
actually find a solar panel in a live camera frame. This is the "Risk to
verify early" step from the solar_dock plan — moondream_cloud_server.py is
loaded with trust_remote_code=True, so the exact .detect() method
name/return shape can vary between checkpoints.

Loads the model directly in-process (reuses InferenceEngine from
moondream_cloud_server.py) — no server, no websocket.

Usage
─────
    python test_moondream_detect.py --model-path /path/to/moondream2 \\
        --device 0 --object "solar panel"

Keys
────
    q / ESC  — quit
    d        — run detect() on the current frame (detect is slow — don't
               run it every frame)
    s        — save the last annotated result to /tmp/moondream_detect.jpg
"""

import argparse
import logging
import sys
import time

import cv2

from moondream_cloud_server import InferenceEngine

log = logging.getLogger("test_moondream_detect")


_BOX_COLOR     = (0, 255, 255)   # bright yellow (BGR) — high contrast against sky/panel/vegetation
_CORNER_LEN    = 24
_BOX_THICKNESS = 4


def _annotate(frame, objects: list[dict], object_name: str, elapsed: float):
    out = frame.copy()
    h, w = out.shape[:2]

    for obj in objects:
        # Clamp well inside the frame — boxes that touch y=0/y=h exactly (common
        # when Moondream's box spans the full visible height) get their top/bottom
        # edges drawn right on the frame border, easy to miss or clipped by
        # downstream resizing. Inset by a couple px so every edge is guaranteed
        # to render on visible pixels.
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
    cv2.putText(out, "d=detect  s=save  q=quit", (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return out


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, metavar="DIR",
                        help="Path to downloaded Moondream2 directory")
    parser.add_argument("--device", type=int, default=0,
                        help="Camera device index (default 0)")
    parser.add_argument("--object", type=str, default="solar panel",
                        help="Object name to ground/detect (default 'solar panel')")
    parser.add_argument("--device-map", default="auto",
                        help="HuggingFace device_map (default: auto)")
    args = parser.parse_args()

    log.info("Loading Moondream2 from %s (this can take a while)…", args.model_path)
    engine = InferenceEngine(model_path=args.model_path, device_map=args.device_map)
    engine.load()

    # ── First: confirm .detect() exists and returns the expected shape,
    # independent of the camera, before touching cv2.VideoCapture at all ──
    if not hasattr(engine._model, "detect"):
        sys.exit(f"FATAL: loaded model has no .detect() method — "
                 f"available attrs: {[a for a in dir(engine._model) if not a.startswith('_')]}")
    log.info(".detect() method found on model — proceeding to camera test")

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("Camera: %dx%d  object='%s'", w, h, args.object)
    log.info("Keys: d=detect (runs .detect() on current frame)  s=save  q=quit")

    last_annotated = None

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Camera read failed")
            break

        disp = last_annotated if last_annotated is not None else frame
        cv2.imshow("moondream detect() smoke test", disp)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        elif key == ord('d'):
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                log.warning("JPEG encode failed")
                continue
            t0 = time.time()
            try:
                result = engine.detect(buf.tobytes(), args.object)
            except Exception as e:
                log.error("detect() raised: %s", e, exc_info=True)
                continue
            elapsed = result.get("elapsed", time.time() - t0)
            objects = result.get("objects", [])
            log.info("detect('%s') -> %d objects  elapsed=%.2fs  raw=%s",
                     args.object, len(objects), elapsed, objects)
            last_annotated = _annotate(frame, objects, args.object, elapsed)

        elif key == ord('s') and last_annotated is not None:
            cv2.imwrite("/tmp/moondream_detect.jpg", last_annotated)
            log.info("Saved /tmp/moondream_detect.jpg")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
