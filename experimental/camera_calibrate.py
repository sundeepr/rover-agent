#!/usr/bin/env python3
"""
camera_calibrate.py — camera centre calibration using red guard rails.

No screen needed — streams the annotated feed to web_server.py so you can
view it in the browser, and prints the offset to the terminal.

The script detects the two red guard rails and shows:
  • Yellow dashed line  — frame centre (50 %)
  • Green dashed line   — midpoint between the two rails
  • Arrow + text        — offset and which direction to move the camera

Move the physical camera until the offset reads ~0 px.

Usage
─────
    python experimental/camera_calibrate.py --device 2
    python experimental/camera_calibrate.py --device 2 --web-server http://192.168.1.10:5001

If rails are not detected, tune HSV bounds for red with --h-lo1/--h-hi1/--h-lo2/--h-hi2/--s-lo/--v-lo.
"""

import argparse
import time
import sys

import cv2
import numpy as np
import requests

# ── Default HSV bounds for red ────────────────────────────────────────────────
DEF_H_LO1, DEF_H_HI1 =   0,  12   # lower red hue range
DEF_H_LO2, DEF_H_HI2 = 165, 180   # upper red hue range (wraps at 180)
DEF_S_LO              =  80        # minimum saturation
DEF_V_LO              =  60        # minimum value (brightness)

MIN_RAIL_WIDTH = 10   # px — blobs narrower than this are ignored
MIN_RAIL_AREA  = 50   # px²


def detect_rails(mask: np.ndarray):
    """Return (left_cx, right_cx) for the two largest red blobs, or (None, None)."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        a = stats[i, cv2.CC_STAT_AREA]
        if w >= MIN_RAIL_WIDTH and a >= MIN_RAIL_AREA:
            blobs.append((int(centroids[i][0]), a))
    if len(blobs) < 2:
        return None, None
    blobs.sort(key=lambda b: b[1], reverse=True)   # biggest two
    xs = sorted([blobs[0][0], blobs[1][0]])
    return xs[0], xs[1]


def draw_vline_dashed(img, x, color, thickness=2, dash=12, gap=8):
    h = img.shape[0]
    y = 0
    while y < h:
        cv2.line(img, (x, y), (x, min(y + dash, h)), color, thickness)
        y += dash + gap


def push_frame(session: requests.Session, web_server: str, frame: np.ndarray):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    try:
        session.post(f"{web_server}/agent/frame?stream=realtime",
                     data=buf.tobytes(),
                     headers={"Content-Type": "image/jpeg"},
                     timeout=1.0)
    except Exception:
        pass


def run(args):
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera device {args.device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    low1 = np.array([args.h_lo1, args.s_lo, args.v_lo])
    hi1  = np.array([args.h_hi1, 255,       255      ])
    low2 = np.array([args.h_lo2, args.s_lo, args.v_lo])
    hi2  = np.array([args.h_hi2, 255,       255      ])

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    sess   = requests.Session()

    print(f"\n=== Camera Calibration  (device={args.device}) ===")
    print(f"Streaming to {args.web_server} — open the browser to see the feed.")
    print("Move the camera until offset ≈ 0 px. Ctrl+C to quit.\n")

    last_log = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        h, w    = frame.shape[:2]
        cx_frame = w // 2

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(cv2.inRange(hsv, low1, hi1),
                              cv2.inRange(hsv, low2, hi2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        left_cx, right_cx = detect_rails(mask)

        display = frame.copy()
        # Tint detected red regions
        tint = display.copy()
        tint[mask > 0] = (0, 0, 200)
        cv2.addWeighted(tint, 0.35, display, 0.65, 0, display)

        # Yellow centre line
        draw_vline_dashed(display, cx_frame, (0, 220, 255), thickness=2)
        cv2.putText(display, "centre", (cx_frame + 4, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        if left_cx is not None and right_cx is not None:
            midpoint  = (left_cx + right_cx) // 2
            offset    = midpoint - cx_frame
            pct       = offset / w * 100
            direction = "RIGHT" if offset > 0 else "LEFT" if offset < 0 else "CENTRED"
            colour    = (0, 255, 80) if abs(offset) < 5 else (0, 200, 255)

            # Rail centre marks
            for rx in (left_cx, right_cx):
                cv2.line(display, (rx, 0), (rx, h), (60, 60, 255), 1)

            # Green midpoint line
            draw_vline_dashed(display, midpoint, (0, 255, 80), thickness=2)
            cv2.putText(display, "rail mid", (midpoint + 4, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 80), 1, cv2.LINE_AA)

            # Arrow
            cv2.arrowedLine(display, (cx_frame, h // 2), (midpoint, h // 2),
                            (255, 200, 0), 2, tipLength=0.2)

            # HUD text
            lines = [
                f"Offset: {offset:+d} px  ({pct:+.1f}%)",
                "Camera centred!" if abs(offset) < 5 else f"Move camera {direction}",
            ]
            for i, txt in enumerate(lines):
                cv2.putText(display, txt, (12, 32 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)

            # Terminal log (1 Hz)
            if time.time() - last_log >= 1.0:
                status = "OK" if abs(offset) < 5 else f"move {direction}"
                print(f"  rails: left={left_cx}  right={right_cx}  mid={midpoint}"
                      f"  centre={cx_frame}  offset={offset:+d}px ({pct:+.1f}%)  → {status}")
                last_log = time.time()
        else:
            cv2.putText(display, "No rails detected — check HSV args (--h-lo1 etc.)",
                        (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2, cv2.LINE_AA)
            if time.time() - last_log >= 2.0:
                print("  No red rails detected. Try adjusting --s-lo or --v-lo.")
                last_log = time.time()

        push_frame(sess, args.web_server, display)
        time.sleep(0.05)   # ~20 fps


def main():
    ap = argparse.ArgumentParser(
        description="Camera centre calibration — streams annotated feed to web_server")
    ap.add_argument("--device",     type=int,   default=0,
                    help="Camera device index (default: 0)")
    ap.add_argument("--web-server", type=str,   default="http://localhost:5001",
                    help="web_server.py URL (default: http://localhost:5001)")
    # HSV tuning
    ap.add_argument("--h-lo1", type=int, default=DEF_H_LO1, help="Lower red hue min (def 0)")
    ap.add_argument("--h-hi1", type=int, default=DEF_H_HI1, help="Lower red hue max (def 12)")
    ap.add_argument("--h-lo2", type=int, default=DEF_H_LO2, help="Upper red hue min (def 165)")
    ap.add_argument("--h-hi2", type=int, default=DEF_H_HI2, help="Upper red hue max (def 180)")
    ap.add_argument("--s-lo",  type=int, default=DEF_S_LO,  help="Min saturation (def 80)")
    ap.add_argument("--v-lo",  type=int, default=DEF_V_LO,  help="Min brightness (def 60)")
    args = ap.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
