#!/usr/bin/env python3
"""
test_live_line_follow.py — live camera test for line_follow detection.

Opens the camera, runs the colour-blob detection on each frame, and
shows the annotated result in a window.

Usage
─────
    python test_live_line_follow.py
    python test_live_line_follow.py --color blue --device 0

Keys
────
    q / ESC  — quit
    s        — save current frame to /tmp/line_snap.jpg
    b/k/o/r  — switch colour on the fly (blue/black/orange/red)
"""

import argparse
import sys

import cv2
import numpy as np

from line_follow_strategy import _detect, _annotate, _COLOUR_BOUNDS


def main():
    parser = argparse.ArgumentParser(description="Live line_follow test")
    parser.add_argument("--device", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--color", default="blue",
                        choices=list(_COLOUR_BOUNDS),
                        help="Target line colour (default: blue)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.device}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {w}x{h}")
    print(f"Colour: {args.color}")
    print("Keys: q=quit  s=save snap  b/k/o/r=switch colour")

    color = args.color

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv_lo, hsv_hi = _COLOUR_BOUNDS[color]
        line_col, error_norm, area, proc, mask, best_stats, strip_y, cx = \
            _detect(frame, hsv_lo, hsv_hi)

        vel    = 80 if line_col is not None else 0
        radius = 0x8000
        result = "following" if line_col is not None else "line_lost"

        out = _annotate(proc, mask, best_stats, line_col, cx, strip_y,
                        vel, radius, error_norm, result, area, color)

        # Full-frame colour mask overlay
        full_hsv  = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
        full_mask = cv2.inRange(full_hsv, hsv_lo, hsv_hi)
        highlight = np.zeros_like(out)
        highlight[full_mask > 0] = (0, 255, 180)
        out = cv2.addWeighted(out, 1.0, highlight, 0.4, 0)

        cv2.imshow("line_follow live", out)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            cv2.imwrite('/tmp/line_snap.jpg', out)
            print("Saved /tmp/line_snap.jpg")
        elif key == ord('b'):
            color = "blue";   print("→ blue")
        elif key == ord('k'):
            color = "black";  print("→ black")
        elif key == ord('o'):
            color = "orange"; print("→ orange")
        elif key == ord('r'):
            color = "red";    print("→ red")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
