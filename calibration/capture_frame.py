#!/usr/bin/env python3
"""
Capture a single frame from the camera and save it as a JPEG.

Usage:
    python calibration/capture_frame.py
    python calibration/capture_frame.py --device 1
    python calibration/capture_frame.py --output my_frame.jpg
"""

import argparse
import sys
import time
import cv2

def main():
    parser = argparse.ArgumentParser(description="Capture one camera frame")
    parser.add_argument("--device", type=int, default=0,
                        help="Camera device index (default 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename (default: frame_<timestamp>.jpg)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup frames before capture (default 20)")
    args = parser.parse_args()

    out = args.output or f"frame_{int(time.time())}.jpg"

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera device {args.device}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"Warming up camera ({args.warmup} frames)…")
    for _ in range(args.warmup):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("ERROR: failed to capture frame")
        sys.exit(1)

    cv2.imwrite(out, frame)
    h, w = frame.shape[:2]
    print(f"Saved {w}×{h} frame → {out}")

if __name__ == "__main__":
    main()
