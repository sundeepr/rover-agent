#!/usr/bin/env python3
"""
exg_threshold_viewer.py — Interactive ExG threshold tuner
==========================================================

Computes the Excess Green (ExG) vegetation index on a live camera feed or
video file and displays the result in real time. Uses the same formula and
morphological pipeline as boundary_guard_strategy.py:

    ExG = 2*G - R - B        (per pixel, float32)
    mask = ExG > threshold   (binary)
    post-process: morphological close then open (5x5 ellipse kernel)

The display shows three panels side by side:
  Left   — original frame with green contours overlaid
  Centre — ExG heatmap (greener = higher ExG value)
  Right  — binary mask after thresholding

A trackbar lets you adjust the threshold live (0–255) to find the right
value for your lighting conditions. The current threshold and the percentage
of pixels above it are printed in the HUD.

Usage
-----
    # Live camera (default device 0)
    python exg_threshold_viewer.py

    # Specific camera device
    python exg_threshold_viewer.py --cam 2

    # Video file
    python exg_threshold_viewer.py --video path/to/video.mp4

    # Start with a specific threshold (default matches rover_agent.py default: 60)
    python exg_threshold_viewer.py --threshold 40

Arguments
---------
  --cam        Camera device index (default: 0)
  --video      Path to a video file (overrides --cam)
  --threshold  Starting ExG threshold 0–255 (default: 60)

Controls
--------
  Trackbar  — adjust threshold live
  Q / ESC   — quit
"""

import argparse
import sys

import cv2
import numpy as np


# ── ExG (identical to boundary_guard_strategy._exg_mask) ─────────────────────

def exg_mask(frame_bgr: np.ndarray, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (raw_exg_u8, binary_mask).
    raw_exg_u8 : ExG values clipped to 0-255, uint8 — used for the heatmap
    binary_mask: pixels above threshold after morph close+open, uint8 0/255
    """
    f   = frame_bgr.astype(np.float32)
    exg = 2.0 * f[:, :, 1] - f[:, :, 0] - f[:, :, 2]
    raw = np.clip(exg, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(raw, threshold, 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return raw, mask


# ── Visualisation helpers ──────────────────────────────────────────────────────

def make_heatmap(raw_exg: np.ndarray) -> np.ndarray:
    """Convert raw ExG uint8 image to a green-tinted colour heatmap."""
    coloured = cv2.applyColorMap(raw_exg, cv2.COLORMAP_SUMMER)
    return coloured


def draw_contours_on_frame(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out


def put_hud(img: np.ndarray, text: str) -> None:
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    cv2.putText(img, text, (11, 26), font, scale, (0,   0,   0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, text, (10, 25), font, scale, (0, 255,   0), thick,     cv2.LINE_AA)


def put_label(img: np.ndarray, text: str) -> None:
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    cv2.putText(img, text, (6, img.shape[0] - 8), font, scale, (0,   0,   0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, text, (5, img.shape[0] - 9), font, scale, (255, 255, 255), thick,   cv2.LINE_AA)


def build_display(frame: np.ndarray, raw_exg: np.ndarray,
                  mask: np.ndarray, threshold: int) -> np.ndarray:
    h, w = frame.shape[:2]
    pct  = 100.0 * np.count_nonzero(mask) / mask.size

    left   = draw_contours_on_frame(frame, mask)
    centre = make_heatmap(raw_exg)
    right  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    put_hud(left,   f"ExG threshold: {threshold}   vegetation: {pct:.1f}%")
    put_label(left,   "original + contours")
    put_label(centre, "ExG heatmap")
    put_label(right,  "binary mask")

    return np.hstack([left, centre, right])


# ── Main ───────────────────────────────────────────────────────────────────────

WINDOW = "ExG Threshold Viewer"


def main():
    parser = argparse.ArgumentParser(description="Interactive ExG threshold tuner")
    parser.add_argument("--cam",       type=int,   default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--video",     type=str,   default=None,
                        help="Path to a video file (overrides --cam)")
    parser.add_argument("--threshold", type=int,   default=60,
                        help="Starting ExG threshold 0-255 (default: 60)")
    args = parser.parse_args()

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            sys.exit(f"ERROR: Cannot open video file {args.video}")
        print(f"Opened video: {args.video}")
    else:
        cap = cv2.VideoCapture(args.cam)
        if not cap.isOpened():
            sys.exit(f"ERROR: Cannot open camera device {args.cam}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera {args.cam} opened at {w}x{h}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1920, 480)
    cv2.createTrackbar("Threshold", WINDOW, args.threshold, 255, lambda v: None)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("WARNING: dropped frame")
                continue

            threshold = cv2.getTrackbarPos("Threshold", WINDOW)
            raw_exg, mask = exg_mask(frame, threshold)
            display = build_display(frame, raw_exg, mask, threshold)

            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):   # Q or ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Print final threshold so user can copy it into rover_agent.py / config
        final = cv2.getTrackbarPos("Threshold", WINDOW) if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 0 else args.threshold
        print(f"\nFinal threshold: {final}")
        print(f"  Use with: python rover_agent.py --exg-threshold {final}")


if __name__ == "__main__":
    main()
