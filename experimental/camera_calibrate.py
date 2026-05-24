#!/usr/bin/env python3
"""
camera_calibrate.py — visual camera-centre calibration tool.

Shows the live camera feed with:
  • Yellow dashed line  — current frame centre (50 %)
  • Red mask overlay    — detected guard rails
  • Green dashed line   — midpoint between the two detected rails
  • Cyan arrow + text   — offset between frame centre and rail midpoint

Move the physical camera until the green line sits on top of the yellow line
(offset ≈ 0 px).

Usage
─────
    python experimental/camera_calibrate.py --device 2
    python experimental/camera_calibrate.py --device 2 --port /dev/ttyUSB0  # if via serial

Keys
────
    q / ESC  — quit
    s        — save a snapshot to calibration_snapshot.jpg
    h        — toggle HSV mask view (full screen red mask for tuning)
"""

import argparse
import time
import cv2
import numpy as np

# ── Default HSV bounds for red ────────────────────────────────────────────────
# Red wraps around 0/180 in HSV — we union two ranges.
RED_LOW1  = np.array([  0,  80,  60])
RED_HIGH1 = np.array([ 12, 255, 255])
RED_LOW2  = np.array([165,  80,  60])
RED_HIGH2 = np.array([180, 255, 255])

# Minimum pixel width a rail blob must span to be counted
MIN_RAIL_WIDTH = 10


def _make_trackbars(win: str):
    cv2.createTrackbar("H_lo1",  win,   0, 180, lambda x: None)
    cv2.createTrackbar("H_hi1",  win,  12, 180, lambda x: None)
    cv2.createTrackbar("H_lo2",  win, 165, 180, lambda x: None)
    cv2.createTrackbar("H_hi2",  win, 180, 180, lambda x: None)
    cv2.createTrackbar("S_lo",   win,  80, 255, lambda x: None)
    cv2.createTrackbar("V_lo",   win,  60, 255, lambda x: None)


def _read_trackbars(win: str):
    h_lo1 = cv2.getTrackbarPos("H_lo1", win)
    h_hi1 = cv2.getTrackbarPos("H_hi1", win)
    h_lo2 = cv2.getTrackbarPos("H_lo2", win)
    h_hi2 = cv2.getTrackbarPos("H_hi2", win)
    s_lo  = cv2.getTrackbarPos("S_lo",  win)
    v_lo  = cv2.getTrackbarPos("V_lo",  win)
    low1  = np.array([h_lo1, s_lo, v_lo])
    hi1   = np.array([h_hi1, 255, 255])
    low2  = np.array([h_lo2, s_lo, v_lo])
    hi2   = np.array([h_hi2, 255, 255])
    return low1, hi1, low2, hi2


def detect_rails(mask: np.ndarray) -> tuple[int | None, int | None]:
    """
    Return (left_cx, right_cx) — horizontal centre-x of the two largest
    red blobs found in `mask`, sorted left-to-right.
    Returns None for a slot if fewer than 2 blobs are found.
    """
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)

    blobs = []
    for i in range(1, n_labels):          # skip background label 0
        w = stats[i, cv2.CC_STAT_WIDTH]
        a = stats[i, cv2.CC_STAT_AREA]
        if w >= MIN_RAIL_WIDTH and a >= 50:
            cx = int(centroids[i][0])
            blobs.append((cx, a))

    if len(blobs) < 2:
        return None, None

    # Take the two largest blobs and sort by x
    blobs.sort(key=lambda b: b[1], reverse=True)
    left_cx  = min(blobs[0][0], blobs[1][0])
    right_cx = max(blobs[0][0], blobs[1][0])
    return left_cx, right_cx


def draw_vline_dashed(img, x, color, thickness=2, dash=12, gap=8):
    h = img.shape[0]
    y = 0
    while y < h:
        y2 = min(y + dash, h)
        cv2.line(img, (x, y), (x, y2), color, thickness)
        y += dash + gap


def run(device: int, show_mask: bool = False):
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera device {device}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    WIN = "Camera Calibration"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 900, 600)
    _make_trackbars(WIN)

    show_hsv = show_mask
    snapshot_saved = False

    print("\n=== Camera Calibration ===")
    print("Move the camera until the green midpoint line aligns with the yellow centre line.")
    print("Keys: q/ESC=quit  s=save snapshot  h=toggle HSV mask\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed — retrying…")
            time.sleep(0.1)
            continue

        h, w = frame.shape[:2]
        cx_frame = w // 2

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        l1, h1, l2, h2 = _read_trackbars(WIN)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, l1, h1),
            cv2.inRange(hsv, l2, h2),
        )
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        left_cx, right_cx = detect_rails(mask)

        # ── Build display frame ───────────────────────────────────────────────
        if show_hsv:
            display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            display = frame.copy()
            # Tint red regions
            tint = display.copy()
            tint[mask > 0] = (0, 0, 200)
            cv2.addWeighted(tint, 0.35, display, 0.65, 0, display)

        # Yellow dashed frame-centre line
        draw_vline_dashed(display, cx_frame, (0, 220, 255), thickness=2)
        cv2.putText(display, "centre", (cx_frame + 4, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        if left_cx is not None and right_cx is not None:
            midpoint = (left_cx + right_cx) // 2
            offset   = midpoint - cx_frame
            pct      = offset / w * 100

            # Draw rail centres
            for rx in (left_cx, right_cx):
                cv2.line(display, (rx, 0), (rx, h), (60, 60, 255), 1)

            # Green dashed midpoint line
            draw_vline_dashed(display, midpoint, (0, 255, 80), thickness=2)
            cv2.putText(display, "rail mid", (midpoint + 4, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 80), 1, cv2.LINE_AA)

            # Arrow from frame centre to midpoint
            arrow_y = h // 2
            cv2.arrowedLine(display,
                            (cx_frame, arrow_y),
                            (midpoint,  arrow_y),
                            (255, 200, 0), 2, tipLength=0.2)

            # Offset text
            direction = "RIGHT" if offset > 0 else "LEFT" if offset < 0 else "CENTRE ✓"
            colour    = (0, 255, 80) if abs(offset) < 5 else (0, 200, 255)
            lines = [
                f"Offset: {offset:+d} px  ({pct:+.1f}%)",
                f"Move camera {direction}" if abs(offset) >= 5 else "Camera centred!",
            ]
            for i, txt in enumerate(lines):
                cv2.putText(display, txt, (12, 32 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
        else:
            cv2.putText(display, "No rails detected — adjust HSV sliders",
                        (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2, cv2.LINE_AA)

        mode_txt = "HSV mask view (h to toggle)" if show_hsv else "Live view (h for HSV mask)"
        cv2.putText(display, mode_txt, (12, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

        cv2.imshow(WIN, display)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('h'):
            show_hsv = not show_hsv
        elif key == ord('s'):
            fname = "calibration_snapshot.jpg"
            cv2.imwrite(fname, display)
            print(f"Snapshot saved → {fname}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Camera centre calibration using red guard rails")
    ap.add_argument("--device", type=int, default=0,
                    help="Camera device index (default: 0)")
    ap.add_argument("--show-mask", action="store_true",
                    help="Start in HSV mask view")
    args = ap.parse_args()
    run(args.device, show_mask=args.show_mask)


if __name__ == "__main__":
    main()
