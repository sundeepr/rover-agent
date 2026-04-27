#!/usr/bin/env python3
"""
exg_viewer.py — test the Excess Green row-centering algorithm on a video file.

Displays the down-camera frame alongside the ExG mask and the gap-detection
overlay so you can tune --exg-threshold and --exg-min-area before a field run.

Controls
--------
  any key    — advance one frame (default: paused on every frame)
  Space      — toggle continuous playback / per-frame stepping
  q / Esc    — quit
  Trackbars  — adjust ExG threshold and min blob area live

Usage
-----
    python utils/exg_viewer.py path/to/down_camera.avi
    python utils/exg_viewer.py path/to/down_camera.avi --threshold 30 --min-area 800
    python utils/exg_viewer.py path/to/down_camera.avi --save out.avi
"""

import argparse
import sys

import cv2
import numpy as np


# ── Core ExG algorithm (mirrors row_centering_omnivla_strategy._find_row_gap_exg) ──

def find_row_gap_exg(
    down_bgr: np.ndarray,
    exg_threshold: int = 20,
    min_area: int = 500,
) -> tuple:
    h, w = down_bgr.shape[:2]
    frame_cx = w // 2

    bgr  = down_bgr.astype(np.float32)
    exg  = 2.0 * bgr[:, :, 1] - bgr[:, :, 0] - bgr[:, :, 2]
    mask = np.clip(exg, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, exg_threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        boxes.append([x, y, x + bw, y + bh])

    left_boxes  = [b for b in boxes if (b[0] + b[2]) / 2 < frame_cx]
    right_boxes = [b for b in boxes if (b[0] + b[2]) / 2 >= frame_cx]

    if not left_boxes and not right_boxes:
        return None, None, None, boxes, mask

    left_wall  = int(max((b[2] for b in left_boxes),  default=0))
    right_wall = int(min((b[0] for b in right_boxes), default=w))
    gap_cx     = (left_wall + right_wall) // 2
    return gap_cx, left_wall, right_wall, boxes, mask


# ── Annotation ────────────────────────────────────────────────────────────────

def annotate(frame, gap_cx, left_wall, right_wall, boxes, mask):
    vis = frame.copy()
    h, w = vis.shape[:2]
    frame_cx = w // 2

    # Tint the ExG mask green (semi-transparent overlay)
    green_layer = np.zeros_like(vis)
    green_layer[:, :, 1] = mask
    vis = cv2.addWeighted(vis, 1.0, green_layer, 0.35, 0)

    # Bounding boxes — orange = left, blue = right
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2
        color = (0, 140, 255) if cx < frame_cx else (255, 100, 0)
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # Wall lines and gap centre
    if left_wall is not None:
        cv2.line(vis, (left_wall, 0), (left_wall, h), (0, 140, 255), 2)
    if right_wall is not None:
        cv2.line(vis, (right_wall, 0), (right_wall, h), (255, 100, 0), 2)
    if gap_cx is not None:
        cv2.line(vis, (gap_cx, 0), (gap_cx, h), (0, 255, 255), 2)

    # Image centre
    cv2.line(vis, (frame_cx, 0), (frame_cx, h), (200, 200, 200), 1)

    # Status label
    if gap_cx is not None:
        error_px = gap_cx - frame_cx
        label = f"gap_cx={gap_cx}  err={error_px:+d}px  blobs={len(boxes)}"
        color = (0, 255, 200)
    else:
        label = f"no vegetation detected  blobs={len(boxes)}"
        color = (0, 60, 220)
    cv2.putText(vis, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return vis


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Test ExG row-gap detection on a video file")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--threshold", type=int, default=20,
                        help="Initial ExG threshold (default: 20)")
    parser.add_argument("--min-area",  type=int, default=500,
                        help="Initial minimum blob area in pixels (default: 500)")
    parser.add_argument("--save",      type=str, default="",
                        metavar="PATH",
                        help="Save annotated output to this .avi path")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open '{args.video}'", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {fw}x{fh}  {fps:.1f} fps  {total} frames")
    print("Controls: Space=pause/resume  s=step  q/Esc=quit  Trackbars=tune")

    win = "ExG viewer — Space pause  s step  q quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, max(fw, 640), fh)

    # Trackbars for live tuning
    cv2.createTrackbar("ExG threshold", win, args.threshold, 100, lambda _: None)
    cv2.createTrackbar("Min area /10",  win, args.min_area // 10, 500, lambda _: None)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (fw, fh))
        if not writer.isOpened():
            print(f"WARNING: cannot open output file '{args.save}'", file=sys.stderr)
            writer = None
        else:
            print(f"Saving annotated video to: {args.save}")

    step_mode = True   # start paused — any key advances one frame
    frame_idx = 0
    frame = None

    while True:
        # Advance one frame (always on first iteration; in continuous mode every loop)
        if frame is None or not step_mode:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break
            frame_idx += 1

        threshold = cv2.getTrackbarPos("ExG threshold", win)
        min_area  = cv2.getTrackbarPos("Min area /10",  win) * 10

        gap_cx, left_wall, right_wall, boxes, mask = find_row_gap_exg(
            frame, threshold, max(min_area, 1)
        )
        vis = annotate(frame, gap_cx, left_wall, right_wall, boxes, mask)

        # Frame counter overlay
        mode_label = "STEP" if step_mode else "PLAY"
        cv2.putText(vis, f"[{mode_label}] frame {frame_idx}/{total}  thr={threshold}  area={min_area}",
                    (8, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)

        cv2.imshow(win, vis)

        if writer:
            writer.write(vis)

        # In step mode wait indefinitely; in play mode run at video fps
        wait_ms = 0 if step_mode else max(1, int(1000 / fps))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            step_mode = not step_mode
        elif step_mode:
            # Any other key advances one frame
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break
            frame_idx += 1

    cap.release()
    if writer:
        writer.release()
        print(f"Saved: {args.save}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
