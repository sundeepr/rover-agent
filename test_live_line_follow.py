#!/usr/bin/env python3
"""
test_live_line_follow.py — live camera segmentation test for pipe/line following.

Segments the darkest connected region in the bottom strip using adaptive
thresholding (pixels significantly darker than their local neighbourhood).
Works for any dark pipe on a lighter floor regardless of lighting.

Usage
─────
    python test_live_line_follow.py [--device 0]

Keys
────
    q / ESC  — quit
    s        — save snapshot to /tmp/line_snap.jpg
    +  -     — increase / decrease adaptive block size
    [  ]     — increase / decrease darkness threshold C
"""

import argparse
import sys

import cv2
import numpy as np

# ── Segmentation parameters ───────────────────────────────────────────────────

# Bottom strip height (rows from bottom of frame to analyse)
STRIP_ROWS   = 120

# Adaptive threshold block size — must be odd.
# Should be wider than the pipe but narrower than the frame.
BLOCK_SIZE   = 61

# Darkness constant: pipe must be this many units darker than local mean.
DARK_C       = 10

# Minimum blob area (px²) to count as the pipe
MIN_AREA     = 300

# Centre-crop fraction (narrows wide-angle FOV before processing)
CENTER_CROP  = 0.7


def _segment_pipe(frame: np.ndarray):
    """
    Segment the darkest connected region (the pipe) in the bottom strip.

    Returns
    -------
    proc_frame   : cropped + resized frame used for detection
    mask         : binary mask of detected dark region (strip coords)
    best_stats   : cv2 CC stats of largest blob, or None
    line_col     : centroid x in proc_frame coords, or None
    error_norm   : lateral error in [-1, 1], 0 if not detected
    strip_y      : y coord where strip starts in proc_frame
    cx           : frame centre x
    """
    h, w = frame.shape[:2]

    # Centre-crop to reduce wide-angle FOV
    crop_w = int(w * CENTER_CROP)
    x0     = (w - crop_w) // 2
    frame  = frame[:, x0: x0 + crop_w]
    # Keep 640px wide
    scale  = 640 / crop_w
    proc_h = int(h * scale)
    frame  = cv2.resize(frame, (640, proc_h), interpolation=cv2.INTER_AREA)
    h, w   = frame.shape[:2]
    cx     = w // 2

    # Bottom strip
    strip_y = max(0, h - STRIP_ROWS)
    strip   = frame[strip_y:, :]

    # ── Adaptive threshold segmentation ──────────────────────────────────────
    gray    = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Pixels darker than local neighbourhood by DARK_C become foreground (255)
    mask = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=BLOCK_SIZE,
        C=DARK_C,
    )

    # Morphological cleanup: close gaps in the pipe, remove tiny specks
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)

    # ── Largest connected component ───────────────────────────────────────────
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)

    best_label = -1
    best_area  = 0
    best_stats = None
    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area  = area
            best_label = lbl
            best_stats = stats[lbl]

    if best_area < MIN_AREA:
        return frame, mask, None, None, 0.0, strip_y, cx

    line_col   = int(centroids[best_label][0])
    error_norm = (line_col - cx) / (w / 2.0)
    return frame, mask, best_stats, line_col, error_norm, strip_y, cx


def _annotate(frame, mask, best_stats, line_col, cx, strip_y,
              error_norm, block_size, dark_c, best_area):
    out = frame.copy()
    h, w = out.shape[:2]

    # Overlay segmentation mask on strip in green
    seg_layer = np.zeros_like(out[strip_y:])
    seg_layer[mask > 0] = (0, 220, 80)
    out[strip_y:] = cv2.addWeighted(out[strip_y:], 0.6, seg_layer, 0.8, 0)

    # Bounding box of largest blob
    if best_stats is not None:
        bx = int(best_stats[cv2.CC_STAT_LEFT])
        by = int(best_stats[cv2.CC_STAT_TOP]) + strip_y
        bw = int(best_stats[cv2.CC_STAT_WIDTH])
        bh = int(best_stats[cv2.CC_STAT_HEIGHT])
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (0, 255, 80), 2)

    # Strip boundary
    cv2.line(out, (0, strip_y), (w, strip_y), (0, 220, 80), 2)

    # Centre reference
    cv2.line(out, (cx, strip_y), (cx, h), (60, 60, 60), 1)

    # Detected pipe centroid
    if line_col is not None:
        cv2.line(out, (line_col, strip_y), (line_col, h), (0, 255, 80), 3)
        mid_y = strip_y + (h - strip_y) // 2
        cv2.circle(out, (line_col, mid_y), 14, (0, 255, 80), -1)
        cv2.line(out, (cx, mid_y), (line_col, mid_y), (0, 255, 80), 2)
        status = f"PIPE  error={error_norm:+.3f}  area={best_area}"
        colour = (0, 255, 80)
    else:
        status = f"NO PIPE  area={best_area} < {MIN_AREA}"
        colour = (0, 60, 255)

    hud = [
        status,
        f"block={block_size}  C={dark_c}  (+/-/[/] to tune)",
    ]
    for i, txt in enumerate(hud):
        cv2.putText(out, txt, (10, 28 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)

    return out


def main():
    global BLOCK_SIZE, DARK_C

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.device}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {w}x{h}")
    print("Keys: q=quit  s=save  +/-=block size  [/]=darkness C")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        proc, mask, best_stats, line_col, error_norm, strip_y, cx = \
            _segment_pipe(frame)

        best_area = int(best_stats[cv2.CC_STAT_AREA]) if best_stats is not None else 0
        out = _annotate(proc, mask, best_stats, line_col, cx, strip_y,
                        error_norm, BLOCK_SIZE, DARK_C, best_area)

        cv2.imshow("pipe segmentation", out)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            cv2.imwrite('/tmp/line_snap.jpg', out)
            print("Saved /tmp/line_snap.jpg")
        elif key == ord('+') and BLOCK_SIZE < 201:
            BLOCK_SIZE += 10
            print(f"block_size={BLOCK_SIZE}")
        elif key == ord('-') and BLOCK_SIZE > 11:
            BLOCK_SIZE -= 10
            print(f"block_size={BLOCK_SIZE}")
        elif key == ord(']') and DARK_C < 50:
            DARK_C += 2
            print(f"dark_C={DARK_C}")
        elif key == ord('[') and DARK_C > 2:
            DARK_C -= 2
            print(f"dark_C={DARK_C}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
