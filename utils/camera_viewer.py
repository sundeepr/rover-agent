#!/usr/bin/env python3
"""
camera_viewer.py — list all attached cameras and show their live streams.

Scans /dev/video* to find capture devices, maps each to a cv2 device index,
then opens all of them and tiles their frames in a single OpenCV window.

Each tile is labelled with its Linux device path (e.g. /dev/video0) and
the index used to open it (e.g. cv2 idx 0).

Controls
--------
  q / Esc  — quit

Usage
-----
    python utils/camera_viewer.py
    python utils/camera_viewer.py --max-devices 8   # scan /dev/video0..7
    python utils/camera_viewer.py --width 320 --height 240
"""

import argparse
import glob
import re
import sys

import cv2
import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────

_FONT        = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE  = 0.55
_FONT_THICK  = 1
_LABEL_H     = 26        # pixels reserved for the label bar at the top of each tile
_TILE_GAP    = 4         # pixels between tiles
_BG_COLOR    = (30, 30, 30)
_LABEL_BG    = (50, 50, 50)
_LABEL_FG    = (220, 220, 220)
_ERR_COLOR   = (60, 60, 200)


# ── Camera discovery ───────────────────────────────────────────────────────────

def _discover_cameras(max_devices: int) -> list[dict]:
    """
    Return a list of dicts describing each usable capture device.

    Each dict has:
        idx     : int   — cv2.VideoCapture index
        dev     : str   — Linux device path (e.g. /dev/video0) or "" on non-Linux
        opened  : bool  — whether the device could be opened
        cap     : cv2.VideoCapture | None
        w, h    : int   — reported resolution
    """
    # Build a sorted list of /dev/videoN paths that actually exist
    dev_paths = sorted(
        glob.glob("/dev/video*"),
        key=lambda p: int(re.search(r"\d+", p).group()),
    )
    # Map /dev/videoN → index N so we can align with cv2's integer indices
    dev_map: dict[int, str] = {}
    for path in dev_paths:
        m = re.search(r"(\d+)$", path)
        if m:
            dev_map[int(m.group(1))] = path

    cameras = []
    for idx in range(max_devices):
        dev_path = dev_map.get(idx, "")
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            # Include in list only if a /dev/videoN path exists for this index
            if dev_path:
                cameras.append({"idx": idx, "dev": dev_path, "opened": False,
                                 "cap": None, "w": 0, "h": 0})
            continue
        # Check we can actually read a frame (some /dev/videoN are metadata devices)
        ret, _ = cap.read()
        if not ret:
            cap.release()
            if dev_path:
                cameras.append({"idx": idx, "dev": dev_path, "opened": False,
                                 "cap": None, "w": 0, "h": 0})
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cameras.append({"idx": idx, "dev": dev_path or f"idx {idx}",
                         "opened": True, "cap": cap, "w": w, "h": h})

    return cameras


# ── Tile layout ────────────────────────────────────────────────────────────────

def _tile_layout(n: int) -> tuple[int, int]:
    """Return (cols, rows) for a grid that fits n tiles as squarely as possible."""
    cols = max(1, int(np.ceil(np.sqrt(n))))
    rows = max(1, int(np.ceil(n / cols)))
    return cols, rows


def _make_error_tile(w: int, h: int, label: str) -> np.ndarray:
    tile = np.full((h + _LABEL_H, w, 3), _BG_COLOR, dtype=np.uint8)
    msg = "no signal"
    tw, _ = cv2.getTextSize(msg, _FONT, _FONT_SCALE, _FONT_THICK)[0], None
    cx = (w - tw[0]) // 2
    cy = _LABEL_H + (h + cv2.getTextSize(msg, _FONT, _FONT_SCALE, _FONT_THICK)[0][1]) // 2
    cv2.putText(tile, msg, (cx, cy), _FONT, _FONT_SCALE, _ERR_COLOR, _FONT_THICK, cv2.LINE_AA)
    _draw_label(tile, label, w)
    return tile


def _draw_label(tile: np.ndarray, label: str, w: int) -> None:
    cv2.rectangle(tile, (0, 0), (w, _LABEL_H), _LABEL_BG, -1)
    cv2.putText(tile, label, (6, _LABEL_H - 8),
                _FONT, _FONT_SCALE, _LABEL_FG, _FONT_THICK, cv2.LINE_AA)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Live viewer for all attached cameras")
    parser.add_argument("--max-devices", type=int, default=10,
                        help="Highest device index to probe (default: 10)")
    parser.add_argument("--width",  type=int, default=400,
                        help="Tile width in pixels (default: 400)")
    parser.add_argument("--height", type=int, default=300,
                        help="Tile height in pixels (default: 300)")
    args = parser.parse_args()

    tw, th = args.width, args.height   # tile frame size (excluding label bar)

    print("Scanning for cameras …")
    cameras = _discover_cameras(args.max_devices)

    if not cameras:
        print("No cameras found.")
        sys.exit(0)

    print(f"\nFound {len(cameras)} device(s):\n")
    for cam in cameras:
        status = f"{cam['w']}x{cam['h']}" if cam["opened"] else "not readable"
        print(f"  cv2 idx {cam['idx']:2d}  {cam['dev']:<20}  {status}")
    print()
    print("Press  q / Esc  to quit.\n")

    cols, rows = _tile_layout(len(cameras))
    canvas_w = cols * tw + (cols + 1) * _TILE_GAP
    canvas_h = rows * (th + _LABEL_H) + (rows + 1) * _TILE_GAP

    win = "Camera viewer — q/Esc to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, canvas_w, canvas_h)

    while True:
        canvas = np.full((canvas_h, canvas_w, 3), _BG_COLOR, dtype=np.uint8)

        for i, cam in enumerate(cameras):
            col = i % cols
            row = i // cols
            x0 = _TILE_GAP + col * (tw + _TILE_GAP)
            y0 = _TILE_GAP + row * (th + _LABEL_H + _TILE_GAP)

            label = f"{cam['dev']}  [cv2 idx {cam['idx']}]"

            if cam["opened"] and cam["cap"] is not None:
                ret, frame = cam["cap"].read()
                if ret:
                    resized = cv2.resize(frame, (tw, th))
                    tile = np.zeros((th + _LABEL_H, tw, 3), dtype=np.uint8)
                    tile[_LABEL_H:] = resized
                    _draw_label(tile, label, tw)
                else:
                    tile = _make_error_tile(tw, th, label)
            else:
                tile = _make_error_tile(tw, th, label)

            canvas[y0:y0 + th + _LABEL_H, x0:x0 + tw] = tile

        cv2.imshow(win, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):   # q or Esc
            break

    for cam in cameras:
        if cam["cap"] is not None:
            cam["cap"].release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
