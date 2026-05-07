"""
LineFollowStrategy — colour-blob line follower for the Atlas rover.

Detection: largest connected component
──────────────────────────────────────
Thresholds the bottom strip of the frame for the target colour, finds
all connected blobs, and picks the LARGEST one.  A pipe or tape running
along the ground is always the biggest continuous region of that colour;
random same-colour specks are tiny and ignored automatically.

Supported colours (--line-color)
─────────────────────────────────
  black  — dark objects (pipes, black tape): low HSV Value
  blue   — light blue tape/pipe
  orange — orange tape/pipe
  red    — red tape/pipe

Pipeline (each step)
────────────────────
  1. Centre-crop the frame to narrow the wide-angle FOV, downscale.
  2. Take the bottom STRIP_ROWS rows (closest ground to rover).
  3. Convert to HSV; apply colour mask.
  4. Find connected components; pick the largest by pixel area.
  5. Confirm: area must exceed MIN_AREA pixels.
  6. Compute centroid x of the largest component.
  7. Lateral error = (centroid_x - frame_cx) / half_width  ∈ [-1, 1].
  8. radius_mm = -kp / error  →  drive_raw for DRIVE_DURATION_S then stop.

Usage
─────
    python rover_agent.py --strategy line_follow --rover atlas \\
        --atlas-port /dev/ttyACM0 --interval 0.4 \\
        --line-color black [--line-vel 80] [--line-kp 2000]
"""

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from navigation_strategy import AgentState, NavigationStrategy

log = logging.getLogger("rover.line_follow")

DRIVE_DURATION_S = 0.3   # seconds to drive per step
STRIP_ROWS       = 80    # rows from bottom of frame to scan
_CENTER_CROP     = 0.5   # keep central 50% of width
_PROC_WIDTH      = 640   # downscale to this width
_MIN_AREA        = 200   # minimum blob area (pixels) to count as line

# HSV bounds per colour (OpenCV: H 0-179, S 0-255, V 0-255)
_COLOUR_BOUNDS = {
    "black":  (np.array([0,   0,   0],   dtype=np.uint8),
               np.array([179, 255, 60],  dtype=np.uint8)),
    "blue":   (np.array([85,  80,  100], dtype=np.uint8),
               np.array([110, 255, 255], dtype=np.uint8)),
    "orange": (np.array([5,   120, 80],  dtype=np.uint8),
               np.array([25,  255, 255], dtype=np.uint8)),
    "red":    (np.array([0,   120, 80],  dtype=np.uint8),
               np.array([10,  255, 255], dtype=np.uint8)),
}


class LineFollowStrategy(NavigationStrategy):

    def __init__(
        self,
        vel_mm_s: int      = 80,
        kp: float          = 2000.0,
        color: str         = "black",
        threshold: int     = 80,    # unused, kept for CLI compat
        roi_frac: float    = 0.4,   # unused, kept for CLI compat
        edge_margin: float = 0.15,  # unused, kept for CLI compat
    ):
        self._vel = vel_mm_s
        self._kp  = kp
        if color not in _COLOUR_BOUNDS:
            raise ValueError(f"Unknown color {color!r}. Choose from: {list(_COLOUR_BOUNDS)}")
        self._hsv_lo, self._hsv_hi = _COLOUR_BOUNDS[color]
        self._color = color

    @property
    def name(self) -> str:
        return "line_follow"

    def on_reset(self) -> None:
        pass

    def run_query(
        self,
        state: AgentState,
        frame: np.ndarray,
        captures_dir: Path,
        rover_ctrl,
    ) -> None:
        t0 = time.time()
        try:
            line_col, error_norm, area, proc, mask, stats, strip_y, cx = \
                _detect(frame, self._hsv_lo, self._hsv_hi)

            if line_col is not None:
                if abs(error_norm) < 0.02:
                    radius = 0x8000
                else:
                    radius = int(-self._kp / error_norm)
                    radius = max(-5000, min(5000, radius))

                vel    = self._vel
                result = "following"
                r_str  = "straight" if radius == 0x8000 else f"{radius}mm"
                log.info("%s line col=%d  error=%.3f  area=%d  vel=%d  r=%s",
                         self._color, line_col, error_norm, area, vel, r_str)
            else:
                error_norm = 0.0
                radius     = 0x8000
                vel        = 0
                result     = "line_lost"
                log.warning("%s line lost (best area=%d < %d) — stopping",
                            self._color, area, _MIN_AREA)

            # ── Drive ─────────────────────────────────────────────────────
            if rover_ctrl and not state.paused.is_set():
                rover_ctrl.drive_raw(vel, radius)
                time.sleep(DRIVE_DURATION_S)
                rover_ctrl.stop()

            # ── Annotate ──────────────────────────────────────────────────
            display = _annotate(proc, mask, stats, line_col, cx, strip_y,
                                vel, radius, error_norm, result, area,
                                self._color)
            with state.llm_lock:
                state.llm_frame = display

            elapsed = time.time() - t0
            with state.result_lock:
                state.llm_response_s  = elapsed
                state.llm_query_start = 0.0
                state.latest_result   = {
                    "strategy":  self.name,
                    "result":    result,
                    "vel_mm_s":  vel,
                    "radius_mm": radius if radius != 0x8000 else None,
                    "error":     round(error_norm, 4),
                    "blob_area": area,
                    "elapsed_s": round(elapsed, 3),
                }

        except Exception:
            log.exception("LineFollowStrategy error")
        finally:
            state.query_in_flight.clear()


# ── Detection (shared with test script) ───────────────────────────────────────

def _detect(frame: np.ndarray, hsv_lo: np.ndarray, hsv_hi: np.ndarray):
    """
    Run detection on one frame.
    Returns (line_col, error_norm, best_area, proc_frame, mask, best_stats,
             strip_y, cx).
    line_col is None if no blob large enough was found.
    """
    h, w = frame.shape[:2]

    # Centre-crop + downscale
    crop_w = int(w * _CENTER_CROP)
    x0     = (w - crop_w) // 2
    frame  = frame[:, x0: x0 + crop_w]
    scale  = _PROC_WIDTH / crop_w
    proc_h = int(h * scale)
    frame  = cv2.resize(frame, (_PROC_WIDTH, proc_h), interpolation=cv2.INTER_AREA)
    h, w   = frame.shape[:2]
    cx     = w // 2

    # Bottom strip
    strip_y = max(0, h - STRIP_ROWS)
    strip   = frame[strip_y:, :]

    # Colour mask
    hsv  = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lo, hsv_hi)

    # Morphological cleanup — join nearby pixels, remove tiny noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    # Connected components
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)

    # Skip label 0 (background); find largest foreground component
    best_label = -1
    best_area  = 0
    best_stats = None
    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area  = area
            best_label = lbl
            best_stats = stats[lbl]

    if best_area < _MIN_AREA:
        return None, 0.0, best_area, frame, mask, None, strip_y, cx

    line_col   = int(centroids[best_label][0])
    error_norm = (line_col - cx) / (w / 2.0)
    return line_col, error_norm, best_area, frame, mask, best_stats, strip_y, cx


# ── Visualisation ─────────────────────────────────────────────────────────────

def _annotate(
    frame: np.ndarray,
    mask: np.ndarray,
    best_stats,
    line_col: Optional[int],
    cx: int,
    strip_y: int,
    vel: int,
    radius: int,
    error_norm: float,
    result: str,
    blob_area: int,
    color: str,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Highlight masked pixels in the strip
    overlay = np.zeros_like(out[strip_y:])
    overlay[mask > 0] = (0, 200, 255)
    out[strip_y:] = cv2.addWeighted(out[strip_y:], 1.0, overlay, 0.6, 0)

    # Bounding box of best blob
    if best_stats is not None:
        bx = int(best_stats[cv2.CC_STAT_LEFT])
        by = int(best_stats[cv2.CC_STAT_TOP]) + strip_y
        bw = int(best_stats[cv2.CC_STAT_WIDTH])
        bh = int(best_stats[cv2.CC_STAT_HEIGHT])
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (0, 255, 200), 2)

    # Strip boundary
    cv2.line(out, (0, strip_y), (w, strip_y), (0, 200, 255), 2)

    # Centre reference
    cv2.line(out, (cx, strip_y), (cx, h), (80, 80, 80), 1)

    # Detected centroid
    if line_col is not None:
        cv2.line(out, (line_col, strip_y), (line_col, h), (0, 255, 200), 3)
        mid_y = strip_y + (h - strip_y) // 2
        cv2.circle(out, (line_col, mid_y), 14, (0, 255, 180), -1)
        cv2.line(out, (cx, mid_y), (line_col, mid_y), (0, 255, 180), 2)

    # HUD
    r_str  = "straight" if radius == 0x8000 else f"r={radius}mm"
    status = "LOST" if line_col is None else f"error={error_norm:+.3f}"
    hud = [
        f"line_follow [{color}]  {result}",
        f"vel={vel}mm/s  {r_str}",
        f"{status}  area={blob_area}px",
    ]
    for i, txt in enumerate(hud):
        cv2.putText(out, txt, (12, 36 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2, cv2.LINE_AA)

    return out
