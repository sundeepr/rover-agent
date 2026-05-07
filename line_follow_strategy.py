"""
LineFollowStrategy — adaptive-segmentation pipe/line follower for Atlas rover.

Detection: adaptive threshold + largest connected component
────────────────────────────────────────────────────────────
Converts the bottom strip to grayscale and marks pixels that are
significantly darker than their local neighbourhood (adaptive threshold).
For coloured pipes/tape, an optional HSV pre-filter keeps only pixels of
the right hue before the darkness test.

This approach works regardless of overall lighting level — only local
contrast matters, so shadows and brightness changes don't confuse it.

Usage
─────
    python rover_agent.py --strategy line_follow --rover atlas \\
        --atlas-port /dev/ttyACM0 --interval 0.4 \\
        --line-color black   # or blue / orange / red
        [--line-vel 80] [--line-kp 2000]
"""

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from navigation_strategy import AgentState, NavigationStrategy

log = logging.getLogger("rover.line_follow")

DRIVE_DURATION_S = 0.3    # seconds to drive per step
STRIP_ROWS       = 240    # rows from bottom of frame to scan (half of 480)
_MIN_AREA        = 15     # minimum blob area to count as the pipe
_BLOCK_SIZE      = 61     # adaptive threshold neighbourhood (must be odd)
_DARK_C          = 5      # pipe must be this much darker than local mean

# Optional HSV pre-filter per colour — limits adaptive search to pixels
# that are already roughly the right hue, reducing false positives.
# "black" uses no pre-filter (dark = dark regardless of hue).
_COLOUR_BOUNDS = {
    "black":  None,
    "grey":   (np.array([0,   0,   150], dtype=np.uint8),   # any hue, low sat, bright
               np.array([179, 60,  255], dtype=np.uint8)),
    "blue":   (np.array([90,  40,  10],  dtype=np.uint8),
               np.array([135, 255, 200], dtype=np.uint8)),
    "orange": (np.array([5,   120, 80],  dtype=np.uint8),
               np.array([25,  255, 255], dtype=np.uint8)),
    "red":    (np.array([0,   120, 80],  dtype=np.uint8),
               np.array([10,  255, 255], dtype=np.uint8)),
}


class LineFollowStrategy(NavigationStrategy):

    def __init__(
        self,
        vel_mm_s: int      = 40,
        kp: float          = 2000.0,
        color: str         = "black",
        threshold: int     = 80,    # unused, kept for CLI compat
        roi_frac: float    = 0.4,   # unused, kept for CLI compat
        edge_margin: float = 0.15,  # unused, kept for CLI compat
    ):
        self._vel   = vel_mm_s
        self._kp    = kp
        self._color = color
        if color not in _COLOUR_BOUNDS:
            raise ValueError(f"Unknown color {color!r}. Choose: {list(_COLOUR_BOUNDS)}")
        self._hsv_bounds = _COLOUR_BOUNDS[color]

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
            line_col, error_norm, area, proc, mask, best_stats, strip_y, cx = \
                _detect(frame, self._hsv_bounds)

            if line_col is not None:
                if abs(error_norm) < 0.02:
                    radius = 0x8000
                else:
                    radius = int(-self._kp / error_norm)
                    radius = max(-5000, min(5000, radius))
                vel    = self._vel
                result = "following"
                r_str  = "straight" if radius == 0x8000 else f"{radius}mm"
                log.info("%s pipe col=%d  err=%.2f  vel=%d  r=%s",
                         self._color, line_col, error_norm, vel, r_str)
            else:
                radius = 0x8000
                vel    = 0
                result = "line_lost"
                log.warning("%s pipe lost (area=%d < %d) — stopping",
                            self._color, area, _MIN_AREA)

            # ── Drive ─────────────────────────────────────────────────────
            if rover_ctrl and not state.paused.is_set():
                if line_col is not None:
                    rover_ctrl.drive_raw(vel, radius)
                else:
                    rover_ctrl.stop()

            # ── Annotate ──────────────────────────────────────────────────
            display = _annotate(proc, mask, best_stats, line_col,
                                cx, strip_y, vel, radius, error_norm,
                                result, area, self._color)
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


# ── Detection (shared with test scripts) ──────────────────────────────────────

def _detect(frame: np.ndarray, hsv_bounds: Optional[tuple]):
    """
    Segment the pipe using adaptive thresholding + largest connected component.

    hsv_bounds: (lo, hi) numpy arrays for optional HSV pre-filter, or None.

    Returns (line_col, error_norm, best_area, proc_frame, mask,
             best_stats, strip_y, cx).
    line_col is None if no blob large enough was found.
    """
    h, w = frame.shape[:2]
    cx   = w // 2

    # Bottom strip
    strip_y = max(0, h - STRIP_ROWS)
    strip   = frame[strip_y:, :]

    # ── Colour detection ───────────────────────────────────────────────────
    # For coloured pipes (blue/orange/red): HSV mask only — the pipe may be
    # lighter than the surrounding soil so adaptive darkness would exclude it.
    # For black: adaptive threshold only (dark relative to surroundings).
    if hsv_bounds is not None:
        hsv_lo, hsv_hi = hsv_bounds
        hsv  = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        # Diagnostic: log actual HSV range in the strip every 10 frames
        if not hasattr(_detect, '_diag_count'):
            _detect._diag_count = 0
        _detect._diag_count += 1
        if _detect._diag_count % 10 == 1:
            test_mask = cv2.inRange(hsv, hsv_lo, hsv_hi)
            match_px  = int(test_mask.sum() // 255)
            log.info("Strip HSV min=%s max=%s mean=%s  blue_px=%d",
                     hsv.min(axis=(0,1)), hsv.max(axis=(0,1)),
                     hsv.mean(axis=(0,1)).astype(int), match_px)
        mask = cv2.inRange(hsv, hsv_lo, hsv_hi)
    else:
        gray    = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        mask    = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=_BLOCK_SIZE,
            C=_DARK_C,
        )

    # Morphological cleanup
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)

    # ── Largest connected component ────────────────────────────────────────
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)

    # Pick the blob closest to the horizontal centre (pipe runs down the middle)
    # among blobs that meet the minimum area threshold.
    best_label = -1
    best_area  = 0
    best_stats = None
    best_dist  = float('inf')
    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < _MIN_AREA:
            continue
        dist = abs(centroids[lbl][0] - cx)
        if dist < best_dist:
            best_dist  = dist
            best_label = lbl
            best_area  = area
            best_stats = stats[lbl]

    if best_label == -1:
        # Report the largest area seen even if below threshold (for diagnostics)
        max_area = max((int(stats[l, cv2.CC_STAT_AREA]) for l in range(1, n)),
                       default=0)
        return None, 0.0, max_area, frame, mask, None, strip_y, cx

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

    # Pipe segmentation mask overlay
    overlay = np.zeros_like(out[strip_y:])
    overlay[mask > 0] = (0, 220, 80)
    out[strip_y:] = cv2.addWeighted(out[strip_y:], 0.6, overlay, 0.8, 0)

    # Bounding box of pipe blob
    if best_stats is not None:
        bx = int(best_stats[cv2.CC_STAT_LEFT])
        by = int(best_stats[cv2.CC_STAT_TOP]) + strip_y
        bw = int(best_stats[cv2.CC_STAT_WIDTH])
        bh = int(best_stats[cv2.CC_STAT_HEIGHT])
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (0, 255, 80), 2)

    # Strip boundary
    cv2.line(out, (0, strip_y), (w, strip_y), (0, 220, 80), 2)

    # Centre reference
    cv2.line(out, (cx, strip_y), (cx, h), (80, 80, 80), 1)

    # Detected pipe centroid
    if line_col is not None:
        cv2.line(out, (line_col, strip_y), (line_col, h), (0, 255, 80), 3)
        mid_y = strip_y + (h - strip_y) // 2
        cv2.circle(out, (line_col, mid_y), 14, (0, 255, 80), -1)
        cv2.line(out, (cx, mid_y), (line_col, mid_y), (0, 255, 80), 2)

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
