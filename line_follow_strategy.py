"""
LineFollowStrategy — light-blue line follower for the Atlas rover.

Scans the bottom STRIP_ROWS rows of the full frame (closest ground to
the rover). Converts to HSV and masks for light blue pixels, then finds
the column with the most blue pixels. No ROI fraction needed.

Usage
─────
    python rover_agent.py --strategy line_follow --rover atlas \\
        --atlas-port /dev/ttyACM0 --interval 0.4 \\
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
STRIP_ROWS       = 80     # number of rows from the bottom of the frame to scan
_SMOOTH_WIN      = 21     # 1-D moving-average window
_MIN_BLUE_PX     = 30     # minimum blue pixels to confirm line present
_CENTER_CROP     = 0.5    # keep central 50% of width (narrows FOV, drops edges)
_PROC_WIDTH      = 640    # downscale cropped frame to this width for processing

# Light blue HSV bounds (OpenCV: H 0-179, S 0-255, V 0-255)
_HSV_LO = np.array([85,  80, 100], dtype=np.uint8)
_HSV_HI = np.array([110, 255, 255], dtype=np.uint8)


class LineFollowStrategy(NavigationStrategy):

    def __init__(
        self,
        vel_mm_s: int      = 80,
        kp: float          = 2000.0,
        threshold: int     = 80,    # unused, kept for CLI compat
        roi_frac: float    = 0.4,   # unused, kept for CLI compat
        edge_margin: float = 0.15,  # unused, kept for CLI compat
    ):
        self._vel = vel_mm_s
        self._kp  = kp

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
            h, w = frame.shape[:2]

            # ── Centre-crop to narrow FOV then downscale ──────────────────
            crop_w  = int(w * _CENTER_CROP)
            x0      = (w - crop_w) // 2
            frame   = frame[:, x0: x0 + crop_w]
            scale   = _PROC_WIDTH / crop_w
            proc_h  = int(h * scale)
            frame   = cv2.resize(frame, (_PROC_WIDTH, proc_h),
                                 interpolation=cv2.INTER_AREA)
            h, w    = frame.shape[:2]
            cx      = w // 2

            # ── Bottom strip: closest ground to the rover ─────────────────
            strip = frame[max(0, h - STRIP_ROWS):, :]

            # ── Light blue HSV mask ───────────────────────────────────────
            hsv  = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, _HSV_LO, _HSV_HI)

            total_px      = int(mask.sum() // 255)
            line_detected = total_px >= _MIN_BLUE_PX

            if line_detected:
                # Column with most blue pixels
                col_counts = mask.sum(axis=0).astype(np.float32)
                kernel     = np.ones(_SMOOTH_WIN, dtype=np.float32) / _SMOOTH_WIN
                smoothed   = np.convolve(col_counts, kernel, mode='same')
                line_col   = int(np.argmax(smoothed))
                error_norm = (line_col - cx) / (w / 2.0)

                if abs(error_norm) < 0.02:
                    radius = 0x8000
                else:
                    radius = int(-self._kp / error_norm)
                    radius = max(-5000, min(5000, radius))

                vel    = self._vel
                result = "following"
                r_str  = "straight" if radius == 0x8000 else f"{radius}mm"
                log.info("Blue line col=%d  error=%.3f  px=%d  vel=%d  r=%s",
                         line_col, error_norm, total_px, vel, r_str)
            else:
                line_col   = None
                error_norm = 0.0
                smoothed   = np.zeros(w, dtype=np.float32)
                vel        = 0
                radius     = 0x8000
                result     = "line_lost"
                log.warning("Blue line lost (px=%d < %d) — stopping",
                            total_px, _MIN_BLUE_PX)

            # ── Drive ─────────────────────────────────────────────────────
            if rover_ctrl and not state.paused.is_set():
                rover_ctrl.drive_raw(vel, radius)
                time.sleep(DRIVE_DURATION_S)
                rover_ctrl.stop()

            # ── Annotate ──────────────────────────────────────────────────
            strip_top = max(0, h - STRIP_ROWS)
            display   = _annotate(frame, strip_top, mask, smoothed,
                                  line_col, cx, vel, radius, error_norm,
                                  result, total_px)
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
                    "blue_px":   total_px,
                    "elapsed_s": round(elapsed, 3),
                }

        except Exception:
            log.exception("LineFollowStrategy error")
        finally:
            state.query_in_flight.clear()


# ── Visualisation ─────────────────────────────────────────────────────────────

def _annotate(
    frame: np.ndarray,
    strip_top: int,
    mask: np.ndarray,
    smoothed: np.ndarray,
    line_col: Optional[int],
    cx: int,
    vel: int,
    radius: int,
    error_norm: float,
    result: str,
    total_px: int,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Highlight detected blue pixels in the strip
    blue_layer = np.zeros_like(out[strip_top:])
    blue_layer[mask > 0] = (255, 180, 0)   # BGR light-blue tint
    out[strip_top:] = cv2.addWeighted(out[strip_top:], 1.0, blue_layer, 0.7, 0)

    # Strip boundary
    cv2.line(out, (0, strip_top), (w, strip_top), (255, 200, 0), 2)

    # Column count graph just above the strip
    graph_h = min(60, strip_top // 3)
    if smoothed.size == w and graph_h > 0:
        s_max = max(smoothed.max(), 1.0)
        for x in range(w):
            bar_h = int(smoothed[x] / s_max * graph_h)
            cv2.line(out, (x, strip_top - 1), (x, strip_top - 1 - bar_h),
                     (200, 140, 0), 1)

    # Centre reference
    cv2.line(out, (cx, strip_top - graph_h - 10), (cx, h), (80, 80, 80), 1)

    # Detected line column
    if line_col is not None:
        cv2.line(out, (line_col, strip_top), (line_col, h), (255, 200, 0), 3)
        mid_y = strip_top + (h - strip_top) // 2
        cv2.circle(out, (line_col, mid_y), 14, (255, 220, 0), -1)
        cv2.line(out, (cx, mid_y), (line_col, mid_y), (255, 220, 0), 2)

    # HUD
    r_str = "straight" if radius == 0x8000 else f"r={radius}mm"
    hud = [
        f"line_follow  {result}",
        f"vel={vel}mm/s  {r_str}",
        f"error={error_norm:+.3f}  px={total_px}",
    ]
    for i, txt in enumerate(hud):
        cv2.putText(out, txt, (12, 36 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2, cv2.LINE_AA)

    return out
