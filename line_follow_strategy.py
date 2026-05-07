"""
LineFollowStrategy — orange-line follower for the Atlas rover.

Detection: HSV colour mask + column projection
──────────────────────────────────────────────
Converts the bottom strip of the frame to HSV and masks for orange pixels.
Colour-based detection is immune to shadows and brightness variation that
fooled the earlier grayscale approach.

Pipeline (each step)
────────────────────
  1. Crop the bottom `roi_frac` of the frame (nearest ground).
  2. Take the bottom 25% of that ROI (immediately in front of rover).
  3. Convert to HSV; apply orange mask (hue 5-25, saturated, bright).
  4. Count masked pixels per column inside the valid (non-vignette) band.
  5. Smooth the column counts with a 1-D moving average.
  6. Line column = argmax of the smoothed count profile.
  7. Confirm: total orange pixel count must exceed MIN_ORANGE_PIXELS.
  8. Lateral error = (line_col - frame_cx) / half_width  ∈ [-1, 1].
  9. radius_mm = -kp / error  (right error → negative radius → turn right).
 10. drive_raw(vel_mm_s, radius_mm) for DRIVE_DURATION_S, then stop.

Usage
─────
    python rover_agent.py --strategy line_follow --rover atlas \\
        --atlas-port /dev/ttyACM0 --interval 0.4 \\
        [--line-vel 80] [--line-kp 2000] [--line-roi-frac 0.4]

Tuning tips
───────────
  --line-roi-frac    Fraction of frame height to look at from the bottom.
  --line-kp          Proportional gain. Reduce if rover oscillates.
  --line-vel         Forward speed in mm/s.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from navigation_strategy import AgentState, NavigationStrategy

log = logging.getLogger("rover.line_follow")

DRIVE_DURATION_S   = 0.3    # seconds to drive per step
_SMOOTH_WIN        = 21     # 1-D moving-average window for column counts
_MIN_ORANGE_PX     = 50     # minimum orange pixels to confirm line detected

# Orange HSV bounds (OpenCV: H 0-179, S 0-255, V 0-255)
_HSV_LO = np.array([5,  120, 80],  dtype=np.uint8)
_HSV_HI = np.array([25, 255, 255], dtype=np.uint8)


class LineFollowStrategy(NavigationStrategy):

    def __init__(
        self,
        vel_mm_s: int      = 80,
        kp: float          = 2000.0,
        threshold: int     = 80,        # unused, kept for CLI compat
        roi_frac: float    = 0.4,
        edge_margin: float = 0.15,
    ):
        self._vel         = vel_mm_s
        self._kp          = kp
        self._roi_frac    = roi_frac
        self._edge_margin = edge_margin

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
            cx   = w // 2

            # ── ROI: bottom roi_frac of frame ─────────────────────────────
            roi_top = int(h * (1.0 - self._roi_frac))
            roi     = frame[roi_top:, :]
            roi_h   = roi.shape[0]

            # ── Detection strip: bottom 25% of ROI (closest ground) ───────
            strip_top = roi_h * 3 // 4
            strip     = roi[strip_top:, :]

            # ── Orange HSV mask ───────────────────────────────────────────
            hsv  = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, _HSV_LO, _HSV_HI)

            # ── Column counts inside valid (non-vignette) band ────────────
            margin    = int(w * self._edge_margin)
            band_mask = mask[:, margin: w - margin].astype(np.float32)
            col_counts = band_mask.sum(axis=0)          # orange px per column

            kernel    = np.ones(_SMOOTH_WIN, dtype=np.float32) / _SMOOTH_WIN
            smoothed  = np.convolve(col_counts, kernel, mode='same')

            total_px      = int(mask[:, margin: w - margin].sum())
            line_detected = total_px >= _MIN_ORANGE_PX

            if line_detected:
                rel_col    = int(np.argmax(smoothed))
                line_col   = rel_col + margin
                centroid_x = line_col
                error_norm = (centroid_x - cx) / (w / 2.0)

                if abs(error_norm) < 0.02:
                    radius = 0x8000
                else:
                    radius = int(-self._kp / error_norm)
                    radius = max(-5000, min(5000, radius))

                vel    = self._vel
                result = "following"
                r_str  = "straight" if radius == 0x8000 else f"{radius}mm"
                log.info("Orange line col=%d  error=%.3f  px=%d  vel=%d  r=%s",
                         line_col, error_norm, total_px, vel, r_str)
            else:
                centroid_x = None
                error_norm = 0.0
                vel        = 0
                radius     = 0x8000
                result     = "line_lost"
                log.warning("Orange line lost (px=%d < %d) — stopping",
                            total_px, _MIN_ORANGE_PX)

            # ── Drive ─────────────────────────────────────────────────────
            if rover_ctrl and not state.paused.is_set():
                rover_ctrl.drive_raw(vel, radius)
                time.sleep(DRIVE_DURATION_S)
                rover_ctrl.stop()

            # ── Annotate display frame ────────────────────────────────────
            strip_top_abs = roi_top + strip_top
            display = _annotate(frame, roi_top, strip_top_abs, mask, smoothed,
                                margin, centroid_x, cx, vel, radius,
                                error_norm, result, total_px)
            with state.llm_lock:
                state.llm_frame = display

            elapsed = time.time() - t0
            with state.result_lock:
                state.llm_response_s  = elapsed
                state.llm_query_start = 0.0
                state.latest_result   = {
                    "strategy":   self.name,
                    "result":     result,
                    "vel_mm_s":   vel,
                    "radius_mm":  radius if radius != 0x8000 else None,
                    "error":      round(error_norm, 4),
                    "orange_px":  total_px,
                    "elapsed_s":  round(elapsed, 3),
                }

        except Exception:
            log.exception("LineFollowStrategy error")
        finally:
            state.query_in_flight.clear()


# ── Visualisation ─────────────────────────────────────────────────────────────

def _annotate(
    frame: np.ndarray,
    roi_top: int,
    strip_top_abs: int,
    mask: np.ndarray,
    smoothed: np.ndarray,
    margin: int,
    centroid_x: Optional[int],
    frame_cx: int,
    vel: int,
    radius: int,
    error_norm: float,
    result: str,
    total_px: int,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Tint full ROI
    out[roi_top:] = (out[roi_top:].astype(np.float32) * 0.7).clip(0, 255).astype(np.uint8)

    # Overlay orange mask on the detection strip in orange colour
    strip_h = h - strip_top_abs
    if strip_h > 0 and mask.shape[0] > 0:
        orange_layer = np.zeros_like(out[strip_top_abs:])
        orange_layer[mask > 0] = (0, 100, 255)   # BGR orange
        out[strip_top_abs:] = cv2.addWeighted(out[strip_top_abs:], 1.0,
                                               orange_layer, 0.6, 0)

    # Detection strip boundary
    cv2.line(out, (0, strip_top_abs), (w, strip_top_abs), (0, 165, 255), 2)
    # ROI boundary
    cv2.line(out, (0, roi_top), (w, roi_top), (0, 200, 255), 1)

    # Column count profile graph
    graph_h = min(60, (strip_top_abs - roi_top) // 2)
    if smoothed.size > 0 and graph_h > 0:
        band_w  = smoothed.size
        s_max   = max(smoothed.max(), 1.0)
        for i, val in enumerate(smoothed):
            x     = i + margin
            bar_h = int(val / s_max * graph_h)
            cv2.line(out, (x, roi_top + graph_h), (x, roi_top + graph_h - bar_h),
                     (0, 140, 255), 1)

    # Centre reference
    cv2.line(out, (frame_cx, roi_top), (frame_cx, h), (80, 80, 80), 1)

    # Detected line position
    if centroid_x is not None:
        cv2.line(out, (centroid_x, roi_top), (centroid_x, h), (0, 165, 255), 3)
        mid_y = strip_top_abs + (h - strip_top_abs) // 2
        cv2.circle(out, (centroid_x, mid_y), 14, (0, 200, 255), -1)
        cv2.line(out, (frame_cx, mid_y), (centroid_x, mid_y), (0, 200, 255), 2)

    # HUD
    r_str = "straight" if radius == 0x8000 else f"r={radius}mm"
    hud = [
        f"line_follow  {result}",
        f"vel={vel}mm/s  {r_str}",
        f"error={error_norm:+.3f}  px={total_px}",
    ]
    for i, txt in enumerate(hud):
        cv2.putText(out, txt, (12, 36 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)

    return out
