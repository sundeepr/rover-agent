"""
LineFollowStrategy — pure-CV black-line follower for the Atlas rover.

Detection: column projection
────────────────────────────
Rather than thresholding (which picks up shadows and random dark patches),
the algorithm averages each column of pixels across the ROI height.  A black
line running straight ahead creates a consistently dark column; isolated
shadows and other marks average out.

Pipeline (each step)
────────────────────
  1. Crop the bottom `roi_frac` of the frame (nearest ground).
  2. Gaussian blur to suppress noise.
  3. Convert to grayscale; compute mean brightness per column → profile[w].
  4. Smooth the profile with a 1-D moving average.
  5. Line column = argmin of the smoothed profile.
  6. Confirm line: contrast = (mean - min) / mean must exceed `min_contrast`.
  7. Lateral error = (line_col - frame_cx) / half_width  ∈ [-1, 1].
  8. radius_mm = -kp / error  (right error → negative radius → turn right).
  9. drive_raw(vel_mm_s, radius_mm) for DRIVE_DURATION_S, then stop.

Usage
─────
    python rover_agent.py --strategy line_follow --rover atlas \\
        --atlas-port /dev/ttyACM0 --interval 0.4 \\
        [--line-vel 80] [--line-kp 2000] [--line-roi-frac 0.4]

Tuning tips
───────────
  --line-roi-frac    Fraction of frame height to look at from the bottom.
                     Larger = sees further ahead but more background clutter.
  --line-kp          Proportional gain. Reduce if rover oscillates.
  --line-vel         Forward speed in mm/s.
  --interval         How often run_query fires. 0.3–0.5 s works well.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from navigation_strategy import AgentState, NavigationStrategy

log = logging.getLogger("rover.line_follow")

DRIVE_DURATION_S  = 0.3    # seconds to drive per step before re-evaluating
_SMOOTH_WIN       = 21     # 1-D moving-average window for the column profile
_MIN_CONTRAST     = 0.08   # (mean - min) / mean must exceed this to confirm a line


class LineFollowStrategy(NavigationStrategy):

    def __init__(
        self,
        vel_mm_s: int   = 80,
        kp: float       = 2000.0,
        threshold: int  = 80,   # kept for CLI compat but unused in projection mode
        roi_frac: float = 0.4,
    ):
        self._vel      = vel_mm_s
        self._kp       = kp
        self._roi_frac = roi_frac

    @property
    def name(self) -> str:
        return "line_follow"

    def on_reset(self) -> None:
        pass   # stateless

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

            # ── Column projection: find the darkest column ────────────────
            gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            roi_h   = blurred.shape[0]
            # Weight rows so bottom (closest, most relevant for centering)
            # contributes more than top (further ahead).
            weights = np.linspace(1.0, 0.1, roi_h, dtype=np.float32)[:, np.newaxis]
            profile = (blurred.astype(np.float32) * weights).sum(axis=0) / weights.sum()

            # 1-D moving average to suppress narrow noise spikes
            kernel   = np.ones(_SMOOTH_WIN, dtype=np.float32) / _SMOOTH_WIN
            smoothed = np.convolve(profile, kernel, mode='same')

            line_col  = int(np.argmin(smoothed))
            contrast  = (smoothed.mean() - smoothed.min()) / max(smoothed.mean(), 1.0)
            line_detected = contrast > _MIN_CONTRAST

            if line_detected:
                centroid_x = line_col
                error_norm = (centroid_x - cx) / (w / 2.0)   # [-1, 1]

                if abs(error_norm) < 0.02:
                    radius = 0x8000   # straight
                else:
                    radius = int(-self._kp / error_norm)
                    radius = max(-5000, min(5000, radius))

                vel    = self._vel
                result = "following"
                r_str  = "straight" if radius == 0x8000 else f"{radius}mm"
                log.info("Line col=%d  error=%.3f  contrast=%.3f  vel=%d  r=%s",
                         line_col, error_norm, contrast, vel, r_str)
            else:
                centroid_x = None
                error_norm = 0.0
                vel        = 0
                radius     = 0x8000
                result     = "line_lost"
                log.warning("Line lost (contrast=%.3f < %.2f) — stopping",
                            contrast, _MIN_CONTRAST)

            # ── Drive ─────────────────────────────────────────────────────
            if rover_ctrl and not state.paused.is_set():
                rover_ctrl.drive_raw(vel, radius)
                time.sleep(DRIVE_DURATION_S)
                rover_ctrl.stop()

            # ── Annotate display frame ────────────────────────────────────
            display = _annotate(frame, roi_top, smoothed, centroid_x,
                                cx, vel, radius, error_norm, result, contrast)
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
                    "contrast":   round(float(contrast), 4),
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
    smoothed_profile: np.ndarray,
    centroid_x: Optional[int],
    frame_cx: int,
    vel: int,
    radius: int,
    error_norm: float,
    result: str,
    contrast: float,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Slight ROI tint
    roi_region = out[roi_top:].astype(np.float32)
    out[roi_top:] = (roi_region * 0.7).clip(0, 255).astype(np.uint8)

    # Draw column brightness profile as a mini graph at the ROI top edge
    graph_h = min(80, (h - roi_top) // 3)
    p_min, p_max = smoothed_profile.min(), smoothed_profile.max()
    p_range = max(p_max - p_min, 1.0)
    for x in range(w):
        bar_h = int((smoothed_profile[x] - p_min) / p_range * graph_h)
        y_top = roi_top + graph_h - bar_h
        cv2.line(out, (x, roi_top + graph_h), (x, y_top), (60, 60, 120), 1)

    # ROI boundary
    cv2.line(out, (0, roi_top), (w, roi_top), (0, 200, 255), 2)

    # Centre reference line
    cv2.line(out, (frame_cx, roi_top), (frame_cx, h), (80, 80, 80), 1)

    # Detected line column — vertical cyan stripe
    if centroid_x is not None:
        cv2.line(out, (centroid_x, roi_top), (centroid_x, h), (0, 255, 220), 3)
        mid_y = roi_top + (h - roi_top) // 2
        cv2.circle(out, (centroid_x, mid_y), 12, (0, 255, 100), -1)
        cv2.line(out, (frame_cx, mid_y), (centroid_x, mid_y), (0, 255, 100), 2)

    # HUD
    r_str = "straight" if radius == 0x8000 else f"r={radius}mm"
    hud = [
        f"line_follow  {result}",
        f"vel={vel}mm/s  {r_str}",
        f"error={error_norm:+.3f}  contrast={contrast:.3f}",
    ]
    for i, txt in enumerate(hud):
        cv2.putText(out, txt, (12, 36 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2, cv2.LINE_AA)

    return out
