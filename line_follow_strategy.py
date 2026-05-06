"""
LineFollowStrategy — pure-CV black-line follower for the Atlas rover.

Detects a black line in the lower portion of the forward camera frame
using a brightness threshold, finds the centroid of the line, and
steers the rover with a proportional controller.  No ML inference.

Pipeline (each step)
────────────────────
  1. Crop the bottom `roi_frac` of the frame (nearest ground).
  2. Convert to grayscale; threshold at `threshold` (dark pixels = line).
  3. Find the centroid of the thresholded region.
  4. Lateral error = (centroid_x - frame_cx) / half_width  ∈ [-1, 1]
       positive = line is to the RIGHT of centre.
  5. radius_mm = -kp / error   (negative error → right → negative radius)
  6. drive_raw(vel_mm_s, radius_mm) for DRIVE_DURATION_S, then stop.

Usage
─────
    python rover_agent.py --strategy line_follow --rover atlas \\
        --atlas-port /dev/ttyACM0 --interval 0.4 \\
        [--line-vel 80] [--line-kp 2000] [--line-threshold 80] \\
        [--line-roi-frac 0.35]

Tuning tips
───────────
  --line-threshold   Lower = only very dark pixels count (stricter).
                     Raise if the line is not pure black.
  --line-kp          Higher = more aggressive steering.
                     Lower if the rover oscillates.
  --line-roi-frac    Larger ROI sees more of the ground but may include
                     the horizon and confuse the detector.
  --interval         Controls how often run_query is called (seconds).
                     0.3–0.5 s works well for line following.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from navigation_strategy import AgentState, NavigationStrategy

log = logging.getLogger("rover.line_follow")

DRIVE_DURATION_S = 0.3   # seconds to drive per step before re-evaluating


class LineFollowStrategy(NavigationStrategy):

    def __init__(
        self,
        vel_mm_s: int   = 80,
        kp: float       = 2000.0,
        threshold: int  = 80,
        roi_frac: float = 0.4,
    ):
        self._vel      = vel_mm_s
        self._kp       = kp
        self._thresh   = threshold
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

            # ── ROI: bottom roi_frac of frame (closest ground) ────────────
            roi_top = int(h * (1.0 - self._roi_frac))
            roi     = frame[roi_top:, :]

            # ── Detect black line ─────────────────────────────────────────
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, self._thresh, 255, cv2.THRESH_BINARY_INV)

            M = cv2.moments(mask)
            line_detected = M["m00"] > 0

            if line_detected:
                centroid_x = int(M["m10"] / M["m00"])
                error_norm = (centroid_x - cx) / (w / 2.0)   # [-1, 1]

                if abs(error_norm) < 0.02:
                    radius = 0x8000   # straight
                else:
                    radius = int(-self._kp / error_norm)
                    radius = max(-5000, min(5000, radius))

                vel    = self._vel
                result = "following"
                r_str  = "straight" if radius == 0x8000 else f"{radius}mm"
                log.info("Line: centroid_x=%d  error=%.3f  vel=%d  r=%s",
                         centroid_x, error_norm, vel, r_str)
            else:
                centroid_x = None
                error_norm = 0.0
                vel        = 0
                radius     = 0x8000
                result     = "line_lost"
                log.warning("Line lost — stopping")

            # ── Drive ─────────────────────────────────────────────────────
            if rover_ctrl and not state.paused.is_set():
                rover_ctrl.drive_raw(vel, radius)
                time.sleep(DRIVE_DURATION_S)
                rover_ctrl.stop()

            # ── Annotate display frame ────────────────────────────────────
            display = _annotate(frame, roi_top, mask, centroid_x,
                                cx, vel, radius, error_norm, result)
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
                    "elapsed_s": round(elapsed, 3),
                }

        except Exception:
            log.exception("LineFollowStrategy error")
        finally:
            state.query_in_flight.clear()


# ── Visualisation ─────────────────────────────────────────────────────────────

def _annotate(
    frame: np.ndarray,
    roi_top: int,
    mask: np.ndarray,
    centroid_x: Optional[int],
    frame_cx: int,
    vel: int,
    radius: int,
    error_norm: float,
    result: str,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Darken ROI slightly to distinguish it visually
    roi_region = out[roi_top:].astype(np.float32)
    roi_region = (roi_region * 0.65).clip(0, 255).astype(np.uint8)
    out[roi_top:] = roi_region

    # Overlay detected line mask in blue
    blue_layer = np.zeros_like(out[roi_top:])
    blue_layer[:, :, 0] = mask   # blue channel
    out[roi_top:] = cv2.addWeighted(out[roi_top:], 1.0, blue_layer, 0.6, 0)

    # ROI boundary
    cv2.line(out, (0, roi_top), (w, roi_top), (0, 200, 255), 2)

    # Centre reference line
    cv2.line(out, (frame_cx, roi_top), (frame_cx, h), (80, 80, 80), 1)

    # Centroid and error line
    if centroid_x is not None:
        mid_y = roi_top + (h - roi_top) // 2
        cv2.circle(out, (centroid_x, mid_y), 14, (0, 255, 100), -1)
        cv2.line(out, (frame_cx, mid_y), (centroid_x, mid_y), (0, 255, 100), 2)

    # HUD
    r_str = "straight" if radius == 0x8000 else f"r={radius}mm"
    hud = [
        f"line_follow  {result}",
        f"vel={vel}mm/s  {r_str}",
        f"error={error_norm:+.3f}",
    ]
    for i, txt in enumerate(hud):
        cv2.putText(out, txt, (12, 36 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2, cv2.LINE_AA)

    return out
