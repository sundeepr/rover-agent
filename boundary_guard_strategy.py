"""
BoundaryGuardStrategy — down-camera-only crop-row navigation.

No cloud inference, no front camera.  The rover drives forward slowly as long
as the green vegetation visible in the down-camera feed is NOT encroaching on
the rover boundary.  Only vegetation blobs that are within a configurable
pixel margin of the rover polygon are considered — distant blobs in the middle
of the field are ignored.

Decision logic per query cycle
───────────────────────────────
  1. Compute ExG vegetation mask from the down-camera frame.
  2. Zero out pixels inside the rover polygon (those can't be trampled).
  3. Dilate the rover polygon outward by `boundary_margin_px` to create a
     "guard zone" just outside the rover body.
  4. Find vegetation blobs whose area in the guard zone exceeds `exg_min_area`.
  5. If any such blobs exist → stop and log which side (left / right / both).
  6. Otherwise → drive straight forward at `boundary_vel_mm_s`.

All tuning lives in rover_geometry.json (hot-reloaded each cycle):

  "boundary_margin_px"  — width (px) of the guard zone around the polygon.
                          Increase if the rover clips plants before stopping.
                          Default 30.
  "boundary_vel_mm_s"   — forward velocity while the path is clear (mm/s).
                          Default 30.
  "exg_threshold"       — ExG binarisation threshold (shared with bev_omnivla).
  "exg_min_area"        — minimum blob area in the guard zone to trigger a stop.
  "rover_polygon_px"    — rover footprint corners in down-camera pixel space.

Usage
─────
  python rover_agent.py --strategy boundary_guard \\
      --down-device 0 \\
      --rover-geometry rover_geometry.json \\
      --rover atlas --atlas-port /dev/ttyACM0
"""

import json
import logging
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from navigation_strategy import AgentState, NavigationStrategy

log = logging.getLogger("rover.boundary_guard")

# ── Geometry defaults ──────────────────────────────────────────────────────────

_GEO_DEFAULTS: dict = {
    "rover_polygon_px":   [[120, 180], [520, 180], [520, 380], [120, 380]],
    "ignore_boxes_px":    [],
    "boundary_margin_px": 30,
    "boundary_vel_mm_s":  30,
    "exg_threshold":      20,
    "exg_min_area":       300,
}


def _load_geometry(path: str | None) -> dict:
    cfg = _GEO_DEFAULTS.copy()
    if path:
        p = Path(path)
        if p.exists():
            try:
                cfg.update(json.loads(p.read_text()))
            except Exception as e:
                log.warning("rover_geometry load error (%s): %s", path, e)
    return cfg


# ── ExG helpers ───────────────────────────────────────────────────────────────

def _exg_mask(frame_bgr: np.ndarray, threshold: int) -> np.ndarray:
    f   = frame_bgr.astype(np.float32)
    exg = 2.0 * f[:, :, 1] - f[:, :, 0] - f[:, :, 2]
    raw = np.clip(exg, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(raw, threshold, 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask


def _build_guard_zone(
    poly: np.ndarray,
    frame_shape: tuple,
    margin_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (rover_mask, guard_zone_mask) both uint8 binary.

    rover_mask    — filled rover polygon (255 inside).
    guard_zone    — the ring of pixels within margin_px outside the polygon.
    """
    h, w = frame_shape[:2]
    rover_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(rover_mask, [poly], 255)

    kernel     = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1))
    dilated    = cv2.dilate(rover_mask, kernel)
    guard_zone = cv2.subtract(dilated, rover_mask)   # ring only
    return rover_mask, guard_zone


# ── Strategy ───────────────────────────────────────────────────────────────────

class BoundaryGuardStrategy(NavigationStrategy):
    """
    Creep forward while green vegetation stays outside the rover boundary.
    Stop the moment any plant encroaches into the guard zone.
    """

    # No language goal needed — signal agent_loop to skip the goal_ready gate
    requires_goal = False

    def __init__(self, geometry_path: str | None = None) -> None:
        self._geometry_path = geometry_path

        self._down_frame:     np.ndarray | None = None
        self._down_annotated: np.ndarray | None = None
        self._down_lock = threading.Lock()

        # Last decision — for annotation between cycles
        self._last_blocked  = False
        self._last_side: str = ""

        log.info("BoundaryGuardStrategy ready  geometry=%s",
                 geometry_path or "(defaults)")

    # ── NavigationStrategy interface ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "boundary_guard"

    def on_reset(self) -> None:
        with self._down_lock:
            self._down_frame     = None
            self._down_annotated = None
        self._last_blocked = False
        self._last_side    = ""

    def run_query(
        self,
        state: AgentState,
        frame: np.ndarray,        # front camera — not used
        captures_dir: Path,
        rover_ctrl,
    ) -> None:
        try:
            self._do_step(state, rover_ctrl)
        finally:
            state.query_in_flight.clear()

    # ── Down-camera interface ─────────────────────────────────────────────────

    def update_down_frame(self, frame: np.ndarray) -> None:
        with self._down_lock:
            self._down_frame = frame.copy()

    def get_down_annotated_frame(self) -> np.ndarray | None:
        with self._down_lock:
            return (self._down_annotated.copy()
                    if self._down_annotated is not None else None)

    def _get_down_frame(self) -> np.ndarray | None:
        with self._down_lock:
            return self._down_frame.copy() if self._down_frame is not None else None

    # ── Core step ─────────────────────────────────────────────────────────────

    def _do_step(self, state: AgentState, rover_ctrl) -> None:
        geo       = _load_geometry(self._geometry_path)
        threshold = int(geo.get("exg_threshold",      20))
        min_area  = int(geo.get("exg_min_area",       300))
        margin    = int(geo.get("boundary_margin_px",  30))
        vel       = int(geo.get("boundary_vel_mm_s",   30))
        poly_raw  = geo.get("rover_polygon_px", _GEO_DEFAULTS["rover_polygon_px"])
        poly      = np.array(poly_raw, dtype=np.int32)

        paused          = state.paused.is_set()
        operator_active = (state.operator_control is not None
                           and state.operator_until > time.time())

        down = self._get_down_frame()
        blocked     = False
        side        = ""
        veg_mask    = None
        guard_zone  = None
        danger_mask = None
        rover_mask  = None

        if down is not None:
            h, w = down.shape[:2]

            # Step 1 — ExG mask
            veg_mask = _exg_mask(down, threshold)

            # Step 2 — zero out rover polygon and ignore boxes
            rover_mask, guard_zone = _build_guard_zone(poly, down.shape, margin)
            veg_mask = cv2.bitwise_and(veg_mask,
                                       cv2.bitwise_not(rover_mask))
            for box in geo.get("ignore_boxes_px", []):
                if len(box) == 4:
                    x1, y1, x2, y2 = (int(v) for v in box)
                    veg_mask[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)] = 0

            # Step 3 — vegetation in the guard zone only
            danger_mask = cv2.bitwise_and(veg_mask, guard_zone)

            # Step 4 — check area and side
            total_danger = int(np.sum(danger_mask > 0))
            if total_danger >= min_area:
                blocked = True
                cx      = w // 2
                left_px  = int(np.sum(danger_mask[:, :cx] > 0))
                right_px = int(np.sum(danger_mask[:, cx:] > 0))
                if left_px > 0 and right_px > 0:
                    side = "both"
                elif left_px >= right_px:
                    side = "left"
                else:
                    side = "right"

            # Annotate
            ann = self._annotate(down, geo, poly, veg_mask, guard_zone,
                                 danger_mask, blocked, side, vel)
            with self._down_lock:
                self._down_annotated = ann

        # Step 5 — drive decision
        self._last_blocked = blocked
        self._last_side    = side

        if paused or operator_active:
            pass   # let operator keep control
        elif blocked:
            if rover_ctrl:
                rover_ctrl.stop()
            log.info("STOP — vegetation in guard zone (%s side, %d px)",
                     side, int(np.sum(danger_mask > 0)) if danger_mask is not None else 0)
        else:
            if rover_ctrl:
                rover_ctrl.drive_raw(vel, 0x8000)   # straight forward
            log.debug("FORWARD  vel=%d mm/s", vel)

        # Update shared state for web display
        result = {
            "strategy":    self.name,
            "step":        state.step,
            "blocked":     blocked,
            "side":        side,
            "vel_mm_s":    0 if blocked else vel,
        }
        with state.result_lock:
            state.latest_result = result
            state.llm_query_start = 0.0

        if down is not None:
            ann = self.get_down_annotated_frame()
            if ann is not None:
                with state.llm_lock:
                    state.llm_frame = ann

    # ── Annotation ────────────────────────────────────────────────────────────

    def _annotate(
        self,
        frame:       np.ndarray,
        geo:         dict,
        poly:        np.ndarray,
        veg_mask:    np.ndarray | None,
        guard_zone:  np.ndarray | None,
        danger_mask: np.ndarray | None,
        blocked:     bool,
        side:        str,
        vel:         int,
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        # ── Vegetation (green tint, outside rover + guard zone) ───────────────
        if veg_mask is not None:
            safe_veg = veg_mask.copy()
            if guard_zone is not None:
                safe_veg = cv2.bitwise_and(safe_veg, cv2.bitwise_not(guard_zone))
            green_layer = np.zeros_like(out)
            green_layer[safe_veg > 0] = (0, 200, 60)
            cv2.addWeighted(green_layer, 0.40, out, 0.60, 0, out)

        # ── Guard zone (yellow semi-transparent ring) ─────────────────────────
        if guard_zone is not None:
            yellow_layer = np.zeros_like(out)
            yellow_layer[guard_zone > 0] = (0, 220, 255)
            cv2.addWeighted(yellow_layer, 0.25, out, 0.75, 0, out)

        # ── Danger vegetation (red — plants in guard zone) ────────────────────
        if danger_mask is not None and np.any(danger_mask):
            red_layer = np.zeros_like(out)
            red_layer[danger_mask > 0] = (0, 0, 255)
            cv2.addWeighted(red_layer, 0.60, out, 0.40, 0, out)

        # ── Rover polygon (purple) ─────────────────────────────────────────────
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], (180, 0, 180))
        cv2.addWeighted(overlay, 0.20, out, 0.80, 0, out)
        cv2.polylines(out, [poly], isClosed=True, color=(220, 0, 220), thickness=2)
        for pt in poly:
            cv2.circle(out, tuple(pt), 5, (255, 80, 255), -1)
        cx_poly = int(poly[:, 0].mean())
        cy_poly = int(poly[:, 1].mean())
        cv2.putText(out, "rover", (cx_poly - 22, cy_poly + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 255), 1, cv2.LINE_AA)

        # ── Ignore boxes ───────────────────────────────────────────────────────
        for i, box in enumerate(geo.get("ignore_boxes_px", [])):
            if len(box) == 4:
                x1, y1, x2, y2 = (int(v) for v in box)
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                ov2 = out.copy()
                cv2.rectangle(ov2, (x1, y1), (x2, y2), (50, 50, 50), -1)
                cv2.addWeighted(ov2, 0.40, out, 0.60, 0, out)
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 80, 220), 2)
                cv2.putText(out, f"ignore{i}", (x1 + 4, y1 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 160, 255), 1)

        # ── Status banner ─────────────────────────────────────────────────────
        if blocked:
            banner_col = (0, 0, 180)
            banner_txt = f"STOP — plant on {side} boundary"
        else:
            banner_col = (0, 120, 0)
            banner_txt = f"FORWARD  {vel} mm/s"

        cv2.rectangle(out, (0, 0), (w, 50), banner_col, -1)
        cv2.putText(out, banner_txt, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        # ── HUD ───────────────────────────────────────────────────────────────
        margin = int(geo.get("boundary_margin_px", 30))
        cv2.putText(out, f"boundary_guard  margin={margin}px",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 180, 255), 1, cv2.LINE_AA)

        return out
