"""
PlantCenterStrategy — down-camera-only crop-row centering.

No cloud inference, no front camera.  Uses only the downward-facing camera.

Algorithm per query cycle
──────────────────────────
  1. Compute ExG vegetation mask from the down-camera frame.
  2. Exclude the rover polygon and any ignore_boxes_px.
  3. Find the nearest green blob on each side of the rover boundary:
       left side  — the blob immediately to the left  of the rover (closest
                    right-edge to the rover's left  edge).
       right side — the blob immediately to the right of the rover (closest
                    left-edge  to the rover's right edge).
  4. Target X = midpoint of those two inner edges  (X coordinates only —
       no Y / forward component).
  5. error_px = target_x − rover_center_x
       positive → target is to the right → steer right (negative radius)
       negative → target is to the left  → steer left  (positive radius)
  6. P-controller:
       ang_rate = centering_gain × error_mm   (rad/s)
       radius   = −vel / ang_rate             (mm)
  7. drive_raw(vel, radius).

Fallback behaviour
──────────────────
  Only left wall visible  → steer slightly right.
  Only right wall visible → steer slightly left.
  No walls detected       → drive straight at reduced speed.

rover_geometry.json keys
─────────────────────────
  rover_polygon_px      corners of rover footprint in down-camera pixels
  ignore_boxes_px       rectangles to exclude (overhangs, brackets)
  down_px_per_mm        ground scale of down camera (px per mm)
  exg_threshold         ExG binarisation threshold
  exg_min_area          minimum blob area to count as a plant wall
  boundary_vel_mm_s     forward velocity while navigating (mm/s, default 30)
  min_radius_mm         tightest allowed turn radius (mm, default 400)
  centering_gain        P-gain: rad/s per mm of lateral error (default 0.003)
                        increase to steer more aggressively, decrease for
                        smoother but slower corrections.

Usage
─────
  python rover_agent.py --strategy plant_center \\
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

log = logging.getLogger("rover.plant_center")

# ── Geometry defaults ──────────────────────────────────────────────────────────

_GEO_DEFAULTS: dict = {
    "rover_polygon_px":   [[120, 180], [520, 180], [520, 380], [120, 380]],
    "ignore_boxes_px":    [],
    "down_px_per_mm":     2.5,
    "exg_threshold":      20,
    "exg_min_area":       300,
    "boundary_vel_mm_s":  30,
    "min_radius_mm":      400,
    "centering_gain":     0.003,   # rad/s per mm of lateral error
}

_STRAIGHT = 0x8000   # atlas sentinel for straight ahead


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


# ── ExG mask ──────────────────────────────────────────────────────────────────

def _exg_mask(frame_bgr: np.ndarray, threshold: int) -> np.ndarray:
    f   = frame_bgr.astype(np.float32)
    exg = 2.0 * f[:, :, 1] - f[:, :, 0] - f[:, :, 2]
    raw = np.clip(exg, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(raw, threshold, 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask


# ── Wall detection relative to rover polygon ───────────────────────────────────

def _find_nearest_walls(
    mask:          np.ndarray,
    rover_left_x:  int,
    rover_right_x: int,
    min_area:      int,
) -> tuple[int | None, int | None, list]:
    """
    Find the nearest vegetation walls on each side of the rover boundary.

    Returns (left_inner_x, right_inner_x, boxes) where:
      left_inner_x  — rightmost edge of the closest left-side blob
                      (the inner/facing edge of the left plant row).
      right_inner_x — leftmost  edge of the closest right-side blob.
      boxes         — all qualifying [x1,y1,x2,y2] bounding boxes.

    Blobs are classified by whether their centre x lies to the LEFT of
    rover_left_x (left blobs) or to the RIGHT of rover_right_x (right blobs).
    Blobs that overlap the rover boundary band are ignored.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list = []
    left_candidates:  list[int] = []   # inner (right) edge x of left blobs
    right_candidates: list[int] = []   # inner (left)  edge x of right blobs

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        cx_blob = x + bw // 2
        boxes.append([x, y, x + bw, y + bh])

        if cx_blob < rover_left_x:
            # Blob is to the left of the rover — record its right (inner) edge
            left_candidates.append(x + bw)
        elif cx_blob > rover_right_x:
            # Blob is to the right of the rover — record its left (inner) edge
            right_candidates.append(x)

    # Nearest left wall  = the candidate with the LARGEST  x (closest to rover)
    # Nearest right wall = the candidate with the SMALLEST x (closest to rover)
    left_wall  = max(left_candidates)  if left_candidates  else None
    right_wall = min(right_candidates) if right_candidates else None

    return left_wall, right_wall, boxes


# ── Strategy ───────────────────────────────────────────────────────────────────

class PlantCenterStrategy(NavigationStrategy):
    """
    Drive through the X-centre of the nearest green plant walls on each side.
    Pure down-camera reactive control — no cloud inference required.
    """

    def __init__(self, geometry_path: str | None = None) -> None:
        self._geometry_path = geometry_path

        self._down_frame:     np.ndarray | None = None
        self._down_annotated: np.ndarray | None = None
        self._down_lock = threading.Lock()

        log.info("PlantCenterStrategy ready  geometry=%s",
                 geometry_path or "(defaults)")

    # ── NavigationStrategy interface ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "plant_center"

    def on_reset(self) -> None:
        with self._down_lock:
            self._down_frame     = None
            self._down_annotated = None

    def run_query(
        self,
        state:        AgentState,
        frame:        np.ndarray,
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
        geo           = _load_geometry(self._geometry_path)
        threshold     = int(geo["exg_threshold"])
        min_area      = int(geo["exg_min_area"])
        px_per_mm     = float(geo["down_px_per_mm"])
        vel           = int(geo["boundary_vel_mm_s"])
        min_r         = int(geo["min_radius_mm"])
        gain          = float(geo["centering_gain"])
        poly_raw      = geo["rover_polygon_px"]
        poly          = np.array(poly_raw, dtype=np.int32)

        rover_left_x  = int(poly[:, 0].min())
        rover_right_x = int(poly[:, 0].max())
        rover_cx      = int(poly[:, 0].mean())

        paused          = state.paused.is_set()
        operator_active = (state.operator_control is not None
                           and state.operator_until > time.time())

        down = self._get_down_frame()
        left_wall = right_wall = None
        boxes:    list = []
        target_cx:    int | None = None
        error_px:     int = 0
        radius:       int = _STRAIGHT
        status:       str = "no down frame"

        if down is None:
            log.warning("plant_center | no down frame")
        else:
            h, w = down.shape[:2]

            # ExG mask — exclude rover polygon and ignore boxes
            mask = _exg_mask(down, threshold)
            cv2.fillPoly(mask, [poly], 0)
            for box in geo.get("ignore_boxes_px", []):
                if len(box) == 4:
                    x1, y1, x2, y2 = (int(v) for v in box)
                    mask[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)] = 0

            left_wall, right_wall, boxes = _find_nearest_walls(
                mask, rover_left_x, rover_right_x, min_area)

            # ── Compute target centre X ───────────────────────────────────────
            if left_wall is not None and right_wall is not None:
                target_cx = (left_wall + right_wall) // 2
                status    = f"L={left_wall}px  R={right_wall}px  target={target_cx}px"
            elif left_wall is not None:
                # Only left wall — aim to keep it at same distance as rover half-width
                half_w    = (rover_right_x - rover_left_x) // 2
                target_cx = left_wall + half_w
                status    = f"L={left_wall}px only → target={target_cx}px"
            elif right_wall is not None:
                half_w    = (rover_right_x - rover_left_x) // 2
                target_cx = right_wall - half_w
                status    = f"R={right_wall}px only → target={target_cx}px"
            else:
                status = "no walls detected — straight"

            # ── P-controller ──────────────────────────────────────────────────
            if target_cx is not None:
                error_px  = target_cx - rover_cx
                error_mm  = error_px / px_per_mm
                # Positive error → target right → steer right → negative radius
                ang_rate  = gain * error_mm   # rad/s, positive = left
                if abs(ang_rate) < 1e-3:
                    radius = _STRAIGHT
                else:
                    radius = int(-vel / ang_rate)
                    if 0 < abs(radius) < min_r:
                        radius = int(math.copysign(min_r, radius))

            log.info(
                "plant_center | %s  err=%+dpx (%+.1fmm)  r=%s",
                status, error_px, error_px / px_per_mm,
                "straight" if radius == _STRAIGHT else f"{radius}mm",
            )

            # Annotate
            ann = self._annotate(
                down, geo, poly, mask, left_wall, right_wall,
                boxes, rover_cx, target_cx, error_px, radius, vel)
            with self._down_lock:
                self._down_annotated = ann

        # ── Drive ─────────────────────────────────────────────────────────────
        if not paused and not operator_active and rover_ctrl and down is not None:
            rover_ctrl.drive_raw(vel, radius)

        # Update shared state
        with state.result_lock:
            state.latest_result = {
                "strategy":  self.name,
                "step":      state.step,
                "left_wall": left_wall,
                "right_wall": right_wall,
                "target_cx": target_cx,
                "error_px":  error_px,
                "radius_mm": None if radius == _STRAIGHT else radius,
                "vel_mm_s":  vel,
            }
            state.llm_query_start = 0.0

        ann = self.get_down_annotated_frame()
        if ann is not None:
            with state.llm_lock:
                state.llm_frame = ann

    # ── Annotation ────────────────────────────────────────────────────────────

    def _annotate(
        self,
        frame:     np.ndarray,
        geo:       dict,
        poly:      np.ndarray,
        veg_mask:  np.ndarray,
        left_wall: int | None,
        right_wall:int | None,
        boxes:     list,
        rover_cx:  int,
        target_cx: int | None,
        error_px:  int,
        radius:    int,
        vel:       int,
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        # ExG vegetation tint (green)
        if veg_mask is not None:
            layer = np.zeros_like(out)
            layer[veg_mask > 0] = (0, 200, 60)
            cv2.addWeighted(layer, 0.40, out, 0.60, 0, out)

        # Blob bounding boxes — orange = left side, blue = right side
        rover_left_x  = int(poly[:, 0].min())
        rover_right_x = int(poly[:, 0].max())
        for b in boxes:
            cx_b  = (b[0] + b[2]) // 2
            color = (0, 140, 255) if cx_b < rover_left_x else (255, 140, 0)
            cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), color, 1)

        # Left wall inner edge (orange vertical)
        if left_wall is not None:
            cv2.line(out, (left_wall, 0), (left_wall, h), (0, 140, 255), 2)
            cv2.putText(out, f"L={left_wall}", (left_wall + 4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)

        # Right wall inner edge (blue vertical)
        if right_wall is not None:
            cv2.line(out, (right_wall, 0), (right_wall, h), (255, 140, 0), 2)
            cv2.putText(out, f"R={right_wall}", (right_wall - 60, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 140, 0), 1)

        # Target centre line (white dashed)
        if target_cx is not None:
            for y in range(0, h, 16):
                cv2.line(out, (target_cx, y), (target_cx, min(y + 10, h)),
                         (255, 255, 255), 2)
            cv2.putText(out, "target", (target_cx + 4, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Rover centre line (yellow dashed)
        for y in range(0, h, 16):
            cv2.line(out, (rover_cx, y), (rover_cx, min(y + 10, h)),
                     (0, 220, 255), 1)

        # Error arrow from rover centre to target
        if target_cx is not None and abs(error_px) > 5:
            mid_y = h // 2
            cv2.arrowedLine(out, (rover_cx, mid_y), (target_cx, mid_y),
                            (0, 255, 255), 2, tipLength=0.2)

        # Rover polygon (purple)
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], (180, 0, 180))
        cv2.addWeighted(overlay, 0.20, out, 0.80, 0, out)
        cv2.polylines(out, [poly], isClosed=True, color=(220, 0, 220), thickness=2)
        for pt in poly:
            cv2.circle(out, tuple(pt), 5, (255, 80, 255), -1)
        cv2.putText(out, "rover",
                    (int(poly[:, 0].mean()) - 22, int(poly[:, 1].mean()) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 255), 1, cv2.LINE_AA)

        # Ignore boxes
        for i, box in enumerate(geo.get("ignore_boxes_px", [])):
            if len(box) == 4:
                x1, y1, x2, y2 = (int(v) for v in box)
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                ov2 = out.copy()
                cv2.rectangle(ov2, (x1, y1), (x2, y2), (50, 50, 50), -1)
                cv2.addWeighted(ov2, 0.40, out, 0.60, 0, out)
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 80, 220), 2)
                cv2.putText(out, f"ign{i}", (x1 + 4, y1 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 160, 255), 1)

        # Status banner
        r_str = "straight" if radius == _STRAIGHT else f"r={radius}mm"
        banner = (f"err={error_px:+d}px  {r_str}  vel={vel}mm/s"
                  if target_cx is not None else "no walls — straight")
        cv2.rectangle(out, (0, 0), (w, 48), (30, 30, 30), -1)
        cv2.putText(out, banner, (8, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)

        cv2.putText(out, "plant_center  down-cam",
                    (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (160, 160, 255), 1, cv2.LINE_AA)
        return out
