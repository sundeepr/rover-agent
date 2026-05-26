"""
BevOmniVLAStrategy — OmniVLA cloud navigation with ICR correction and
down-camera vegetation intersection guard.

Extends CloudOmniVLAStrategy with two additional capabilities built up
across multiple implementation steps:

  Step 1 (done)   — ICR bearing correction in _waypoint_to_drive
  Step 2 (this)   — Strategy skeleton: name, down-camera frame ingestion,
                     rover_geometry.json hot-reload, registration hook
  Step 3 (next)   — Rover footprint polygon overlay on down-camera feed
  Step 4          — ExG vegetation mask from down camera
  Step 5          — Trajectory arc projection onto down-camera frame
  Step 6          — Intersection check + corrective goal string

All tunable rover measurements live in rover_geometry.json (gitignored).
That file is re-read every query cycle so you can tweak values while the
rover is running — no restart needed.

Usage
─────
    python rover_agent.py --strategy bev_omnivla \\
        --cloud-server ws://<GPU-IP>:8765 \\
        --down-device 0 \\
        --rover-geometry rover_geometry.json \\
        --rover atlas --atlas-port /dev/ttyACM0 \\
        --goal "follow the crop row"
"""

import json
import logging
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from cloud_omnivla_strategy import CloudOmniVLAStrategy

log = logging.getLogger("rover.bev_omnivla")


# ── ExG vegetation detection ───────────────────────────────────────────────────

def _exg_mask(frame_bgr: np.ndarray, threshold: int = 20) -> np.ndarray:
    """Return a binary uint8 mask where ExG = 2G - R - B > threshold."""
    f   = frame_bgr.astype(np.float32)
    exg = 2.0 * f[:, :, 1] - f[:, :, 0] - f[:, :, 2]
    raw = np.clip(exg, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(raw, threshold, 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask


def _find_veg_walls(
    mask: np.ndarray, min_area: int = 500
) -> tuple[int | None, int | None, list]:
    """
    From a binary ExG mask find left/right vegetation wall x-positions.

    Returns (left_wall_x, right_wall_x, boxes) where boxes are [x1,y1,x2,y2].
    left_wall_x  — inner (rightmost) edge of left-side vegetation.
    right_wall_x — inner (leftmost)  edge of right-side vegetation.
    Either can be None if that side has no vegetation blobs.
    """
    h, w = mask.shape[:2]
    cx   = w // 2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        boxes.append([x, y, x + bw, y + bh])

    left_boxes  = [b for b in boxes if (b[0] + b[2]) / 2 < cx]
    right_boxes = [b for b in boxes if (b[0] + b[2]) / 2 >= cx]

    left_wall  = int(max((b[2] for b in left_boxes),  default=None or -1)) if left_boxes  else None
    right_wall = int(min((b[0] for b in right_boxes), default=None or w))  if right_boxes else None
    return left_wall, right_wall, boxes

# ── Geometry defaults (mirrors rover_agent._GEOMETRY_DEFAULTS) ─────────────────
_GEO_DEFAULTS = {
    "icr_offset_mm":          480,
    "down_px_per_mm":         2.5,
    "rover_polygon_px":       [[120, 180], [520, 180], [520, 380], [120, 380]],
    "lookahead_s":            1.0,
    "arc_steps":              10,
    "exg_threshold":          20,
    "exg_min_area":           500,
    "correction_goal_suffix": "steer slightly {direction} to avoid vegetation",
}


# Steps before reverting to original goal once intersection clears
_CORRECTION_CLEAR_STEPS = 3
# Thickness (px) of the arc corridor sampled against the vegetation mask
_ARC_CHECK_THICKNESS = 10

# Atlas wheel base / reference speed (mirrors atlas_controller constants)
_WHEEL_BASE_MM        = 650
_MAX_VEL_REF_MM_S     = 200


def _vel_radius_to_lr(vel: int, radius: int) -> tuple[int, int]:
    """Replicate atlas_controller._velocity_radius_to_lr for data logging."""
    if vel == 0:
        return 0, 0
    if radius == 0x8000:
        pct = max(-100, min(100, int(vel / _MAX_VEL_REF_MM_S * 100)))
        return pct, pct
    if radius == 1:
        return 60, -60   # DRIVE_SPEED_PCT
    if radius == -1:
        return -60, 60
    ratio = max(-0.9, min(0.9, _WHEEL_BASE_MM / (2 * radius)))
    v_r   = vel * (1 + ratio)
    v_l   = vel * (1 - ratio)
    max_v = max(abs(v_r), abs(v_l), _MAX_VEL_REF_MM_S)
    return (max(-100, min(100, int(v_l / max_v * 100))),
            max(-100, min(100, int(v_r / max_v * 100))))


def _load_geometry(path: str | None) -> dict:
    """Read rover_geometry.json, filling missing keys from defaults."""
    cfg = _GEO_DEFAULTS.copy()
    if path:
        p = Path(path)
        if p.exists():
            try:
                cfg.update(json.loads(p.read_text()))
            except Exception as e:
                log.warning("rover_geometry load error (%s): %s — using defaults", path, e)
    return cfg


class BevOmniVLAStrategy(CloudOmniVLAStrategy):
    """
    OmniVLA cloud strategy with ICR-corrected steering and (future) BEV
    vegetation intersection guard using the downward-facing camera.

    Parameters
    ----------
    server_url : str
        WebSocket URL of omnivla_cloud_server.py.
    goal : str
        Initial navigation goal.
    max_lin_mm_s : int
        Maximum forward velocity (mm/s).
    geometry_path : str | None
        Path to rover_geometry.json.  Hot-reloaded each query cycle.
    """

    def __init__(
        self,
        server_url: str,
        goal: str        = "",
        max_lin_mm_s: int = 150,
        geometry_path: str | None = None,
    ):
        self._geometry_path = geometry_path
        geo = _load_geometry(geometry_path)

        super().__init__(
            server_url    = server_url,
            goal          = goal,
            max_lin_mm_s  = max_lin_mm_s,
            icr_offset_m  = geo["icr_offset_mm"] / 1000.0,
        )

        # ── Down-camera state ─────────────────────────────────────────────────
        self._down_frame: np.ndarray | None = None
        self._down_lock  = threading.Lock()
        self._down_annotated: np.ndarray | None = None  # updated each query cycle

        # Vegetation detection results (set in _do_query, used by Step 6)
        self._veg_mask:   np.ndarray | None = None   # binary ExG mask
        self._left_wall:  int | None = None           # inner x of left veg row
        self._right_wall: int | None = None           # inner x of right veg row

        # Latest drive command + waypoints (captured via _write_result, used for arc + logging)
        self._last_vel:       int             = 0
        self._last_radius:    int             = 0x8000
        self._last_waypoints: np.ndarray | None = None

        # Intersection guard state (Step 6)
        self._base_goal              = goal   # original goal before any correction
        self._correcting             = False  # True while corrective goal is active
        self._correction_clear_steps = 0      # consecutive clear steps since correction

        log.info(
            "BevOmniVLAStrategy ready  icr_offset=%.0fmm  geometry=%s",
            geo["icr_offset_mm"],
            geometry_path or "(defaults)",
        )

    # ── NavigationStrategy interface ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "bev_omnivla"

    # ── Down-camera interface (called by rover_agent._down_camera_loop) ───────

    def update_down_frame(self, frame: np.ndarray) -> None:
        """Receive a new frame from the downward-facing camera (thread-safe)."""
        with self._down_lock:
            self._down_frame = frame.copy()

    def get_down_annotated_frame(self) -> np.ndarray | None:
        """Return the latest annotated down-camera frame for web display."""
        with self._down_lock:
            return (self._down_annotated.copy()
                    if self._down_annotated is not None else None)

    def _get_down_frame(self) -> np.ndarray | None:
        """Retrieve a copy of the latest raw down-camera frame."""
        with self._down_lock:
            return self._down_frame.copy() if self._down_frame is not None else None

    # ── Geometry hot-reload ───────────────────────────────────────────────────

    def _reload_geometry(self) -> dict:
        """Re-read rover_geometry.json and update icr_offset on the fly."""
        geo = _load_geometry(self._geometry_path)
        new_icr = geo["icr_offset_mm"] / 1000.0
        if abs(new_icr - self._icr_offset_m) > 1e-4:
            log.info("ICR offset updated: %.0fmm → %.0fmm",
                     self._icr_offset_m * 1000, new_icr * 1000)
            self._icr_offset_m = new_icr
        return geo

    # ── Query override ────────────────────────────────────────────────────────

    def _do_query(self, state, frame: np.ndarray, rover_ctrl) -> None:
        # Hot-reload geometry on every cycle
        geo = self._reload_geometry()

        # Run the parent cloud inference + ICR-corrected drive
        super()._do_query(state, frame, rover_ctrl)

        # ── Down-camera: ExG mask + arc + intersection guard ─────────────────
        down = self._get_down_frame()
        intersect_side: str | None = None   # 'left' | 'right' | None

        if down is not None:
            # Step 4: compute vegetation mask and wall positions.
            # Blank out the rover polygon area first so the rover body cannot
            # trigger a false-positive vegetation detection.
            threshold = int(geo.get("exg_threshold", 20))
            min_area  = int(geo.get("exg_min_area",  500))
            mask = _exg_mask(down, threshold)

            poly_raw = geo.get("rover_polygon_px", _GEO_DEFAULTS["rover_polygon_px"])
            rover_poly = np.array(poly_raw, dtype=np.int32)
            cv2.fillPoly(mask, [rover_poly], 0)   # zero out rover footprint

            left_wall, right_wall, boxes = _find_veg_walls(mask, min_area)

            self._veg_mask   = mask
            self._left_wall  = left_wall
            self._right_wall = right_wall

            # Step 5: project drive arc onto down-camera frame
            arc_px = self._compute_arc_pixels(geo)

            # Step 6: intersection check
            intersect_side = self._check_arc_intersection(arc_px, mask, down.shape[1])
            self._handle_intersection(intersect_side, geo, rover_ctrl, state)

            # Write comprehensive per-step data record
            self._write_step_data(state, geo, left_wall, right_wall,
                                  len(boxes), arc_px, intersect_side)

            annotated = self._annotate_down(
                down, geo, mask, left_wall, right_wall, boxes, arc_px, intersect_side)
            with self._down_lock:
                self._down_annotated = annotated

    # ── Per-step data logging ─────────────────────────────────────────────────

    def _write_step_data(
        self,
        state,
        geo:          dict,
        left_wall:    int | None,
        right_wall:   int | None,
        blob_count:   int,
        arc_px:       list[tuple[int, int]],
        intersect_side: str | None,
    ) -> None:
        """Write one record to data.jsonl with all computed values for this step."""
        if state.recorder is None:
            return

        vel    = self._last_vel
        radius = self._last_radius
        wps    = self._last_waypoints
        L, R   = _vel_radius_to_lr(vel, radius)

        # Angular rate at ICR
        if radius not in (0, 0x8000, 1, -1) and vel != 0:
            ang_rad_s = round(vel / radius, 4)
        else:
            ang_rad_s = 0.0

        # Waypoints: raw model units and converted to metres
        wps_raw: list | None = None
        wps_metric: list | None = None
        if wps is not None:
            from omnivla_strategy import METRIC_SPACING, WAYPOINT_IDX
            wps_raw    = wps.tolist()
            wps_metric = [[round(float(w[0]) * METRIC_SPACING, 4),
                           round(float(w[1]) * METRIC_SPACING, 4)]
                          for w in wps]

        gap_cx = ((left_wall + right_wall) // 2
                  if left_wall is not None and right_wall is not None else None)

        record = {
            "step":    state.step,
            "goal":    self._goal,
            "correcting":       self._correcting,
            "corrective_active": self._correcting,
            "base_goal":        self._base_goal if self._correcting else None,

            "drive": {
                "vel_mm_s":        vel,
                "radius_mm":       None if radius == 0x8000 else radius,
                "angular_rate_rad_s": ang_rad_s,
                "wheel_L_pct":     L,
                "wheel_R_pct":     R,
                "icr_offset_mm":   geo.get("icr_offset_mm", 480),
                "straight":        radius == 0x8000,
            },

            "waypoints": {
                "used_idx":     4,   # WAYPOINT_IDX
                "raw_units":    wps_raw,
                "metric_m":     wps_metric,
            },

            "vegetation": {
                "detected":      left_wall is not None or right_wall is not None,
                "left_wall_px":  left_wall,
                "right_wall_px": right_wall,
                "gap_cx_px":     gap_cx,
                "blob_count":    blob_count,
                "exg_threshold": geo.get("exg_threshold", 20),
                "exg_min_area":  geo.get("exg_min_area",  500),
            },

            "arc": {
                "lookahead_s":  geo.get("lookahead_s",    1.0),
                "steps":        geo.get("arc_steps",      10),
                "px_per_mm":    geo.get("down_px_per_mm", 2.5),
                "points_px":    arc_px,
            },

            "intersection": {
                "detected": intersect_side is not None,
                "side":     intersect_side,
            },
        }

        try:
            state.recorder.write_data(record)
        except Exception as e:
            log.debug("write_data error: %s", e)

    # ── Capture vel/radius from parent's result writer ────────────────────────

    def _write_result(self, state, step, phase, waypoints,
                      vel, radius, goal_status, elapsed) -> None:
        """Intercept drive command before writing result (arc + data logging)."""
        self._last_vel       = vel
        self._last_radius    = radius
        self._last_waypoints = waypoints.copy() if waypoints is not None else None
        super()._write_result(state, step, phase, waypoints,
                              vel, radius, goal_status, elapsed)

    # ── Arc computation (Step 5) ──────────────────────────────────────────────

    def _compute_arc_pixels(self, geo: dict) -> list[tuple[int, int]]:
        """
        Integrate the planned drive arc in rover-frame mm, then project into
        down-camera pixel coordinates.

        Rover frame convention (matches the down-camera view):
          x — lateral  (positive = right in image, increasing pixel-x)
          y — forward  (positive = ahead of rover, decreasing pixel-y)
          θ — heading  (0 = straight ahead; positive = turning left)

        Reference pixel: top-centre of rover polygon (forward edge of rover).
        """
        vel_mm_s   = self._last_vel
        radius_mm  = self._last_radius
        px_per_mm  = float(geo.get("down_px_per_mm", 2.5))
        lookahead  = float(geo.get("lookahead_s",    1.0))
        steps      = int(geo.get("arc_steps",        10))
        poly_raw   = geo.get("rover_polygon_px", _GEO_DEFAULTS["rover_polygon_px"])
        poly       = np.array(poly_raw, dtype=np.float32)

        # Forward edge of rover = minimum y in image (top of polygon)
        ref_px = int(poly[:, 0].mean())
        ref_py = int(poly[:, 1].min())

        dt = lookahead / max(steps, 1)
        x, y, theta = 0.0, 0.0, 0.0   # rover-frame pose in mm / radians

        pixel_pts: list[tuple[int, int]] = []
        for _ in range(steps + 1):
            px = int(ref_px + x * px_per_mm)
            py = int(ref_py - y * px_per_mm)
            pixel_pts.append((px, py))

            if vel_mm_s == 0:
                break                             # stationary — single point

            if radius_mm == 0x8000:               # straight ahead
                y += vel_mm_s * dt
            else:
                ang_rate = vel_mm_s / radius_mm   # rad/s (positive = left)
                dtheta   = ang_rate * dt
                # Euler integration in rover frame
                x     += vel_mm_s * math.sin(theta) * dt
                y     += vel_mm_s * math.cos(theta) * dt
                theta += dtheta

        return pixel_pts

    # ── Intersection check + goal correction (Step 6) ────────────────────────

    def _check_arc_intersection(
        self,
        arc_px: list[tuple[int, int]],
        veg_mask: np.ndarray,
        frame_w: int,
    ) -> str | None:
        """
        Return 'left', 'right', or None depending on whether the planned arc
        corridor overlaps with the vegetation mask and which side it's on.

        The arc is rasterised as a thick line (_ARC_CHECK_THICKNESS px wide)
        onto a blank canvas, then ANDed with the ExG mask.  Only the first
        half of the arc is checked ('immediate' intersection).
        """
        if not arc_px or veg_mask is None:
            return None

        h, w = veg_mask.shape[:2]
        # Only check the near half of the arc (immediate collision)
        near_arc = arc_px[: max(2, len(arc_px) // 2)]
        arc_canvas = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(near_arc, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(arc_canvas, [pts], False, 255, _ARC_CHECK_THICKNESS)

        overlap = cv2.bitwise_and(arc_canvas, veg_mask)
        if not np.any(overlap):
            return None

        # Determine side by comparing left vs right pixel counts
        ys, xs = np.where(overlap > 0)
        cx = frame_w // 2
        return "left" if int(np.sum(xs < cx)) >= int(np.sum(xs >= cx)) else "right"

    def _handle_intersection(
        self,
        side: str | None,
        geo: dict,
        rover_ctrl,
        state,
    ) -> None:
        """Stop rover and issue corrective goal when arc intersects vegetation."""
        operator_active = (
            state.operator_control is not None
            and state.operator_until > time.time()
        )

        if side is not None:
            # ── Intersection detected ─────────────────────────────────────────
            if not operator_active and rover_ctrl and not state.paused.is_set():
                rover_ctrl.stop()
                log.info("Arc intersects vegetation on %s — stopping rover", side)

            if not self._correcting:
                # Save original goal and send corrective one
                self._base_goal  = self._goal
                self._correcting = True
                self._correction_clear_steps = 0
                suffix  = geo.get("correction_goal_suffix",
                                  "steer slightly {direction} to avoid vegetation")
                # opposite direction: veg on left → steer right, and vice versa
                steer   = "right" if side == "left" else "left"
                corrective = (f"{self._base_goal}, "
                              f"{suffix.format(direction=steer)}")
                log.info("Sending corrective goal: '%s'", corrective)
                self.set_goal(corrective)
            else:
                # Still intersecting — reset the clear counter
                self._correction_clear_steps = 0

        else:
            # ── No intersection ───────────────────────────────────────────────
            if self._correcting:
                self._correction_clear_steps += 1
                log.debug("Intersection clear for %d/%d steps",
                          self._correction_clear_steps, _CORRECTION_CLEAR_STEPS)
                if self._correction_clear_steps >= _CORRECTION_CLEAR_STEPS:
                    log.info("Arc clear — reverting to base goal: '%s'", self._base_goal)
                    self.set_goal(self._base_goal)
                    self._correcting             = False
                    self._correction_clear_steps = 0

    # ── Down-camera annotation (grows with each step) ─────────────────────────

    def _annotate_down(
        self,
        frame: np.ndarray,
        geo: dict,
        veg_mask: np.ndarray | None = None,
        left_wall: int | None = None,
        right_wall: int | None = None,
        boxes: list | None = None,
        arc_px: list | None = None,
        intersect_side: str | None = None,
    ) -> np.ndarray:
        """Draw overlays onto the down-camera frame.

        Step 2: HUD label.
        Step 3: Rover footprint polygon (purple).
        Step 4: ExG vegetation mask (green tint) + wall markers.
        Step 5: Planned drive arc (orange polyline).
        Step 6: Red alert banner when arc intersects vegetation.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # ── Step 4: ExG vegetation overlay ────────────────────────────────────
        if veg_mask is not None:
            # Green tint wherever vegetation is detected
            green_layer = np.zeros_like(out)
            green_layer[veg_mask > 0] = (0, 200, 60)
            cv2.addWeighted(green_layer, 0.45, out, 0.55, 0, out)

            # Draw bounding boxes (orange=left side, blue=right side)
            if boxes:
                cx = w // 2
                for b in boxes:
                    color = (0, 140, 255) if (b[0] + b[2]) / 2 < cx else (255, 140, 0)
                    cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), color, 1)

            # Left wall edge (orange vertical line)
            if left_wall is not None:
                cv2.line(out, (left_wall, 0), (left_wall, h), (0, 140, 255), 2)
                cv2.putText(out, f"L={left_wall}px", (left_wall + 4, h - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)

            # Right wall edge (blue vertical line)
            if right_wall is not None:
                cv2.line(out, (right_wall, 0), (right_wall, h), (255, 140, 0), 2)
                cv2.putText(out, f"R={right_wall}px",
                            (max(right_wall - 70, 0), h - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 140, 0), 1)

            # Gap centre dashed line (yellow)
            if left_wall is not None and right_wall is not None:
                gap_cx = (left_wall + right_wall) // 2
                for y in range(0, h, 18):
                    cv2.line(out, (gap_cx, y), (gap_cx, min(y + 10, h)),
                             (0, 220, 255), 1)

        # ── Step 5: Planned drive arc ──────────────────────────────────────────
        if arc_px and len(arc_px) >= 2:
            pts = np.array(arc_px, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], isClosed=False,
                          color=(0, 140, 255), thickness=3)
            # Dot at each step
            for i, pt in enumerate(arc_px):
                r = 6 if i == 0 else 4
                cv2.circle(out, pt, r, (0, 100, 255), -1)
            # Arrowhead at final point
            if len(arc_px) >= 2:
                p1, p2 = arc_px[-2], arc_px[-1]
                cv2.arrowedLine(out, p1, p2, (0, 80, 255), 3, tipLength=0.5)

        # ── Step 3: Rover footprint polygon ───────────────────────────────────
        poly_raw = geo.get("rover_polygon_px", _GEO_DEFAULTS["rover_polygon_px"])
        poly = np.array(poly_raw, dtype=np.int32)

        # Semi-transparent purple fill
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], (180, 0, 180))
        cv2.addWeighted(overlay, 0.20, out, 0.80, 0, out)

        # Solid purple outline + corner dots
        cv2.polylines(out, [poly], isClosed=True, color=(220, 0, 220), thickness=2)
        for pt in poly:
            cv2.circle(out, tuple(pt), 5, (255, 80, 255), -1)

        # Label inside the polygon (centroid)
        cx_poly = int(poly[:, 0].mean())
        cy_poly = int(poly[:, 1].mean())
        cv2.putText(out, "rover", (cx_poly - 22, cy_poly + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 255), 1, cv2.LINE_AA)

        # ── Step 6: Intersection alert banner ─────────────────────────────────
        if intersect_side is not None:
            steer = "RIGHT" if intersect_side == "left" else "LEFT"
            banner = f"⚠ VEGETATION {intersect_side.upper()} — STEER {steer}"
            # Red semi-transparent bar across the top
            cv2.rectangle(out, (0, 0), (w, 70), (0, 0, 180), -1)
            cv2.addWeighted(out, 0.6, frame.copy(), 0.4, 0, out)
            # Re-draw rectangle cleanly after blend
            cv2.rectangle(out, (0, 0), (w, 70), (0, 0, 200), -1)
            cv2.putText(out, banner, (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        elif self._correcting:
            # Intersection cleared but still in correction mode
            cv2.rectangle(out, (0, 0), (w, 70), (0, 140, 0), -1)
            cv2.putText(out, f"Clearing... ({self._correction_clear_steps}/{_CORRECTION_CLEAR_STEPS})",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2, cv2.LINE_AA)

        # ── HUD ───────────────────────────────────────────────────────────────
        cv2.putText(out, "bev_omnivla  down-cam",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f"icr={geo['icr_offset_mm']:.0f}mm  "
                         f"px/mm={geo['down_px_per_mm']}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 200, 220), 1, cv2.LINE_AA)
        return out
