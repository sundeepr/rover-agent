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
import threading
from pathlib import Path

import cv2
import numpy as np

from cloud_omnivla_strategy import CloudOmniVLAStrategy

log = logging.getLogger("rover.bev_omnivla")

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

        # Annotate down-camera frame (currently: pass-through; overlays added in
        # later steps as rover footprint, ExG mask and arc are implemented)
        down = self._get_down_frame()
        if down is not None:
            annotated = self._annotate_down(down, geo)
            with self._down_lock:
                self._down_annotated = annotated

    # ── Down-camera annotation (grows with each step) ─────────────────────────

    def _annotate_down(self, frame: np.ndarray, geo: dict) -> np.ndarray:
        """Draw overlays onto the down-camera frame.

        Currently just stamps a HUD label so we know the feed is live.
        Rover polygon, ExG mask and arc are added in Steps 3-5.
        """
        out = frame.copy()
        h, w = out.shape[:2]
        cv2.putText(out, "bev_omnivla  down-cam",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f"icr={geo['icr_offset_mm']:.0f}mm  "
                         f"px/mm={geo['down_px_per_mm']}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 200, 220), 1, cv2.LINE_AA)
        return out
