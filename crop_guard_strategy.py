"""
CropGuardStrategy — cloud OmniVLA navigation + dual wheel camera crop-trampling guard.

Camera layout
─────────────
  Front camera  (--device N)        : forward-facing, used for OmniVLA navigation
  Left wheel camera  (--left-cam N) : looks down at left front wheel,
                                      mounted 90° anti-clockwise (CCW)
  Right wheel camera (--right-cam N): looks down at right front wheel,
                                      mounted 90° clockwise (CW)

In the raw image from each wheel camera the bottom half is occupied by the wheel.
The top half shows the ground ahead of the wheel — this is the trampling-risk zone.

Rotation correction for display
────────────────────────────────
  Left  cam: rotate 90° CW  (cv2.ROTATE_90_CLOCKWISE)         to undo 90° CCW mount
  Right cam: rotate 90° CCW (cv2.ROTATE_90_COUNTERCLOCKWISE)  to undo 90° CW  mount

After correction both images show: top = forward, bottom = backward.
The corrected images are stitched side-by-side in the "down" camera web slot.

Trampling detection (ExG — Excess Green Index)
────────────────────────────────────────────────
  ExG = 2·G − R − B
  A pixel is vegetation when ExG > exg_threshold AND it is in the top half
  of the RAW wheel camera image (ahead-of-wheel zone).
  If the vegetation blob area > exg_min_area → trampling risk on that side.

Corrective behaviour
────────────────────
  Left only  → steer right  (pass corrective goal suffix to OmniVLA)
  Right only → steer left
  Both       → stop immediately; update OmniVLA goal
  Neither    → normal OmniVLA navigation

The correction is applied INSTANTLY in the 10 Hz motor loop; OmniVLA also
receives an updated goal text so its next prediction steers around the crop.

Usage
─────
  python rover_agent.py --strategy crop_guard \\
      --cloud-server ws://192.168.1.100:8765 \\
      --left-cam 1 --right-cam 2 \\
      --goal "Follow the crop row" \\
      --atlas-port /dev/ttyACM0
"""

import asyncio
import base64
import io
import json
import logging
import threading
import time
from enum import Enum, auto

import cv2
import numpy as np

from navigation_strategy import AgentState, NavigationStrategy
from omnivla_strategy import (
    METRIC_SPACING, WAYPOINT_IDX,
    _waypoint_to_drive, _annotate,
    load_camera_calibration, bev_to_pixel,
)

log = logging.getLogger("rover.crop_guard")

# ── Constants ─────────────────────────────────────────────────────────────────
_CLOUD_INTERVAL_S    = 1.5    # how often to send a frame to the cloud OmniVLA
_JPEG_QUALITY        = 85
_SEND_W, _SEND_H     = 640, 480
_RECONNECT_BASE      = 3.0
_RECONNECT_MAX       = 30.0

# Corrective steering (in-place spin radius returned to rover_ctrl)
_CORRECT_VEL_MM_S    = 40     # slow creep while correcting
_CORRECT_RADIUS_LEFT  = -800  # steer right  (negative radius = right turn)
_CORRECT_RADIUS_RIGHT = +800  # steer left   (positive radius = left turn)


class _CloudState(Enum):
    CONNECTING   = auto()
    WAITING_GOAL = auto()
    NAVIGATING   = auto()


# ── ExG helpers ───────────────────────────────────────────────────────────────

def _exg_mask(frame_bgr: np.ndarray, threshold: int = 20) -> np.ndarray:
    """Return binary ExG mask: ExG = 2G − R − B > threshold."""
    b = frame_bgr[:, :, 0].astype(np.int16)
    g = frame_bgr[:, :, 1].astype(np.int16)
    r = frame_bgr[:, :, 2].astype(np.int16)
    exg = (2 * g - r - b).clip(0, 255).astype(np.uint8)
    _, mask = cv2.threshold(exg, threshold, 255, cv2.THRESH_BINARY)
    return mask


def _vegetation_area(mask: np.ndarray, min_area: int = 500) -> int:
    """Return total vegetation area if any blob > min_area, else 0."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    total = sum(cv2.contourArea(c) for c in contours
                if cv2.contourArea(c) >= min_area)
    return int(total)


# ── Wheel frame processing ────────────────────────────────────────────────────

def _process_wheel_frame(raw: np.ndarray,
                         side: str,          # "left" or "right"
                         exg_threshold: int,
                         exg_min_area: int,
                         exg_density_pct: float = 8.0) -> tuple[bool, np.ndarray]:
    """
    Analyse one wheel camera frame. No rotation applied — images shown as-is.

    Detection zones (in raw frame):
      Left  cam: top half     (raw[:h//2, :])   — wheel is in top half
      Right cam: bottom half  (raw[h//2:, :])   — wheel is in bottom half

    Trampling is declared only when BOTH conditions are true:
      1. veg_area  > exg_min_area         (a connected blob of vegetation exists)
      2. pct_above > exg_density_pct      (enough of the zone is actually green)
    The density guard prevents scattered pixels on gravelly/bright soil from
    triggering a false alarm when only ~5–10 % of the zone is marginally above
    the threshold.
    """
    h, w = raw.shape[:2]

    # ── Detection zone ────────────────────────────────────────────────────────
    if side == "left":
        wheel_zone = raw[:h // 2, :]    # top half  — wheel sits in top half
    else:
        wheel_zone = raw[h // 2:, :]    # bottom half — wheel sits in bottom half

    # ── ExG stats ─────────────────────────────────────────────────────────────
    b, g, r   = cv2.split(wheel_zone.astype(np.int16))
    exg_raw   = (2 * g - r - b)
    exg_mean  = float(exg_raw.mean())
    exg_max   = int(exg_raw.max())
    exg_p90   = float(np.percentile(exg_raw, 90))
    pct_above = float((exg_raw > exg_threshold).mean() * 100)

    veg_mask  = _exg_mask(wheel_zone, exg_threshold)
    veg_area  = _vegetation_area(veg_mask, exg_min_area)

    # Both conditions must pass: connected blob AND sufficient green density
    trampling = veg_area > 0 and pct_above >= exg_density_pct

    log.info("%s ExG | mean=%.1f  max=%d  p90=%.1f  above_thresh(>%d)=%.1f%%  "
             "veg_area=%d  density_ok=%s  trampling=%s",
             side.upper(), exg_mean, exg_max, exg_p90,
             exg_threshold, pct_above, veg_area,
             f"{pct_above:.1f}>={exg_density_pct}", trampling)

    # ── Display: raw frame with overlays ─────────────────────────────────────
    display = raw.copy()

    # ── ExG heatmap across the FULL frame (faint blue tint outside zone) ─────
    # ExG values clamped to [0,255] and used as an alpha channel so even
    # low-ExG areas are visible.  Inside the wheel zone the mask is bright
    # lime-green so it's easy to compare zone vs non-zone.
    b_f = raw[:, :, 0].astype(np.int16)
    g_f = raw[:, :, 1].astype(np.int16)
    r_f = raw[:, :, 2].astype(np.int16)
    exg_full = np.clip(2 * g_f - r_f - b_f, 0, 255).astype(np.uint8)

    # Faint cyan tint outside the wheel zone — shows ExG signal everywhere
    outside_color = np.zeros_like(display)
    outside_color[:, :] = (180, 180, 0)   # cyan-ish (BGR)
    alpha_outside = (exg_full.astype(np.float32) / 255.0 * 0.35)[..., np.newaxis]
    if side == "left":
        # outside zone = bottom half (plants, not the wheel)
        display[h // 2:, :] = np.clip(
            display[h // 2:, :] * (1 - alpha_outside[h // 2:]) +
            outside_color[h // 2:, :] * alpha_outside[h // 2:], 0, 255
        ).astype(np.uint8)
    else:
        # outside zone = top half (plants, not the wheel)
        display[:h // 2, :] = np.clip(
            display[:h // 2, :] * (1 - alpha_outside[:h // 2]) +
            outside_color[:h // 2, :] * alpha_outside[:h // 2], 0, 255
        ).astype(np.uint8)

    # Bright lime-green mask inside the wheel zone — pixels above threshold
    veg_mask_zone = _exg_mask(wheel_zone, exg_threshold)
    # Semi-transparent green where ExG is above threshold (60% blend)
    zone_overlay = wheel_zone.copy()
    zone_overlay[veg_mask_zone > 0] = (0, 255, 60)   # bright lime green
    blended_zone = cv2.addWeighted(wheel_zone, 0.4, zone_overlay, 0.6, 0)
    if side == "left":
        display[:h // 2, :] = blended_zone   # wheel zone = top half
    else:
        display[h // 2:, :] = blended_zone   # wheel zone = bottom half

    # Cyan zone-boundary line (2 px, easy to see)
    mid_y = h // 2
    cv2.line(display, (0, mid_y), (w, mid_y), (0, 220, 220), 2)
    # Label arrow points INTO the wheel zone
    zone_y = mid_y - 5 if side == "left" else mid_y + 16
    cv2.putText(display, "WHEEL ZONE ^" if side == "left" else "WHEEL ZONE v",
                (8, zone_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 220), 1)

    # ExG stats at bottom
    density_str = f"dens={pct_above:.1f}%>={exg_density_pct:.0f}%"
    cv2.putText(display,
                f"ExG mean={exg_mean:.0f} p90={exg_p90:.0f} area={veg_area} "
                f"thr={exg_threshold} {density_str}",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (220, 220, 0), 1)

    # Label + border
    label = f"{'!! TRAMPLE' if trampling else 'CLEAR'}  area={veg_area}"
    cv2.putText(display, f"{side.upper()} WHEEL  {label}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 220) if trampling else (0, 220, 80), 2)
    border_col = (0, 0, 220) if trampling else (0, 200, 50)
    cv2.rectangle(display, (0, 0), (w - 1, h - 1), border_col, 6)

    return trampling, display


# ── Strategy ──────────────────────────────────────────────────────────────────

class CropGuardStrategy(NavigationStrategy):
    """
    Cloud OmniVLA navigation with real-time crop trampling detection
    using left and right front-wheel cameras.
    """

    requires_goal   = True
    cycle_interval  = 0.10      # 10 Hz — fast wheel check + watchdog keepalive

    def __init__(self,
                 server_url:         str,
                 goal:               str  = "",
                 left_device:        int  = 1,
                 right_device:       int  = 2,
                 max_lin_mm_s:       int  = 150,
                 icr_offset_m:       float = 0.480,
                 exg_threshold:      int  = 60,
                 exg_min_area:       int  = 500,
                 exg_density_pct:    float = 8.0,
                 cloud_interval_s:   float = _CLOUD_INTERVAL_S,
                 camera_calibration: dict | None = None):

        self._server_url        = server_url
        self._goal              = goal
        self._left_device       = left_device   # stored for lazy open in _wheel_thread
        self._right_device      = right_device
        self._max_lin_mm_s      = max_lin_mm_s
        self._icr_offset_m      = icr_offset_m
        self._exg_threshold     = exg_threshold
        self._exg_min_area      = exg_min_area
        self._exg_density_pct   = exg_density_pct
        self._cloud_interval_s  = cloud_interval_s
        self._calib             = camera_calibration or {}

        # ── Cloud navigation state ────────────────────────────────────────────
        self._cloud_state      = _CloudState.CONNECTING
        self._cloud_lock       = threading.Lock()
        self._nav_vel          = 0
        self._nav_radius       = 0x8000          # "straight" sentinel
        self._nav_waypoints    = None            # latest [8×4] array from cloud
        self._last_cloud_query = 0.0
        self._cloud_in_flight  = threading.Event()

        # WebSocket asyncio machinery (same pattern as CloudOmniVLAStrategy)
        self._loop   = asyncio.new_event_loop()
        self._ws     = None
        self._resp_event   = threading.Event()
        self._pending_resp: dict | None = None
        threading.Thread(target=self._start_ws_loop, daemon=True,
                         name="crop-guard-ws").start()

        # ── Wheel camera state ────────────────────────────────────────────────
        self._trample_left   = False
        self._trample_right  = False
        self._trample_lock   = threading.Lock()
        self._left_vis: np.ndarray | None   = None
        self._right_vis: np.ndarray | None  = None
        self._wheel_vis_lock = threading.Lock()

        # Wheel cameras are opened lazily inside _wheel_thread after a short
        # startup delay.  Opening them immediately in __init__ races with the
        # main camera (device 0) initialisation — on Jetson all USB cameras
        # share one controller, and grabbing frames on 2+4 before 0 is ready
        # starves device 0's warmup reads.
        self._left_cap  = None
        self._right_cap = None

        # Background wheel-capture thread (10 Hz independent of OmniVLA timing)
        self._running = True
        threading.Thread(target=self._wheel_thread, daemon=True,
                         name="wheel-cam").start()

        log.info("CropGuardStrategy: left_cam=%s  right_cam=%s  server=%s",
                 left_device, right_device, server_url)

        if goal:
            with self._cloud_lock:
                self._cloud_state = _CloudState.WAITING_GOAL

    # ── NavigationStrategy interface ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "crop_guard"

    def set_goal(self, goal: str) -> None:
        self._goal = goal
        asyncio.run_coroutine_threadsafe(
            self._ws_send_goal(goal), self._loop
        )
        with self._cloud_lock:
            if self._cloud_state != _CloudState.CONNECTING:
                self._cloud_state = _CloudState.NAVIGATING
        log.info("Goal set: '%s'", goal)

    def on_reset(self) -> None:
        with self._cloud_lock:
            if self._cloud_state == _CloudState.NAVIGATING:
                self._cloud_state = _CloudState.WAITING_GOAL

    def run_query(self,
                  state:        AgentState,
                  frame:        np.ndarray,
                  captures_dir,
                  rover_ctrl) -> None:
        """
        Called at 10 Hz by the agent loop.

        1. Read latest wheel trampling state (updated by background thread).
        2. Decide motor command (trampling override or latest cloud nav).
        3. Send motor command (keepalive + navigation).
        4. Fire a cloud OmniVLA query if interval elapsed and not in flight.
        5. Update web displays.
        6. Clear query_in_flight immediately (we never block here).
        """
        try:
            self._do_step(state, frame, rover_ctrl)
        except Exception as e:
            log.error("CropGuardStrategy error: %s", e, exc_info=True)
        finally:
            state.query_in_flight.clear()

    # ── 10 Hz step ────────────────────────────────────────────────────────────

    def _do_step(self, state: AgentState, frame: np.ndarray, rover_ctrl) -> None:
        t0 = time.time()
        with state.result_lock:
            step  = state.step
            phase = state.phase

        # ── 1. Get trampling state ────────────────────────────────────────────
        with self._trample_lock:
            tramp_l = self._trample_left
            tramp_r = self._trample_right

        # ── 2. Choose motor command ───────────────────────────────────────────
        with self._cloud_lock:
            nav_vel    = self._nav_vel
            nav_radius = self._nav_radius
            cloud_st   = self._cloud_state

        # ── Periodic diagnostic (once per second) ────────────────────────────
        if not hasattr(self, "_diag_t") or t0 - self._diag_t >= 1.0:
            self._diag_t = t0
            r_str = "straight" if nav_radius == 0x8000 else f"r={nav_radius}"
            log.info(
                "DIAG | cloud=%s  goal=%r  vel=%d %s  "
                "tramp=L%s/R%s  in_flight=%s  ws=%s",
                cloud_st.name, self._goal[:30] if self._goal else "",
                nav_vel, r_str,
                "Y" if tramp_l else "N", "Y" if tramp_r else "N",
                "Y" if self._cloud_in_flight.is_set() else "N",
                "connected" if self._ws else "DISCONNECTED",
            )

        operator_active = (state.operator_control is not None
                           and state.operator_until > time.time())
        paused          = state.paused.is_set()

        if rover_ctrl and not paused and not operator_active:
            if tramp_l and tramp_r:
                # Both wheels → immediate stop
                rover_ctrl.drive_raw(0, 0x8000)
                log.warning("Step %d | BOTH wheels trampling — STOP", step)
                goal_override = "stop, vegetation detected under both wheels"
            elif tramp_l:
                # Left wheel only → steer right
                rover_ctrl.drive_raw(_CORRECT_VEL_MM_S, _CORRECT_RADIUS_LEFT)
                log.warning("Step %d | LEFT wheel trampling — steer RIGHT", step)
                goal_override = "steer right, vegetation under left wheel"
            elif tramp_r:
                # Right wheel only → steer left
                rover_ctrl.drive_raw(_CORRECT_VEL_MM_S, _CORRECT_RADIUS_RIGHT)
                log.warning("Step %d | RIGHT wheel trampling — steer LEFT", step)
                goal_override = "steer left, vegetation under right wheel"
            else:
                # No trampling — apply latest cloud navigation command
                if cloud_st == _CloudState.NAVIGATING and nav_vel > 0:
                    rover_ctrl.drive_raw(nav_vel, nav_radius)
                goal_override = None
        else:
            goal_override = None

        # If goal override needed, update OmniVLA goal for next cloud query
        if goal_override and goal_override != getattr(self, "_last_override", ""):
            self._last_override = goal_override
            combined = f"{self._goal}. {goal_override}." if self._goal else goal_override
            asyncio.run_coroutine_threadsafe(
                self._ws_send_goal(combined), self._loop
            )

        # ── 3. Fire cloud OmniVLA query if due ───────────────────────────────
        now = time.time()
        since_last = now - self._last_cloud_query
        can_query  = (cloud_st != _CloudState.CONNECTING
                      and self._goal
                      and not self._cloud_in_flight.is_set()
                      and since_last >= self._cloud_interval_s)
        if can_query:
            log.info("Cloud query firing (since_last=%.1fs  state=%s)",
                     since_last, cloud_st.name)
            self._last_cloud_query = now
            self._cloud_in_flight.set()
            threading.Thread(target=self._cloud_query,
                             args=(state, frame.copy()),
                             daemon=True, name="cloud-query").start()
        elif not can_query and since_last >= self._cloud_interval_s * 2:
            # Haven't queried for 2× the interval — log why
            log.warning(
                "Cloud query BLOCKED | state=%s  goal=%r  in_flight=%s  since=%.1fs",
                cloud_st.name, bool(self._goal),
                self._cloud_in_flight.is_set(), since_last)

        # ── 4. Update displays ────────────────────────────────────────────────
        # Front camera annotated frame
        with self._cloud_lock:
            wps = self._nav_waypoints
        if wps is not None:
            display = _annotate(frame, wps, nav_vel, nav_radius,
                                self._goal, calib=self._calib)
        else:
            display = frame.copy()

        # Trample warning overlay on front camera
        if tramp_l or tramp_r:
            sides   = ("LEFT" if tramp_l else "") + (" + " if tramp_l and tramp_r else "") + ("RIGHT" if tramp_r else "")
            cv2.rectangle(display, (0, 0),
                          (display.shape[1]-1, display.shape[0]-1),
                          (0, 0, 220), 8)
            cv2.putText(display, f"⚠ CROP TRAMPLE: {sides}",
                        (10, display.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        with state.llm_lock:
            state.llm_frame = display

        # Wheel camera stitched view is served via _get_down_frame() below.
        # (agent_publisher reads that method — no state.down_frame needed)

        # ── 5. Write result dict ──────────────────────────────────────────────
        h, w   = frame.shape[:2]
        ui_wps = []
        if wps is not None:
            for i, wp_i in enumerate(wps[:3]):
                x_m = float(wp_i[1]) * METRIC_SPACING
                y_m = float(wp_i[0]) * METRIC_SPACING
                pt  = bev_to_pixel(x_m, y_m, self._calib, w, h)
                if pt:
                    ui_wps.append({
                        "rank":        i + 1,
                        "x":           pt[0],
                        "y":           pt[1],
                        "description": f"wp[{i}] +{y_m:.2f}m",
                        "probability": round(1.0 - i * 0.1, 1),
                    })

        r_str  = "straight" if nav_radius == 0x8000 else f"r={nav_radius}mm"
        tramp_str = (("⚠L" if tramp_l else "") + ("⚠R" if tramp_r else "") or "OK")
        with state.result_lock:
            state.latest_result = {
                "phase":           phase,
                "navigation_mode": "crop_guard",
                "goal_status":     "trampling" if (tramp_l or tramp_r) else "navigating",
                "reasoning":       (
                    f"wheels={tramp_str}  vel={nav_vel}mm/s {r_str}"
                    f" | goal='{self._goal}'"
                ),
                "waypoints":    ui_wps,
                "confidence":   0.3 if (tramp_l or tramp_r) else 1.0,
            }
            state.llm_query_start = 0.0

    # ── Cloud OmniVLA query (runs in its own thread, ~1 Hz) ───────────────────

    def _cloud_query(self, state: AgentState, frame: np.ndarray) -> None:
        """Send one frame to cloud OmniVLA server; update nav state on response."""
        try:
            import math as _math

            h_lbox  = max(_SEND_W * frame.shape[0] // frame.shape[1], 1)
            send_f  = cv2.resize(frame, (_SEND_W, _SEND_H))
            _, buf  = cv2.imencode(".jpg", send_f,
                                   [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
            f64     = base64.b64encode(buf.tobytes()).decode()

            self._resp_event.clear()
            self._pending_resp = None

            asyncio.run_coroutine_threadsafe(
                self._ws_send_infer(f64, self._goal), self._loop
            ).result(timeout=5.0)

            if not self._resp_event.wait(timeout=30.0):
                log.warning("Cloud OmniVLA timed out")
                return

            resp = self._pending_resp
            if not resp:
                log.warning("Cloud query: empty response")
                return
            if resp.get("type") != "waypoints":
                log.warning("Cloud query: unexpected response type=%s  full=%s",
                            resp.get("type"), resp)
                return

            wps = np.array(resp["waypoints"])   # [8, 4]
            vel, radius = _waypoint_to_drive(wps, self._max_lin_mm_s,
                                              self._icr_offset_m)

            with self._cloud_lock:
                self._nav_vel       = vel
                self._nav_radius    = radius
                self._nav_waypoints = wps
                self._cloud_state   = _CloudState.NAVIGATING

            r_str = "straight" if radius == 0x8000 else f"r={radius}mm"
            log.info("Cloud nav: vel=%d  %s  (wps[0]=%s)",
                     vel, r_str, wps[0].tolist() if len(wps) else "?")

        except Exception as e:
            log.warning("Cloud query error: %s", e)
        finally:
            self._cloud_in_flight.clear()

    # ── Background wheel-camera thread (10 Hz) ────────────────────────────────

    def _wheel_thread(self) -> None:
        # Wait for the main front camera (device 0) to finish its warmup
        # before opening wheel cameras.  Without this delay, all three USB
        # cameras try to initialise simultaneously on the same controller and
        # the main camera's warmup reads starve.
        _STARTUP_DELAY_S = 4.0
        log.info("Wheel cameras: waiting %.0fs for main camera to initialise…",
                 _STARTUP_DELAY_S)
        time.sleep(_STARTUP_DELAY_S)
        self._left_cap  = self._open_cam(self._left_device,  "left")
        time.sleep(5.0)   # stagger: both cams on same USB hub — give left time to fully stabilise
        self._right_cap = self._open_cam(self._right_device, "right")

        while self._running:
            try:
                lf = self._grab(self._left_cap)
                rf = self._grab(self._right_cap)

                if lf is not None:
                    tl, lvis = _process_wheel_frame(
                        lf, "left", self._exg_threshold, self._exg_min_area,
                        self._exg_density_pct)
                else:
                    tl   = False
                    lvis = self._blank_vis("LEFT CAM MISSING")

                if rf is not None:
                    tr, rvis = _process_wheel_frame(
                        rf, "right", self._exg_threshold, self._exg_min_area,
                        self._exg_density_pct)
                else:
                    tr   = False
                    rvis = self._blank_vis("RIGHT CAM MISSING")

                with self._trample_lock:
                    self._trample_left  = tl
                    self._trample_right = tr
                with self._wheel_vis_lock:
                    self._left_vis  = lvis
                    self._right_vis = rvis

            except Exception as e:
                log.debug("Wheel thread error: %s", e)

            time.sleep(0.10)   # 10 Hz

    # ── Camera helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _open_cam(device, label: str):
        """Open a wheel camera using an explicit /dev/videoN path.

        `device` can be an int (2 → /dev/video2) or a path string
        (/dev/cam-left).  Using a path bypasses OpenCV's index remapping
        which can open the wrong node when multiple cameras share a VID:PID.

        HD USB cameras on this rover don't support MJPEG; we let the driver
        choose the format (YUYV) and just cap FPS at 10 to save USB bandwidth.
        """
        path = device if isinstance(device, str) else f"/dev/video{device}"
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(path)        # fallback to default backend
        if not cap.isOpened():
            log.warning("%s wheel camera NOT FOUND at %s", label, path)
            return None

        # Try MJPEG first; if the camera doesn't support it the driver will
        # silently stay on YUYV — we check below and warn.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 10)

        # Reduce gain and backlight_compensation to prevent sun-glint washing
        # out the image outdoors.  These cameras have no exposure control —
        # gain and backlight_compensation are the effective levers.
        # Defaults: gain=136, backlight_compensation=136, brightness=128.
        # Tune live with:
        #   v4l2-ctl --device <path> --set-ctrl=gain=64
        #   v4l2-ctl --device <path> --set-ctrl=backlight_compensation=0
        cap.set(cv2.CAP_PROP_GAIN,       64)   # default 136 — lower = darker
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 90)   # default 128 — slightly darker
        # backlight_compensation has no OpenCV constant — set via v4l2-ctl after open
        import subprocess, shlex
        try:
            subprocess.run(
                shlex.split(f"v4l2-ctl --device {path} --set-ctrl=backlight_compensation=0"),
                check=False, capture_output=True)
        except Exception:
            pass
        log.info("%s wheel camera %s: gain=64 brightness=90 backlight_comp=0", label, path)

        actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))
        actual_fps  = cap.get(cv2.CAP_PROP_FPS)

        if fourcc_str != "MJPG":
            log.info("%s wheel camera %s: MJPEG not supported, using %s",
                     label, path, fourcc_str)

        # Warmup reads — driver needs a few frames to stabilise
        for _ in range(30):
            ret, _ = cap.read()
            if ret:
                log.info("%s wheel camera ready at %s  (%dx%d)  fourcc=%s  fps=%.0f",
                         label, path,
                         int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                         fourcc_str, actual_fps)
                return cap
            time.sleep(0.05)

        log.warning("%s wheel camera %s opened but delivers no frames", label, path)
        cap.release()
        return None

    @staticmethod
    def _grab(cap) -> np.ndarray | None:
        if cap is None or not cap.isOpened():
            return None
        ret, frame = cap.read()
        return frame if ret else None

    @staticmethod
    def _blank_vis(label: str) -> np.ndarray:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, label, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        return img

    # ── WebSocket (same pattern as CloudOmniVLAStrategy) ─────────────────────

    def _start_ws_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._ws_connect_loop())

    async def _ws_connect_loop(self) -> None:
        import websockets
        delay = _RECONNECT_BASE
        while True:
            try:
                async with websockets.connect(self._server_url,
                                              ping_interval=20,
                                              ping_timeout=10) as ws:
                    self._ws = ws
                    with self._cloud_lock:
                        if self._cloud_state == _CloudState.CONNECTING:
                            self._cloud_state = _CloudState.WAITING_GOAL
                    log.info("Connected to cloud server: %s", self._server_url)
                    delay = _RECONNECT_BASE
                    if self._goal:
                        await self._ws_send_goal(self._goal)
                    async for raw_msg in ws:
                        self._pending_resp = json.loads(raw_msg)
                        self._resp_event.set()
            except Exception as e:
                self._ws = None
                with self._cloud_lock:
                    self._cloud_state = _CloudState.CONNECTING
                log.warning("WS disconnected (%s) — retry in %.0fs", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, _RECONNECT_MAX)

    async def _ws_send_goal(self, goal: str) -> None:
        if self._ws:
            await self._ws.send(json.dumps({"type": "set_goal", "goal": goal}))

    async def _ws_send_infer(self, frame_b64: str, goal: str) -> None:
        if self._ws:
            await self._ws.send(json.dumps({
                "type":      "infer",
                "frame_b64": frame_b64,
                "goal":      goal,
            }))

    # ── Down-camera feed for agent_publisher ──────────────────────────────────
    # agent_publisher checks hasattr(strategy, "_get_down_frame") and calls it
    # to get the frame to push to /video/down every publish cycle.

    def _get_down_frame(self) -> np.ndarray | None:
        """
        Return the stitched left+right wheel camera view for the browser.

        Layout: [LEFT wheel | RIGHT wheel] side by side.
        Both images are rotation-corrected; ExG vegetation is highlighted.
        Red border = trampling detected, green border = clear.
        """
        with self._wheel_vis_lock:
            lv = self._left_vis
            rv = self._right_vis

        if lv is None and rv is None:
            return None

        # Fallback if one cam is missing
        if lv is None:
            lv = self._blank_vis("LEFT CAM MISSING")
        if rv is None:
            rv = self._blank_vis("RIGHT CAM MISSING")

        # Resize both to the same height (480px), keeping aspect ratio
        target_h = 480

        def _resize_h(img: np.ndarray) -> np.ndarray:
            h, w = img.shape[:2]
            new_w = max(1, int(w * target_h / h))
            return cv2.resize(img, (new_w, target_h))

        lv = _resize_h(lv)
        rv = _resize_h(rv)

        # Divider line between the two cams
        divider = np.zeros((target_h, 4, 3), dtype=np.uint8)
        divider[:] = (60, 60, 60)

        stitched = np.concatenate([lv, divider, rv], axis=1)
        return stitched

    def _get_left_frame(self) -> np.ndarray | None:
        """Return the annotated left wheel camera view for the browser."""
        with self._wheel_vis_lock:
            lv = self._left_vis
        if lv is None:
            return self._blank_vis("LEFT CAM MISSING")
        return lv

    def _get_right_frame(self) -> np.ndarray | None:
        """Return the annotated right wheel camera view for the browser."""
        with self._wheel_vis_lock:
            rv = self._right_vis
        if rv is None:
            return self._blank_vis("RIGHT CAM MISSING")
        return rv
