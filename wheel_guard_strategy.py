"""
WheelGuardStrategy — forward-only navigation with crop trampling detection.

No cloud server, no language goal.  The rover moves forward at the slowest
usable speed (deadband-compensated by atlas_controller) and steers away from
any wheel that detects vegetation.

Camera layout
─────────────
  Left  wheel cam (--left-cam)  : bottom half = wheel zone
  Right wheel cam (--right-cam) : bottom half = wheel zone

Each camera can be a local USB device OR a WebSocket stream:
  --left-cam /dev/cam-left          → local V4L2 device
  --left-cam ws://192.168.1.10:5010 → remote JPEG-over-WebSocket stream

Usage
─────
  python rover_agent.py --strategy wheel_guard \\
      --rover atlas --atlas-port /dev/ttyACM0 \\
      --left-cam /dev/cam-left --right-cam /dev/cam-right
"""

import logging
import threading
import time

import cv2
import numpy as np

from navigation_strategy import AgentState, NavigationStrategy
from crop_guard_strategy import _process_wheel_frame
from frame_source import open_frame_source, FrameSource

log = logging.getLogger("rover.wheel_guard")

# Corrective spin radii (mm) — same as crop_guard
_CORRECT_VEL         = 35     # mm/s while steering (deadband ensures movement)
_CORRECT_RADIUS_LEFT  = -800  # steer right
_CORRECT_RADIUS_RIGHT = +800  # steer left


class WheelGuardStrategy(NavigationStrategy):
    """Forward-only crop trampling avoidance using left/right wheel cameras."""

    requires_goal  = False
    cycle_interval = 0.05   # 20 Hz

    def __init__(self,
                 left_device              = 1,
                 right_device             = 2,
                 forward_vel:       int   = 10,
                 exg_threshold:     int   = 60,
                 exg_min_area:      int   = 500,
                 exg_density_pct:   float = 8.0,
                 veg_index:         str   = "ngrdi",
                 clahe:             bool  = False,
                 clahe_clip:        float = 2.0,
                 cam_controls:      dict | None = None):

        self._left_device     = left_device
        self._right_device    = right_device
        self._forward_vel     = forward_vel
        self._exg_threshold   = exg_threshold
        self._exg_min_area    = exg_min_area
        self._exg_density_pct = exg_density_pct
        self._veg_index       = veg_index
        self._clahe           = clahe
        self._clahe_clip      = clahe_clip
        self._cam_controls    = cam_controls or {}
        self._recorder        = None   # set via set_recorder() after construction

        self._trample_left  = False
        self._trample_right = False
        self._warn_left     = False
        self._warn_right    = False
        self._trample_lock  = threading.Lock()

        self._left_vis:  np.ndarray | None = None
        self._right_vis: np.ndarray | None = None
        self._wheel_vis_lock = threading.Lock()

        self._left_src:  FrameSource | None = None
        self._right_src: FrameSource | None = None

        self._running = True
        threading.Thread(target=self._wheel_thread, daemon=True,
                         name="wheel-cam").start()

        log.info("WheelGuardStrategy: left=%s  right=%s  vel=%d mm/s",
                 left_device, right_device, forward_vel)

    # ── NavigationStrategy interface ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "wheel_guard"

    def set_recorder(self, recorder) -> None:
        self._recorder = recorder

    def set_goal(self, goal: str) -> None:
        pass   # no goal needed

    def on_reset(self) -> None:
        pass

    def cameras_ready(self) -> tuple[bool, bool, bool]:
        left_ok  = self._left_src  is not None and self._left_src.is_open()
        right_ok = self._right_src is not None and self._right_src.is_open()
        return True, left_ok, right_ok

    def run_query(self, state: AgentState, frame: np.ndarray,
                  captures_dir, rover_ctrl) -> None:
        try:
            self._do_step(state, rover_ctrl)
        except Exception as e:
            log.error("WheelGuardStrategy error: %s", e, exc_info=True)
        finally:
            state.query_in_flight.clear()

    # ── 20 Hz step ────────────────────────────────────────────────────────────

    def _do_step(self, state: AgentState, rover_ctrl) -> None:
        with self._trample_lock:
            tramp_l = self._trample_left
            tramp_r = self._trample_right

        with state.result_lock:
            step = state.step

        paused          = state.paused.is_set()
        operator_active = (state.operator_control is not None
                           and state.operator_until > time.time())

        if rover_ctrl and not paused and not operator_active:
            if tramp_l and tramp_r:
                rover_ctrl.drive_raw(0, 0x8000)
                log.warning("Step %d | BOTH trampling — STOP", step)
            elif tramp_l:
                rover_ctrl.drive_raw(_CORRECT_VEL, _CORRECT_RADIUS_LEFT)
                log.warning("Step %d | LEFT trampling — steer RIGHT", step)
            elif tramp_r:
                rover_ctrl.drive_raw(_CORRECT_VEL, _CORRECT_RADIUS_RIGHT)
                log.warning("Step %d | RIGHT trampling — steer LEFT", step)
            else:
                rover_ctrl.drive_raw(self._forward_vel, 0x8000)

        # Update result for web UI
        tramp_str = (("L" if tramp_l else "") + ("R" if tramp_r else "")) or "clear"
        with state.result_lock:
            state.latest_result = {
                "navigation_mode": "wheel_guard",
                "goal_status":     "trampling" if (tramp_l or tramp_r) else "forward",
                "reasoning":       f"wheels={tramp_str}  vel={self._forward_vel}mm/s",
                "waypoints":       [],
                "confidence":      0.2 if (tramp_l or tramp_r) else 1.0,
            }

    # ── Wheel camera thread ───────────────────────────────────────────────────

    def _wheel_thread(self) -> None:
        log.info("Wheel cameras: waiting 4s for main camera to initialise…")
        time.sleep(4.0)
        self._left_src  = open_frame_source(self._left_device,  "left",  self._cam_controls)
        time.sleep(5.0)
        self._right_src = open_frame_source(self._right_device, "right", self._cam_controls)

        if not hasattr(self, "_fps_times"):
            self._fps_times = {"left": [], "right": []}

        while self._running:
            try:
                lf = self._left_src.read()  if self._left_src  else None
                rf = self._right_src.read() if self._right_src else None

                both_ready = (self._left_src  is not None and self._left_src.is_open() and
                              self._right_src is not None and self._right_src.is_open())

                now_fps = time.time()
                for side in ("left", "right"):
                    self._fps_times[side].append(now_fps)
                    self._fps_times[side] = [t for t in self._fps_times[side]
                                             if now_fps - t <= 2.0]
                fps_l = len(self._fps_times["left"])  / 2.0
                fps_r = len(self._fps_times["right"]) / 2.0

                # Record raw frames before annotation
                if lf is not None and self._recorder:
                    self._recorder.record("left_wheel", lf, fps=10)
                if rf is not None and self._recorder:
                    self._recorder.record("right_wheel", rf, fps=10)

                if lf is not None:
                    tl, wl, lvis = _process_wheel_frame(
                        lf, "left", self._exg_threshold, self._exg_min_area,
                        self._exg_density_pct, verbose=both_ready, fps=fps_l,
                        veg_index=self._veg_index,
                        clahe=self._clahe, clahe_clip=self._clahe_clip)
                else:
                    tl = wl = False
                    lvis = self._blank_vis("LEFT CAM MISSING")

                if rf is not None:
                    tr, wr, rvis = _process_wheel_frame(
                        rf, "right", self._exg_threshold, self._exg_min_area,
                        self._exg_density_pct, verbose=both_ready, fps=fps_r,
                        veg_index=self._veg_index,
                        clahe=self._clahe, clahe_clip=self._clahe_clip)
                else:
                    tr = wr = False
                    rvis = self._blank_vis("RIGHT CAM MISSING")

                with self._trample_lock:
                    self._trample_left  = tl
                    self._trample_right = tr
                    self._warn_left     = wl
                    self._warn_right    = wr
                with self._wheel_vis_lock:
                    self._left_vis  = lvis
                    self._right_vis = rvis

            except Exception as e:
                log.debug("Wheel thread error: %s", e)

            time.sleep(0.04)   # 25 Hz

    # ── Camera helpers (shared with crop_guard) ───────────────────────────────

    @staticmethod
    def _blank_vis(label: str) -> np.ndarray:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, label, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        return img

    # ── Frames for agent_publisher ────────────────────────────────────────────

    def _get_left_frame(self) -> np.ndarray | None:
        with self._wheel_vis_lock:
            lv = self._left_vis
        return lv if lv is not None else self._blank_vis("LEFT CAM MISSING")

    def _get_right_frame(self) -> np.ndarray | None:
        with self._wheel_vis_lock:
            rv = self._right_vis
        return rv if rv is not None else self._blank_vis("RIGHT CAM MISSING")
