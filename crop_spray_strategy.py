"""
CropSprayStrategy — alternating crop-guard navigation and arm spray sweep.

Phase loop
----------
  GUARD (2 s)   : Left + right wheel cameras active.  Rover moves forward
                  at forward_vel.  Trampling detected → steer / stop.
                  When 2 s elapse with both wheels clear, transition → SPRAY.

  SPRAY         : Rover stops.  Wheel cameras released (free USB bandwidth).
                  Arm camera opens.  Arm sweeps base -120° → +120°; LED on
                  whenever the largest green contour is centred in frame.
                  After sweep completes, arm returns to home, arm camera
                  released, wheel cameras reopen → back to GUARD.

Hardware
--------
  Left  wheel cam  --left-cam  (device or ws:// URL)
  Right wheel cam  --right-cam (device or ws:// URL)
  Arm camera       --arm-cam   (device index, default 0)
  Arm serial       --arm-port  (default /dev/ttyUSB0)
  Arm config JSON  --arm-config (default experimental/arm_scan_config.json)

Usage
-----
  python rover_agent.py --strategy crop_spray \\
      --rover atlas --atlas-port /dev/ttyACM0 \\
      --left-cam /dev/cam-left --right-cam /dev/cam-right \\
      --arm-port /dev/ttyUSB0 --arm-cam 2 \\
      --guard-duration 2.0
"""

import json
import logging
import threading
import time
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
import serial

from navigation_strategy import AgentState, NavigationStrategy
from crop_guard_strategy import _process_wheel_frame
from frame_source import open_frame_source, FrameSource

log = logging.getLogger("rover.crop_spray")

# ── Constants ─────────────────────────────────────────────────────────────────
_DEFAULT_ARM_PORT   = "/dev/ttyUSB0"
_DEFAULT_ARM_CONFIG = Path(__file__).parent / "experimental" / "arm_scan_config.json"
_CORRECT_VEL         = 35
_CORRECT_RADIUS_LEFT  = -800
_CORRECT_RADIUS_RIGHT = +800

# HSV green range for arm camera detection (same as arm_panorama_capture.py)
_GREEN_HSV_LO = np.array([ 35,  60,  60], dtype=np.uint8)
_GREEN_HSV_HI = np.array([ 85, 255, 255], dtype=np.uint8)
_MIN_CONTOUR_AREA = 500


class _Phase(Enum):
    GUARD = auto()   # wheel cams active, rover moving
    SPRAY = auto()   # arm cam active, rover stopped


# ── Arm serial helpers (mirrors arm_panorama_capture.py) ─────────────────────

def _arm_send(ser: serial.Serial, cmd: dict) -> None:
    ser.write((json.dumps(cmd) + "\n").encode())


def _arm_home(ser: serial.Serial, cfg: dict) -> None:
    h = cfg["home"]
    log.info("Arm → scan home: base=%d° shoulder=%d° elbow=%d° eoat=%d°",
             h["base_deg"], h["shoulder_deg"], h["elbow_deg"], h["eoat_deg"])
    _arm_send(ser, {"T": 121, "joint": 2, "angle": h["shoulder_deg"], "spd": 30, "acc": 10})
    time.sleep(2)
    _arm_send(ser, {"T": 121, "joint": 3, "angle": h["elbow_deg"],    "spd": 30, "acc": 10})
    time.sleep(2)
    _arm_send(ser, {"T": 121, "joint": 4, "angle": h["eoat_deg"],     "spd": 30, "acc": 10})
    time.sleep(1)
    _arm_send(ser, {"T": 121, "joint": 1, "angle": h["base_deg"],     "spd": 30, "acc": 10})
    time.sleep(3)


def _arm_feedback(ser: serial.Serial) -> dict:
    _arm_send(ser, {"T": 105})
    deadline = time.time() + 0.5
    buf = b""
    while time.time() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    data = json.loads(line.strip())
                    if isinstance(data, dict) and data.get("T") == 1051:
                        return data
                except Exception:
                    pass
    return {}


def _arm_detect_green(frame: np.ndarray) -> tuple[float, list]:
    hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask   = cv2.inRange(hsv, _GREEN_HSV_LO, _GREEN_HSV_HI)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    ratio  = float(np.count_nonzero(mask)) / mask.size
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(
        [c for c in contours if cv2.contourArea(c) >= _MIN_CONTOUR_AREA],
        key=cv2.contourArea, reverse=True)
    return ratio, contours


# ── Strategy ──────────────────────────────────────────────────────────────────

class CropSprayStrategy(NavigationStrategy):
    """Alternating crop-guard navigation and arm camera spray sweep."""

    requires_goal  = False
    cycle_interval = 0.05   # 20 Hz

    def __init__(self,
                 left_device               = 1,
                 right_device              = 2,
                 forward_vel:        int   = 10,
                 guard_duration_s:   float = 2.0,
                 exg_threshold:      int   = 60,
                 exg_min_area:       int   = 500,
                 exg_density_pct:    float = 8.0,
                 veg_index:          str   = "ngrdi",
                 clahe:              bool  = False,
                 clahe_clip:         float = 2.0,
                 cam_controls:       dict | None = None,
                 arm_port:           str   = _DEFAULT_ARM_PORT,
                 arm_cam_device:     int   = 0,
                 arm_config_path:    str   = str(_DEFAULT_ARM_CONFIG),
                 arm_sweep_spd:      int   = 5):

        self._left_device     = left_device
        self._right_device    = right_device
        self._forward_vel     = forward_vel
        self._guard_duration  = guard_duration_s
        self._exg_threshold   = exg_threshold
        self._exg_min_area    = exg_min_area
        self._exg_density_pct = exg_density_pct
        self._veg_index       = veg_index
        self._clahe           = clahe
        self._clahe_clip      = clahe_clip
        self._cam_controls    = cam_controls or {}
        self._arm_port        = arm_port
        self._arm_cam_device  = arm_cam_device
        self._arm_sweep_spd   = arm_sweep_spd
        self._recorder        = None
        self._arm_ser         = None   # kept open between sweeps, reused

        # Load arm config
        cfg_path = Path(arm_config_path)
        if cfg_path.exists():
            with open(cfg_path) as f:
                self._arm_cfg = json.load(f)
            log.info("Arm config loaded from %s", cfg_path)
        else:
            log.warning("Arm config not found at %s — using defaults", cfg_path)
            self._arm_cfg = {"home": {
                "base_deg": 0, "shoulder_deg": -60,
                "elbow_deg": 150, "eoat_deg": 180}}

        # Wheel camera state
        self._left_src:  FrameSource | None = None
        self._right_src: FrameSource | None = None
        self._trample_left  = False
        self._trample_right = False
        self._warn_left     = False
        self._warn_right    = False
        self._trample_lock  = threading.Lock()
        self._left_vis:  np.ndarray | None = None
        self._right_vis: np.ndarray | None = None
        self._wheel_vis_lock = threading.Lock()
        if not hasattr(self, "_fps_times"):
            self._fps_times = {"left": [], "right": []}

        # Phase state
        self._phase       = _Phase.GUARD
        self._phase_lock  = threading.Lock()
        self._guard_start = time.time()
        self._spray_done  = threading.Event()
        self._spray_done.set()   # not currently spraying

        self._running = True
        threading.Thread(target=self._wheel_thread, daemon=True,
                         name="wheel-cam").start()

        log.info("CropSprayStrategy: left=%s right=%s arm_port=%s arm_cam=%d guard=%.1fs",
                 left_device, right_device, arm_port, arm_cam_device, guard_duration_s)

    # ── NavigationStrategy interface ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "crop_spray"

    def set_recorder(self, recorder) -> None:
        self._recorder = recorder

    def set_goal(self, goal: str) -> None:
        pass

    def on_reset(self) -> None:
        with self._phase_lock:
            self._phase       = _Phase.GUARD
            self._guard_start = time.time()

    def cameras_ready(self) -> tuple[bool, bool, bool]:
        left_ok  = self._left_src  is not None and self._left_src.is_open()
        right_ok = self._right_src is not None and self._right_src.is_open()
        return True, left_ok, right_ok

    def run_query(self, state: AgentState, frame: np.ndarray,
                  captures_dir, rover_ctrl) -> None:
        try:
            with self._phase_lock:
                phase = self._phase

            if phase == _Phase.GUARD:
                self._do_guard(state, rover_ctrl)
            else:
                self._do_spray_wait(state, rover_ctrl)
        except Exception as e:
            log.error("CropSprayStrategy error: %s", e, exc_info=True)
        finally:
            state.query_in_flight.clear()

    # ── GUARD phase ───────────────────────────────────────────────────────────

    def _do_guard(self, state: AgentState, rover_ctrl) -> None:
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
                self._guard_start = time.time()   # reset guard timer on trampling
            elif tramp_l:
                rover_ctrl.drive_raw(_CORRECT_VEL, _CORRECT_RADIUS_LEFT)
                log.warning("Step %d | LEFT trampling — steer RIGHT", step)
                self._guard_start = time.time()
            elif tramp_r:
                rover_ctrl.drive_raw(_CORRECT_VEL, _CORRECT_RADIUS_RIGHT)
                log.warning("Step %d | RIGHT trampling — steer LEFT", step)
                self._guard_start = time.time()
            else:
                rover_ctrl.drive_raw(self._forward_vel, 0x8000)

                # Transition to SPRAY after guard_duration with clear wheels
                elapsed = time.time() - self._guard_start
                if elapsed >= self._guard_duration:
                    log.info("Guard duration elapsed (%.1fs) — starting arm sweep", elapsed)
                    self._transition_to_spray(rover_ctrl)

        with state.result_lock:
            state.latest_result = {
                "navigation_mode": "crop_spray/guard",
                "goal_status":     "trampling" if (tramp_l or tramp_r) else "forward",
                "reasoning":       f"guard  elapsed={time.time()-self._guard_start:.1f}s",
                "waypoints":       [],
                "confidence":      0.3 if (tramp_l or tramp_r) else 1.0,
            }

    # ── SPRAY wait phase (rover stopped, arm sweeping) ────────────────────────

    def _do_spray_wait(self, state: AgentState, rover_ctrl) -> None:
        if rover_ctrl:
            rover_ctrl.drive_raw(0, 0x8000)   # hold stopped

        if self._spray_done.is_set():
            log.info("Arm sweep complete — transitioning back to GUARD")
            self._transition_to_guard()
            return

        with state.result_lock:
            state.latest_result = {
                "navigation_mode": "crop_spray/spray",
                "goal_status":     "spraying",
                "reasoning":       "arm sweep in progress — rover stopped",
                "waypoints":       [],
                "confidence":      1.0,
            }

    # ── Phase transitions ─────────────────────────────────────────────────────

    def _transition_to_spray(self, rover_ctrl) -> None:
        """Release wheel cameras, stop rover, launch arm sweep thread."""
        with self._phase_lock:
            self._phase = _Phase.SPRAY
        self._spray_done.clear()

        if rover_ctrl:
            rover_ctrl.drive_raw(0, 0x8000)

        # Release wheel cameras to free USB bandwidth for arm camera
        if self._left_src:
            self._left_src.release()
            self._left_src = None
        if self._right_src:
            self._right_src.release()
            self._right_src = None

        log.info("Wheel cameras released — launching arm sweep")
        threading.Thread(target=self._arm_sweep, daemon=True,
                         name="arm-sweep").start()

    def _transition_to_guard(self) -> None:
        """Reopen wheel cameras, reset guard timer, resume forward motion."""
        with self._phase_lock:
            self._phase = _Phase.GUARD
        self._guard_start = time.time()

        # Reopen wheel cameras in background (staggered, same as startup)
        threading.Thread(target=self._reopen_wheel_cams, daemon=True,
                         name="wheel-reopen").start()

    def _reopen_wheel_cams(self) -> None:
        # Retry opening left camera until it succeeds — the arm camera
        # may take a moment to fully release USB bandwidth on the Jetson.
        log.info("Waiting for arm camera to release USB bandwidth…")
        for attempt in range(20):   # up to 10s (20 × 0.5s)
            time.sleep(0.5)
            src = open_frame_source(self._left_device, "left", self._cam_controls)
            if src is not None and src.is_open():
                self._left_src = src
                log.info("Left wheel camera ready after %.1fs", (attempt + 1) * 0.5)
                break
            if src:
                src.release()
        else:
            log.warning("Left wheel camera could not reopen after arm sweep")

        time.sleep(5.0)   # stagger
        self._right_src = open_frame_source(self._right_device, "right", self._cam_controls)

    # ── Arm sweep (runs on dedicated thread) ──────────────────────────────────

    def _arm_sweep(self) -> None:
        cap = None
        try:
            # Reuse existing serial port; open only on first sweep or after error
            if self._arm_ser is None or not self._arm_ser.is_open:
                log.info("Arm sweep: opening serial %s", self._arm_port)
                self._arm_ser = serial.Serial(self._arm_port, 115200, timeout=0.5,
                                              dsrdtr=False, rtscts=False)
                self._arm_ser.dtr = False
                self._arm_ser.rts = False
                time.sleep(4)   # only needed on first open — ESP32 boot time
            else:
                log.info("Arm sweep: reusing open serial %s", self._arm_port)
            ser = self._arm_ser

            cap = cv2.VideoCapture(self._arm_cam_device)
            if not cap.isOpened():
                log.error("Arm camera device %d could not be opened", self._arm_cam_device)
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            log.info("Arm camera %d opened", self._arm_cam_device)

            # Read sweep params from config (fall back to constructor args)
            sweep     = self._arm_cfg.get("sweep", {})
            start_deg = sweep.get("start_deg",   -120)
            end_deg   = sweep.get("end_deg",       120)
            sweep_spd = sweep.get("spd",  self._arm_sweep_spd)

            # Home arm and move to sweep start position
            _arm_home(ser, self._arm_cfg)
            log.info("Arm sweep: moving base to %d°…", start_deg)
            _arm_send(ser, {"T": 121, "joint": 1, "angle": start_deg, "spd": 30, "acc": 10})
            travel_s = abs(start_deg) / 30 + 3
            time.sleep(travel_s)

            # Start continuous rotation
            log.info("Arm sweep: starting rotation %d°→%d° spd=%d",
                     start_deg, end_deg, sweep_spd)
            _arm_send(ser, {"T": 123, "m": 0, "axis": 1, "cmd": 1,
                            "spd": sweep_spd})
            sweep_start_time = time.time()
            max_sweep_s = sweep.get("timeout_s", 60)

            led_on = False
            frames_processed = 0

            while True:
                # ── Stop checks run FIRST — before any camera or feedback call ──
                elapsed = time.time() - sweep_start_time
                if elapsed >= max_sweep_s:
                    log.warning("Arm sweep time limit (%.0fs) reached — stopping", max_sweep_s)
                    break

                # Poll feedback (0.5s timeout) and check angle
                feedback = _arm_feedback(ser)
                b_rad    = feedback.get("b", float("nan"))
                if not np.isnan(b_rad) and np.degrees(b_rad) >= end_deg - 5.0:
                    log.info("Arm sweep: base at %.1f° — sweep complete", np.degrees(b_rad))
                    break

                # ── Camera frame (non-blocking best-effort) ───────────────────
                ret, frame = cap.read()
                if not ret:
                    continue

                ratio, contours = _arm_detect_green(frame)

                # LED on when largest green blob is centred
                plant_centred = False
                if contours:
                    fh, fw = frame.shape[:2]
                    x, y, w, h = cv2.boundingRect(contours[0])
                    cx, cy = x + w / 2, y + h / 2
                    if (abs(cx - fw / 2) <= fw * 0.15 and
                            abs(cy - fh / 2) <= fh * 0.15):
                        plant_centred = True

                if plant_centred and not led_on:
                    _arm_send(ser, {"T": 114, "led": 255})
                    led_on = True
                    log.info("Arm LED ON  — plant centred (base=%.1f°)",
                             np.degrees(b_rad) if not np.isnan(b_rad) else float("nan"))
                elif not plant_centred and led_on:
                    _arm_send(ser, {"T": 114, "led": 0})
                    led_on = False

                if self._recorder:
                    self._recorder.record("arm_cam", frame, fps=10)

                frames_processed += 1

        except Exception as e:
            log.error("Arm sweep error: %s", e, exc_info=True)

        finally:
            # Stop rotation + LED off — serial stays open for next sweep
            if self._arm_ser and self._arm_ser.is_open:
                try:
                    _arm_send(self._arm_ser, {"T": 123, "m": 0, "axis": 1, "cmd": 0, "spd": 0})
                    _arm_send(self._arm_ser, {"T": 114, "led": 0})
                    time.sleep(0.5)
                    log.info("Arm sweep: returning base to 0°")
                    _arm_send(self._arm_ser, {"T": 121, "joint": 1, "angle": 0, "spd": 30, "acc": 10})
                    time.sleep(4)
                    _arm_home(self._arm_ser, self._arm_cfg)
                except Exception as e:
                    log.warning("Arm cleanup error: %s — serial will reopen next sweep", e)
                    try:
                        self._arm_ser.close()
                    except Exception:
                        pass
                    self._arm_ser = None   # force reopen on next sweep

            # Release arm camera — wheel cams reopen after this
            if cap:
                cap.release()
                cap = None
            log.info("Arm camera released — signalling GUARD transition")
            self._spray_done.set()

    # ── Wheel camera thread ───────────────────────────────────────────────────

    def _wheel_thread(self) -> None:
        log.info("Wheel cameras: waiting 4s for main camera to initialise…")
        time.sleep(4.0)
        self._left_src  = open_frame_source(self._left_device,  "left",  self._cam_controls)
        time.sleep(5.0)
        self._right_src = open_frame_source(self._right_device, "right", self._cam_controls)

        while self._running:
            with self._phase_lock:
                phase = self._phase

            if phase != _Phase.GUARD or (self._left_src is None and self._right_src is None):
                time.sleep(0.1)
                continue

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

                if lf is not None and self._recorder:
                    self._recorder.record("left_wheel",  lf, fps=10)
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

            time.sleep(0.04)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _blank_vis(label: str) -> np.ndarray:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, label, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        return img

    def _get_left_frame(self) -> np.ndarray | None:
        with self._wheel_vis_lock:
            lv = self._left_vis
        return lv if lv is not None else self._blank_vis("LEFT CAM MISSING")

    def _get_right_frame(self) -> np.ndarray | None:
        with self._wheel_vis_lock:
            rv = self._right_vis
        return rv if rv is not None else self._blank_vis("RIGHT CAM MISSING")
