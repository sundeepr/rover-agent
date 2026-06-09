"""
CropRowStrategy — autonomous crop row following with headland turn and realignment.

Uses the left wheel camera to follow a crop row via optical flow — plant leaves
create chaotic residual motion on top of the smooth background camera translation.
When the row ends the residual flow drops to near zero.  The rover then executes
a headland turn and realigns with the next row using front-camera EXG balance.

Camera layout
─────────────
  Front camera  (--device N)     : forward-facing; used during alignment rocking
                                   to balance green on left vs right side
  Left wheel cam (--left-cam N)  : looks down at the left front wheel;
                                   optical flow residual drives all row detection

Optical flow detection
──────────────────────
  Each frame pair → Farneback dense flow → subtract median (global camera motion)
  → residual magnitude = plant leaf motion.

  Zones on the residual magnitude map (same layout as EXG zones):
    wheel zone    : bottom half — high residual → wheel is over plants → steer right
    look-ahead    : top-right   — high residual → plants approaching → go straight
    full frame    : any residual above threshold → row is present

State machine
─────────────
  FOLLOWING     — optical flow residual tracks row; steer by zone balance
  OVERSHOOT     — drive forward past end-of-row (overshoot_s seconds)
  TURN_1        — 90° spin turn (turn_90_s seconds)
  INTER_ROW     — drive forward across row gap (inter_row_s seconds)
  TURN_2        — 90° spin in same direction (turn_90_s seconds)
  ROCK_FWD      — rock forward; measure front-cam EXG balance
  ROCK_BWD      — rock backward (completes one rock cycle)
  ALIGN_FORWARD — creep forward until left-cam flow detects the new row → FOLLOWING

Usage
─────
  python rover_agent.py --strategy crop_row \\
      --rover atlas --atlas-port /dev/ttyACM0 \\
      --left-cam /dev/cam-left \\
      --flow-threshold 1.5 --turn-90-duration 4.5
"""

import logging
import threading
import time
from enum import Enum, auto

import cv2
import numpy as np

from navigation_strategy import AgentState, NavigationStrategy
from crop_guard_strategy import _veg_index   # still used for front-cam EXG balance
from frame_source import open_frame_source, FrameSource

log = logging.getLogger("rover.crop_row")

# ── Steering radii (mm) — Roomba OI convention ────────────────────────────────
_STEER_RIGHT   = -800   # gentle arc right  (negative = right turn)
_STEER_LEFT    = +800   # gentle arc left   (positive = left turn)
_SPIN_RIGHT    =  1     # tank spin clockwise
_SPIN_LEFT     = -1     # tank spin counter-clockwise



class _Phase(Enum):
    FOLLOWING     = auto()
    OVERSHOOT     = auto()
    TURN_1        = auto()
    INTER_ROW     = auto()
    TURN_2        = auto()
    ROCK_FWD      = auto()
    ROCK_BWD      = auto()
    ALIGN_FORWARD = auto()


class CropRowStrategy(NavigationStrategy):
    """
    Follows a crop row with the left wheel camera via optical flow, turns at
    the headland, and realigns with the next row using front-camera EXG balance.

    Parameters
    ----------
    left_device : int | str
        Left wheel camera device index or path (or ws:// URL).
    forward_vel : int
        Normal forward speed in mm/s.
    flow_threshold : float
        Residual optical flow magnitude (pixels/frame) above which plants are
        detected.  Residual = raw flow minus median (global camera translation).
        Start with 1.0–2.0 and tune from the LEFT CAM FLOW diagnostic log.
    row_end_frames : int
        Consecutive 20 Hz cycles with flow below threshold before row-end.
    overshoot_s : float
        Seconds to continue driving after row-end before first 90° turn.
    turn_90_s : float
        Seconds for a 90° tank spin.  Atlas-1 at DRIVE_SPEED_PCT=60 ≈ 4.25 s.
    inter_row_s : float
        Seconds to drive forward between the two 90° headland turns.
    turn_direction : str
        Direction of both headland turns: "right" or "left".
    rock_fwd_s : float
        Duration of each forward rock burst during front-cam alignment.
    rock_bwd_s : float
        Duration of each backward rock burst during alignment.
    rock_max_cycles : int
        Maximum rock cycles before proceeding to ALIGN_FORWARD regardless.
    balance_threshold : float
        Front-cam EXG mean difference (0–255) below which alignment is declared.
    balance_frames : int
        Consecutive balanced readings required to exit rocking.
    align_fwd_vel : int
        Forward speed (mm/s) while creeping toward the new row.
    veg_index : str
        Vegetation index for front-cam alignment only: "exg", "ngrdi", etc.
    exg_threshold : int
        EXG threshold for front-cam alignment measurement only.
    cam_controls : dict | None
        V4L2 camera control overrides passed to open_frame_source().
    """

    requires_goal  = False
    cycle_interval = 0.05   # 20 Hz

    def __init__(
        self,
        left_device              = 1,
        forward_vel:      int    = 35,
        flow_threshold:   float  = 1.5,
        row_end_frames:   int    = 10,
        overshoot_s:      float  = 1.5,
        turn_90_s:        float  = 4.5,
        inter_row_s:      float  = 2.0,
        turn_direction:   str    = "right",
        rock_fwd_s:       float  = 0.4,
        rock_bwd_s:       float  = 0.4,
        rock_max_cycles:  int    = 20,
        balance_threshold: float = 15.0,
        balance_frames:   int    = 5,
        align_fwd_vel:    int    = 20,
        veg_index:        str    = "exg",
        exg_threshold:    int    = 30,
        cam_controls:     dict | None = None,
    ) -> None:

        self._left_device       = left_device
        self._forward_vel       = forward_vel
        self._flow_threshold    = flow_threshold
        self._row_end_frames    = row_end_frames
        self._overshoot_s       = overshoot_s
        self._turn_90_s         = turn_90_s
        self._inter_row_s       = inter_row_s
        self._spin_radius       = _SPIN_RIGHT if turn_direction == "right" else _SPIN_LEFT
        self._rock_fwd_s        = rock_fwd_s
        self._rock_bwd_s        = rock_bwd_s
        self._rock_max_cycles   = rock_max_cycles
        self._balance_threshold = balance_threshold
        self._balance_frames    = balance_frames
        self._align_fwd_vel     = align_fwd_vel
        self._veg_index         = veg_index
        self._exg_threshold     = exg_threshold
        self._cam_controls      = cam_controls or {}
        self._recorder          = None

        # Left-wheel camera state (updated by _wheel_thread)
        self._trample_left     = False
        self._warn_left        = False
        self._left_exg_present = False
        self._trample_lock     = threading.Lock()
        self._left_vis:  np.ndarray | None = None
        self._wheel_vis_lock = threading.Lock()
        self._left_src: FrameSource | None = None

        # Front-camera EXG balance (updated in run_query from main frame)
        self._front_left_exg  = 0.0
        self._front_right_exg = 0.0
        self._front_lock      = threading.Lock()

        # State machine
        self._phase            = _Phase.FOLLOWING
        self._phase_start      = time.time()
        self._phase_lock       = threading.Lock()
        self._row_end_count    = 0
        self._balance_count    = 0
        self._rock_cycles    = 0
        self._last_imbalance = 0.0
        # Set True once the full-frame EXG mask confirms plants are present.
        # Prevents row-end counting and forward motion before the row is confirmed.
        self._plants_seen    = False

        self._fps_times: list[float] = []

        self._running = True
        threading.Thread(target=self._wheel_thread, daemon=True,
                         name="crop-row-cam").start()

        log.info(
            "CropRowStrategy: left=%s  vel=%d mm/s  turn=%s  turn_90_s=%.1f s",
            left_device, forward_vel, turn_direction, turn_90_s,
        )

    # ── NavigationStrategy interface ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "crop_row"

    def set_recorder(self, recorder) -> None:
        self._recorder = recorder

    def set_goal(self, goal: str) -> None:
        pass

    def on_reset(self) -> None:
        with self._phase_lock:
            self._phase       = _Phase.FOLLOWING
            self._phase_start = time.time()
        self._row_end_count = 0
        self._balance_count = 0
        self._plants_seen   = False

    def cameras_ready(self) -> tuple[bool, bool, bool]:
        left_ok = self._left_src is not None and self._left_src.is_open()
        return True, left_ok, False

    def run_query(self, state: AgentState, frame: np.ndarray,
                  captures_dir, rover_ctrl) -> None:
        try:
            self._measure_front_exg(frame)
            self._do_step(state, frame, rover_ctrl)
        except Exception as e:
            log.error("CropRowStrategy error: %s", e, exc_info=True)
        finally:
            state.query_in_flight.clear()

    # ── Front-camera EXG measurement ──────────────────────────────────────────

    def _measure_front_exg(self, frame: np.ndarray) -> None:
        """Measure mean EXG in the left and right thirds of the front camera."""
        h, w = frame.shape[:2]
        third = w // 3
        l_mean = float(_veg_index(frame[:, :third],       "exg").mean())
        r_mean = float(_veg_index(frame[:, 2 * third:],   "exg").mean())
        with self._front_lock:
            self._front_left_exg  = l_mean
            self._front_right_exg = r_mean

    # ── Main 20 Hz step ───────────────────────────────────────────────────────

    def _do_step(self, state: AgentState, frame: np.ndarray, rover_ctrl) -> None:
        with self._trample_lock:
            tramp_l = self._trample_left
            warn_l  = self._warn_left
            exg_any = self._left_exg_present

        with self._front_lock:
            fl = self._front_left_exg
            fr = self._front_right_exg

        paused          = state.paused.is_set()
        operator_active = (state.operator_control is not None
                           and state.operator_until > time.time())

        # Don't start until the left camera is open and producing frames
        left_cam_ready = self._left_src is not None and self._left_src.is_open()
        if not left_cam_ready:
            if rover_ctrl:
                rover_ctrl.drive_raw(0, 0x8000)
            log.debug("Waiting for left cam…")
            return

        with self._phase_lock:
            phase       = self._phase
            phase_start = self._phase_start

        elapsed = time.time() - phase_start

        if paused or operator_active:
            if rover_ctrl:
                rover_ctrl.drive_raw(0, 0x8000)
            self._update_result(state, phase, fl, fr, tramp_l, warn_l, exg_any)
            return

        # ── Phase dispatch ────────────────────────────────────────────────────

        if phase == _Phase.FOLLOWING:
            self._step_following(rover_ctrl, tramp_l, warn_l, exg_any)

        elif phase == _Phase.OVERSHOOT:
            if rover_ctrl:
                rover_ctrl.drive_raw(self._forward_vel, 0x8000)
            if elapsed >= self._overshoot_s:
                log.info("Overshoot complete → TURN_1")
                self._enter(_Phase.TURN_1)

        elif phase == _Phase.TURN_1:
            if rover_ctrl:
                rover_ctrl.drive_raw(0, self._spin_radius)
            if elapsed >= self._turn_90_s:
                log.info("TURN_1 complete → INTER_ROW")
                if rover_ctrl:
                    rover_ctrl.drive_raw(0, 0x8000)
                self._enter(_Phase.INTER_ROW)

        elif phase == _Phase.INTER_ROW:
            if rover_ctrl:
                rover_ctrl.drive_raw(self._forward_vel, 0x8000)
            if elapsed >= self._inter_row_s:
                log.info("INTER_ROW complete → TURN_2")
                self._enter(_Phase.TURN_2)

        elif phase == _Phase.TURN_2:
            if rover_ctrl:
                rover_ctrl.drive_raw(0, self._spin_radius)
            if elapsed >= self._turn_90_s:
                log.info("TURN_2 complete → ROCK_FWD (alignment)")
                if rover_ctrl:
                    rover_ctrl.drive_raw(0, 0x8000)
                self._rock_cycles  = 0
                self._balance_count = 0
                self._last_imbalance = fl - fr
                self._enter(_Phase.ROCK_FWD)

        elif phase == _Phase.ROCK_FWD:
            # Check balance on every cycle; exit early when converged
            imbalance = fl - fr
            if abs(imbalance) < self._balance_threshold:
                self._balance_count += 1
                if self._balance_count >= self._balance_frames:
                    log.info("Front-cam EXG balanced (L=%.0f R=%.0f) → ALIGN_FORWARD",
                             fl, fr)
                    if rover_ctrl:
                        rover_ctrl.drive_raw(0, 0x8000)
                    self._enter(_Phase.ALIGN_FORWARD)
                    self._update_result(state, _Phase.ALIGN_FORWARD,
                                        fl, fr, tramp_l, warn_l, exg_any)
                    return
            else:
                self._balance_count = 0

            # Proportional steering: arc toward the side with more green
            # left > right → rover too far right → steer left (and vice versa)
            if abs(imbalance) >= self._balance_threshold:
                steer = _STEER_LEFT if imbalance > 0 else _STEER_RIGHT
            else:
                steer = 0x8000  # straight when nearly balanced
            if rover_ctrl:
                rover_ctrl.drive_raw(self._forward_vel, steer)

            if elapsed >= self._rock_fwd_s:
                if rover_ctrl:
                    rover_ctrl.drive_raw(0, 0x8000)
                self._last_imbalance = imbalance
                self._enter(_Phase.ROCK_BWD)

        elif phase == _Phase.ROCK_BWD:
            # Rock straight backward — no steering on reverse
            if rover_ctrl:
                rover_ctrl.drive_raw(-self._forward_vel, 0x8000)
            if elapsed >= self._rock_bwd_s:
                if rover_ctrl:
                    rover_ctrl.drive_raw(0, 0x8000)
                self._rock_cycles += 1
                log.debug("Rock cycle %d | imbalance=%.1f", self._rock_cycles,
                          self._last_imbalance)
                if self._rock_cycles >= self._rock_max_cycles:
                    log.warning("Rock max cycles (%d) reached → ALIGN_FORWARD",
                                self._rock_max_cycles)
                    self._enter(_Phase.ALIGN_FORWARD)
                else:
                    self._enter(_Phase.ROCK_FWD)

        elif phase == _Phase.ALIGN_FORWARD:
            if rover_ctrl:
                rover_ctrl.drive_raw(self._align_fwd_vel, 0x8000)
            if exg_any:
                log.info("Left-cam EXG detected → FOLLOWING")
                self._row_end_count = 0
                self._plants_seen   = True   # already on row — count from first loss
                self._enter(_Phase.FOLLOWING)

        self._update_result(state, phase, fl, fr, tramp_l, warn_l, exg_any)
        self._update_llm_frame(state, frame, phase, fl, fr)

    # ── Row-following step ────────────────────────────────────────────────────

    def _step_following(self, rover_ctrl, tramp_l: bool, warn_l: bool,
                        exg_any: bool) -> None:
        """
        Steer based on left-wheel-cam plant presence, detect row end.

        The full-frame EXG mask drives all decisions:
          exg_any=True  → plants confirmed; drive with zone-based steering correction
          exg_any=False, never seen yet → hold still; wait for plants to appear
          exg_any=False, was following  → row end; drive straight and count frames
        """
        if exg_any:
            # Plants confirmed in frame — drive with zone-based steering
            self._plants_seen   = True
            self._row_end_count = 0
            if rover_ctrl:
                if tramp_l:
                    # Wheel on plants → steer right to move wheel away from row
                    rover_ctrl.drive_raw(self._forward_vel, _STEER_RIGHT)
                elif warn_l:
                    # Plants in look-ahead, not under wheel → aligned → straight
                    rover_ctrl.drive_raw(self._forward_vel, 0x8000)
                else:
                    # Full-frame EXG present but not in zones yet → steer left
                    rover_ctrl.drive_raw(self._forward_vel, _STEER_LEFT)

        elif self._plants_seen:
            # Was following, now no EXG → row end; keep driving straight while counting
            self._row_end_count += 1
            if rover_ctrl:
                rover_ctrl.drive_raw(self._forward_vel, 0x8000)
            if self._row_end_count >= self._row_end_frames:
                log.info("Row end detected (%d no-EXG frames) → OVERSHOOT",
                         self._row_end_count)
                self._row_end_count = 0
                self._enter(_Phase.OVERSHOOT)

        else:
            # Plants not yet seen — hold still until EXG confirms we are at the row
            log.debug("Waiting for EXG confirmation before driving…")
            if rover_ctrl:
                rover_ctrl.drive_raw(0, 0x8000)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _enter(self, phase: _Phase) -> None:
        with self._phase_lock:
            self._phase       = phase
            self._phase_start = time.time()
        log.info("Phase → %s", phase.name)

    def _update_result(self, state: AgentState, phase: _Phase,
                       fl: float, fr: float,
                       tramp_l: bool, warn_l: bool, exg_any: bool) -> None:
        with state.result_lock:
            state.latest_result = {
                "navigation_mode": "crop_row",
                "phase":           phase.name,
                "goal_status":     "following" if phase == _Phase.FOLLOWING else phase.name.lower(),
                "reasoning": (
                    f"phase={phase.name}"
                    f"  tramp={'Y' if tramp_l else 'N'}"
                    f"  warn={'Y' if warn_l else 'N'}"
                    f"  exg={'Y' if exg_any else 'N'}"
                    f"  front L={fl:.0f} R={fr:.0f}"
                ),
                "waypoints":  [],
                "confidence": 1.0,
            }

    def _update_llm_frame(self, state: AgentState, frame: np.ndarray,
                          phase: _Phase, fl: float, fr: float) -> None:
        disp = frame.copy()
        h, w = disp.shape[:2]
        third = w // 3

        # Draw left/right EXG zone boundaries on front frame
        cv2.line(disp, (third,     0), (third,     h), (0, 200, 80),  1)
        cv2.line(disp, (2 * third, 0), (2 * third, h), (0, 200, 80),  1)

        # Phase label
        col = (0, 220, 80) if phase == _Phase.FOLLOWING else (0, 180, 220)
        cv2.putText(disp, f"{phase.name}  L={fl:.0f} R={fr:.0f}",
                    (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        with state.llm_lock:
            state.llm_frame = disp

    # ── Left-wheel camera background thread ───────────────────────────────────

    def _wheel_thread(self) -> None:
        log.info("CropRowStrategy: waiting 4 s before opening left cam…")
        time.sleep(4.0)
        self._left_src = open_frame_source(
            self._left_device, "left", self._cam_controls
        )

        prev_gray: np.ndarray | None = None

        while self._running:
            try:
                lf = self._left_src.read() if self._left_src else None

                now = time.time()
                self._fps_times.append(now)
                self._fps_times = [t for t in self._fps_times if now - t <= 2.0]
                fps = len(self._fps_times) / 2.0

                if lf is not None and self._recorder:
                    self._recorder.record("left_wheel", lf, fps=10)

                if lf is not None and prev_gray is not None:
                    gray = cv2.cvtColor(lf, cv2.COLOR_BGR2GRAY)

                    # Dense optical flow (Farneback)
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                    )

                    # Subtract median to remove global camera translation —
                    # residual = local leaf/plant motion on top of camera motion
                    med_x = float(np.median(flow[..., 0]))
                    med_y = float(np.median(flow[..., 1]))
                    res_x = flow[..., 0] - med_x
                    res_y = flow[..., 1] - med_y
                    res_mag, _ = cv2.cartToPolar(res_x, res_y)

                    h, w = res_mag.shape
                    # Zone magnitudes (same spatial layout as EXG zones)
                    wheel_mean    = float(res_mag[h // 2:, :].mean())       # bottom half
                    lookahead_mean = float(res_mag[:h // 2, w // 2:].mean()) # top-right
                    full_mean     = float(res_mag.mean())

                    thr  = self._flow_threshold
                    tl      = wheel_mean    > thr        # plants under wheel
                    wl      = lookahead_mean > thr * 0.7 # plants approaching
                    flow_any = full_mean    > thr * 0.5  # any plant motion in frame

                    # Diagnostic (1 Hz)
                    if not hasattr(self, "_diag_t"):
                        self._diag_t = 0.0
                    if now - self._diag_t >= 1.0:
                        self._diag_t = now
                        log.info(
                            "LEFT CAM FLOW | full=%.2f  wheel=%.2f  ahead=%.2f  "
                            "thr=%.1f  tl=%s  wl=%s  flow=%s  plants_seen=%s",
                            full_mean, wheel_mean, lookahead_mean, thr,
                            tl, wl, flow_any, self._plants_seen,
                        )

                    # Visualisation — flow magnitude heatmap overlaid on raw frame
                    mag_norm = np.clip(res_mag * (255.0 / max(thr * 4, 0.1)),
                                       0, 255).astype(np.uint8)
                    heat     = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)
                    lvis     = cv2.addWeighted(lf, 0.55, heat, 0.45, 0)

                    # Zone boundary lines
                    cv2.line(lvis, (0, h // 2), (w, h // 2),
                             (0, 0, 200) if tl else (0, 200, 220), 2)
                    cv2.line(lvis, (w // 2, 0), (w // 2, h // 2),
                             (0, 0, 200) if wl else (0, 200, 80), 1)

                    label = (f"flow full={full_mean:.1f} wheel={wheel_mean:.1f} "
                             f"ahead={lookahead_mean:.1f}  thr={thr:.1f}")
                    col   = (0, 220, 80) if flow_any else (60, 60, 200)
                    cv2.putText(lvis, label, (8, h - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
                    border_col = (0, 0, 200) if tl else (0, 220, 80) if flow_any else (80, 80, 80)
                    cv2.rectangle(lvis, (0, 0), (w - 1, h - 1), border_col, 4)

                    prev_gray = gray

                elif lf is not None:
                    # First frame — no previous to compare; just initialise
                    prev_gray = cv2.cvtColor(lf, cv2.COLOR_BGR2GRAY)
                    tl = wl = flow_any = False
                    lvis = lf.copy()
                    cv2.putText(lvis, "FLOW INIT", (8, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                else:
                    tl = wl = flow_any = False
                    lvis = self._blank_vis("LEFT CAM MISSING")

                exg_any = flow_any  # shared name used by state machine

                with self._trample_lock:
                    self._trample_left     = tl
                    self._warn_left        = wl
                    self._left_exg_present = exg_any
                with self._wheel_vis_lock:
                    self._left_vis = lvis

            except Exception as exc:
                log.debug("Wheel thread error: %s", exc)

            time.sleep(0.04)   # 25 Hz

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
