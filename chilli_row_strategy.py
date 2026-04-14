"""
ChilliRowStrategy — crop-row navigation using YOLOv8 chilli plant detection.

Detects chilli plants on the left and right sides of the frame using a
YOLOv8 model, computes the centre of the gap between the plant walls, and
steers the rover to keep that gap centred in the frame.

The rover always moves forward at a constant velocity.  No path-detection
gate — if no plants are detected the rover drives straight.

State machine:
    INITIALIZING → RUNNING
    RUNNING      : detecting plants, steering toward gap centre

Usage:
    python rover_agent.py --strategy chilli_row \\
        --yolo-model chilli.pt \\
        --interval 0.2 \\
        --roomba-port /dev/ttyUSB0

Model:
    Any YOLOv8/YOLOv11 model exported with ultralytics.
    Default: yolov8n.pt (general COCO model; works for prototyping but a
    fine-tuned chilli detection model gives much better results).

    Recommended fine-tuned models:
        https://huggingface.co/datasets/roboflow/chilli-plant-detection
    Download and pass via --yolo-model chilli.pt

Steering logic:
    - Detections are split: centre_x < frame_cx → left wall, else right wall
    - left_wall  = rightmost edge of all left-side detections
    - right_wall = leftmost edge of all right-side detections
    - gap_cx     = (left_wall + right_wall) / 2
    - error      = gap_cx - frame_cx  (px, positive = gap right of centre)
    - radius     = -Kp * vel / error  (kinematic: r = v/ω, ω ∝ error)
    - Clamped to ±MAX_RADIUS_MM
"""

import logging
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from navigation_strategy import AgentState, NavigationStrategy

log = logging.getLogger("rover.chilli_row")

# ── Tunable constants ──────────────────────────────────────────────────────────

# Forward velocity — constant, rover always moves forward.
FWD_VEL_MM_S   = 80      # mm/s

# Proportional gain: how aggressively to correct lateral error.
# error_px * Kp → angular_rate (rad/s).  Start low and tune up.
KP             = 0.003   # rad/s per pixel of error

# Hard clamp on arc radius sent to the rover.
MAX_RADIUS_MM  = 2000    # mm  (tighter than this is too aggressive)
MIN_RADIUS_MM  = 200     # mm  (below this is a spin, not a gentle arc)

# Minimum detection confidence to accept a YOLO box.
CONF_THRESHOLD = 0.35

# If the gap centre is within this many pixels of frame centre, go straight.
STRAIGHT_DEAD_ZONE_PX = 20


# ── Strategy ──────────────────────────────────────────────────────────────────

class ChilliRowStrategy(NavigationStrategy):
    """
    Crop-row navigation via YOLOv8 chilli plant detection.

    Parameters
    ----------
    model_path : str
        Path to a YOLOv8/YOLOv11 .pt weights file, or a Ultralytics model
        tag (e.g. "yolov8n.pt" auto-downloads from ultralytics).
    class_ids : list[int] | None
        YOLO class IDs to treat as chilli plants.  None = all classes.
    fwd_vel : int
        Forward velocity in mm/s (default FWD_VEL_MM_S).
    kp : float
        Proportional steering gain (default KP).
    conf : float
        Minimum detection confidence (default CONF_THRESHOLD).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        class_ids: list | None = None,
        fwd_vel: int = FWD_VEL_MM_S,
        kp: float = KP,
        conf: float = CONF_THRESHOLD,
    ):
        self._model_path = model_path
        self._class_ids  = set(class_ids) if class_ids else None
        self._fwd_vel    = fwd_vel
        self._kp         = kp
        self._conf       = conf

        self._model   = None
        self._loaded  = threading.Event()

        threading.Thread(target=self._load, daemon=True, name="chilli-load").start()

    @property
    def name(self) -> str:
        return "chilli_row"

    def on_reset(self) -> None:
        log.info("ChilliRowStrategy reset")

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            log.error("ultralytics not installed — run: pip install ultralytics")
            return
        log.info("ChilliRow: loading YOLO model '%s'…", self._model_path)
        self._model = YOLO(self._model_path)
        self._loaded.set()
        log.info("ChilliRow: model ready — class_ids=%s conf=%.2f",
                 self._class_ids or "all", self._conf)

    # ── Query ─────────────────────────────────────────────────────────────────

    def run_query(
        self,
        state: AgentState,
        frame: np.ndarray,
        captures_dir: Path,
        rover_ctrl,
    ) -> None:
        try:
            self._do_query(state, frame, rover_ctrl)
        except Exception as e:
            with state.result_lock:
                state.llm_query_start = 0.0
            log.error("ChilliRow error: %s", e, exc_info=True)
        finally:
            state.query_in_flight.clear()

    def _do_query(self, state: AgentState, frame: np.ndarray, rover_ctrl) -> None:
        if not self._loaded.is_set():
            log.info("ChilliRow: model not ready — skipping")
            return

        t0 = time.time()
        with state.result_lock:
            step  = state.step
            phase = state.phase
            state.llm_query_start = t0

        h, w = frame.shape[:2]
        frame_cx = w // 2

        # ── Run YOLO detection ────────────────────────────────────────────────
        boxes = self._detect(frame)

        # ── Split into left / right plant walls ───────────────────────────────
        left_boxes  = [b for b in boxes if _box_cx(b) < frame_cx]
        right_boxes = [b for b in boxes if _box_cx(b) >= frame_cx]

        # Wall edges: rightmost edge of left plants, leftmost edge of right plants
        left_wall  = max((b[2] for b in left_boxes),  default=0)       # x2 of left boxes
        right_wall = min((b[0] for b in right_boxes), default=w)       # x1 of right boxes

        gap_cx    = (left_wall + right_wall) / 2
        error_px  = gap_cx - frame_cx   # positive = gap is right of centre

        # ── Compute steering ──────────────────────────────────────────────────
        vel    = self._fwd_vel
        radius = self._error_to_radius(error_px, vel)

        # ── Drive ─────────────────────────────────────────────────────────────
        operator_active = (state.operator_control is not None and
                           state.operator_until > time.time())
        if rover_ctrl and not state.paused.is_set() and not operator_active:
            rover_ctrl.drive_raw(vel, radius)

        elapsed = time.time() - t0

        # ── Annotate ──────────────────────────────────────────────────────────
        annotated = self._annotate(frame.copy(), boxes, left_wall, right_wall,
                                   gap_cx, frame_cx, error_px, vel, radius)
        with state.llm_lock:
            state.llm_frame = annotated

        # ── Result ────────────────────────────────────────────────────────────
        r_str = "straight" if radius == 0x8000 else f"r={radius}mm"
        reasoning = (f"plants L={len(left_boxes)} R={len(right_boxes)}  "
                     f"gap_cx={gap_cx:.0f}  err={error_px:+.0f}px  "
                     f"vel={vel}mm/s {r_str}")
        log.info("Step %d | %s | %.3fs", step, reasoning, elapsed)

        result = {
            "phase":           phase,
            "navigation_mode": "following",
            "goal_status":     "in_progress",
            "reasoning":       reasoning,
            "waypoints":       [],
            "confidence":      round(min(1.0, (len(left_boxes) + len(right_boxes)) / 4), 2),
        }

        with state.result_lock:
            state.latest_result   = result
            state.llm_query_start = 0.0
            state.llm_response_s  = elapsed

        if state.recorder:
            state.recorder.write_decision({
                "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
                "step":          step,
                "phase":         phase,
                "elapsed_s":     round(elapsed, 3),
                "strategy":      self.name,
                "left_boxes":    len(left_boxes),
                "right_boxes":   len(right_boxes),
                "left_wall_px":  left_wall,
                "right_wall_px": right_wall,
                "gap_cx_px":     round(gap_cx, 1),
                "error_px":      round(error_px, 1),
                "vel_mm_s":      vel,
                "radius_mm":     radius if radius != 0x8000 else None,
            })

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray) -> list:
        """Run YOLO and return accepted boxes as list of [x1,y1,x2,y2]."""
        results = self._model(frame, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            if float(box.conf[0]) < self._conf:
                continue
            cls = int(box.cls[0])
            if self._class_ids is not None and cls not in self._class_ids:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append([x1, y1, x2, y2])
        return boxes

    # ── Steering calculation ───────────────────────────────────────────────────

    def _error_to_radius(self, error_px: float, vel: int) -> int:
        """Convert lateral pixel error to a Roomba radius value.

        Uses the same kinematic formula as the joystick and OmniVLA:
            radius = velocity / angular_rate
            angular_rate = Kp * error_px

        Positive error (gap right of centre) → steer right → negative radius.
        """
        if abs(error_px) < STRAIGHT_DEAD_ZONE_PX:
            return 0x8000   # straight

        ang_rad_s = self._kp * error_px   # rad/s
        # radius = v / ω; negate because Roomba negative=right, positive error=right
        raw_radius = vel / ang_rad_s
        radius = int(math.copysign(
            min(MAX_RADIUS_MM, max(MIN_RADIUS_MM, abs(raw_radius))),
            -raw_radius
        ))
        return radius

    # ── Annotation ────────────────────────────────────────────────────────────

    def _annotate(self, frame, boxes, left_wall, right_wall,
                  gap_cx, frame_cx, error_px, vel, radius) -> np.ndarray:
        h, w = frame.shape[:2]

        # Draw all bounding boxes
        for x1, y1, x2, y2 in boxes:
            cx = (x1 + x2) / 2
            color = (255, 100, 0) if cx < frame_cx else (0, 100, 255)  # orange=left, blue=right
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Left wall line (orange)
        cv2.line(frame, (int(left_wall), 0), (int(left_wall), h), (255, 140, 0), 2)
        # Right wall line (blue)
        cv2.line(frame, (int(right_wall), 0), (int(right_wall), h), (0, 140, 255), 2)

        # Gap centre line (green)
        cv2.line(frame, (int(gap_cx), 0), (int(gap_cx), h), (0, 220, 80), 2)
        # Frame centre line (white dashed reference)
        for y in range(0, h, 20):
            cv2.line(frame, (int(frame_cx), y), (int(frame_cx), min(h, y + 10)),
                     (200, 200, 200), 1)

        # Error arrow: frame centre → gap centre
        if abs(error_px) > STRAIGHT_DEAD_ZONE_PX:
            cv2.arrowedLine(frame,
                            (int(frame_cx), h // 2),
                            (int(gap_cx),   h // 2),
                            (0, 255, 255), 2, tipLength=0.3)

        # HUD text
        r_str = "STRAIGHT" if radius == 0x8000 else f"r={radius}mm"
        cv2.putText(frame, f"err={error_px:+.0f}px  {r_str}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 80), 2)
        cv2.putText(frame, f"vel={vel}mm/s  L={sum(1 for b in boxes if _box_cx(b)<frame_cx)}"
                           f"  R={sum(1 for b in boxes if _box_cx(b)>=frame_cx)}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        return frame


# ── Helpers ────────────────────────────────────────────────────────────────────

def _box_cx(box) -> float:
    return (box[0] + box[2]) / 2
