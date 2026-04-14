"""
CropRowStrategy — crop-row navigation using YOLOv8 plant detection.

The user provides a crop type (e.g. "chilli", "tomato", "corn") and a matching
YOLOv8 model.  The strategy detects plants on either side of the rover, computes
the centre of the gap between the plant walls, and steers to keep that gap
centred in the frame.

The rover always moves forward at a constant velocity.  No path-detection
gate — if no plants are detected the rover drives straight.

Usage:
    python rover_agent.py --strategy crop_row \\
        --crop-type chilli \\
        --yolo-model chilli.pt \\
        --interval 0.2 \\
        --roomba-port /dev/ttyUSB0

Model:
    Any YOLOv8/YOLOv11 model exported with ultralytics.
    Default: yolov8n.pt (general COCO model — useful for prototyping,
    but a crop-specific fine-tuned model gives much better results).

    Fine-tuned models can be found on Roboflow Universe:
        https://universe.roboflow.com  (search your crop type)
    Download as "YOLOv8 PyTorch" and pass via --yolo-model <file>.pt

Steering logic:
    - Detections split: box centre_x < frame_cx → left wall, else right wall
    - left_wall  = rightmost x2 of left-side detections
    - right_wall = leftmost  x1 of right-side detections
    - gap_cx     = (left_wall + right_wall) / 2
    - error_px   = gap_cx - frame_cx  (positive = gap right of centre)
    - radius     = vel / (Kp * error_px)  — same kinematic formula as joystick
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

log = logging.getLogger("rover.crop_row")

# ── Tunable constants ──────────────────────────────────────────────────────────

FWD_VEL_MM_S        = 80      # mm/s — constant forward speed
KP                  = 0.003   # rad/s per pixel of lateral error
MAX_RADIUS_MM       = 2000    # mm — gentlest arc limit
MIN_RADIUS_MM       = 200     # mm — tightest arc limit (below = spin)
CONF_THRESHOLD      = 0.35    # minimum YOLO detection confidence
STRAIGHT_DEAD_ZONE_PX = 20   # pixels — gap errors smaller than this → straight


# ── Strategy ──────────────────────────────────────────────────────────────────

class CropRowStrategy(NavigationStrategy):
    """
    Crop-row navigation via YOLOv8 plant detection.

    Parameters
    ----------
    crop_type : str
        Human-readable crop name used in logging and the web UI HUD
        (e.g. "chilli", "tomato", "corn").
    model_path : str
        Path to YOLOv8/YOLOv11 .pt weights, or an ultralytics tag
        (e.g. "yolov8n.pt" auto-downloads).  Defaults to "<crop_type>.pt".
    class_ids : list[int] | None
        YOLO class IDs to treat as crop plants.  None = accept all classes.
    fwd_vel : int
        Constant forward velocity in mm/s.
    kp : float
        Proportional steering gain.
    conf : float
        Minimum detection confidence.
    """

    def __init__(
        self,
        crop_type: str = "plant",
        model_path: str | None = None,
        class_ids: list | None = None,
        fwd_vel: int = FWD_VEL_MM_S,
        kp: float = KP,
        conf: float = CONF_THRESHOLD,
    ):
        self._crop_type  = crop_type
        self._model_path = model_path or f"{crop_type}.pt"
        self._class_ids  = set(class_ids) if class_ids else None
        self._fwd_vel    = fwd_vel
        self._kp         = kp
        self._conf       = conf

        self._model  = None
        self._loaded = threading.Event()

        threading.Thread(target=self._load, daemon=True, name="crop-row-load").start()

    @property
    def name(self) -> str:
        return "crop_row"

    def on_reset(self) -> None:
        log.info("CropRowStrategy reset (crop_type=%s)", self._crop_type)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            log.error("ultralytics not installed — run: pip install ultralytics")
            return
        log.info("CropRow[%s]: loading YOLO model '%s'…", self._crop_type, self._model_path)
        self._model = YOLO(self._model_path)
        self._loaded.set()
        log.info("CropRow[%s]: model ready — class_ids=%s conf=%.2f",
                 self._crop_type, self._class_ids or "all", self._conf)

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
            log.error("CropRow[%s] error: %s", self._crop_type, e, exc_info=True)
        finally:
            state.query_in_flight.clear()

    def _do_query(self, state: AgentState, frame: np.ndarray, rover_ctrl) -> None:
        if not self._loaded.is_set():
            log.info("CropRow[%s]: model not ready — skipping", self._crop_type)
            return

        t0 = time.time()
        with state.result_lock:
            step  = state.step
            phase = state.phase
            state.llm_query_start = t0

        h, w = frame.shape[:2]
        frame_cx = w // 2

        # ── Detect plants ─────────────────────────────────────────────────────
        boxes = self._detect(frame)

        # ── Split left / right and find wall edges ────────────────────────────
        left_boxes  = [b for b in boxes if _box_cx(b) < frame_cx]
        right_boxes = [b for b in boxes if _box_cx(b) >= frame_cx]

        left_wall  = max((b[2] for b in left_boxes),  default=0)   # rightmost x2 of left plants
        right_wall = min((b[0] for b in right_boxes), default=w)   # leftmost  x1 of right plants

        gap_cx   = (left_wall + right_wall) / 2
        error_px = gap_cx - frame_cx   # positive = gap right of centre → steer right

        # ── Steering ─────────────────────────────────────────────────────────
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
        r_str     = "straight" if radius == 0x8000 else f"r={radius}mm"
        reasoning = (f"{self._crop_type} L={len(left_boxes)} R={len(right_boxes)}  "
                     f"gap_cx={gap_cx:.0f}px  err={error_px:+.0f}px  "
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
                "crop_type":     self._crop_type,
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
        """Run YOLO; return accepted boxes as list of [x1, y1, x2, y2]."""
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

    # ── Steering ──────────────────────────────────────────────────────────────

    def _error_to_radius(self, error_px: float, vel: int) -> int:
        """Convert lateral pixel error to a Roomba radius.

        Same kinematic formula as joystick and OmniVLA: radius = vel / ang_rate
        where ang_rate = Kp * error_px.

        Positive error → gap right of centre → steer right → negative radius.
        """
        if abs(error_px) < STRAIGHT_DEAD_ZONE_PX:
            return 0x8000

        ang_rad_s  = self._kp * error_px
        raw_radius = vel / ang_rad_s
        return int(math.copysign(
            min(MAX_RADIUS_MM, max(MIN_RADIUS_MM, abs(raw_radius))),
            -raw_radius,
        ))

    # ── Annotation ────────────────────────────────────────────────────────────

    def _annotate(self, frame, boxes, left_wall, right_wall,
                  gap_cx, frame_cx, error_px, vel, radius) -> np.ndarray:
        h, w = frame.shape[:2]

        # Bounding boxes — orange = left plant, blue = right plant
        for x1, y1, x2, y2 in boxes:
            color = (255, 100, 0) if _box_cx([x1, y1, x2, y2]) < frame_cx else (0, 100, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Wall edge lines
        cv2.line(frame, (int(left_wall),  0), (int(left_wall),  h), (255, 140, 0), 2)
        cv2.line(frame, (int(right_wall), 0), (int(right_wall), h), (0, 140, 255), 2)

        # Gap centre (green) and frame centre (dashed white)
        cv2.line(frame, (int(gap_cx), 0), (int(gap_cx), h), (0, 220, 80), 2)
        for y in range(0, h, 20):
            cv2.line(frame, (int(frame_cx), y), (int(frame_cx), min(h, y + 10)),
                     (200, 200, 200), 1)

        # Error arrow
        if abs(error_px) > STRAIGHT_DEAD_ZONE_PX:
            cv2.arrowedLine(frame,
                            (int(frame_cx), h // 2),
                            (int(gap_cx),   h // 2),
                            (0, 255, 255), 2, tipLength=0.3)

        # HUD
        r_str = "STRAIGHT" if radius == 0x8000 else f"r={radius}mm"
        cv2.putText(frame, f"{self._crop_type}  err={error_px:+.0f}px  {r_str}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 80), 2)
        cv2.putText(frame,
                    f"vel={vel}mm/s  "
                    f"L={sum(1 for b in boxes if _box_cx(b) < frame_cx)}  "
                    f"R={sum(1 for b in boxes if _box_cx(b) >= frame_cx)}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        return frame


# ── Helpers ────────────────────────────────────────────────────────────────────

def _box_cx(box) -> float:
    return (box[0] + box[2]) / 2
