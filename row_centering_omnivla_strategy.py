"""
RowCenteringOmniVLAStrategy — CLIP/OmniVLA front navigation + downward-camera row centering.

Extends the ClipOmniVLA strategy with a second downward-facing camera that
monitors the crop rows on either side of the rover and corrects lateral drift.

Cameras:
    Primary (--device)       : front-facing; used by CLIP path detection and OmniVLA
                               trajectory prediction — identical to clip_omnivla.
    Secondary (--down-device): downward-facing; used exclusively for row centering.

Row-centering pipeline (downward frame each cycle):
    1. Compute Excess Green (ExG = 2G - R - B) on the downward frame.
    2. Threshold ExG mask → vegetation blobs → bounding boxes via contours.
    3. Split boxes: box centre_x < frame_cx → left wall, else right wall.
    4. left_wall  = rightmost x2 of left-side boxes
       right_wall = leftmost  x1 of right-side boxes
    5. gap_centre_x  = (left_wall + right_wall) / 2
       lateral_error = gap_centre_x - frame_width / 2  (px; +ve → gap right of centre)
    6. ang_correction = -lateral_error * centering_gain            (rad/s)
    7. final_angular  = omnivla_angular + centering_alpha * ang_correction
       final_radius   = vel / final_angular  (clamped to [-2000, 2000] mm)

State machine (identical to clip_omnivla):
    INITIALIZING → PATH_LOST → NAVIGATING ↔ PATH_LOST

Usage:
    python rover_agent.py --strategy row_centering_omnivla \\
        --omnivla-server localhost:5100 \\
        --goal "Follow the crop row" \\
        --device 0 --down-device 1 \\
        --exg-threshold 20 --exg-min-area 500 \\
        --centering-gain 0.001 --centering-alpha 0.4 \\
        --interval 1.0 --rover atlas --atlas-port /dev/ttyACM0
"""

import collections
import io
import logging
import threading
import time
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

from navigation_strategy import AgentState, NavigationStrategy
from prompt_generator import generate_clip_prompts
from omnivla_strategy import (
    CONTEXT_SIZE, TRAJ_LEN, IMG_OBS, IMG_CLIP, IMG_MAP,
    METRIC_SPACING, ENC_SIZE,
    MAX_LIN_MM_S, MAX_ANG_RAD_S,
    MODALITY_LANG, MODALITY_GOAL_IMG,
    _waypoint_to_drive, _annotate,
)

log = logging.getLogger("rover.row_centering_omnivla")

# CLIP path detection floor (same as clip_omnivla)
_MIN_PATH_POS_SIM = 0.18

# Resolution fed to the OmniVLA/CLIP server. Both cameras capture at their
# maximum native resolution; frames are downscaled here before JPEG-encoding
# for the inference server. Local inference paths use torchvision T.Resize()
# and need no explicit resize.
_MODEL_W, _MODEL_H = 640, 480

# ── Row-centering constants ────────────────────────────────────────────────────
_MAX_ERROR_PX     = 160    # lateral error clamped at ±this value (px)

# PiP thumbnail size (pixels) embedded in the annotated front-camera frame
_PIP_W = 213
_PIP_H = 160


# ── State machine ─────────────────────────────────────────────────────────────

class _NavState(Enum):
    INITIALIZING = auto()
    NAVIGATING   = auto()
    PATH_LOST    = auto()


# ── ExG-based row-gap detector ────────────────────────────────────────────────

def _find_row_gap_exg(
    down_bgr: np.ndarray,
    exg_threshold: int = 20,
    min_area: int = 500,
) -> tuple:
    """
    Detect the crop-row gap centre from a downward-facing BGR frame using ExG.

    ExG = 2*G - R - B pixels above exg_threshold are vegetation.
    Contours above min_area become bounding boxes.
    Plants on the left half define the left wall (rightmost x2 of left boxes).
    Plants on the right half define the right wall (leftmost x1 of right boxes).
    The gap centre is the midpoint between the two walls.

    Returns
    -------
    gap_cx      : int | None     — x-pixel of gap midpoint (None if no detections)
    left_wall   : int | None     — x-pixel of inner edge of left row
    right_wall  : int | None     — x-pixel of inner edge of right row
    boxes       : list           — all accepted [x1,y1,x2,y2] boxes
    """
    h, w = down_bgr.shape[:2]
    frame_cx = w // 2

    bgr   = down_bgr.astype(np.float32)
    exg   = 2.0 * bgr[:, :, 1] - bgr[:, :, 0] - bgr[:, :, 2]
    mask  = np.clip(exg, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, exg_threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        boxes.append([x, y, x + bw, y + bh])

    left_boxes  = [b for b in boxes if (b[0] + b[2]) / 2 < frame_cx]
    right_boxes = [b for b in boxes if (b[0] + b[2]) / 2 >= frame_cx]

    if not left_boxes and not right_boxes:
        return None, None, None, boxes

    left_wall  = int(max((b[2] for b in left_boxes),  default=0))
    right_wall = int(min((b[0] for b in right_boxes), default=w))
    gap_cx     = (left_wall + right_wall) // 2
    return gap_cx, left_wall, right_wall, boxes


# ── Centering correction ───────────────────────────────────────────────────────

def _apply_centering(
    vel_mm_s: int,
    radius_mm: int,
    lateral_error_px: float,
    gain: float,
    alpha: float,
) -> tuple[int, int]:
    """
    Blend a lateral centering correction into an OmniVLA drive command.

    A positive lateral_error_px means the row gap is to the right of the image
    centre, so the rover has drifted left and needs to steer right (negative
    angular velocity in iRobot convention).

    Returns
    -------
    (corrected_vel_mm_s, corrected_radius_mm)
    """
    if vel_mm_s == 0:
        return vel_mm_s, radius_mm

    # Decode OmniVLA turn as angular velocity (rad/s)
    if radius_mm == 0x8000:
        omnivla_ang = 0.0
    else:
        r = float(radius_mm) if radius_mm != 0 else 1e-3
        omnivla_ang = float(np.clip(vel_mm_s / r, -MAX_ANG_RAD_S, MAX_ANG_RAD_S))

    # Centering correction: right-shifted gap → steer right → negative angular
    error = float(np.clip(lateral_error_px, -_MAX_ERROR_PX, _MAX_ERROR_PX))
    centering_ang = float(np.clip(-error * gain, -MAX_ANG_RAD_S, MAX_ANG_RAD_S))

    # Blend
    final_ang = omnivla_ang + alpha * centering_ang
    final_ang = float(np.clip(final_ang, -MAX_ANG_RAD_S, MAX_ANG_RAD_S))

    if abs(final_ang) < 0.01:
        return vel_mm_s, 0x8000
    corrected_radius = int(np.clip(vel_mm_s / final_ang, -2000, 2000))
    return vel_mm_s, corrected_radius


# ── Down-camera annotator ──────────────────────────────────────────────────────

def _annotate_down_frame(
    down_bgr: np.ndarray,
    gap_cx: int | None,
    left_wall: int | None,
    right_wall: int | None,
    boxes: list,
    error_px: float,
) -> np.ndarray:
    """Draw ExG bounding boxes and row-gap overlay on the downward-camera frame for the PiP thumbnail."""
    vis = down_bgr.copy()
    h, w = vis.shape[:2]
    frame_cx = w // 2

    # ExG blobs — orange = left plant, blue = right plant
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2
        color = (255, 100, 0) if cx < frame_cx else (0, 100, 255)
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

    # Wall edges and gap centre
    if left_wall is not None:
        cv2.line(vis, (left_wall, 0), (left_wall, h), (255, 140, 0), 2)
    if right_wall is not None:
        cv2.line(vis, (right_wall, 0), (right_wall, h), (0, 140, 255), 2)
    if gap_cx is not None:
        cv2.line(vis, (gap_cx, 0), (gap_cx, h), (0, 255, 255), 2)

    # Image centre reference
    cv2.line(vis, (frame_cx, 0), (frame_cx, h), (200, 200, 200), 1)

    label = f"err={error_px:+.0f}px" if gap_cx is not None else "no detections"
    color = (0, 200, 255) if gap_cx is not None else (0, 60, 220)
    cv2.putText(vis, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


def _embed_pip(main: np.ndarray, thumb: np.ndarray) -> None:
    """Embed a thumbnail Picture-in-Picture in the bottom-right corner of main (in-place)."""
    h, w = main.shape[:2]
    th, tw = _PIP_H, _PIP_W
    resized = cv2.resize(thumb, (tw, th))
    x0, y0 = w - tw - 8, h - th - 8
    # Border
    cv2.rectangle(main, (x0 - 2, y0 - 2), (x0 + tw + 1, y0 + th + 1), (80, 80, 80), 2)
    main[y0:y0 + th, x0:x0 + tw] = resized
    cv2.putText(main, "DOWN CAM", (x0 + 4, y0 + th - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)


# ── Path HUD (same style as clip_omnivla) ─────────────────────────────────────

def _draw_hud(
    frame: np.ndarray,
    path_score: float,
    threshold: float,
    nav_state: _NavState,
    pos_sim: float,
    neg_sim: float,
    error_px: float | None,
    corrected: bool,
) -> None:
    if nav_state == _NavState.NAVIGATING:
        color = (0, 220, 80)
        label = f"path: {path_score*100:.0f}%  NAVIGATING"
    elif nav_state == _NavState.PATH_LOST:
        color = (0, 60, 220)
        label = f"PATH LOST  {path_score*100:.0f}%"
    else:
        color = (180, 180, 180)
        label = "initializing..."

    cv2.putText(frame, label, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"pos={pos_sim:.4f}  neg={neg_sim:.4f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)

    # Row centering status
    if error_px is not None:
        corr_color = (0, 220, 200) if corrected else (160, 160, 0)
        cv2.putText(frame, f"row err={error_px:+.0f}px {'corrected' if corrected else 'no rows'}",
                    (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.40, corr_color, 1, cv2.LINE_AA)

    # Score bar with threshold marker
    bx, by, bw, bh = 10, 96, 120, 8
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
    fill = int(bw * max(0.0, min(1.0, path_score)))
    cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), color, -1)
    tx = bx + int(bw * threshold)
    cv2.line(frame, (tx, by - 2), (tx, by + bh + 2), (255, 255, 255), 1)


# ── Strategy ──────────────────────────────────────────────────────────────────

class RowCenteringOmniVLAStrategy(NavigationStrategy):
    """
    OmniVLA/CLIP front-camera navigation extended with downward-camera row centering.

    The front camera drives CLIP path detection and OmniVLA waypoint prediction
    (identical to clip_omnivla).  Every cycle the downward camera provides a
    lateral correction signal: the detected row-gap midpoint is compared to the
    image centre and the resulting pixel error is blended into OmniVLA's turn
    radius before the drive command is issued.

    Parameters
    ----------
    goal : str
        Language navigation goal for OmniVLA and CLIP prompts.
    goal_image_path : str | None
        Optional goal image for OmniVLA (modality 6).
    server_addr : str | None
        "host:port" of a running omnivla_server.py.
    path_threshold : float
        CLIP path score above which the path is considered present (default 0.5).
    ollama_url : str
        Ollama API URL used to generate CLIP prompts.
    weights_path : str | None
        Custom OmniVLA-edge weights (.pth), or None to download from HuggingFace.
    centering_gain : float
        Proportional gain: rad/s per pixel of lateral error (default 0.001).
        Increase if corrections feel sluggish; decrease to reduce oscillation.
    centering_alpha : float
        Blending weight applied to the centering correction before adding to
        OmniVLA's angular velocity.  0 = no centering, 1 = full override (default 0.4).

    Notes
    -----
    The downward camera is NOT opened by this strategy. rover_agent.py opens
    it separately (--down-device) and feeds frames via update_down_frame().
    This keeps all camera lifecycle management in one place.
    """

    def __init__(
        self,
        goal: str = "",
        goal_image_path: str | None = None,
        server_addr: str | None = None,
        path_threshold: float = 0.5,
        ollama_url: str = "http://localhost:11434",
        weights_path: str | None = None,
        centering_gain: float = 0.001,
        centering_alpha: float = 0.4,
        exg_threshold: int = 20,
        exg_min_area: int = 500,
    ):
        self._goal            = goal
        self._goal_image_path = goal_image_path
        self._server_addr     = server_addr
        self._path_threshold  = path_threshold
        self._ollama_url      = ollama_url
        self._weights_path    = weights_path
        self._centering_gain  = centering_gain
        self._centering_alpha = centering_alpha
        self._exg_threshold   = exg_threshold
        self._exg_min_area    = exg_min_area
        self._path_cache: dict = {}

        self._pos_prompts: list = []
        self._neg_prompts: list = []

        self._nav_state  = _NavState.INITIALIZING
        self._state_lock = threading.Lock()

        self._context: collections.deque = collections.deque(maxlen=CONTEXT_SIZE + 1)
        self._context_lock = threading.Lock()

        self._loaded = threading.Event()

        # Down-camera frames are pushed externally via update_down_frame()
        self._down_frame: np.ndarray | None = None
        self._down_lock  = threading.Lock()

        # Latest annotated down frame (ExG blobs + gap overlay), updated at
        # inference rate. Held between inferences so down.avi stays populated,
        # mirroring how state.llm_frame works for the front camera.
        self._down_ann_frame: np.ndarray | None = None
        self._down_ann_lock  = threading.Lock()

        # ── Front-camera model setup (mirrors ClipOmniVLAStrategy) ────────────
        if server_addr:
            host, port_str = server_addr.rsplit(":", 1)
            from omnivla_server import OmniVLAManager, DEFAULT_AUTHKEY
            self._manager = OmniVLAManager(
                address=(host, int(port_str)), authkey=DEFAULT_AUTHKEY
            )
            self._manager.connect()
            self._engine = self._manager.engine()
            self._goal_image_bytes: bytes | None = None
            if goal_image_path:
                with open(goal_image_path, "rb") as fh:
                    self._goal_image_bytes = fh.read()
            self._loaded.set()
            with self._state_lock:
                self._nav_state = _NavState.PATH_LOST
            log.info("RowCenteringOmniVLA: connected to OmniVLA server at %s", server_addr)
        else:
            self._clip_model    = None
            self._clip_tf       = None
            self._path_pos_feat = None
            self._path_neg_feat = None
            self._model         = None
            self._feat_text     = None
            self._device        = None
            self._obs_tf        = None
            self._dummy_pose    = None
            self._dummy_map     = None
            self._goal_img      = None
            self._modality_id   = None
            threading.Thread(target=self._load, daemon=True, name="row-center-load").start()

    # ── Down-camera frame interface ───────────────────────────────────────────

    def update_down_frame(self, frame: np.ndarray) -> None:
        """Called by rover_agent's down-camera loop with each captured frame."""
        with self._down_lock:
            self._down_frame = frame.copy()

    def _get_down_frame(self) -> np.ndarray | None:
        with self._down_lock:
            return self._down_frame.copy() if self._down_frame is not None else None

    def get_down_annotated_frame(self) -> np.ndarray | None:
        """Return the latest annotated down frame (ExG + gap overlay), or None."""
        with self._down_ann_lock:
            return self._down_ann_frame.copy() if self._down_ann_frame is not None else None

    # ── NavigationStrategy interface ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "row_centering_omnivla"

    def on_reset(self) -> None:
        with self._context_lock:
            self._context.clear()
        with self._state_lock:
            if self._nav_state != _NavState.INITIALIZING:
                self._nav_state = _NavState.PATH_LOST
        log.info("RowCenteringOmniVLAStrategy reset")

    def set_goal(self, goal: str) -> None:
        self._goal = goal
        self._path_cache.clear()
        if not self._loaded.is_set():
            log.info("Goal stored (models loading): '%s'", goal)
            return
        _prompts = generate_clip_prompts(goal, ollama_url=self._ollama_url)
        self._pos_prompts = _prompts["positive"]
        self._neg_prompts = _prompts["negative"]
        if not self._server_addr and self._clip_model is not None:
            self._encode_path_prompts()
        with self._state_lock:
            self._nav_state = _NavState.PATH_LOST
        log.info("Goal updated: '%s' — entering PATH_LOST", goal)

    # ── Local model loading ───────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            import torch
            import torchvision.transforms as T
            import clip as clip_lib
            from PIL import Image as PIL_Image
        except ImportError as e:
            log.error("OmniVLA/CLIP dependencies not installed: %s", e)
            log.error("Run: pip install -r requirements-omnivla.txt")
            return

        try:
            from omnivla_model import OmniVLA_edge
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            log.error("Cannot import OmniVLA model: %s", e)
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        log.info("RowCenteringOmniVLA: loading OmniVLA-edge on %s…", device)

        if self._weights_path:
            weights_path = self._weights_path
            log.info("RowCenteringOmniVLA: using custom weights '%s'", weights_path)
        else:
            weights_path = hf_hub_download("NHirose/omnivla-edge", "omnivla-edge.pth")
        model = OmniVLA_edge(
            context_size=CONTEXT_SIZE, len_traj_pred=TRAJ_LEN, learn_angle=True,
            obs_encoder="efficientnet-b0", obs_encoding_size=ENC_SIZE,
            late_fusion=False, mha_num_attention_heads=4,
            mha_num_attention_layers=4, mha_ff_dim_factor=4,
        )
        ckpt = torch.load(weights_path, map_location=device)
        ckpt = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        model.to(device).eval()
        self._model = model
        log.info("RowCenteringOmniVLA: OmniVLA-edge weights loaded")

        log.info("RowCenteringOmniVLA: loading CLIP ViT-B/32…")
        clip_model, _ = clip_lib.load("ViT-B/32", device=device)
        clip_model.eval()
        self._clip_model = clip_model

        self._obs_tf = T.Compose([
            T.Resize(IMG_OBS), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._clip_tf = T.Compose([
            T.Resize(IMG_CLIP), T.ToTensor(),
            T.Normalize([0.48145466, 0.4578275, 0.40821073],
                        [0.26862954, 0.26130258, 0.27577711]),
        ])

        self._dummy_pose = torch.zeros(1, 4, device=device)
        self._dummy_map  = torch.zeros(1, 9, *IMG_MAP, device=device)

        if self._goal_image_path:
            goal_pil = PIL_Image.open(self._goal_image_path).convert("RGB")
            self._goal_img    = self._obs_tf(goal_pil).unsqueeze(0).to(device)
            self._modality_id = torch.tensor([MODALITY_GOAL_IMG], device=device)
            with torch.no_grad():
                self._feat_text = clip_model.encode_text(
                    clip_lib.tokenize([self._goal], truncate=True).to(device)
                ).float()
            log.info("RowCenteringOmniVLA: image+language goal (modality %d)", MODALITY_GOAL_IMG)
        else:
            if self._goal:
                with torch.no_grad():
                    self._feat_text = clip_model.encode_text(
                        clip_lib.tokenize([self._goal], truncate=True).to(device)
                    ).float()
            else:
                self._feat_text = torch.zeros(1, ENC_SIZE, device=device)
            self._goal_img    = torch.zeros(1, 3, *IMG_OBS, device=device)
            self._modality_id = torch.tensor([MODALITY_LANG], device=device)
            log.info("RowCenteringOmniVLA: language-only goal (modality %d)", MODALITY_LANG)

        # Generate CLIP prompts after OmniVLA+CLIP are on GPU (Qwen3 cannot displace them)
        if self._goal:
            log.info("RowCenteringOmniVLA: generating CLIP prompts via Ollama…")
            _prompts = generate_clip_prompts(self._goal, ollama_url=self._ollama_url)
            self._pos_prompts = _prompts["positive"]
            self._neg_prompts = _prompts["negative"]

        if self._pos_prompts and self._neg_prompts:
            with torch.no_grad():
                pos = clip_model.encode_text(
                    clip_lib.tokenize(self._pos_prompts, truncate=True).to(device)
                ).float()
                neg = clip_model.encode_text(
                    clip_lib.tokenize(self._neg_prompts, truncate=True).to(device)
                ).float()
            self._path_pos_feat = (pos / pos.norm(dim=-1, keepdim=True)).mean(dim=0, keepdim=True)
            self._path_neg_feat = (neg / neg.norm(dim=-1, keepdim=True)).mean(dim=0, keepdim=True)
            log.info("RowCenteringOmniVLA: CLIP prompts encoded (%d pos, %d neg)",
                     len(self._pos_prompts), len(self._neg_prompts))
        else:
            self._path_pos_feat = None
            self._path_neg_feat = None

        self._loaded.set()
        with self._state_lock:
            if self._goal:
                self._nav_state = _NavState.PATH_LOST
        log.info("RowCenteringOmniVLAStrategy ready — goal: '%s'", self._goal or "(none)")

    def _encode_path_prompts(self) -> None:
        import torch
        import clip as clip_lib
        with torch.no_grad():
            pos = self._clip_model.encode_text(
                clip_lib.tokenize(self._pos_prompts, truncate=True).to(self._device)
            ).float()
            neg = self._clip_model.encode_text(
                clip_lib.tokenize(self._neg_prompts, truncate=True).to(self._device)
            ).float()
        self._path_pos_feat = (pos / pos.norm(dim=-1, keepdim=True)).mean(dim=0, keepdim=True)
        self._path_neg_feat = (neg / neg.norm(dim=-1, keepdim=True)).mean(dim=0, keepdim=True)

    # ── CLIP path detection (local) ───────────────────────────────────────────

    def _detect_path_local(self, pil_frame) -> dict:
        import torch
        with torch.no_grad():
            img_feat = self._clip_model.encode_image(
                self._clip_tf(pil_frame).unsqueeze(0).to(self._device)
            ).float()
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        pos_sim = float((img_feat @ self._path_pos_feat.T).squeeze())
        neg_sim = float((img_feat @ self._path_neg_feat.T).squeeze())
        if pos_sim < _MIN_PATH_POS_SIM:
            return {"score": 0.0, "pos_sim": pos_sim, "neg_sim": neg_sim}
        scale = float(self._clip_model.logit_scale.exp())
        score = float(torch.softmax(
            torch.tensor([scale * pos_sim, scale * neg_sim]), dim=0
        )[0])
        return {"score": score, "pos_sim": pos_sim, "neg_sim": neg_sim}

    # ── OmniVLA inference ─────────────────────────────────────────────────────

    def _run_inference(self, pil_frame, current_jpeg=None) -> tuple:
        if self._server_addr:
            buf = io.BytesIO()
            pil_frame.save(buf, format="JPEG", quality=85)
            if current_jpeg is None:
                current_jpeg = buf.getvalue()
            with self._context_lock:
                self._context.append(current_jpeg)
                ctx = list(self._context)
            while len(ctx) < CONTEXT_SIZE + 1:
                ctx.insert(0, ctx[0])
            result = self._engine.infer(ctx, current_jpeg, self._goal, self._goal_image_bytes)
            return np.array(result["waypoints"]), result["vel"], result["radius"]
        else:
            return self._run_omnivla_local(pil_frame)

    def _run_omnivla_local(self, pil_frame) -> tuple:
        import torch
        with self._context_lock:
            self._context.append(pil_frame)
            frames = list(self._context)
        while len(frames) < CONTEXT_SIZE + 1:
            frames.insert(0, frames[0])
        obs_images = torch.stack([self._obs_tf(f) for f in frames]).unsqueeze(0)
        obs_images = obs_images.view(1, -1, *IMG_OBS).to(self._device)
        cur_large  = self._clip_tf(pil_frame).unsqueeze(0).to(self._device)
        with torch.no_grad():
            actions, _, _ = self._model(
                obs_images, self._dummy_pose, self._dummy_map,
                self._goal_img, self._modality_id,
                self._feat_text, cur_large,
            )
        waypoints = actions[0].cpu().numpy()
        vel, radius = _waypoint_to_drive(waypoints)
        return waypoints, vel, radius

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
            log.error("RowCenteringOmniVLA error: %s", e, exc_info=True)
        finally:
            state.query_in_flight.clear()

    def _do_query(self, state: AgentState, frame: np.ndarray, rover_ctrl) -> None:
        from PIL import Image as PIL_Image

        if not self._loaded.is_set():
            log.info("RowCenteringOmniVLA: models not ready — skipping step")
            return

        if not self._goal or not self._pos_prompts:
            log.info("RowCenteringOmniVLA: no goal yet — waiting for goal from web UI")
            return

        t0 = time.time()
        with state.result_lock:
            step  = state.step
            phase = state.phase
            state.llm_query_start = t0

        pil = PIL_Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Encode front frame as JPEG once (shared by detect_path and infer in server mode).
        # Downscale to model input size first — camera captures at full/max resolution.
        if self._server_addr:
            pil_enc = (pil.resize((_MODEL_W, _MODEL_H))
                       if pil.size != (_MODEL_W, _MODEL_H) else pil)
            buf = io.BytesIO()
            pil_enc.save(buf, format="JPEG", quality=85)
            current_jpeg = buf.getvalue()
        else:
            current_jpeg = None

        # ── Row centering from downward camera ────────────────────────────────
        down_frame = self._get_down_frame()
        gap_cx = left_inner = right_inner = None
        lateral_error_px: float = 0.0
        down_annotated: np.ndarray | None = None

        left_wall = right_wall = None
        down_boxes: list = []
        if down_frame is not None:
            gap_cx, left_wall, right_wall, down_boxes = _find_row_gap_exg(
                down_frame, self._exg_threshold, self._exg_min_area
            )
            if gap_cx is not None:
                lateral_error_px = float(gap_cx - down_frame.shape[1] / 2)
            down_annotated = _annotate_down_frame(
                down_frame, gap_cx, left_wall, right_wall, down_boxes, lateral_error_px
            )
            with self._down_ann_lock:
                self._down_ann_frame = down_annotated
            log.info(
                "RowCentering(ExG): gap_cx=%s  error=%+.1f px  left_wall=%s  right_wall=%s  blobs=%d",
                gap_cx, lateral_error_px, left_wall, right_wall, len(down_boxes),
            )
        else:
            log.debug("RowCentering: no down-camera frame yet")

        # ── CLIP path detection ───────────────────────────────────────────────
        if self._server_addr:
            det = self._engine.detect_path(current_jpeg, self._pos_prompts, self._neg_prompts)
        else:
            det = self._detect_path_local(pil)
        path_score = det["score"]
        pos_sim    = det["pos_sim"]
        neg_sim    = det["neg_sim"]
        log.info("CLIP detect | score=%.3f  pos=%.4f  neg=%.4f  threshold=%.2f",
                 path_score, pos_sim, neg_sim, self._path_threshold)

        with self._state_lock:
            current_state = self._nav_state

        # ── State machine + OmniVLA + centering blend ─────────────────────────
        waypoints = None
        vel = radius = 0
        centering_applied = False
        operator_active = (state.operator_control is not None and
                           state.operator_until > time.time())

        if current_state == _NavState.INITIALIZING:
            log.info("RowCenteringOmniVLA: still initializing — skipping")
            self._write_result(state, step, phase, None, 0, 0x8000,
                               path_score, pos_sim, neg_sim, lateral_error_px,
                               False, "initializing", time.time() - t0)
            return

        elif current_state == _NavState.NAVIGATING:
            if path_score < self._path_threshold:
                with self._state_lock:
                    self._nav_state = _NavState.PATH_LOST
                log.info("Step %d | PATH LOST (score=%.2f) — stopping rover", step, path_score)
                if rover_ctrl and not state.paused.is_set() and not operator_active:
                    rover_ctrl.stop()
            else:
                waypoints, vel, radius = self._run_inference(pil, current_jpeg)
                # Apply row-centering correction if rows detected
                if gap_cx is not None:
                    vel, radius = _apply_centering(
                        vel, radius, lateral_error_px,
                        self._centering_gain, self._centering_alpha,
                    )
                    centering_applied = True
                if rover_ctrl and not state.paused.is_set() and not operator_active:
                    rover_ctrl.drive_raw(vel, radius)
                elif operator_active:
                    log.info("Step %d | operator override active — skipping drive", step)

        elif current_state == _NavState.PATH_LOST:
            if path_score >= self._path_threshold:
                with self._state_lock:
                    self._nav_state = _NavState.NAVIGATING
                log.info("Step %d | PATH FOUND (score=%.2f) — resuming", step, path_score)
                waypoints, vel, radius = self._run_inference(pil, current_jpeg)
                if gap_cx is not None:
                    vel, radius = _apply_centering(
                        vel, radius, lateral_error_px,
                        self._centering_gain, self._centering_alpha,
                    )
                    centering_applied = True
                if rover_ctrl and not state.paused.is_set() and not operator_active:
                    rover_ctrl.drive_raw(vel, radius)
                elif operator_active:
                    log.info("Step %d | operator override active — skipping drive", step)
            else:
                log.info("Step %d | path_lost (score=%.2f)", step, path_score)

        with self._state_lock:
            goal_status = {
                _NavState.NAVIGATING:   "in_progress",
                _NavState.PATH_LOST:    "path_lost",
                _NavState.INITIALIZING: "initializing",
            }[self._nav_state]

        elapsed = time.time() - t0
        log.info("Step %d | state=%s score=%.2f err=%+.1f vel=%d | %.2fs",
                 step, self._nav_state.name, path_score, lateral_error_px, vel, elapsed)

        # ── Compose display frame ─────────────────────────────────────────────
        if waypoints is not None:
            display = _annotate(frame, waypoints, vel, radius, self._goal)
        else:
            display = frame.copy()
        _draw_hud(display, path_score, self._path_threshold, self._nav_state,
                  pos_sim, neg_sim,
                  lateral_error_px if gap_cx is not None else None,
                  centering_applied)
        if down_annotated is not None:
            _embed_pip(display, down_annotated)

        with state.llm_lock:
            state.llm_frame = display

        self._write_result(state, step, phase, waypoints, vel, radius,
                           path_score, pos_sim, neg_sim, lateral_error_px,
                           centering_applied, goal_status, elapsed)

    # ── Result writer ─────────────────────────────────────────────────────────

    def _write_result(
        self, state, step, phase, waypoints, vel, radius,
        path_score, pos_sim, neg_sim, lateral_error_px,
        centering_applied, goal_status, elapsed,
    ) -> None:
        h, w = 480, 640
        ui_waypoints = []
        if waypoints is not None:
            with state.raw_lock:
                if state.raw_frame is not None:
                    h, w = state.raw_frame.shape[:2]
            cx, cy = w // 2, h
            scale = min(h, w) * 0.3
            for i, wp_i in enumerate(waypoints[:3]):
                px = int(cx - float(wp_i[1]) * METRIC_SPACING * scale)
                py = int(cy - float(wp_i[0]) * METRIC_SPACING * scale)
                ui_waypoints.append({
                    "rank": i + 1,
                    "x":    max(0, min(w - 1, px)),
                    "y":    max(0, min(h - 1, py)),
                    "description": f"wp[{i}] +{wp_i[0]*METRIC_SPACING:.2f}m",
                    "probability": round(1.0 - i * 0.1, 1),
                })

        r_str = "straight" if radius == 0x8000 else f"r={radius}mm"
        result = {
            "phase":            phase,
            "navigation_mode":  "following",
            "goal_status":      goal_status,
            "reasoning": (
                f"path={path_score:.2f} (pos={pos_sim:.4f} neg={neg_sim:.4f})"
                f" vel={vel}mm/s {r_str}"
                f" row_err={lateral_error_px:+.1f}px"
                f" centering={'on' if centering_applied else 'off'}"
                f" | goal='{self._goal}'"
            ),
            "waypoints":        ui_waypoints,
            "confidence":       round(path_score, 2),
            "row_lateral_error_px": round(lateral_error_px, 1),
            "centering_applied":    centering_applied,
        }

        with state.result_lock:
            state.latest_result   = result
            state.llm_query_start = 0.0
            state.llm_response_s  = elapsed
            if ui_waypoints:
                top = ui_waypoints[0]
                state.trajectory.append({
                    "step": step, "phase": phase,
                    "x": top["x"], "y": top["y"],
                    "description": top["description"],
                })

        if state.recorder:
            state.recorder.write_decision({
                "timestamp":           time.strftime("%Y-%m-%dT%H:%M:%S"),
                "step":                step,
                "phase":               phase,
                "elapsed_s":           round(elapsed, 3),
                "strategy":            self.name,
                "path_score":          round(path_score, 4),
                "pos_sim":             round(pos_sim, 4),
                "neg_sim":             round(neg_sim, 4),
                "row_lateral_error_px": round(lateral_error_px, 1),
                "centering_applied":   centering_applied,
                "vel_mm_s":            vel,
                "radius_mm":           radius if radius != 0x8000 else None,
                "result":              result,
            })
