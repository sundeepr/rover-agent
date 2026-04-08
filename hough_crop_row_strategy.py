"""
HoughCropRowStrategy — Hough-preprocessed crop-row navigation.

Pipeline per step:

  1. Run the Hough crop-row detector on the raw camera frame:
       ExG mask → Otsu threshold → strip-centre canvas → Hough lines → filter
       → draw vegetation overlay (crop=neon-green, soil=brown) + row lines
  2. Feed the Hough-annotated frame to CLIP for path-presence detection.
       The high-contrast overlay (green rows on brown soil) gives CLIP a much
       cleaner signal than the raw image.
  3. Feed the same Hough-annotated frame to OmniVLA for motion commands.
       OmniVLA navigates the visual structure of the highlighted rows.
  4. OmniVLA trajectory dots are drawn on top of the Hough overlay for display.

State machine (rover-centric):
    INITIALIZING → PATH_LOST → NAVIGATING ↔ PATH_LOST

    INITIALIZING : OmniVLA / CLIP loading; rover idle
    NAVIGATING   : CLIP says path present; OmniVLA driving
    PATH_LOST    : CLIP says path gone; rover stopped

Usage:
    python rover_agent.py --strategy hough_crop_row \\
        --goal "Follow the crop row" \\
        --omnivla-server localhost:5100 \\
        --interval 1.0 \\
        --rover atlas --atlas-port /dev/ttyACM0
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
    METRIC_SPACING, WAYPOINT_IDX, ENC_SIZE,
    MAX_LIN_MM_S, MAX_ANG_RAD_S,
    MODALITY_LANG, MODALITY_GOAL_IMG,
    _waypoint_to_drive, _annotate,
)

log = logging.getLogger("rover.hough_crop_row")

# ── Hough pipeline constants (mirror crop_row_hough.py defaults) ──────────────

_N_STRIPS       = 10
_SUM_THRESH     = 2
_DIFF_NOISE     = 8
_HOUGH_RHO      = 2
_HOUGH_ANGLE    = np.pi * 4 / 180    # 4° angular resolution
_HOUGH_THRESH   = 6                  # min votes
_ANGLE_THRESH   = np.pi * 30 / 180   # ±30° around expected row direction
_THETA_SIM      = np.pi * 6 / 180    # merge lines closer than 6°
_RHO_SIM        = 20                  # merge lines closer than 20 px
_EXPECTED_THETA = np.pi / 2          # π/2 = vertical rows (forward-facing camera)
_ALPHA          = 0.5                 # overlay blend opacity

_CROP_BGR  = np.array((0,   255,  57), dtype=np.uint8)   # neon green
_SOIL_BGR  = np.array((19,   69, 139), dtype=np.uint8)   # brown

# CLIP threshold floor (from clip_omnivla_strategy)
_MIN_PATH_POS_SIM = 0.18


# ── Hough pipeline ────────────────────────────────────────────────────────────

def _vegetation_mask(frame_bgr: np.ndarray) -> np.ndarray:
    b = frame_bgr[:, :, 0].astype(np.float32)
    g = frame_bgr[:, :, 1].astype(np.float32)
    r = frame_bgr[:, :, 2].astype(np.float32)
    exg = 2.0 * g - r - b
    lo, hi = exg.min(), exg.max()
    if hi == lo:
        return np.zeros(exg.shape, dtype=np.uint8)
    exg_u8 = ((exg - lo) / (hi - lo) * 255).astype(np.uint8)
    _, mask = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def _strip_centre_canvas(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    strip_h = h // _N_STRIPS
    for s in range(_N_STRIPS):
        y0   = s * strip_h
        y1   = y0 + strip_h if s < _N_STRIPS - 1 else h
        y_mid = (y0 + y1) // 2
        col_sum = mask[y0:y1, :].sum(axis=0) // 255
        in_region, region_start = False, 0
        for x in range(w):
            if col_sum[x] > _SUM_THRESH:
                if not in_region:
                    in_region, region_start = True, x
            else:
                if in_region:
                    in_region = False
                    if (x - 1 - region_start + 1) >= _DIFF_NOISE:
                        cv2.circle(canvas, ((region_start + x - 1) // 2, y_mid), 2, 255, -1)
        if in_region and (w - 1 - region_start + 1) >= _DIFF_NOISE:
            cv2.circle(canvas, ((region_start + w - 1) // 2, y_mid), 2, 255, -1)
    return canvas


def _hough_filter(canvas: np.ndarray) -> list:
    raw = cv2.HoughLines(canvas, _HOUGH_RHO, _HOUGH_ANGLE, _HOUGH_THRESH)
    if raw is None:
        return []
    rho_theta = [(float(l[0][0]), float(l[0][1])) for l in raw]

    filtered = [
        (rv, th) for rv, th in rho_theta
        if min(abs(th - _EXPECTED_THETA), np.pi - abs(th - _EXPECTED_THETA)) <= _ANGLE_THRESH
    ]

    deduped_theta = []
    for rv, th in filtered:
        if not any(min(abs(th - t), np.pi - abs(th - t)) < _THETA_SIM for _, t in deduped_theta):
            deduped_theta.append((rv, th))

    deduped_rho = []
    for rv, th in deduped_theta:
        if not any(abs(rv - r) < _RHO_SIM for r, _ in deduped_rho):
            deduped_rho.append((rv, th))

    return deduped_rho


def hough_annotate(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Run the full Hough crop-row pipeline on one BGR frame.

    Returns:
        annotated  — BGR frame with vegetation overlay + row lines drawn
        mask       — uint8 vegetation mask (0/255)
        lines      — list of (rho, theta) pairs that survived all filters
    """
    mask   = _vegetation_mask(frame_bgr)
    canvas = _strip_centre_canvas(mask)
    lines  = _hough_filter(canvas)

    # Vegetation colour overlay only — no row lines drawn
    overlay = frame_bgr.copy()
    overlay[mask == 0] = _SOIL_BGR
    overlay[mask >  0] = _CROP_BGR
    annotated = cv2.addWeighted(frame_bgr, 1 - _ALPHA, overlay, _ALPHA, 0)

    return annotated, mask, lines


# ── State machine ─────────────────────────────────────────────────────────────

class _NavState(Enum):
    INITIALIZING = auto()
    NAVIGATING   = auto()
    PATH_LOST    = auto()


# ── Strategy ──────────────────────────────────────────────────────────────────

class HoughCropRowStrategy(NavigationStrategy):
    """
    Crop-row navigation: Hough preprocessing → CLIP path detection + OmniVLA control.

    Both CLIP and OmniVLA receive the Hough-annotated frame (vegetation overlay
    with row lines) rather than the raw camera image.

    Parameters
    ----------
    goal : str
        Language navigation goal.
    goal_image_path : str | None
        Optional goal image for OmniVLA (modality 6).
    server_addr : str | None
        "host:port" of a running omnivla_server.py.
    path_threshold : float
        CLIP path score above which crop rows are considered present (default 0.5).
    ollama_url : str
        Ollama URL used by generate_clip_prompts for dynamic prompt generation.
    """

    def __init__(
        self,
        goal: str = "follow the crop row",
        goal_image_path: str | None = None,
        server_addr: str | None = None,
        path_threshold: float = 0.5,
        ollama_url: str = "http://localhost:11434",
    ):
        self._goal            = goal
        self._goal_image_path = goal_image_path
        self._server_addr     = server_addr
        self._path_threshold  = path_threshold

        # Generate CLIP prompts from goal (same as clip_omnivla_strategy)
        _prompts = generate_clip_prompts(goal, ollama_url=ollama_url)
        self._pos_prompts: list = _prompts["positive"]
        self._neg_prompts: list = _prompts["negative"]

        self._nav_state  = _NavState.INITIALIZING
        self._state_lock = threading.Lock()

        self._context: collections.deque = collections.deque(maxlen=CONTEXT_SIZE + 1)
        self._context_lock = threading.Lock()

        self._loaded = threading.Event()

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
            log.info("HoughCropRowStrategy: connected to OmniVLA server at %s", server_addr)
        else:
            self._clip_model        = None
            self._clip_tf           = None
            self._path_pos_feat     = None
            self._path_neg_feat     = None
            self._model             = None
            self._feat_text         = None
            self._device            = None
            self._obs_tf            = None
            self._dummy_pose        = None
            self._dummy_map         = None
            self._goal_img          = None
            self._modality_id       = None
            threading.Thread(target=self._load, daemon=True).start()

    @property
    def name(self) -> str:
        return "hough_crop_row"

    def on_reset(self) -> None:
        with self._context_lock:
            self._context.clear()
        with self._state_lock:
            if self._nav_state != _NavState.INITIALIZING:
                self._nav_state = _NavState.PATH_LOST
        log.info("HoughCropRowStrategy reset")

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
        log.info("HoughCropRow: loading OmniVLA-edge on %s…", device)

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
        log.info("HoughCropRow: OmniVLA-edge weights loaded")

        log.info("HoughCropRow: loading CLIP ViT-B/32…")
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

        # OmniVLA goal setup
        if self._goal_image_path:
            from PIL import Image as PIL_Image
            goal_pil = PIL_Image.open(self._goal_image_path).convert("RGB")
            self._goal_img    = self._obs_tf(goal_pil).unsqueeze(0).to(device)
            self._modality_id = torch.tensor([MODALITY_GOAL_IMG], device=device)
            with torch.no_grad():
                self._feat_text = clip_model.encode_text(
                    clip_lib.tokenize([self._goal], truncate=True).to(device)
                ).float()
            log.info("HoughCropRow: image+language goal (modality %d)", MODALITY_GOAL_IMG)
        else:
            log.info("HoughCropRow: encoding goal '%s' with CLIP…", self._goal)
            with torch.no_grad():
                self._feat_text = clip_model.encode_text(
                    clip_lib.tokenize([self._goal], truncate=True).to(device)
                ).float()
            self._goal_img    = torch.zeros(1, 3, *IMG_OBS, device=device)
            self._modality_id = torch.tensor([MODALITY_LANG], device=device)
            log.info("HoughCropRow: language-only goal (modality %d)", MODALITY_LANG)

        # Encode CLIP path-detection prompts
        with torch.no_grad():
            pos = clip_model.encode_text(
                clip_lib.tokenize(self._pos_prompts, truncate=True).to(device)
            ).float()
            neg = clip_model.encode_text(
                clip_lib.tokenize(self._neg_prompts, truncate=True).to(device)
            ).float()
        self._path_pos_feat = (pos / pos.norm(dim=-1, keepdim=True)).mean(dim=0, keepdim=True)
        self._path_neg_feat = (neg / neg.norm(dim=-1, keepdim=True)).mean(dim=0, keepdim=True)
        log.info("HoughCropRow: CLIP prompts encoded (%d pos, %d neg)",
                 len(self._pos_prompts), len(self._neg_prompts))

        self._loaded.set()
        with self._state_lock:
            self._nav_state = _NavState.PATH_LOST
        log.info("HoughCropRowStrategy ready — goal: '%s'", self._goal)

    # ── CLIP path detection on Hough-annotated frame (local) ─────────────────

    def _detect_path_local(self, pil_hough) -> dict:
        """Run CLIP on the Hough-annotated PIL image."""
        import torch
        with torch.no_grad():
            img_feat = self._clip_model.encode_image(
                self._clip_tf(pil_hough).unsqueeze(0).to(self._device)
            ).float()
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        pos_sim = float((img_feat @ self._path_pos_feat.T).squeeze())
        neg_sim = float((img_feat @ self._path_neg_feat.T).squeeze())
        if pos_sim < _MIN_PATH_POS_SIM:
            return {"score": 0.0, "pos_sim": pos_sim, "neg_sim": neg_sim}
        scale = float(self._clip_model.logit_scale.exp())
        import torch as _torch
        score = float(_torch.softmax(
            _torch.tensor([scale * pos_sim, scale * neg_sim]), dim=0
        )[0])
        return {"score": score, "pos_sim": pos_sim, "neg_sim": neg_sim}

    # ── OmniVLA inference on Hough-annotated frame ────────────────────────────

    def _run_inference(self, pil_hough, jpeg_hough: bytes | None = None):
        """Run OmniVLA on the Hough-annotated frame (server or local)."""
        if self._server_addr:
            if jpeg_hough is None:
                buf = io.BytesIO()
                pil_hough.save(buf, format="JPEG", quality=85)
                jpeg_hough = buf.getvalue()
            with self._context_lock:
                self._context.append(jpeg_hough)
                context_jpegs = list(self._context)
            while len(context_jpegs) < CONTEXT_SIZE + 1:
                context_jpegs.insert(0, context_jpegs[0])
            result = self._engine.infer(
                context_jpegs, jpeg_hough, self._goal, self._goal_image_bytes
            )
            return np.array(result["waypoints"]), result["vel"], result["radius"]
        else:
            return self._run_omnivla_local(pil_hough)

    def _run_omnivla_local(self, pil_hough):
        import torch
        with self._context_lock:
            self._context.append(pil_hough)
            frames = list(self._context)
        while len(frames) < CONTEXT_SIZE + 1:
            frames.insert(0, frames[0])
        obs_images = torch.stack([self._obs_tf(f) for f in frames]).unsqueeze(0)
        obs_images = obs_images.view(1, -1, *IMG_OBS).to(self._device)
        cur_large  = self._clip_tf(pil_hough).unsqueeze(0).to(self._device)
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
            log.error("HoughCropRow error: %s", e, exc_info=True)
        finally:
            state.query_in_flight.clear()

    def _do_query(self, state: AgentState, frame: np.ndarray, rover_ctrl) -> None:
        from PIL import Image as PIL_Image

        if not self._loaded.is_set():
            log.info("HoughCropRow: models not ready — skipping step")
            return

        t0 = time.time()
        with state.result_lock:
            step  = state.step
            phase = state.phase
            state.llm_query_start = t0

        # ── Step 1: Hough preprocessing ───────────────────────────────────────
        hough_frame, mask, lines = hough_annotate(frame)
        n_rows = len(lines)

        # Convert to PIL once — reused for both CLIP and OmniVLA
        pil_hough = PIL_Image.fromarray(cv2.cvtColor(hough_frame, cv2.COLOR_BGR2RGB))

        # Pre-encode as JPEG for server mode (single encode shared by both)
        jpeg_hough: bytes | None = None
        if self._server_addr:
            buf = io.BytesIO()
            pil_hough.save(buf, format="JPEG", quality=85)
            jpeg_hough = buf.getvalue()

        # ── Step 2: CLIP path detection on Hough-annotated frame ─────────────
        if self._server_addr:
            det = self._engine.detect_path(jpeg_hough, self._pos_prompts, self._neg_prompts)
        else:
            det = self._detect_path_local(pil_hough)

        path_score = det["score"]
        pos_sim    = det["pos_sim"]
        neg_sim    = det["neg_sim"]
        log.info("Hough rows=%d | CLIP score=%.3f pos=%.4f neg=%.4f threshold=%.2f",
                 n_rows, path_score, pos_sim, neg_sim, self._path_threshold)

        with self._state_lock:
            current_state = self._nav_state

        # ── Step 3: State machine + OmniVLA inference ─────────────────────────
        waypoints = None
        vel = radius = 0

        if current_state == _NavState.INITIALIZING:
            log.info("HoughCropRow: still initializing — skipping")
            self._write_result(state, step, phase, None, 0, 0x8000,
                               n_rows, lines, path_score, pos_sim, neg_sim,
                               "initializing", time.time() - t0)
            return

        elif current_state == _NavState.NAVIGATING:
            if path_score < self._path_threshold:
                with self._state_lock:
                    self._nav_state = _NavState.PATH_LOST
                log.info("Step %d | PATH LOST (score=%.2f rows=%d) — stopping rover",
                         step, path_score, n_rows)
                if rover_ctrl and not state.paused.is_set():
                    rover_ctrl.stop()
            else:
                # OmniVLA also sees the Hough-annotated frame
                waypoints, vel, radius = self._run_inference(pil_hough, jpeg_hough)
                if rover_ctrl and not state.paused.is_set():
                    rover_ctrl.drive_raw(vel, radius)

        elif current_state == _NavState.PATH_LOST:
            if path_score >= self._path_threshold:
                with self._state_lock:
                    self._nav_state = _NavState.NAVIGATING
                log.info("Step %d | PATH FOUND (score=%.2f rows=%d) — resuming",
                         step, path_score, n_rows)
                waypoints, vel, radius = self._run_inference(pil_hough, jpeg_hough)
                if rover_ctrl and not state.paused.is_set():
                    rover_ctrl.drive_raw(vel, radius)
            else:
                log.info("Step %d | path_lost (score=%.2f rows=%d)", step, path_score, n_rows)

        with self._state_lock:
            goal_status = {
                _NavState.NAVIGATING:   "in_progress",
                _NavState.PATH_LOST:    "path_lost",
                _NavState.INITIALIZING: "initializing",
            }[self._nav_state]

        elapsed = time.time() - t0
        log.info("Step %d | state=%s score=%.2f rows=%d vel=%d | %.2fs",
                 step, self._nav_state.name, path_score, n_rows, vel, elapsed)

        # ── Step 4: Build display frame ───────────────────────────────────────
        # Base = Hough overlay; OmniVLA trajectory dots drawn on top if navigating
        display = hough_frame.copy()
        if waypoints is not None:
            display = _annotate(display, waypoints, vel, radius, self._goal)
        _draw_hud(display, n_rows, path_score, self._path_threshold,
                  self._nav_state, pos_sim, neg_sim)
        with state.llm_lock:
            state.llm_frame = display

        self._write_result(state, step, phase, waypoints, vel, radius,
                           n_rows, lines, path_score, pos_sim, neg_sim,
                           goal_status, elapsed)

    def _write_result(self, state, step, phase, waypoints, vel, radius,
                      n_rows, lines, path_score, pos_sim, neg_sim,
                      goal_status, elapsed):
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
            "phase":           phase,
            "navigation_mode": "following",
            "goal_status":     goal_status,
            "reasoning":       (f"rows={n_rows} clip={path_score:.2f}"
                                f" (pos={pos_sim:.4f} neg={neg_sim:.4f})"
                                f" vel={vel}mm/s {r_str} | goal='{self._goal}'"),
            "waypoints":       ui_waypoints,
            "confidence":      round(path_score, 2),
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
                "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
                "step":       step,
                "phase":      phase,
                "elapsed_s":  round(elapsed, 3),
                "strategy":   self.name,
                "n_rows":     n_rows,
                "lines":      [{"rho": round(r, 1), "theta_deg": round(np.degrees(t), 1)}
                                for r, t in lines],
                "path_score": round(path_score, 4),
                "pos_sim":    round(pos_sim, 4),
                "neg_sim":    round(neg_sim, 4),
                "vel_mm_s":   vel,
                "radius_mm":  radius if radius != 0x8000 else None,
                "result":     result,
            })


# ── HUD ───────────────────────────────────────────────────────────────────────

def _draw_hud(frame: np.ndarray, n_rows: int, score: float, threshold: float,
              nav_state: _NavState, pos_sim: float, neg_sim: float) -> None:
    """Draw Hough row count, CLIP score, and nav state onto the frame in-place."""
    if nav_state == _NavState.NAVIGATING:
        color = (0, 220, 80)
        label = f"rows={n_rows}  clip={score*100:.0f}%  NAVIGATING"
    elif nav_state == _NavState.PATH_LOST:
        color = (0, 60, 220)
        label = f"rows={n_rows}  clip={score*100:.0f}%  PATH LOST"
    else:
        color = (180, 180, 180)
        label = "initializing..."

    cv2.putText(frame, label, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"pos={pos_sim:.4f}  neg={neg_sim:.4f}", (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

    bar_x, bar_y, bar_w, bar_h = 10, 80, 120, 8
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill = int(bar_w * max(0.0, min(1.0, score)))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)
    thr_x = bar_x + int(bar_w * threshold)
    cv2.line(frame, (thr_x, bar_y - 2), (thr_x, bar_y + bar_h + 2), (255, 255, 255), 1)
