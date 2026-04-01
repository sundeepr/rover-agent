"""
ClipOmniVLAStrategy — CLIP path reasoner + OmniVLA navigation state machine.

Uses CLIP zero-shot image classification to detect whether the brown path is
present, then drives the rover with OmniVLA only while the path is visible.

State machine (rover-centric):

    INITIALIZING → PATH_LOST → NAVIGATING ↔ PATH_LOST

    INITIALIZING : server connecting / local models loading; rover idle
    NAVIGATING   : brown path detected; OmniVLA running; rover driving
    PATH_LOST    : path not detected; rover stopped; OmniVLA skipped

Server mode (--omnivla-server):
    Path detection and OmniVLA inference both run in omnivla_server.py.
    No heavy models are loaded in this process.

Local mode (default):
    CLIP + OmniVLA-edge are loaded on a background thread at startup.

Usage:
    python rover_agent.py --strategy clip_omnivla \\
        --omnivla-server localhost:5100 \\
        --goal "Follow the brown path" \\
        --path-threshold 0.5 --interval 1.0 \\
        --roomba-port /dev/ttyUSB0
"""

import collections
import io
import logging
import math
import threading
import time
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

from navigation_strategy import AgentState, NavigationStrategy
from omnivla_strategy import (
    CONTEXT_SIZE, TRAJ_LEN, IMG_OBS, IMG_CLIP, IMG_MAP,
    METRIC_SPACING, DT, WAYPOINT_IDX, ENC_SIZE,
    MAX_LIN_MM_S, MAX_ANG_RAD_S,
    MODALITY_LANG, MODALITY_GOAL_IMG,
    _waypoint_to_drive, _annotate,
)

log = logging.getLogger("rover.clip_omnivla")

# ── Prompts for zero-shot path detection ──────────────────────────────────────

_POSITIVE_PROMPTS = [
    "brown path ahead",
    "brown tape on floor",
    "brown strip on ground",
]
_NEGATIVE_PROMPTS = [
    "white floor",
    "no brown path visible",
    "end of brown path",
]

# Minimum pos_sim before trusting logit_scale-amplified softmax.
# Below this floor CLIP has no real signal — return score=0.0 (PATH_LOST).
MIN_PATH_POS_SIM = 0.18


# ── State machine ─────────────────────────────────────────────────────────────

class _NavState(Enum):
    INITIALIZING = auto()   # waiting for server / local load
    NAVIGATING   = auto()   # path visible; OmniVLA driving
    PATH_LOST    = auto()   # path gone; rover stopped


# ── Strategy ──────────────────────────────────────────────────────────────────

class ClipOmniVLAStrategy(NavigationStrategy):
    """
    Navigation strategy using CLIP for path detection and OmniVLA for control.

    Parameters
    ----------
    goal : str
        Language navigation goal (always used for OmniVLA FiLM conditioning).
    goal_image_path : str | None
        Optional goal image path (OmniVLA modality 6).
    server_addr : str | None
        "host:port" of a running omnivla_server.py. When set, both path
        detection and OmniVLA inference run in the server process.
    path_threshold : float
        CLIP path score above which the brown path is considered present.
        Default 0.5. Tune by watching the score in the web UI.
    """

    def __init__(
        self,
        goal: str = "navigate forward",
        goal_image_path: str | None = None,
        server_addr: str | None = None,
        path_threshold: float = 0.5,
    ):
        self._goal            = goal
        self._goal_image_path = goal_image_path
        self._server_addr     = server_addr
        self._path_threshold  = path_threshold

        self._nav_state = _NavState.INITIALIZING
        # query_in_flight serialises query threads so no separate lock needed,
        # but use one anyway for clarity.
        self._state_lock = threading.Lock()

        self._context: collections.deque = collections.deque(maxlen=CONTEXT_SIZE + 1)
        self._context_lock = threading.Lock()

        self._loaded = threading.Event()

        if server_addr:
            # ── Server mode ───────────────────────────────────────────────────
            host, port_str = server_addr.rsplit(":", 1)
            from omnivla_server import OmniVLAManager, DEFAULT_AUTHKEY
            self._manager = OmniVLAManager(
                address=(host, int(port_str)), authkey=DEFAULT_AUTHKEY
            )
            self._manager.connect()
            self._engine = self._manager.engine()   # proxy with infer + detect_path
            # Pre-read goal image bytes once (sent on every infer request)
            self._goal_image_bytes: bytes | None = None
            if goal_image_path:
                with open(goal_image_path, "rb") as fh:
                    self._goal_image_bytes = fh.read()
            self._loaded.set()
            with self._state_lock:
                self._nav_state = _NavState.PATH_LOST  # ready; wait for path
            log.info("ClipOmniVLAStrategy: connected to server at %s", server_addr)
        else:
            # ── Local mode ────────────────────────────────────────────────────
            self._clip_model   = None
            self._clip_tf      = None
            self._path_pos_feat = None
            self._path_neg_feat = None
            self._model        = None
            self._feat_text    = None
            self._device       = None
            self._obs_tf       = None
            self._dummy_pose   = None
            self._dummy_map    = None
            self._goal_img     = None
            self._modality_id  = None
            threading.Thread(target=self._load, daemon=True).start()

    @property
    def name(self) -> str:
        return "clip_omnivla"

    def on_reset(self) -> None:
        with self._context_lock:
            self._context.clear()
        with self._state_lock:
            if self._nav_state != _NavState.INITIALIZING:
                self._nav_state = _NavState.PATH_LOST
        log.info("ClipOmniVLAStrategy reset")

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
        log.info("ClipOmniVLA: loading OmniVLA-edge on %s…", device)

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
        log.info("ClipOmniVLA: OmniVLA-edge weights loaded")

        log.info("ClipOmniVLA: loading CLIP ViT-B/32…")
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
            log.info("ClipOmniVLA: loading goal image '%s'…", self._goal_image_path)
            goal_pil = PIL_Image.open(self._goal_image_path).convert("RGB")
            self._goal_img    = self._obs_tf(goal_pil).unsqueeze(0).to(device)
            self._modality_id = torch.tensor([MODALITY_GOAL_IMG], device=device)
            self._feat_text   = torch.zeros(1, ENC_SIZE, device=device)
            # Still need CLIP text for FiLM even in image-goal mode
            with torch.no_grad():
                self._feat_text = clip_model.encode_text(
                    clip_lib.tokenize([self._goal], truncate=True).to(device)
                ).float()
            log.info("ClipOmniVLA: image+language goal (modality %d)", MODALITY_GOAL_IMG)
        else:
            log.info("ClipOmniVLA: encoding goal '%s' with CLIP…", self._goal)
            with torch.no_grad():
                self._feat_text = clip_model.encode_text(
                    clip_lib.tokenize([self._goal], truncate=True).to(device)
                ).float()
            self._goal_img    = torch.zeros(1, 3, *IMG_OBS, device=device)
            self._modality_id = torch.tensor([MODALITY_LANG], device=device)
            log.info("ClipOmniVLA: language-only goal (modality %d)", MODALITY_LANG)

        # Pre-encode path detection prompts
        with torch.no_grad():
            pos = clip_model.encode_text(
                clip_lib.tokenize(_POSITIVE_PROMPTS, truncate=True).to(device)
            ).float()
            neg = clip_model.encode_text(
                clip_lib.tokenize(_NEGATIVE_PROMPTS, truncate=True).to(device)
            ).float()
        self._path_pos_feat = (pos / pos.norm(dim=-1, keepdim=True)).mean(dim=0, keepdim=True)
        self._path_neg_feat = (neg / neg.norm(dim=-1, keepdim=True)).mean(dim=0, keepdim=True)
        log.info("ClipOmniVLA: path detection prompts encoded")

        self._loaded.set()
        with self._state_lock:
            self._nav_state = _NavState.PATH_LOST   # ready; wait for first detection
        log.info("ClipOmniVLAStrategy ready — goal: '%s'", self._goal)

    # ── Path detection (local) ────────────────────────────────────────────────

    def _detect_path_local(self, pil_frame) -> dict:
        import torch
        with torch.no_grad():
            img_feat = self._clip_model.encode_image(
                self._clip_tf(pil_frame).unsqueeze(0).to(self._device)
            ).float()
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        pos_sim = float((img_feat @ self._path_pos_feat.T).squeeze())
        neg_sim = float((img_feat @ self._path_neg_feat.T).squeeze())
        if pos_sim < MIN_PATH_POS_SIM:
            return {"score": 0.0, "pos_sim": pos_sim, "neg_sim": neg_sim}
        scale   = float(self._clip_model.logit_scale.exp())
        score   = float(torch.softmax(
            torch.tensor([scale * pos_sim, scale * neg_sim]), dim=0
        )[0])
        return {"score": score, "pos_sim": pos_sim, "neg_sim": neg_sim}

    # ── OmniVLA inference (local) ─────────────────────────────────────────────

    def _run_omnivla_local(self, pil_frame) -> tuple[np.ndarray, int, int]:
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
        waypoints = actions[0].cpu().numpy()   # [8, 4]
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
            log.error("ClipOmniVLA error: %s", e, exc_info=True)
        finally:
            state.query_in_flight.clear()

    def _do_query(
        self,
        state: AgentState,
        frame: np.ndarray,
        rover_ctrl,
    ) -> None:
        from PIL import Image as PIL_Image

        if not self._loaded.is_set():
            log.info("ClipOmniVLA: models not ready — skipping step")
            return

        t0 = time.time()
        with state.result_lock:
            step  = state.step
            phase = state.phase
            state.llm_query_start = t0

        pil = PIL_Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Encode current frame as JPEG once (used for both detect_path and infer)
        if self._server_addr:
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=85)
            current_jpeg = buf.getvalue()
        else:
            current_jpeg = None

        # ── Path detection ────────────────────────────────────────────────────
        if self._server_addr:
            det = self._engine.detect_path(current_jpeg)
        else:
            det = self._detect_path_local(pil)
        path_score = det["score"]
        pos_sim    = det["pos_sim"]
        neg_sim    = det["neg_sim"]
        log.info("CLIP detect | score=%.3f  pos_sim=%.4f  neg_sim=%.4f  threshold=%.2f",
                 path_score, pos_sim, neg_sim, self._path_threshold)

        with self._state_lock:
            current_state = self._nav_state

        # ── State machine ─────────────────────────────────────────────────────
        waypoints = None
        vel = radius = 0

        if current_state == _NavState.INITIALIZING:
            # Should not reach here (_loaded guards entry), but be safe
            log.info("ClipOmniVLA: still initializing — skipping")
            self._write_result(state, step, phase, None, 0, 0x8000,
                               path_score, pos_sim, neg_sim,
                               "initializing", time.time() - t0)
            return

        elif current_state == _NavState.NAVIGATING:
            if path_score < self._path_threshold:
                # Path lost — stop rover
                with self._state_lock:
                    self._nav_state = _NavState.PATH_LOST
                log.info("Step %d | PATH LOST (score=%.2f < %.2f) — stopping rover",
                         step, path_score, self._path_threshold)
                if rover_ctrl and not state.paused.is_set():
                    rover_ctrl.stop()
            else:
                # Continue navigating
                waypoints, vel, radius = self._run_inference(pil, current_jpeg)
                if rover_ctrl and not state.paused.is_set():
                    rover_ctrl.drive_raw(vel, radius)

        elif current_state == _NavState.PATH_LOST:
            if path_score >= self._path_threshold:
                # Path reappeared — resume
                with self._state_lock:
                    self._nav_state = _NavState.NAVIGATING
                log.info("Step %d | PATH FOUND (score=%.2f ≥ %.2f) — resuming",
                         step, path_score, self._path_threshold)
                waypoints, vel, radius = self._run_inference(pil, current_jpeg)
                if rover_ctrl and not state.paused.is_set():
                    rover_ctrl.drive_raw(vel, radius)
            else:
                log.info("Step %d | path_lost (score=%.2f)", step, path_score)

        with self._state_lock:
            goal_status = {
                _NavState.NAVIGATING:   "in_progress",
                _NavState.PATH_LOST:    "path_lost",
                _NavState.INITIALIZING: "initializing",
            }[self._nav_state]

        elapsed = time.time() - t0
        log.info("Step %d | state=%s score=%.2f vel=%d | %.2fs",
                 step, self._nav_state.name, path_score, vel, elapsed)

        # Annotate frame
        if waypoints is not None:
            annotated = _annotate(frame, waypoints, vel, radius, self._goal)
        else:
            annotated = frame.copy()
        _draw_path_hud(annotated, path_score, self._path_threshold,
                       self._nav_state, pos_sim, neg_sim)
        with state.llm_lock:
            state.llm_frame = annotated

        self._write_result(state, step, phase, waypoints, vel, radius,
                           path_score, pos_sim, neg_sim, goal_status, elapsed)

    def _run_inference(self, pil_frame, current_jpeg=None):
        """Run OmniVLA inference in either server or local mode."""
        if self._server_addr:
            buf = io.BytesIO()
            pil_frame.save(buf, format="JPEG", quality=85)
            if current_jpeg is None:
                current_jpeg = buf.getvalue()
            with self._context_lock:
                self._context.append(current_jpeg)
                context_jpegs = list(self._context)
            while len(context_jpegs) < CONTEXT_SIZE + 1:
                context_jpegs.insert(0, context_jpegs[0])
            result = self._engine.infer(
                context_jpegs, current_jpeg, self._goal, self._goal_image_bytes
            )
            waypoints = np.array(result["waypoints"])
            return waypoints, result["vel"], result["radius"]
        else:
            return self._run_omnivla_local(pil_frame)

    def _write_result(self, state, step, phase, waypoints, vel, radius,
                      path_score, pos_sim, neg_sim, goal_status, elapsed):
        h, w = 480, 640   # fallback; overwritten if waypoints present
        ui_waypoints = []
        if waypoints is not None:
            # Use actual frame size from raw_frame if available
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
            "reasoning":       (f"path={path_score:.2f} (pos={pos_sim:.4f} neg={neg_sim:.4f})"
                                f" vel={vel}mm/s {r_str} | goal='{self._goal}'"),
            "waypoints":       ui_waypoints,
            "confidence":      round(path_score, 2),
        }

        with state.result_lock:
            state.latest_result  = result
            state.llm_query_start = 0.0
            state.llm_response_s = elapsed
            if ui_waypoints:
                top = ui_waypoints[0]
                state.trajectory.append({
                    "step": step, "phase": phase,
                    "x": top["x"], "y": top["y"],
                    "description": top["description"],
                })


# ── HUD annotation ────────────────────────────────────────────────────────────

def _draw_path_hud(frame: np.ndarray, score: float, threshold: float,
                   nav_state: _NavState, pos_sim: float, neg_sim: float) -> None:
    """Draw path score, raw CLIP sims, and threshold bar onto frame in-place."""
    if nav_state == _NavState.NAVIGATING:
        color = (0, 220, 80)
        label = f"path: {score*100:.0f}%"
    elif nav_state == _NavState.PATH_LOST:
        color = (0, 60, 220)
        label = f"PATH LOST  {score*100:.0f}%"
    else:
        color = (180, 180, 180)
        label = "initializing..."

    cv2.putText(frame, label, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Raw cosine similarities — useful for tuning prompts and threshold
    cv2.putText(frame, f"pos={pos_sim:.4f}  neg={neg_sim:.4f}", (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    # Score bar with threshold marker
    bar_x, bar_y, bar_w, bar_h = 10, 80, 120, 8
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    fill = int(bar_w * max(0.0, min(1.0, score)))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)
    thr_x = bar_x + int(bar_w * threshold)
    cv2.line(frame, (thr_x, bar_y - 2), (thr_x, bar_y + bar_h + 2), (255, 255, 255), 1)
