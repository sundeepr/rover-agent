"""
OmniVLAStrategy — local neural-network navigation using OmniVLA-edge.

Uses a vision-language model that predicts 8 trajectory waypoints from a
sliding window of camera frames and a language goal (e.g. "blue trash bin").
Runs inference locally (CPU or CUDA) — no cloud API calls.

Model weights are downloaded automatically from HuggingFace on first use:
    NHirose/omnivla-edge/omnivla-edge.pth

Recommended interval: --interval 1.0  (OmniVLA control rate is 1 Hz)

The drive command is sent once per step and the Roomba keeps moving at that
velocity/radius until the next command arrives — no stop between steps.

Dependencies (beyond requirements.txt):
    pip install -r requirements-omnivla.txt
"""

import collections
import logging
import math
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from navigation_strategy import AgentState, NavigationStrategy

log = logging.getLogger("rover.omnivla_strategy")

# ── Constants (mirror run_rover.py) ───────────────────────────────────────────

CONTEXT_SIZE   = 5          # previous frames to include (5 + current = 6 total)
TRAJ_LEN       = 8          # waypoints predicted by the model
IMG_OBS        = (96, 96)   # observation image size (for trajectory encoder)
IMG_CLIP       = (224, 224) # image size for FiLM language-conditioning
IMG_MAP        = (352, 352) # satellite map size (unused; dummy zeros)
METRIC_SPACING = 0.1        # 1 model unit = 0.1 m
DT             = 1.0        # intended control period (seconds)
WAYPOINT_IDX   = 4          # which of the 8 predicted waypoints to execute
MODALITY_LANG  = 7          # modality ID for language-only goal
ENC_SIZE       = 1024
MAX_LIN_MM_S   = 50         # max forward velocity sent to Roomba
MAX_ANG_RAD_S  = 0.5        # max angular velocity

# omnivla source directory — needed to import model.py
_OMNIVLA_DIR = Path(__file__).parent.parent / "omnivla"


# ── Pure functions (no torch imports needed) ───────────────────────────────────

def _waypoint_to_drive(waypoints: np.ndarray) -> tuple[int, int]:
    """Convert predicted waypoints to a Roomba (velocity_mm_s, radius_mm) pair."""
    wp = waypoints[WAYPOINT_IDX].copy()
    dx = float(wp[0]) * METRIC_SPACING   # forward (m)
    dy = float(wp[1]) * METRIC_SPACING   # lateral (m)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0, 0x8000
    lin_m_s   = np.clip(dx / DT, 0.0, MAX_LIN_MM_S / 1000.0)
    ang_rad_s = np.clip(math.atan2(dy, dx) / DT, -MAX_ANG_RAD_S, MAX_ANG_RAD_S)
    lin_mm_s  = int(lin_m_s * 1000)
    if abs(ang_rad_s) < 0.01:
        return lin_mm_s, 0x8000
    radius_mm = int(np.clip(lin_mm_s / ang_rad_s, -2000, 2000))
    return lin_mm_s, radius_mm


def _annotate(frame: np.ndarray, waypoints: np.ndarray,
              vel: int, radius: int, goal: str) -> np.ndarray:
    """Draw predicted trajectory dots and HUD text onto a copy of frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h
    scale = min(h, w) * 0.3

    for i, wp in enumerate(waypoints):
        dx = float(wp[0]) * METRIC_SPACING
        dy = float(wp[1]) * METRIC_SPACING
        px = int(cx - dy * scale)
        py = int(cy - dx * scale)
        color = (0, 255, 100) if i == WAYPOINT_IDX else (0, 180, 60)
        dot_r = 6 if i == WAYPOINT_IDX else 3
        cv2.circle(out, (px, py), dot_r, color, -1)
        if i > 0:
            prev  = waypoints[i - 1]
            ppx   = int(cx - float(prev[1]) * METRIC_SPACING * scale)
            ppy   = int(cy - float(prev[0]) * METRIC_SPACING * scale)
            cv2.line(out, (ppx, ppy), (px, py), (0, 200, 80), 1)

    r_str = "straight" if radius == 0x8000 else f"r={radius}mm"
    cv2.putText(out, f"vel {vel} mm/s  {r_str}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 230, 255), 1)
    cv2.putText(out, f"goal: {goal}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 220, 255), 1)
    return out


# ── Strategy ───────────────────────────────────────────────────────────────────

class OmniVLAStrategy(NavigationStrategy):
    """
    Navigation strategy using OmniVLA-edge.

    The model is downloaded from HuggingFace and loaded on a background thread
    at construction time. Steps received before the model finishes loading are
    silently skipped.

    Parameters
    ----------
    goal : str
        Language navigation goal (e.g. "blue trash bin", "go forward").
    """

    def __init__(self, goal: str = "navigate forward"):
        self._goal = goal
        self._context: collections.deque = collections.deque(maxlen=CONTEXT_SIZE + 1)
        self._context_lock = threading.Lock()

        # Set by _load() on the background thread
        self._model        = None
        self._feat_text    = None
        self._device       = None
        self._obs_tf       = None   # obs_transform
        self._clip_tf      = None   # clip_transform
        self._dummy_pose   = None
        self._dummy_map    = None
        self._dummy_goal   = None
        self._modality_id  = None
        self._loaded       = threading.Event()

        threading.Thread(target=self._load, daemon=True).start()

    @property
    def name(self) -> str:
        return "omnivla"

    def on_reset(self) -> None:
        with self._context_lock:
            self._context.clear()
        log.info("OmniVLAStrategy frame context cleared")

    # ── Model loading (background thread) ─────────────────────────────────────

    def _load(self) -> None:
        try:
            import torch
            import torchvision.transforms as T
            import clip as clip_lib
        except ImportError as e:
            log.error("OmniVLA dependencies not installed: %s", e)
            log.error("Run: pip install -r requirements-omnivla.txt")
            return

        if str(_OMNIVLA_DIR) not in sys.path:
            sys.path.insert(0, str(_OMNIVLA_DIR))
        try:
            from model import OmniVLA_edge
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            log.error("Cannot import OmniVLA model: %s", e)
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        log.info("OmniVLA: loading weights on %s…", device)

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
        log.info("OmniVLA-edge weights loaded")

        log.info("OmniVLA: encoding goal '%s' with CLIP…", self._goal)
        text_encoder, _ = clip_lib.load("ViT-B/32", device=device)
        text_encoder.eval()
        with torch.no_grad():
            self._feat_text = text_encoder.encode_text(
                clip_lib.tokenize([self._goal], truncate=True).to(device)
            ).float()

        self._obs_tf = T.Compose([
            T.Resize(IMG_OBS), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._clip_tf = T.Compose([
            T.Resize(IMG_CLIP), T.ToTensor(),
            T.Normalize([0.48145466, 0.4578275, 0.40821073],
                        [0.26862954, 0.26130258, 0.27577711]),
        ])

        self._dummy_pose  = torch.zeros(1, 4, device=device)
        self._dummy_map   = torch.zeros(1, 9, *IMG_MAP, device=device)
        self._dummy_goal  = torch.zeros(1, 3, *IMG_OBS, device=device)
        self._modality_id = torch.tensor([MODALITY_LANG], device=device)

        self._loaded.set()
        log.info("OmniVLAStrategy ready — goal: '%s'", self._goal)

    # ── Query ──────────────────────────────────────────────────────────────────

    def run_query(
        self,
        state: AgentState,
        frame: np.ndarray,
        captures_dir: Path,
        roomba_ctrl,
    ) -> None:
        try:
            self._do_query(state, frame, roomba_ctrl)
        except Exception as e:
            with state.result_lock:
                state.llm_query_start = 0.0
            log.error("OmniVLA error: %s", e, exc_info=True)
        finally:
            state.query_in_flight.clear()

    def _do_query(
        self,
        state: AgentState,
        frame: np.ndarray,
        roomba_ctrl,
    ) -> None:
        import torch
        from PIL import Image as PIL_Image

        if not self._loaded.is_set():
            log.info("OmniVLA model still loading — skipping step")
            return

        t0 = time.time()
        with state.result_lock:
            step  = state.step
            phase = state.phase
            state.llm_query_start = t0

        # Build frame context (sliding window of PIL images)
        pil = PIL_Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with self._context_lock:
            self._context.append(pil)
            frames = list(self._context)
        # Pad with first frame if context not yet full
        while len(frames) < CONTEXT_SIZE + 1:
            frames.insert(0, frames[0])

        # Prepare model inputs
        obs_images = torch.stack([self._obs_tf(f) for f in frames]).unsqueeze(0)
        obs_images = obs_images.view(1, -1, *IMG_OBS).to(self._device)
        cur_large  = self._clip_tf(pil).unsqueeze(0).to(self._device)

        # Run inference
        with torch.no_grad():
            actions, _, _ = self._model(
                obs_images, self._dummy_pose, self._dummy_map,
                self._dummy_goal, self._modality_id,
                self._feat_text, cur_large,
            )

        waypoints = actions[0].cpu().numpy()   # [8, 4]: (dx, dy, cos θ, sin θ)
        vel, radius = _waypoint_to_drive(waypoints)
        elapsed = time.time() - t0

        wp = waypoints[WAYPOINT_IDX]
        log.info("Step %d | wp=(%.2fm, %.2fm) vel=%d mm/s %s | %.2fs",
                 step,
                 wp[0] * METRIC_SPACING, wp[1] * METRIC_SPACING,
                 vel,
                 "straight" if radius == 0x8000 else f"r={radius}mm",
                 elapsed)

        # Annotate frame for web display
        annotated = _annotate(frame, waypoints, vel, radius, self._goal)
        with state.llm_lock:
            state.llm_frame = annotated.copy()

        # Build result dict in the shape the web UI expects
        h, w = frame.shape[:2]
        cx, cy = w // 2, h
        scale = min(h, w) * 0.3
        ui_waypoints = []
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
            "goal_status":     "in_progress",
            "reasoning":       f"vel={vel}mm/s {r_str} | goal='{self._goal}'",
            "waypoints":       ui_waypoints,
            "confidence":      1.0,
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

        # Send drive command — Roomba keeps moving at this velocity until next step
        if roomba_ctrl:
            try:
                roomba_ctrl.drive_raw(vel, radius)
            except Exception as e:
                log.error("Roomba drive error: %s", e, exc_info=True)
