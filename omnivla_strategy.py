"""
OmniVLAStrategy — local neural-network navigation using OmniVLA-edge.

Uses a vision-language model that predicts 8 trajectory waypoints from a
sliding window of camera frames and a language goal (e.g. "blue trash bin").
Runs inference locally (CPU or CUDA) — no cloud API calls.

Model weights are downloaded automatically from HuggingFace on first use:
    NHirose/omnivla-edge/omnivla-edge.pth

Recommended interval: --interval 1.0  (OmniVLA control rate is 1 Hz)

The drive command is sent once per step and the rover keeps moving at that
velocity/radius until the next command arrives — no stop between steps.

Server mode (optional):
    Start the model server once with:
        python omnivla_server.py
    Then point rover_agent at it:
        python rover_agent.py --strategy omnivla --omnivla-server localhost:5100 ...
    The model stays loaded in the server process between rover_agent runs,
    eliminating the ~30 s reload on each restart.

Dependencies (beyond requirements.txt):
    pip install -r requirements-omnivla.txt
"""

import collections
import logging
import math
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
DT             = 1.0 / 3.0  # control period matching run_omnivla.py (tick_rate=3)
WAYPOINT_IDX   = 4          # which of the 8 predicted waypoints to execute
ENC_SIZE       = 1024
MAX_LIN_MM_S   = 50         # max forward velocity mm/s
MAX_ANG_RAD_S  = 0.3        # max angular velocity rad/s
MIN_RADIUS_MM  = 200        # minimum turn radius — below this drive straight

# Modality IDs (defined by the OmniVLA-edge model architecture):
#   7 = language only          — language token in transformer
#   6 = image only             — goal image token in transformer
#                                (language still conditions FiLM in both cases)
MODALITY_LANG     = 7
MODALITY_GOAL_IMG = 6



# ── Pure functions (no torch imports needed) ───────────────────────────────────

def _waypoint_to_drive(waypoints: np.ndarray) -> tuple[int, int]:
    """Convert predicted waypoints to a Roomba (velocity_mm_s, radius_mm) pair.

    Matches the PD controller in run_omnivla.py exactly.
    """
    wp = waypoints[WAYPOINT_IDX].copy()
    dx = float(wp[0]) * METRIC_SPACING   # forward (m)
    dy = float(wp[1]) * METRIC_SPACING   # lateral (m)

    EPS = 1e-8
    if abs(dx) < EPS and abs(dy) < EPS:
        return 0, 0x8000
    elif abs(dx) < EPS:
        lin_m_s   = 0.0
        ang_rad_s = math.copysign(math.pi / (2 * DT), dy)
    else:
        lin_m_s   = dx / DT
        ang_rad_s = math.atan(dy / dx) / DT

    # Velocity limits (matching run_omnivla.py maxv=0.3, maxw=0.3)
    maxv = MAX_LIN_MM_S / 1000.0
    maxw = MAX_ANG_RAD_S
    lin_m_s   = max(0.0, min(maxv, lin_m_s))   # no reversing
    if abs(lin_m_s) <= maxv:
        if abs(ang_rad_s) <= maxw:
            lin_lim, ang_lim = lin_m_s, ang_rad_s
        else:
            rd = lin_m_s / ang_rad_s if abs(ang_rad_s) > EPS else 0
            lin_lim = maxw * math.copysign(1, lin_m_s) * abs(rd)
            ang_lim = maxw * math.copysign(1, ang_rad_s)
    else:
        if abs(ang_rad_s) <= 0.001:
            lin_lim = maxv * math.copysign(1, lin_m_s)
            ang_lim = 0.0
        else:
            rd = lin_m_s / ang_rad_s
            if abs(rd) >= maxv / maxw:
                lin_lim = maxv * math.copysign(1, lin_m_s)
                ang_lim = maxv * math.copysign(1, ang_rad_s) / abs(rd)
            else:
                lin_lim = maxw * math.copysign(1, lin_m_s) * abs(rd)
                ang_lim = maxw * math.copysign(1, ang_rad_s)

    lin_mm_s = int(lin_lim * 1000)
    if abs(ang_lim) < 0.01:
        return lin_mm_s, 0x8000
    radius_mm = int(np.clip(lin_mm_s / ang_lim, -2000, 2000))
    # Clamp to minimum radius — very tight turns at low speed stall the Roomba
    if 0 < abs(radius_mm) < MIN_RADIUS_MM:
        radius_mm = int(math.copysign(MIN_RADIUS_MM, radius_mm))
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

    Two operating modes:

    Local mode (default):
        The model is downloaded from HuggingFace and loaded on a background
        thread at construction time. Steps received before the model finishes
        loading are silently skipped.

    Server mode (server_addr is set):
        Connects to a running omnivla_server.py via TCP. The server keeps
        weights loaded between rover_agent restarts, saving ~30 s reload time.
        Context frames are sent as JPEG bytes; goal text/image are sent on
        every request so the server needs no persistent goal state.

    Parameters
    ----------
    goal : str
        Language navigation goal (e.g. "blue trash bin", "go forward").
        Always used to condition the FiLM visual encoder.
    goal_image_path : str | None
        Optional path to a goal image (any format PIL can read).
    server_addr : str | None
        "host:port" of a running omnivla_server.py. When set, inference is
        delegated to the server instead of running locally.
    """

    def __init__(self, goal: str = "navigate forward",
                 goal_image_path: str | None = None,
                 server_addr: str | None = None):
        self._goal            = goal
        self._goal_image_path = goal_image_path
        self._server_addr     = server_addr

        # Context deque — PIL images in local mode, JPEG bytes in server mode
        self._context: collections.deque = collections.deque(maxlen=CONTEXT_SIZE + 1)
        self._context_lock = threading.Lock()

        if server_addr:
            # ── Server mode ───────────────────────────────────────────────────
            host, port_str = server_addr.rsplit(":", 1)
            from omnivla_server import OmniVLAManager, DEFAULT_AUTHKEY
            self._manager = OmniVLAManager(
                address=(host, int(port_str)), authkey=DEFAULT_AUTHKEY
            )
            self._manager.connect()
            # engine() returns a proxy to the InferenceEngine instance;
            # calling .infer() on the proxy sends args to the server and
            # returns the dict result via pickle (not wrapped in another proxy).
            self._infer_fn = self._manager.engine().infer
            # Pre-load goal image bytes once (sent on every request)
            self._goal_image_bytes: bytes | None = None
            if goal_image_path:
                with open(goal_image_path, "rb") as fh:
                    self._goal_image_bytes = fh.read()
            self._loaded = threading.Event()
            self._loaded.set()   # server is already ready
            log.info("OmniVLAStrategy: connected to server at %s", server_addr)
        else:
            # ── Local mode ────────────────────────────────────────────────────
            # Set by _load() on the background thread
            self._model        = None
            self._feat_text    = None
            self._device       = None
            self._obs_tf       = None   # obs_transform
            self._clip_tf      = None   # clip_transform
            self._dummy_pose   = None
            self._dummy_map    = None
            self._goal_img     = None   # processed goal image tensor, or dummy zeros
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
            from PIL import Image as PIL_Image
        except ImportError as e:
            log.error("OmniVLA dependencies not installed: %s", e)
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

        # CLIP text always conditions the FiLM visual backbone regardless of modality
        log.info("OmniVLA: encoding goal '%s' with CLIP…", self._goal)
        text_encoder, _ = clip_lib.load("ViT-B/32", device=device)
        text_encoder.eval()
        with torch.no_grad():
            self._feat_text = text_encoder.encode_text(
                clip_lib.tokenize([self._goal], truncate=True).to(device)
            ).float()

        if self._goal_image_path:
            # Modality 6: goal image token in transformer + language via FiLM
            log.info("OmniVLA: loading goal image '%s'…", self._goal_image_path)
            goal_pil = PIL_Image.open(self._goal_image_path).convert("RGB")
            self._goal_img    = self._obs_tf(goal_pil).unsqueeze(0).to(device)
            self._modality_id = torch.tensor([MODALITY_GOAL_IMG], device=device)
            log.info("OmniVLA: using image+language goal (modality %d)", MODALITY_GOAL_IMG)
        else:
            # Modality 7: language token in transformer + language via FiLM
            self._goal_img    = torch.zeros(1, 3, *IMG_OBS, device=device)
            self._modality_id = torch.tensor([MODALITY_LANG], device=device)
            log.info("OmniVLA: using language-only goal (modality %d)", MODALITY_LANG)

        self._loaded.set()
        log.info("OmniVLAStrategy ready — goal: '%s'", self._goal)

    # ── Query ──────────────────────────────────────────────────────────────────

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
            log.error("OmniVLA error: %s", e, exc_info=True)
        finally:
            state.query_in_flight.clear()

    def _do_query(
        self,
        state: AgentState,
        frame: np.ndarray,
        rover_ctrl,
    ) -> None:
        import io as _io
        import numpy as np_local
        from PIL import Image as PIL_Image

        if not self._loaded.is_set():
            log.info("OmniVLA model still loading — skipping step")
            return

        t0 = time.time()
        with state.result_lock:
            step  = state.step
            phase = state.phase
            state.llm_query_start = t0

        # Encode current frame as PIL/JPEG (both modes need it)
        pil = PIL_Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if self._server_addr:
            # ── Server mode: send JPEG bytes, receive waypoints ────────────────
            buf = _io.BytesIO()
            pil.save(buf, format="JPEG", quality=85)
            current_jpeg = buf.getvalue()

            with self._context_lock:
                self._context.append(current_jpeg)
                context_jpegs = list(self._context)
            while len(context_jpegs) < CONTEXT_SIZE + 1:
                context_jpegs.insert(0, context_jpegs[0])

            result_srv = self._infer_fn(
                context_jpegs,
                current_jpeg,
                self._goal,
                self._goal_image_bytes,
            )
            waypoints = np_local.array(result_srv["waypoints"])   # [8, 4]
            vel       = result_srv["vel"]
            radius    = result_srv["radius"]
            elapsed   = time.time() - t0

        else:
            # ── Local mode: run inference in-process ───────────────────────────
            import torch

            with self._context_lock:
                self._context.append(pil)
                frames = list(self._context)
            while len(frames) < CONTEXT_SIZE + 1:
                frames.insert(0, frames[0])

            obs_images = torch.stack([self._obs_tf(f) for f in frames]).unsqueeze(0)
            obs_images = obs_images.view(1, -1, *IMG_OBS).to(self._device)
            cur_large  = self._clip_tf(pil).unsqueeze(0).to(self._device)

            with torch.no_grad():
                actions, _, _ = self._model(
                    obs_images, self._dummy_pose, self._dummy_map,
                    self._goal_img, self._modality_id,
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

        if state.recorder:
            state.recorder.write_decision({
                "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
                "step":             step,
                "phase":            phase,
                "elapsed_s":        round(elapsed, 3),
                "strategy":         self.name,
                "vel_mm_s":         vel,
                "radius_mm":        radius if radius != 0x8000 else None,
                "waypoints_metric": waypoints[:, :2].tolist(),
                "result":           result,
            })

        # Send drive command — skip if paused (pause handler already sent stop)
        if rover_ctrl and not state.paused.is_set():
            try:
                rover_ctrl.drive_raw(vel, radius)
            except Exception as e:
                log.error("Rover drive error: %s", e, exc_info=True)
