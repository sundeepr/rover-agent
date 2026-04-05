"""
QwenOmniVLAStrategy — Qwen2.5-VL path detection + OmniVLA navigation.

Replaces CLIP entirely. Uses Qwen2.5-VL (via Ollama) to detect whether the
navigation target is visible in the camera frame, then drives the rover with
OmniVLA when it is. No CLIP image encoding needed — only CLIP text encoding
remains in local mode, used to produce feat_text for OmniVLA's FiLM layer.

State machine (rover-centric):
    INITIALIZING → PATH_LOST → NAVIGATING ↔ PATH_LOST

    INITIALIZING : Ollama/OmniVLA not yet ready; rover idle
    NAVIGATING   : Qwen says target visible; OmniVLA driving
    PATH_LOST    : Qwen says target not visible; rover stopped

Server mode (--omnivla-server):
    OmniVLA inference runs in omnivla_server.py.
    Qwen2.5-VL detection always runs via Ollama (separate service).

Local mode (default):
    OmniVLA-edge and CLIP (text encoder only) loaded on a background thread.
    Qwen2.5-VL detection always via Ollama.

Usage:
    # Start Ollama with Qwen2.5-VL:
    ollama pull qwen2.5vl:3b
    ollama serve

    # Start OmniVLA server (optional):
    python omnivla_server.py

    python rover_agent.py --strategy qwen_omnivla \\
        --ollama-server http://localhost:11434 \\
        --omnivla-server localhost:5100 \\
        --goal "Follow the brown path" \\
        --path-threshold 0.6 --interval 3.0 \\
        --rover atlas --atlas-port /dev/ttyACM0
"""

import base64
import collections
import io
import json
import logging
import re
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

log = logging.getLogger("rover.qwen_omnivla")

# Default Qwen vision model served by Ollama
_VISION_MODEL = "qwen2.5vl:3b"


# ── State machine ──────────────────────────────────────────────────────────────

class _NavState(Enum):
    INITIALIZING = auto()
    NAVIGATING   = auto()
    PATH_LOST    = auto()


# ── Strategy ───────────────────────────────────────────────────────────────────

class QwenOmniVLAStrategy(NavigationStrategy):
    """
    Navigation strategy using Qwen2.5-VL for path detection and OmniVLA
    for waypoint-based control.

    Parameters
    ----------
    goal : str
        Language navigation goal.
    goal_image_path : str | None
        Optional goal image path (OmniVLA modality 6).
    server_addr : str | None
        "host:port" of a running omnivla_server.py. When set OmniVLA
        inference runs in the server process.
    path_threshold : float
        Qwen confidence score above which target is considered visible.
    ollama_url : str
        URL of the Ollama API server (default: http://localhost:11434).
    vision_model : str
        Ollama model tag for vision detection (default: qwen2.5vl:3b).
    """

    def __init__(
        self,
        goal: str = "navigate forward",
        goal_image_path: str | None = None,
        server_addr: str | None = None,
        path_threshold: float = 0.6,
        ollama_url: str = "http://localhost:11434",
        vision_model: str = _VISION_MODEL,
    ):
        self._goal           = goal
        self._goal_image_path = goal_image_path
        self._server_addr    = server_addr
        self._path_threshold = path_threshold
        self._ollama_url     = ollama_url.rstrip("/")
        self._vision_model   = vision_model

        # Detection prompt built once from goal
        self._detection_prompt = (
            f"Goal: {goal}\n\n"
            "Look at this robot camera image. "
            "Is the navigation target currently visible in the image?\n\n"
            "Respond with JSON only:\n"
            '{"visible": true/false, "confidence": 0.0-1.0, '
            '"reason": "one sentence"}'
        )

        self._nav_state  = _NavState.INITIALIZING
        self._state_lock = threading.Lock()

        self._context      = collections.deque(maxlen=CONTEXT_SIZE + 1)
        self._context_lock = threading.Lock()

        self._loaded = threading.Event()

        self._check_ollama()

        if server_addr:
            # ── Server mode ───────────────────────────────────────────────
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
            log.info("QwenOmniVLAStrategy: connected to OmniVLA server at %s", server_addr)
        else:
            # ── Local mode ────────────────────────────────────────────────
            self._clip_model  = None
            self._clip_tf     = None
            self._model       = None
            self._feat_text   = None
            self._device      = None
            self._obs_tf      = None
            self._dummy_pose  = None
            self._dummy_map   = None
            self._goal_img    = None
            self._modality_id = None
            threading.Thread(target=self._load, daemon=True).start()

    @property
    def name(self) -> str:
        return "qwen_omnivla"

    def on_reset(self) -> None:
        with self._context_lock:
            self._context.clear()
        with self._state_lock:
            if self._nav_state != _NavState.INITIALIZING:
                self._nav_state = _NavState.PATH_LOST
        log.info("QwenOmniVLAStrategy reset")

    # ── Ollama health check ────────────────────────────────────────────────────

    def _check_ollama(self) -> None:
        try:
            import requests
            requests.get(f"{self._ollama_url}/api/tags", timeout=3)
            log.info("QwenOmniVLA: Ollama reachable at %s", self._ollama_url)
        except Exception as e:
            log.warning("QwenOmniVLA: Ollama not reachable (%s) — "
                        "detection will fail until Ollama is available", e)

    # ── Local model loading ────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            import torch
            import torchvision.transforms as T
            import clip as clip_lib
            from PIL import Image as PIL_Image
        except ImportError as e:
            log.error("OmniVLA/CLIP dependencies not installed: %s", e)
            return

        try:
            from omnivla_model import OmniVLA_edge
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            log.error("Cannot import OmniVLA model: %s", e)
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        log.info("QwenOmniVLA: loading OmniVLA-edge on %s…", device)

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
        log.info("QwenOmniVLA: OmniVLA-edge weights loaded")

        # CLIP is loaded for text encoding only (feat_text for OmniVLA FiLM).
        # Image encoding for path detection is handled by Qwen2.5-VL via Ollama.
        log.info("QwenOmniVLA: loading CLIP ViT-B/32 (text encoder for OmniVLA FiLM)…")
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

        # Encode goal text for OmniVLA FiLM conditioning
        if self._goal_image_path:
            from PIL import Image as PIL_Image
            log.info("QwenOmniVLA: loading goal image '%s'…", self._goal_image_path)
            goal_pil = PIL_Image.open(self._goal_image_path).convert("RGB")
            self._goal_img    = self._obs_tf(goal_pil).unsqueeze(0).to(device)
            self._modality_id = torch.tensor([MODALITY_GOAL_IMG], device=device)
            with torch.no_grad():
                self._feat_text = clip_model.encode_text(
                    clip_lib.tokenize([self._goal], truncate=True).to(device)
                ).float()
            log.info("QwenOmniVLA: image+language goal (modality %d)", MODALITY_GOAL_IMG)
        else:
            log.info("QwenOmniVLA: encoding goal '%s' with CLIP…", self._goal)
            with torch.no_grad():
                self._feat_text = clip_model.encode_text(
                    clip_lib.tokenize([self._goal], truncate=True).to(device)
                ).float()
            self._goal_img    = torch.zeros(1, 3, *IMG_OBS, device=device)
            self._modality_id = torch.tensor([MODALITY_LANG], device=device)
            log.info("QwenOmniVLA: language-only goal (modality %d)", MODALITY_LANG)

        self._loaded.set()
        with self._state_lock:
            self._nav_state = _NavState.PATH_LOST
        log.info("QwenOmniVLAStrategy ready — goal: '%s'", self._goal)

    # ── Path detection via Qwen2.5-VL ─────────────────────────────────────────

    def _detect_path(self, jpeg_bytes: bytes) -> dict:
        """
        Ask Qwen2.5-VL via Ollama whether the navigation target is visible.

        Returns {"visible": bool, "confidence": float, "reason": str}.
        On any error returns visible=False, confidence=0.0.

        num_ctx=2048 caps the context window, which reduces KV cache size
        and allows the model to fit within available memory.
        """
        import requests

        b64 = base64.b64encode(jpeg_bytes).decode()

        payload = {
            "model": self._vision_model,
            "messages": [{
                "role":    "user",
                "content": self._detection_prompt,
                "images":  [b64],
            }],
            "stream":  False,
            # No format=json — not supported by all Ollama versions for vision
            # models. We extract JSON from free-text response instead.
            # num_ctx=2048: our prompt + image tokens + short answer fit easily
            # in 2048 tokens; smaller context = much smaller KV cache.
            "options": {"temperature": 0.1, "num_ctx": 2048},
        }
        try:
            r = requests.post(
                f"{self._ollama_url}/api/chat",
                json=payload,
                timeout=30,
            )
            if not r.ok:
                try:
                    err_body = r.json().get("error", r.text[:200])
                except Exception:
                    err_body = r.text[:200]
                log.warning("Qwen detection failed: HTTP %d — %s", r.status_code, err_body)
                return {"visible": False, "confidence": 0.0, "reason": f"ollama: {err_body}"}

            content = r.json()["message"]["content"]
            # Extract the first JSON object from the response text — handles
            # cases where the model adds preamble or markdown code fences.
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if not m:
                raise ValueError(f"no JSON found in response: {content[:200]}")
            data = json.loads(m.group())
            return {
                "visible":    bool(data.get("visible", False)),
                "confidence": float(data.get("confidence", 0.0)),
                "reason":     str(data.get("reason", "")),
            }
        except Exception as e:
            log.warning("Qwen detection failed: %s", e)
            return {"visible": False, "confidence": 0.0, "reason": f"error: {e}"}

    # ── OmniVLA inference (local) ──────────────────────────────────────────────

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
        waypoints = actions[0].cpu().numpy()
        vel, radius = _waypoint_to_drive(waypoints)
        return waypoints, vel, radius

    def _run_inference(self, pil_frame, current_jpeg=None):
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
            log.error("QwenOmniVLA error: %s", e, exc_info=True)
        finally:
            state.query_in_flight.clear()

    def _do_query(self, state, frame, rover_ctrl) -> None:
        from PIL import Image as PIL_Image

        if not self._loaded.is_set():
            log.info("QwenOmniVLA: models not ready — skipping step")
            return

        t0 = time.time()
        with state.result_lock:
            step  = state.step
            phase = state.phase
            state.llm_query_start = t0

        pil = PIL_Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Encode frame as JPEG for Qwen detection (and OmniVLA server if used)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85)
        current_jpeg = buf.getvalue()

        # ── Path detection via Qwen2.5-VL ─────────────────────────────────
        det = self._detect_path(current_jpeg)
        visible    = det["visible"]
        confidence = det["confidence"]
        reason     = det["reason"]
        path_active = visible and confidence >= self._path_threshold

        log.info("Qwen detect | visible=%s conf=%.2f threshold=%.2f | %s",
                 visible, confidence, self._path_threshold, reason)

        with self._state_lock:
            current_state = self._nav_state

        # ── State machine ──────────────────────────────────────────────────
        waypoints = None
        vel = radius = 0

        if current_state == _NavState.INITIALIZING:
            log.info("QwenOmniVLA: still initializing — skipping")
            self._write_result(state, step, phase, None, 0, 0x8000,
                               det, "initializing", time.time() - t0)
            return

        elif current_state == _NavState.NAVIGATING:
            if not path_active:
                with self._state_lock:
                    self._nav_state = _NavState.PATH_LOST
                log.info("Step %d | PATH LOST (conf=%.2f) — stopping rover", step, confidence)
                if rover_ctrl and not state.paused.is_set():
                    rover_ctrl.stop()
            else:
                waypoints, vel, radius = self._run_inference(pil, current_jpeg)
                if rover_ctrl and not state.paused.is_set():
                    rover_ctrl.drive_raw(vel, radius)

        elif current_state == _NavState.PATH_LOST:
            if path_active:
                with self._state_lock:
                    self._nav_state = _NavState.NAVIGATING
                log.info("Step %d | PATH FOUND (conf=%.2f) — resuming", step, confidence)
                waypoints, vel, radius = self._run_inference(pil, current_jpeg)
                if rover_ctrl and not state.paused.is_set():
                    rover_ctrl.drive_raw(vel, radius)
            else:
                log.info("Step %d | path_lost (conf=%.2f)", step, confidence)

        with self._state_lock:
            goal_status = {
                _NavState.NAVIGATING:   "in_progress",
                _NavState.PATH_LOST:    "path_lost",
                _NavState.INITIALIZING: "initializing",
            }[self._nav_state]

        elapsed = time.time() - t0
        log.info("Step %d | state=%s conf=%.2f vel=%d | %.2fs",
                 step, self._nav_state.name, confidence, vel, elapsed)

        if waypoints is not None:
            annotated = _annotate(frame, waypoints, vel, radius, self._goal)
        else:
            annotated = frame.copy()
        _draw_qwen_hud(annotated, det, self._path_threshold, self._nav_state)
        with state.llm_lock:
            state.llm_frame = annotated

        self._write_result(state, step, phase, waypoints, vel, radius,
                           det, goal_status, elapsed)

    def _write_result(self, state, step, phase, waypoints, vel, radius,
                      det, goal_status, elapsed):
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
            "reasoning": (
                f"visible={det['visible']} conf={det['confidence']:.2f} | "
                f"{det['reason']} | vel={vel}mm/s {r_str} | goal='{self._goal}'"
            ),
            "waypoints":  ui_waypoints,
            "confidence": round(det["confidence"], 2),
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


# ── HUD annotation ─────────────────────────────────────────────────────────────

def _draw_qwen_hud(frame: np.ndarray, det: dict, threshold: float,
                   nav_state: _NavState) -> None:
    """Draw Qwen detection result onto frame in-place."""
    confidence = det["confidence"]
    reason     = det["reason"]

    if nav_state == _NavState.NAVIGATING:
        color = (0, 220, 80)
        label = f"visible: {confidence*100:.0f}%"
    elif nav_state == _NavState.PATH_LOST:
        color = (0, 60, 220)
        label = f"NOT VISIBLE  {confidence*100:.0f}%"
    else:
        color = (180, 180, 180)
        label = "initializing..."

    cv2.putText(frame, label, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Qwen reason text (truncated to fit frame width)
    if reason:
        cv2.putText(frame, reason[:60], (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    # Confidence bar with threshold marker
    bar_x, bar_y, bar_w, bar_h = 10, 80, 120, 8
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    fill = int(bar_w * max(0.0, min(1.0, confidence)))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)
    thr_x = bar_x + int(bar_w * threshold)
    cv2.line(frame, (thr_x, bar_y - 2), (thr_x, bar_y + bar_h + 2),
             (255, 255, 255), 1)
