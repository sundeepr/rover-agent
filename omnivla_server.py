#!/usr/bin/env python3
"""
OmniVLA model server — run once, keep alive between rover_agent sessions.

Loads OmniVLA-edge weights into memory once and serves inference requests
over TCP using Python's built-in multiprocessing.managers (no extra deps).

The goal text and optional goal image are sent by the client on every
request. CLIP text encoding is cached per unique goal string so a repeated
goal costs nothing after the first call.

Usage:
    python omnivla_server.py
    python omnivla_server.py --host 127.0.0.1 --port 5100

Then start rover_agent with:
    python rover_agent.py --strategy omnivla --omnivla-server localhost:5100 \\
        --goal "Follow the brown path" --interval 1.0
"""

import argparse
import io
import logging
import sys
import time
from multiprocessing.managers import BaseManager
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("omnivla_server")

# ── RPC constants (imported by omnivla_strategy when in server mode) ──────────

DEFAULT_HOST    = "127.0.0.1"
DEFAULT_PORT    = 5100
DEFAULT_AUTHKEY = b"omnivla-edge"

# ── OmniVLA constants (mirrors omnivla_strategy.py) ──────────────────────────

CONTEXT_SIZE   = 5
TRAJ_LEN       = 8
IMG_OBS        = (96, 96)
IMG_CLIP       = (224, 224)
IMG_MAP        = (352, 352)
METRIC_SPACING = 0.1
DT             = 1.0
WAYPOINT_IDX   = 4
ENC_SIZE       = 1024
MODALITY_LANG     = 7
MODALITY_GOAL_IMG = 6

_OMNIVLA_DIR = Path(__file__).parent.parent / "omnivla"

# ── Manager (shared definition used by both server and client) ────────────────

class OmniVLAManager(BaseManager):
    pass

# Register at module level so the client side gets the proxy by importing this
# module. "engine" exposes InferenceEngine.infer as a proxied method; calling
# a method on a proxy returns the pickled return value directly (unlike the
# manager creation pattern which wraps the return in another proxy).
OmniVLAManager.register("engine", exposed=["infer"])


# ── Inference engine (lives in the server process) ───────────────────────────

class InferenceEngine:
    """
    Holds the loaded OmniVLA-edge model and serves inference requests.

    Goal text is encoded with CLIP on the first call and cached — switching
    goals between rover_agent restarts costs only a CLIP encode (~200 ms),
    not a full model reload.
    """

    def __init__(self):
        self._model       = None
        self._device      = None
        self._obs_tf      = None
        self._clip_tf     = None
        self._text_encoder = None
        self._dummy_pose  = None
        self._dummy_map   = None

        # CLIP text cache: goal_string → feat_text tensor
        self._text_cache: dict = {}
        # Goal image cache: image bytes → goal_img tensor
        self._img_cache: dict = {}

        self._load()

    def _load(self) -> None:
        import torch
        import torchvision.transforms as T
        import clip as clip_lib
        from huggingface_hub import hf_hub_download

        if str(_OMNIVLA_DIR) not in sys.path:
            sys.path.insert(0, str(_OMNIVLA_DIR))
        from model import OmniVLA_edge

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        log.info("Loading OmniVLA-edge on %s…", device)

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

        log.info("Loading CLIP text encoder…")
        self._text_encoder, _ = clip_lib.load("ViT-B/32", device=device)
        self._text_encoder.eval()
        log.info("CLIP loaded")

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

        log.info("InferenceEngine ready — waiting for requests")

    def _encode_text(self, goal: str):
        """Encode goal text with CLIP; result is cached per unique string."""
        import torch
        import clip as clip_lib
        if goal not in self._text_cache:
            log.info("CLIP: encoding new goal '%s'", goal)
            with torch.no_grad():
                feat = self._text_encoder.encode_text(
                    clip_lib.tokenize([goal], truncate=True).to(self._device)
                ).float()
            self._text_cache[goal] = feat
        return self._text_cache[goal]

    def _encode_goal_image(self, goal_image_bytes: bytes):
        """Encode goal image; result is cached by content hash."""
        import torch
        from PIL import Image as PIL_Image
        key = hash(goal_image_bytes)
        if key not in self._img_cache:
            log.info("Encoding new goal image (%d bytes)", len(goal_image_bytes))
            pil = PIL_Image.open(io.BytesIO(goal_image_bytes)).convert("RGB")
            tensor = self._obs_tf(pil).unsqueeze(0).to(self._device)
            self._img_cache[key] = tensor
        return self._img_cache[key]

    def infer(
        self,
        context_jpegs: list,
        current_jpeg: bytes,
        goal: str,
        goal_image_bytes: bytes | None = None,
    ) -> dict:
        """
        Run one inference step.

        Parameters
        ----------
        context_jpegs : list[bytes]
            Exactly CONTEXT_SIZE+1 JPEG frames, oldest first (client pads
            with the first frame if the context window is not yet full).
        current_jpeg : bytes
            Current frame as JPEG, used for the CLIP FiLM encoder (224×224).
        goal : str
            Language navigation goal.
        goal_image_bytes : bytes | None
            Optional goal image as raw bytes (any PIL-readable format).
            When provided, modality switches to MODALITY_GOAL_IMG (6).

        Returns
        -------
        dict with keys:
            waypoints : list[list[float]]  — 8 × 4 array as nested list
            vel       : int                — mm/s
            radius    : int                — mm (0x8000 = straight)
            elapsed   : float              — inference time in seconds
        """
        import torch
        import math
        import numpy as np
        from PIL import Image as PIL_Image

        t0 = time.time()

        feat_text = self._encode_text(goal)

        if goal_image_bytes:
            goal_img    = self._encode_goal_image(goal_image_bytes)
            modality_id = torch.tensor([MODALITY_GOAL_IMG], device=self._device)
        else:
            goal_img    = torch.zeros(1, 3, *IMG_OBS, device=self._device)
            modality_id = torch.tensor([MODALITY_LANG], device=self._device)

        # Decode context frames
        frames = [
            PIL_Image.open(io.BytesIO(j)).convert("RGB")
            for j in context_jpegs
        ]
        cur_pil = PIL_Image.open(io.BytesIO(current_jpeg)).convert("RGB")

        obs_images = torch.stack([self._obs_tf(f) for f in frames]).unsqueeze(0)
        obs_images = obs_images.view(1, -1, *IMG_OBS).to(self._device)
        cur_large  = self._clip_tf(cur_pil).unsqueeze(0).to(self._device)

        with torch.no_grad():
            actions, _, _ = self._model(
                obs_images, self._dummy_pose, self._dummy_map,
                goal_img, modality_id, feat_text, cur_large,
            )

        waypoints = actions[0].cpu().numpy()  # [8, 4]

        # Convert waypoint to drive command (mirrors omnivla_strategy._waypoint_to_drive)
        wp  = waypoints[WAYPOINT_IDX].copy()
        dx  = float(wp[0]) * METRIC_SPACING
        dy  = float(wp[1]) * METRIC_SPACING
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            vel, radius = 0, 0x8000
        else:
            from omnivla_strategy import MAX_LIN_MM_S, MAX_ANG_RAD_S
            lin_m_s   = float(np.clip(dx / DT, 0.0, MAX_LIN_MM_S / 1000.0))
            ang_rad_s = float(np.clip(math.atan2(dy, dx) / DT, -MAX_ANG_RAD_S, MAX_ANG_RAD_S))
            lin_mm_s  = int(lin_m_s * 1000)
            if abs(ang_rad_s) < 0.01:
                vel, radius = lin_mm_s, 0x8000
            else:
                vel    = lin_mm_s
                radius = int(np.clip(lin_mm_s / ang_rad_s, -2000, 2000))

        elapsed = time.time() - t0
        return {
            "waypoints": waypoints.tolist(),
            "vel":       vel,
            "radius":    radius,
            "elapsed":   elapsed,
        }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OmniVLA model server")
    parser.add_argument("--host",    default=DEFAULT_HOST,
                        help=f"Bind address (default: {DEFAULT_HOST})")
    parser.add_argument("--port",    default=DEFAULT_PORT, type=int,
                        help=f"TCP port (default: {DEFAULT_PORT})")
    parser.add_argument("--authkey", default=DEFAULT_AUTHKEY.decode(),
                        help="Shared secret between server and client")
    args = parser.parse_args()

    authkey = args.authkey.encode()

    engine = InferenceEngine()

    OmniVLAManager.register("engine", callable=lambda: engine, exposed=["infer"])
    manager = OmniVLAManager(address=(args.host, args.port), authkey=authkey)
    server  = manager.get_server()

    log.info("OmniVLA server listening on %s:%d", args.host, args.port)
    log.info("Connect with: --omnivla-server %s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
