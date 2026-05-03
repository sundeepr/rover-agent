#!/usr/bin/env python3
"""
omnivla_cloud_server.py — WebSocket inference server for full OmniVLA (cloud GPU).

Loads the full OmniVLA model (VLA backbone + pose projector + action head) and
serves waypoint predictions to rover_agent (cloud_omnivla_strategy.py) over a
persistent WebSocket connection.

Unlike OmniVLA-edge, the full model takes a SINGLE current frame (no rolling
context window) plus a language goal.  A black dummy image is used as the goal
image when running in language-only mode (modality 7).

The output waypoint format is identical to OmniVLA-edge:
    8 × [dx, dy, cos_heading, sin_heading]  in 0.1 m units, robot frame.

Setup (on the cloud instance)
──────────────────────────────
    # Clone OmniVLA repo (provides the prismatic library)
    git clone https://github.com/NHirose/OmniVLA
    cd OmniVLA && pip install -e .   # installs prismatic

    # Download model weights
    git clone https://huggingface.co/NHirose/omnivla-original   # or omnivla-finetuned-cast

    # Start server
    python omnivla_cloud_server.py --model-path ./omnivla-original --host 0.0.0.0 --port 8765

Protocol (UTF-8 JSON over WebSocket)
─────────────────────────────────────
Client → Server

  {"type": "goal",  "goal": "<text>"}
      Update the navigation goal without running inference.  Sent on
      connect and whenever the user changes the goal via the web UI.

  {"type": "infer", "goal": "<text>", "frame_b64": "<base64 JPEG>"}
      Run one inference step with the given frame and goal.

Server → Client

  {"type": "ready"}
      Model loaded; accepting "infer" requests.

  {"type": "waypoints",
   "waypoints": [[dx, dy, cos_h, sin_h], …],   # 8 × 4  (0.1 m units)
   "elapsed": <float>}

  {"type": "error", "message": "<text>"}

On the rover:
    python rover_agent.py --strategy cloud_omnivla \\
        --cloud-server ws://<cloud-ip>:8765 \\
        --goal "Follow the crop row" --interval 1.0
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import sys
import time
from pathlib import Path

log = logging.getLogger("omnivla_cloud_server")

# OmniVLA constants (mirrors run_omnivla.py / omnivla_strategy.py)
NUM_IMAGES_IN_INPUT    = 2       # current frame + goal image
NUM_ACTIONS_CHUNK      = 8       # waypoints predicted per step
ACTION_DIM             = 4       # [dx, dy, cos_heading, sin_heading]
POSE_DIM               = 4       # goal pose vector size
MODALITY_LANG          = 7       # language-only navigation
METRIC_WAYPOINT_SPACING = 0.1    # model units → metres
GOAL_IMAGE_SIZE        = (224, 224)


# ── Inference engine ──────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Wraps the full OmniVLA model for single-frame waypoint prediction.

    The prismatic library (from the OmniVLA repo) must be importable.
    Add the repo root to sys.path before starting the server if needed:
        python omnivla_cloud_server.py --omnivla-repo /path/to/OmniVLA ...
    """

    def __init__(self, model_path: str, unnorm_key: str = "bridge_orig"):
        self._model_path  = Path(model_path)
        self._unnorm_key  = unnorm_key
        self._vla         = None
        self._processor   = None
        self._pose_proj   = None
        self._action_head = None
        self._action_tok  = None
        self._device      = None
        self._num_patches = 0
        self._black_goal_tensor = None   # cached dummy black goal image tensor
        self._goal_cache: dict = {}      # goal str → tokenized human-turn token ids

    def load(self) -> None:
        """Load all model components.  Call once before starting the server."""
        import torch
        import numpy as np
        from PIL import Image as PIL_Image

        # prismatic imports — these live in the OmniVLA repo
        from transformers import (AutoConfig, AutoProcessor,
                                   AutoModelForVision2Seq, AutoImageProcessor)
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from prismatic.models.projectors import ProprioProjector
        from prismatic.models.action_heads import L1RegressionActionHead_idcat
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.processing_prismatic import (
            PrismaticImageProcessor, PrismaticProcessor)

        # Register custom HuggingFace classes
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        log.info("Loading full OmniVLA on %s …", device)

        model_path = str(self._model_path)

        # ── VLA backbone ──────────────────────────────────────────────────────
        self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self._vla = AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(device)
        self._vla.vision_backbone.set_num_images_in_input(NUM_IMAGES_IN_INPUT)
        self._vla.eval()
        log.info("VLA backbone loaded")

        # ── Pose projector ────────────────────────────────────────────────────
        ckpt_pose = _find_checkpoint(self._model_path, "proprio_projector")
        self._pose_proj = ProprioProjector(
            llm_dim=self._vla.llm_dim, proprio_dim=POSE_DIM
        ).to(device)
        pose_ckpt = torch.load(ckpt_pose, map_location=device)
        pose_ckpt = {k[7:] if k.startswith("module.") else k: v for k, v in pose_ckpt.items()}
        self._pose_proj.load_state_dict(pose_ckpt)
        self._pose_proj.eval()
        log.info("Pose projector loaded from %s", ckpt_pose.name)

        # ── Action head ───────────────────────────────────────────────────────
        ckpt_head = _find_checkpoint(self._model_path, "action_head")
        self._action_head = L1RegressionActionHead_idcat(
            input_dim=self._vla.llm_dim,
            hidden_dim=self._vla.llm_dim,
            action_dim=ACTION_DIM,
        ).to(torch.bfloat16).to(device)
        head_ckpt = torch.load(ckpt_head, map_location=device)
        head_ckpt = {k[7:] if k.startswith("module.") else k: v for k, v in head_ckpt.items()}
        self._action_head.load_state_dict(head_ckpt)
        self._action_head.eval()
        log.info("Action head loaded from %s", ckpt_head.name)

        # ── Action tokenizer ──────────────────────────────────────────────────
        self._action_tok = ActionTokenizer(self._processor.tokenizer)

        # ── num_patches: patches × images + 1 (goal pose token) ──────────────
        self._num_patches = (
            self._vla.vision_backbone.get_num_patches()
            * self._vla.vision_backbone.get_num_images_in_input()
            + 1
        )

        # ── Cache black dummy goal image ──────────────────────────────────────
        black_pil = PIL_Image.new("RGB", GOAL_IMAGE_SIZE, (0, 0, 0))
        self._black_goal_tensor = (
            self._processor.image_processor.apply_transform(black_pil)
            .unsqueeze(0)
        )

        log.info("InferenceEngine ready — num_patches=%d", self._num_patches)

    def infer(self, frame_jpeg: bytes, goal: str) -> dict:
        """
        Run one inference step (blocking — call from a thread executor).

        frame_jpeg : current camera frame as JPEG bytes.
        goal       : language navigation goal.

        Returns {"waypoints": list[list[float]] (8×4), "elapsed": float}.
        """
        import torch
        import numpy as np
        from PIL import Image as PIL_Image
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder

        import numpy as np
        from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask

        t0 = time.time()

        # ── Build batch (mirrors run_omnivla.data_transformer_omnivla) ────────
        cur_pil    = PIL_Image.open(io.BytesIO(frame_jpeg)).convert("RGB")
        cur_tensor = self._processor.image_processor.apply_transform(cur_pil).unsqueeze(0)
        pixel_values = torch.cat(
            [cur_tensor, self._black_goal_tensor], dim=1
        ).to(torch.bfloat16).to(self._device)

        if goal not in self._goal_cache:
            self._goal_cache[goal] = self._build_input_ids(goal)
        input_ids, labels = self._goal_cache[goal]
        input_ids  = input_ids.to(self._device)
        labels     = labels.to(self._device)
        attention_mask = input_ids.ne(self._processor.tokenizer.pad_token_id)

        goal_pose   = torch.zeros(1, POSE_DIM, dtype=torch.bfloat16, device=self._device)
        modality_id = torch.as_tensor([MODALITY_LANG], dtype=torch.float32).to(torch.bfloat16).to(self._device)

        # ── Forward pass (matches run_forward_pass exactly) ───────────────────
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = self._vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                modality_id=modality_id,
                labels=labels,
                output_hidden_states=True,
                proprio=goal_pose,
                proprio_projector=self._pose_proj,
            )

        # ── Extract action hidden states (mirrors run_forward_pass) ───────────
        ground_truth_token_ids = labels[:, 1:].to(self._device)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask   = get_next_actions_mask(ground_truth_token_ids)

        last_hidden_states   = output.hidden_states[-1]
        text_hidden_states   = last_hidden_states[:, self._num_patches:-1]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(1, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )

        with torch.no_grad():
            predicted = self._action_head.predict_action(actions_hidden_states, modality_id)

        waypoints = predicted.float().cpu().numpy().reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        return {"waypoints": waypoints.tolist(), "elapsed": round(time.time() - t0, 3)}

    def _build_input_ids(self, goal: str):
        """Build (input_ids, labels) matching transform_datatype in run_omnivla.py."""
        import torch
        import numpy as np
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder

        IGNORE_INDEX = -100

        # Build action string from dummy actions
        dummy_actions = np.zeros((NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)
        current_action_string  = self._action_tok(dummy_actions[0])
        future_actions_strings = "".join(self._action_tok(dummy_actions[1:]))
        action_chunk_string    = current_action_string + future_actions_strings
        action_chunk_len       = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {goal}?"},
            {"from": "gpt",   "value": action_chunk_string},
        ]
        pb = PurePromptBuilder("openvla")
        for turn in conversation:
            pb.add_turn(turn["from"], turn["value"])

        input_ids = torch.tensor(
            self._processor.tokenizer(
                pb.get_prompt(), add_special_tokens=True
            ).input_ids
        ).unsqueeze(0)  # [1, seq_len]

        labels = input_ids.clone()
        labels[:, :-(action_chunk_len + 1)] = IGNORE_INDEX

        return input_ids, labels


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_checkpoint(model_dir: Path, prefix: str) -> Path:
    """Return the checkpoint file matching prefix inside model_dir."""
    candidates = sorted(model_dir.glob(f"{prefix}--*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint matching '{prefix}--*.pt' in {model_dir}"
        )
    # Pick highest step number if multiple exist
    return max(candidates, key=lambda p: int(p.stem.split("--")[1].split("_")[0]))


# ── Per-connection session ─────────────────────────────────────────────────────

class ConnectionSession:
    """Manages goal text for one WebSocket connection."""

    def __init__(self, engine: InferenceEngine, loop: asyncio.AbstractEventLoop):
        self._engine = engine
        self._loop   = loop
        self._goal   = ""

    async def handle(self, websocket) -> None:
        addr = getattr(websocket, "remote_address", "?")
        log.info("Client connected: %s", addr)
        await websocket.send(json.dumps({"type": "ready"}))

        try:
            async for raw in websocket:
                await self._dispatch(websocket, raw)
        except Exception as e:
            log.info("Client %s disconnected: %s", addr, e)

    async def _dispatch(self, websocket, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send(json.dumps(
                {"type": "error", "message": "invalid JSON"}))
            return

        mtype = msg.get("type")

        if mtype == "goal":
            self._goal = msg.get("goal", "")
            log.info("Goal updated: '%s'", self._goal)

        elif mtype == "infer":
            if msg.get("goal"):
                self._goal = msg["goal"]
            if not self._goal:
                await websocket.send(json.dumps(
                    {"type": "error",
                     "message": "no goal — send a 'goal' message first"}))
                return
            frame_b64 = msg.get("frame_b64", "")
            if not frame_b64:
                await websocket.send(json.dumps(
                    {"type": "error", "message": "missing frame_b64"}))
                return

            frame_bytes = base64.b64decode(frame_b64)
            goal = self._goal
            try:
                result = await self._loop.run_in_executor(
                    None, self._engine.infer, frame_bytes, goal
                )
                await websocket.send(json.dumps({"type": "waypoints", **result}))
                log.info("Infer OK  goal='%s'  elapsed=%.2fs", goal, result["elapsed"])
            except Exception as e:
                log.error("Inference error: %s", e, exc_info=True)
                await websocket.send(json.dumps(
                    {"type": "error", "message": str(e)}))

        else:
            await websocket.send(json.dumps(
                {"type": "error", "message": f"unknown type: {mtype!r}"}))


# ── Server entry point ────────────────────────────────────────────────────────

async def _serve(engine: InferenceEngine, host: str, port: int) -> None:
    import websockets

    loop = asyncio.get_running_loop()

    async def _handler(ws):
        session = ConnectionSession(engine, loop)
        await session.handle(ws)

    async with websockets.serve(_handler, host, port):
        log.info("WebSocket server listening on ws://%s:%d", host, port)
        log.info("Rover command:  --strategy cloud_omnivla --cloud-server ws://%s:%d",
                 host, port)
        await asyncio.Future()   # run until cancelled


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Full OmniVLA cloud WebSocket inference server")
    parser.add_argument("--model-path", required=True, metavar="DIR",
                        help="Path to OmniVLA model directory "
                             "(e.g. ./omnivla-original or ./omnivla-finetuned-cast)")
    parser.add_argument("--host",    default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port",    default=8765, type=int,
                        help="WebSocket port (default: 8765)")
    parser.add_argument("--omnivla-repo", default=None, metavar="DIR",
                        help="Path to cloned OmniVLA repo root — prepended to "
                             "sys.path so 'prismatic' is importable. "
                             "Not needed if you ran 'pip install -e .' in the repo.")
    parser.add_argument("--unnorm-key", default="bridge_orig",
                        help="Dataset key for action un-normalization stats. "
                             "Run without this flag to see available keys. "
                             "(default: bridge_orig)")
    args = parser.parse_args()

    if args.omnivla_repo:
        sys.path.insert(0, args.omnivla_repo)
        log.info("Added to sys.path: %s", args.omnivla_repo)

    engine = InferenceEngine(model_path=args.model_path, unnorm_key=args.unnorm_key)
    engine.load()

    asyncio.run(_serve(engine, args.host, args.port))


if __name__ == "__main__":
    main()
