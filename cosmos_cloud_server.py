#!/usr/bin/env python3
"""
cosmos_cloud_server.py — WebSocket inference server for Cosmos3-Edge (cloud GPU).

Supports four operating modes selected at startup via --mode:

  reasoning_supervisor  (Option 1)
      Receives a camera frame + goal.  Returns a short structured reasoning
      response: drift direction, drift_mm estimate, row_end flag, and a
      free-text observation.  Designed to supervise a fast local strategy.

  reasoning_driver  (Option 2)
      Receives a camera frame + goal.  Returns a vel + radius drive command
      derived from Cosmos's reasoning about the scene.  Cosmos drives the
      robot directly at low frequency (~0.2 Hz).

  av_policy  (Option 4)
      Receives a short video clip (last N frames) + goal.  Runs
      CosmosActionCondition(mode="policy", domain_name="av") and returns
      a chunk of 16 × 9D actions.  The rover-side strategy maps the
      relevant dimensions to [vel, radius].

  trajectory_ranking  (Option 6)
      Receives a camera frame + goal.  Samples the policy N times
      (default 5), scores each by the model's value prediction, and returns
      all trajectories ranked by score so the rover can pick the best one
      and display all candidates on the UI.

Setup (cloud GPU — H100 / A100 / B200 recommended)
────────────────────────────────────────────────────
    pip install diffusers transformers torch accelerate websockets

    # reasoning modes:
    python cosmos_cloud_server.py \\
        --mode reasoning_supervisor \\
        --model-path nvidia/Cosmos3-Edge \\
        --host 0.0.0.0 --port 8767

    # av policy:
    python cosmos_cloud_server.py \\
        --mode av_policy \\
        --model-path nvidia/Cosmos3-Edge \\
        --host 0.0.0.0 --port 8767

    # trajectory ranking:
    python cosmos_cloud_server.py \\
        --mode trajectory_ranking \\
        --model-path nvidia/Cosmos3-Edge \\
        --num-samples 5 \\
        --host 0.0.0.0 --port 8767

Protocol (UTF-8 JSON over WebSocket)
──────────────────────────────────────
Client → Server

  {"type": "goal",  "goal": "<text>"}
      Update goal without running inference.

  {"type": "infer",
   "goal": "<text>",
   "frame_b64": "<base64 JPEG>"}          ← reasoning_supervisor / reasoning_driver / trajectory_ranking
      OR
  {"type": "infer",
   "goal": "<text>",
   "frames_b64": ["<b64>", …]}            ← av_policy (list of JPEGs, newest last)

Server → Client

  {"type": "ready", "mode": "<mode>"}

  -- reasoning_supervisor --
  {"type": "supervision",
   "drift":      "left"|"right"|"center",
   "drift_mm":   <int>,
   "row_end":    true|false,
   "observation": "<text>",
   "elapsed":    <float>}

  -- reasoning_driver --
  {"type": "drive",
   "velocity":   <int mm/s>,
   "radius":     <int mm>,   (32767 = straight)
   "reasoning":  "<text>",
   "elapsed":    <float>}

  -- av_policy --
  {"type": "actions",
   "actions":   [[…9 floats…], …],   # 16 × 9
   "elapsed":   <float>}

  -- trajectory_ranking --
  {"type": "trajectories",
   "trajectories": [
     {"rank": 1, "score": <float>, "actions": [[…], …], "description": "<text>"},
     …
   ],
   "elapsed": <float>}

  -- all modes --
  {"type": "error", "message": "<text>"}

On the rover:
    python rover_agent.py --strategy cosmos_supervisor  --cosmos-server ws://<ip>:8767
    python rover_agent.py --strategy cosmos_driver      --cosmos-server ws://<ip>:8767
    python rover_agent.py --strategy cosmos_av          --cosmos-server ws://<ip>:8767
    python rover_agent.py --strategy cosmos_trajectory  --cosmos-server ws://<ip>:8767
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

log = logging.getLogger("cosmos_cloud_server")

# ── Shared helpers ─────────────────────────────────────────────────────────────

def _decode_jpeg(b64: str):
    """Decode a base64 JPEG string to a PIL Image."""
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _load_pipeline(model_path: str):
    """Load Cosmos3OmniPipeline in bfloat16 on CUDA."""
    import torch
    from diffusers import Cosmos3OmniPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    log.info("Loading Cosmos3OmniPipeline from %s …", model_path)
    t0 = time.time()
    pipe = Cosmos3OmniPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
    )
    log.info("Pipeline loaded in %.1fs", time.time() - t0)
    return pipe


# ── Mode 1 & 2: Reasoning engine ──────────────────────────────────────────────
#
# Cosmos3OmniPipeline is a VIDEO DIFFUSION model — it generates pixels, not
# text.  There is no text output path.
#
# For navigation reasoning we use CosmosActionCondition(mode="policy") to get
# the model's action prediction from the current frame, then interpret the
# action dimensions as navigation signals:
#
#   AV 9-D action layout (inferred from Cosmos3 AV domain):
#     [0] forward velocity  (positive = forward)
#     [1] lateral offset    (positive = right of path, negative = left)
#     [2] yaw rate          (positive = turn right)
#     [3–8] other dims (height, orientation, etc.) — not used here
#
# reasoning_supervisor  →  interprets lateral offset as drift direction
# reasoning_driver      →  maps forward + yaw directly to vel + radius


class ReasoningEngine:
    """
    Options 1 & 2: Navigation reasoning via Cosmos3OmniPipeline action prediction.

    Cosmos3 is a video diffusion model — it generates pixels, not text.
    We use CosmosActionCondition(mode="policy", domain_name="av") to get the
    model's predicted action from the current camera frame, then interpret
    the action dimensions as navigation signals:

      AV 9-D action (inferred layout):
        [0] forward component  (higher = more forward motion predicted)
        [1] lateral component  (positive = rightward, negative = leftward)
        [2] yaw component      (positive = turn right)

    reasoning_supervisor → drift direction derived from lateral component
    reasoning_driver     → vel + radius derived from forward + yaw components
    """

    # Thresholds for lateral drift classification
    _LAT_DEADBAND  = 0.05   # |lat| < this → center
    _LAT_LARGE     = 0.20   # |lat| > this → hard correction

    # Scale yaw → Roomba radius (mm).  Tune by observation.
    _YAW_TO_RADIUS = 800.0  # radius = _YAW_TO_RADIUS / |yaw|  (clamped)
    _MIN_RADIUS    = 200
    _MAX_RADIUS    = 3000
    _STRAIGHT      = 32767

    def __init__(self, model_path: str, mode: str):
        self._model_path = model_path
        self._mode       = mode   # "reasoning_supervisor" | "reasoning_driver"
        self._pipe       = None

    def load(self) -> None:
        self._pipe = _load_pipeline(self._model_path)
        log.info("ReasoningEngine ready (mode=%s)", self._mode)

    def infer(self, frame_jpeg: bytes, goal: str) -> dict:
        """
        Run one inference step.  Returns a supervision or drive dict.
        """
        import torch
        from PIL import Image
        from diffusers import CosmosActionCondition

        t0    = time.time()
        image = Image.open(io.BytesIO(frame_jpeg)).convert("RGB")

        result = self._pipe(
            prompt=goal,
            action=CosmosActionCondition(
                mode="policy",
                chunk_size=8,           # only need first few steps for direction
                domain_name="av",
                resolution_tier=480,
                image=image,
                view_point="ego_view",
            ),
            num_frames=1,
            height=480,
            width=832,
            num_inference_steps=20,    # fewer steps → faster (~8–12s on H100)
            guidance_scale=1.0,
            use_system_prompt=False,
            enable_safety_check=False,
        )

        elapsed = round(time.time() - t0, 3)

        # Extract action chunk — shape [chunk_size, 9] or None
        actions = result.action[0].tolist() if (
            result.action is not None and len(result.action) > 0
        ) else []

        log.info("ReasoningEngine inference (%.2fs)  actions[0]=%s",
                 elapsed, actions[0] if actions else "none")

        if self._mode == "reasoning_supervisor":
            return self._to_supervision(actions, elapsed)
        else:
            return self._to_drive(actions, elapsed)

    def _to_supervision(self, actions: list, elapsed: float) -> dict:
        """Convert action chunk to supervision signal."""
        if not actions:
            return {
                "type": "supervision", "drift": "center", "drift_mm": 0,
                "row_end": False, "observation": "no action predicted",
                "elapsed": elapsed,
            }

        # Average lateral over first 4 steps for stability
        n   = min(4, len(actions))
        lat = sum(float(a[1]) for a in actions[:n] if len(a) > 1) / n
        fwd = float(actions[0][0]) if len(actions[0]) > 0 else 1.0

        if abs(lat) < self._LAT_DEADBAND:
            drift = "center"
        elif lat < 0:
            drift = "left"
        else:
            drift = "right"

        drift_mm  = int(lat * 500)   # rough mm estimate (tune as needed)
        row_end   = fwd < 0.05       # very low forward motion → path ending
        obs       = (f"drift={drift} lat={lat:.3f} fwd={fwd:.3f} "
                     f"({'row end' if row_end else 'path clear'})")

        return {
            "type":        "supervision",
            "drift":       drift,
            "drift_mm":    drift_mm,
            "row_end":     row_end,
            "observation": obs,
            "elapsed":     elapsed,
        }

    def _to_drive(self, actions: list, elapsed: float) -> dict:
        """Convert action chunk to vel + radius drive command."""
        if not actions:
            return {
                "type": "drive", "velocity": 0, "radius": self._STRAIGHT,
                "reasoning": "no action predicted", "elapsed": elapsed,
            }

        fwd = float(actions[0][0]) if len(actions[0]) > 0 else 0.0
        yaw = float(actions[0][2]) if len(actions[0]) > 2 else 0.0

        # Map forward component → velocity (0–200 mm/s)
        vel = int(max(0, min(200, fwd * 200.0)))

        # Map yaw → radius
        if abs(yaw) < 0.02:
            radius = self._STRAIGHT
        else:
            r = int(self._YAW_TO_RADIUS / abs(yaw))
            r = max(self._MIN_RADIUS, min(self._MAX_RADIUS, r))
            radius = r if yaw < 0 else -r   # negative yaw = left turn

        reasoning = (f"cosmos_driver: fwd={fwd:.3f} yaw={yaw:.3f} "
                     f"→ vel={vel} radius={radius}")

        return {
            "type":      "drive",
            "velocity":  vel,
            "radius":    radius,
            "reasoning": reasoning,
            "elapsed":   elapsed,
        }


# ── Mode 4: AV Policy engine ───────────────────────────────────────────────────

class AvPolicyEngine:
    """
    Uses CosmosActionCondition(mode='policy', domain_name='av') to predict
    a 16-step action chunk from a short video clip.
    """

    # How many conditioning frames to pass (keep small for latency)
    NUM_COND_FRAMES = 5

    def __init__(self, model_path: str, chunk_size: int = 16):
        self._model_path = model_path
        self._chunk_size = chunk_size
        self._pipe       = None

    def load(self) -> None:
        self._pipe = _load_pipeline(self._model_path)
        log.info("AvPolicyEngine ready (chunk_size=%d)", self._chunk_size)

    def infer(self, frames_jpeg: list[bytes], goal: str) -> dict:
        """
        frames_jpeg : list of JPEG bytes, newest last, up to NUM_COND_FRAMES used.
        Returns {"type": "actions", "actions": [[…9…], …16…], "elapsed": float}.
        """
        import torch
        from PIL import Image
        from diffusers import CosmosActionCondition

        t0 = time.time()

        # Build PIL frame list (use last NUM_COND_FRAMES)
        pil_frames = [
            Image.open(io.BytesIO(j)).convert("RGB")
            for j in frames_jpeg[-self.NUM_COND_FRAMES:]
        ]

        result = self._pipe(
            prompt=goal,
            action=CosmosActionCondition(
                mode="policy",
                chunk_size=self._chunk_size,
                domain_name="av",
                resolution_tier=480,
                video=pil_frames,
                view_point="ego_view",
            ),
            fps=5,
            num_inference_steps=30,
            guidance_scale=1.0,
            use_system_prompt=False,
            enable_safety_check=False,
        )

        actions = result.action[0].tolist() if result.action is not None else []
        return {
            "type":    "actions",
            "actions": actions,   # list of 16 × 9 floats
            "elapsed": round(time.time() - t0, 3),
        }


# ── Mode 6: Trajectory ranking engine ─────────────────────────────────────────

class TrajectoryRankingEngine:
    """
    Samples the policy num_samples times, scores each trajectory by its
    predicted value, and returns all candidates ranked best-first.
    """

    def __init__(self, model_path: str, num_samples: int = 5, chunk_size: int = 16):
        self._model_path  = model_path
        self._num_samples = num_samples
        self._chunk_size  = chunk_size
        self._pipe        = None

    def load(self) -> None:
        self._pipe = _load_pipeline(self._model_path)
        log.info("TrajectoryRankingEngine ready (num_samples=%d)", self._num_samples)

    def infer(self, frame_jpeg: bytes, goal: str) -> dict:
        """
        Returns {"type": "trajectories", "trajectories": [...ranked...], "elapsed": float}.
        Each trajectory: {"rank", "score", "actions": [[9-float], …16], "description"}.
        """
        import torch
        from PIL import Image
        from diffusers import CosmosActionCondition

        t0    = time.time()
        image = Image.open(io.BytesIO(frame_jpeg)).convert("RGB")

        candidates = []
        for i in range(self._num_samples):
            try:
                result = self._pipe(
                    prompt=goal,
                    action=CosmosActionCondition(
                        mode="policy",
                        chunk_size=self._chunk_size,
                        domain_name="av",
                        resolution_tier=480,
                        image=image,
                        view_point="ego_view",
                    ),
                    fps=5,
                    num_inference_steps=30,
                    guidance_scale=1.0,
                    use_system_prompt=False,
                    enable_safety_check=False,
                )
                actions = result.action[0].tolist() if result.action is not None else []

                # Score: use the mean forward component of first few actions
                # as a proxy for value when no explicit value head is available.
                # Replace with result.value when/if diffusers exposes it.
                score = _score_trajectory(actions, goal)
                candidates.append({"actions": actions, "score": score})
                log.debug("Sample %d/%d  score=%.3f", i + 1, self._num_samples, score)
            except Exception as e:
                log.warning("Sample %d failed: %s", i + 1, e)

        # Rank by score descending
        candidates.sort(key=lambda c: c["score"], reverse=True)

        trajectories = [
            {
                "rank":        rank + 1,
                "score":       round(c["score"], 4),
                "actions":     c["actions"],
                "description": _describe_trajectory(c["actions"], rank),
            }
            for rank, c in enumerate(candidates)
        ]

        return {
            "type":         "trajectories",
            "trajectories": trajectories,
            "elapsed":      round(time.time() - t0, 3),
        }


def _score_trajectory(actions: list, goal: str) -> float:
    """
    Heuristic score for an action chunk from the 'av' domain.
    AV 9D likely encodes [x, y, z, qx, qy, qz, qw, vel, steering] or similar.
    We use forward motion (dim 0) minus abs(lateral offset, dim 1) as a proxy.
    Replace with model value head output when available.
    """
    if not actions:
        return 0.0
    import math
    score = 0.0
    for a in actions[:8]:   # score on first 8 steps
        fwd  = float(a[0]) if len(a) > 0 else 0.0
        lat  = float(a[1]) if len(a) > 1 else 0.0
        score += fwd - 0.5 * abs(lat)
    return score / min(8, len(actions))


def _describe_trajectory(actions: list, rank: int) -> str:
    """Generate a human-readable one-liner for a trajectory."""
    if not actions:
        return "no actions"
    fwd = float(actions[0][0]) if actions[0] else 0.0
    lat = float(actions[0][1]) if len(actions[0]) > 1 else 0.0
    direction = "straight" if abs(lat) < 0.05 else ("left" if lat < 0 else "right")
    return f"rank {rank+1}: {direction} fwd={fwd:.2f} lat={lat:.2f}"


# ── Per-connection session ─────────────────────────────────────────────────────

class ConnectionSession:

    def __init__(self, engine, mode: str, loop: asyncio.AbstractEventLoop):
        self._engine = engine
        self._mode   = mode
        self._loop   = loop
        self._goal   = ""

    async def handle(self, websocket) -> None:
        addr = getattr(websocket, "remote_address", "?")
        log.info("Client connected: %s", addr)
        await websocket.send(json.dumps({"type": "ready", "mode": self._mode}))
        try:
            async for raw in websocket:
                await self._dispatch(websocket, raw)
        except Exception as e:
            log.info("Client %s disconnected: %s", addr, e)

    async def _dispatch(self, websocket, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send(json.dumps({"type": "error", "message": "invalid JSON"}))
            return

        mtype = msg.get("type")

        if mtype == "goal":
            self._goal = msg.get("goal", "")
            log.info("Goal updated: '%s'", self._goal)
            return

        if mtype != "infer":
            await websocket.send(json.dumps(
                {"type": "error", "message": f"unknown type: {mtype!r}"}))
            return

        if msg.get("goal"):
            self._goal = msg["goal"]
        if not self._goal:
            await websocket.send(json.dumps(
                {"type": "error", "message": "no goal set — send a 'goal' message first"}))
            return

        goal = self._goal

        # av_policy accepts frames_b64 list; others accept frame_b64 single frame
        if self._mode == "av_policy":
            frames_b64 = msg.get("frames_b64", [])
            if not frames_b64:
                # fall back to single frame
                fb = msg.get("frame_b64", "")
                frames_b64 = [fb] if fb else []
            if not frames_b64:
                await websocket.send(json.dumps(
                    {"type": "error", "message": "missing frames_b64"}))
                return
            frames_jpeg = [base64.b64decode(f) for f in frames_b64]
            try:
                result = await self._loop.run_in_executor(
                    None, self._engine.infer, frames_jpeg, goal)
            except Exception as e:
                log.error("Inference error: %s", e, exc_info=True)
                await websocket.send(json.dumps({"type": "error", "message": str(e)}))
                return
        else:
            frame_b64 = msg.get("frame_b64", "")
            if not frame_b64:
                await websocket.send(json.dumps(
                    {"type": "error", "message": "missing frame_b64"}))
                return
            frame_jpeg = base64.b64decode(frame_b64)
            try:
                result = await self._loop.run_in_executor(
                    None, self._engine.infer, frame_jpeg, goal)
            except Exception as e:
                log.error("Inference error: %s", e, exc_info=True)
                await websocket.send(json.dumps({"type": "error", "message": str(e)}))
                return

        await websocket.send(json.dumps(result))
        log.info("Infer OK  mode=%s  goal='%s'  elapsed=%.2fs",
                 self._mode, goal, result.get("elapsed", 0))


# ── Server entry point ─────────────────────────────────────────────────────────

async def _serve(engine, mode: str, host: str, port: int) -> None:
    import websockets

    loop = asyncio.get_running_loop()

    async def _handler(ws):
        session = ConnectionSession(engine, mode, loop)
        await session.handle(ws)

    async with websockets.serve(_handler, host, port):
        log.info("Cosmos cloud server listening on ws://%s:%d  mode=%s", host, port, mode)
        await asyncio.Future()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Cosmos3-Edge cloud WebSocket inference server")
    parser.add_argument("--mode", required=True,
                        choices=["reasoning_supervisor", "reasoning_driver",
                                 "av_policy", "trajectory_ranking"],
                        help="Operating mode")
    parser.add_argument("--model-path", default="nvidia/Cosmos3-Edge",
                        help="HF repo id or local path to Cosmos3-Edge weights "
                             "(default: nvidia/Cosmos3-Edge)")
    parser.add_argument("--host",        default="0.0.0.0")
    parser.add_argument("--port",        default=8767, type=int)
    parser.add_argument("--num-samples",    default=5,   type=int,
                        help="Trajectory samples for trajectory_ranking mode (default 5)")
    parser.add_argument("--chunk-size",     default=16,  type=int,
                        help="Action chunk size for policy modes (default 16)")
    parser.add_argument("--max-new-tokens", default=128, type=int,
                        help="Max tokens to generate in reasoning modes (default 128)")
    args = parser.parse_args()

    # Instantiate the right engine
    if args.mode in ("reasoning_supervisor", "reasoning_driver"):
        engine = ReasoningEngine(args.model_path, args.mode,
                                 max_new_tokens=args.max_new_tokens)
    elif args.mode == "av_policy":
        engine = AvPolicyEngine(args.model_path, chunk_size=args.chunk_size)
    elif args.mode == "trajectory_ranking":
        engine = TrajectoryRankingEngine(
            args.model_path,
            num_samples=args.num_samples,
            chunk_size=args.chunk_size,
        )
    else:
        sys.exit(f"Unknown mode: {args.mode}")

    engine.load()
    asyncio.run(_serve(engine, args.mode, args.host, args.port))


if __name__ == "__main__":
    main()
