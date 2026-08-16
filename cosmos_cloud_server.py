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

Setup (Jetson AGX — 64 GB unified memory, ARM64 + integrated GPU)
──────────────────────────────────────────────────────────────────
    pip install diffusers torch accelerate websockets openai

    # reasoning modes — requires vLLM serving Cosmos3-Edge for text output:
    #
    #   docker pull vllm/vllm-openai:cosmos3
    #   docker run --gpus all -p 8000:8000 vllm/vllm-openai:cosmos3 \\
    #       vllm serve nvidia/Cosmos3-Edge \\
    #       --host 0.0.0.0 --port 8000 \\
    #       --max-model-len 131072 \\
    #       --allowed-local-media-path / \\
    #       --mm-processor-kwargs '{"do_resize":true,"min_pixels":4096,"max_pixels":16777216}'
    #
    python cosmos_cloud_server.py \\
        --mode reasoning_supervisor \\
        --vllm-url http://localhost:8000 \\
        --host 0.0.0.0 --port 8767

    # av policy (uses Cosmos3OmniPipeline + CosmosActionCondition):
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
    """
    Load Cosmos3OmniPipeline for Jetson AGX (ARM64 + integrated GPU).

    Jetson-specific choices:
      - torch.float16 instead of bfloat16 (bfloat16 not supported on Jetson GPU)
      - .to("cuda") instead of device_map="cuda" (device_map needs accelerate
        configured for x86; plain .to() works on Jetson)

    Do NOT override the scheduler — the pipeline's built-in scheduler handles
    both video generation and action policy modes correctly. Overriding with
    UniPCMultistepScheduler breaks action mode (assert this_order > 0 fails).
    """
    import torch
    from diffusers import Cosmos3OmniPipeline

    log.info("Loading Cosmos3OmniPipeline from %s …", model_path)
    t0 = time.time()
    pipe = Cosmos3OmniPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,   # bfloat16 not supported on Jetson GPU
    )
    pipe.to("cuda")                  # device_map= is x86/accelerate specific
    log.info("Pipeline loaded in %.1fs  scheduler=%s",
             time.time() - t0, type(pipe.scheduler).__name__)
    return pipe


# ── Mode 1 & 2: Reasoning engine ──────────────────────────────────────────────
#
# Cosmos3-Edge is an omni-model: it does text, video, image, and action.
# Text output (reasoning) is accessed via vLLM serving Cosmos3-Edge with an
# OpenAI-compatible chat completions API.  The diffusers pipeline handles
# video/action modes (options 4 & 6).
#
# Start vLLM before launching this server in reasoning mode:
#
#   docker pull vllm/vllm-openai:cosmos3
#   docker run --gpus all -p 8000:8000 vllm/vllm-openai:cosmos3 \
#       vllm serve nvidia/Cosmos3-Edge \
#       --host 0.0.0.0 --port 8000 \
#       --max-model-len 131072 \
#       --allowed-local-media-path / \
#       --mm-processor-kwargs '{"do_resize":true,"min_pixels":4096,"max_pixels":16777216}'

_SUPERVISOR_PROMPT = """\
You are a navigation assistant for a Roomba robot with a forward-facing camera.
The robot's goal is: "{goal}"

Look at the camera image and respond with ONLY a JSON object, no other text:
{{"drift": "left"|"right"|"center", "drift_mm": <int, lateral offset mm positive=right>, "row_end": <true|false>, "observation": "<one sentence>"}}"""

_DRIVER_PROMPT = """\
You are the navigation controller for a Roomba robot.
drive_raw(velocity mm/s 0-200, radius mm: 32767=straight, positive=left, negative=right, 1=spin).
The robot's goal is: "{goal}"

Look at the camera image and respond with ONLY a JSON object, no other text:
{{"velocity": <int 0-200>, "radius": <int>, "reasoning": "<one sentence>"}}"""


def _parse_json(text: str) -> dict:
    """Extract the first JSON object from model output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    log.warning("Could not parse JSON from: %r", text[:300])
    return {}


class ReasoningEngine:
    """
    Options 1 & 2: Text reasoning via vLLM serving Cosmos3-Edge.

    Cosmos3-Edge is an omni-model — it outputs text, video, image, and actions.
    Text output is accessed via vLLM's OpenAI-compatible chat completions API
    with multimodal (image + text) messages.

    The vLLM server must be running before this engine is used:
        vllm serve nvidia/Cosmos3-Edge --port 8000 ...
    """

    def __init__(self, mode: str, vllm_url: str = "http://localhost:8000",
                 max_tokens: int = 256):
        self._mode      = mode   # "reasoning_supervisor" | "reasoning_driver"
        self._vllm_url  = vllm_url.rstrip("/")
        self._max_tokens = max_tokens
        self._model_id  = None   # resolved at load() from vLLM /v1/models

    def load(self) -> None:
        """Verify vLLM is reachable and resolve the served model ID."""
        try:
            import openai
        except ImportError:
            raise RuntimeError("pip install openai  — required for reasoning modes")

        client = openai.OpenAI(api_key="EMPTY", base_url=f"{self._vllm_url}/v1")
        try:
            models = client.models.list()
            self._model_id = models.data[0].id
            log.info("ReasoningEngine connected to vLLM  model=%s  url=%s",
                     self._model_id, self._vllm_url)
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach vLLM at {self._vllm_url}: {e}\n"
                "Start vLLM first:  vllm serve nvidia/Cosmos3-Edge --port 8000 ..."
            ) from e

        log.info("ReasoningEngine ready (mode=%s)", self._mode)

    def infer(self, frame_jpeg: bytes, goal: str) -> dict:
        import openai

        t0 = time.time()

        # Encode frame as base64 data URL
        img_b64  = base64.b64encode(frame_jpeg).decode()
        data_url = f"data:image/jpeg;base64,{img_b64}"

        prompt = (_SUPERVISOR_PROMPT if self._mode == "reasoning_supervisor"
                  else _DRIVER_PROMPT).format(goal=goal)

        client = openai.OpenAI(api_key="EMPTY", base_url=f"{self._vllm_url}/v1")
        response = client.chat.completions.create(
            model=self._model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text",      "text": prompt},
                ],
            }],
            max_tokens=self._max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        raw_text = response.choices[0].message.content or ""
        elapsed  = round(time.time() - t0, 3)
        log.info("Reasoning (%.2fs): %s", elapsed, raw_text[:200])

        parsed = _parse_json(raw_text)

        if self._mode == "reasoning_supervisor":
            return {
                "type":        "supervision",
                "drift":       parsed.get("drift", "center"),
                "drift_mm":    int(parsed.get("drift_mm", 0)),
                "row_end":     bool(parsed.get("row_end", False)),
                "observation": parsed.get("observation", raw_text[:200]),
                "elapsed":     elapsed,
            }
        else:
            return {
                "type":      "drive",
                "velocity":  int(max(0, min(200, parsed.get("velocity", 100)))),
                "radius":    int(parsed.get("radius", 32767)),
                "reasoning": parsed.get("reasoning", raw_text[:200]),
                "elapsed":   elapsed,
            }


# ── Mode 4: AV Policy engine ───────────────────────────────────────────────────

class AvPolicyEngine:
    """
    Uses CosmosActionCondition(mode='policy', domain_name='av') to predict
    a 16-step action chunk from a short video clip.
    """

    def __init__(self, model_path: str, chunk_size: int = 5):
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
            for j in frames_jpeg[-self._chunk_size:]
        ]

        result = self._pipe(
            prompt=goal,
            action=CosmosActionCondition(
                mode="policy",
                chunk_size=self._chunk_size,
                domain_name="av",
                resolution_tier=256,   # lowest tier → fastest on Jetson AGX
                video=pil_frames,
                view_point="ego_view",
            ),
            fps=5,
            num_inference_steps=5,     # 5 steps ~10s on Jetson vs 30 steps ~60s
            guidance_scale=1.0,
            use_system_prompt=False,
            enable_safety_check=False,
        )

        actions = result.action[0].tolist() if result.action is not None else []
        return {
            "type":    "actions",
            "actions": actions,   # list of chunk_size × 9 floats
            "elapsed": round(time.time() - t0, 3),
        }


# ── Mode 6: Trajectory ranking engine ─────────────────────────────────────────

class TrajectoryRankingEngine:
    """
    Samples the policy num_samples times, scores each trajectory by its
    predicted value, and returns all candidates ranked best-first.
    """

    def __init__(self, model_path: str, num_samples: int = 3, chunk_size: int = 8):
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
                        resolution_tier=256,   # lowest tier → fastest on Jetson AGX
                        image=image,
                        view_point="ego_view",
                    ),
                    fps=5,
                    num_inference_steps=5,     # 5 steps ~10s per sample on Jetson
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
    parser.add_argument("--vllm-url", default="http://localhost:8000",
                        help="Base URL of the vLLM server for reasoning modes "
                             "(default: http://localhost:8000)")
    parser.add_argument("--max-tokens", default=256, type=int,
                        help="Max tokens to generate in reasoning modes (default 256)")
    args = parser.parse_args()

    # Instantiate the right engine
    if args.mode in ("reasoning_supervisor", "reasoning_driver"):
        engine = ReasoningEngine(args.mode,
                                 vllm_url=args.vllm_url,
                                 max_tokens=args.max_tokens)
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
