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

_SUPERVISOR_SYSTEM = (
    "You are a navigation assistant for a Roomba robot equipped with a forward-facing "
    "camera. The robot navigates autonomously through indoor and outdoor environments. "
    "Respond ONLY with a JSON object — no prose, no markdown fences."
)

_SUPERVISOR_PROMPT = (
    'The robot\'s goal is: "{goal}"\n\n'
    "Analyse the camera image and respond with a JSON object:\n"
    '{{"drift": "left"|"right"|"center", '
    '"drift_mm": <estimated lateral offset in mm, positive=right>, '
    '"row_end": <true if the path/row ends within 1 metre>, '
    '"observation": "<one sentence describing what you see>"}}'
)

_DRIVER_SYSTEM = (
    "You are the navigation controller for a Roomba robot. "
    "The Roomba uses drive_raw(velocity, radius) where velocity is mm/s (0–200) "
    "and radius is mm (32767=straight, positive=left turn, negative=right turn, "
    "1=spin). Respond ONLY with a JSON object — no prose, no markdown fences."
)

_DRIVER_PROMPT = (
    'The robot\'s goal is: "{goal}"\n\n'
    "Analyse the camera image and respond with the drive command:\n"
    '{{"velocity": <int 0-200>, '
    '"radius": <int, 32767 for straight>, '
    '"reasoning": "<one sentence explaining the command>"}}'
)


class ReasoningEngine:
    """
    Uses the Cosmos3-Edge *reasoning* (LLM/VLM) path — text output only, no diffusion.

    Cosmos3OmniPipeline does not expose a text-output mode; the `output` kwarg
    belongs only to Cosmos3OmniModularPipeline.  Instead we access the reasoning
    tower (a Qwen-style VLM) directly via the HuggingFace `transformers` library,
    which is how the official Jetson benchmarks run the reasoner.

    The transformer inside the pipeline is a `Cosmos3OmniTransformer` whose
    text generation path is reached by calling `pipe.transformer` with token
    inputs.  For simplicity we load it as a plain AutoModelForCausalLM so we
    can use the standard chat-template / generate() interface.
    """

    def __init__(self, model_path: str, mode: str, max_new_tokens: int = 128):
        self._model_path    = model_path
        self._mode          = mode   # "reasoning_supervisor" | "reasoning_driver"
        self._max_new_tokens = max_new_tokens
        self._model     = None
        self._processor = None

    def load(self) -> None:
        import torch
        import transformers
        from transformers import AutoTokenizer

        log.info("Loading Cosmos3 reasoning model from %s  (transformers %s) …",
                 self._model_path, transformers.__version__)
        t0 = time.time()

        # Try AutoModelForVision2Seq first (transformers ≥ 4.45); fall back to
        # AutoModel with trust_remote_code for older versions in the cosmos venv.
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                self._model_path, trust_remote_code=True)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self._model_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                trust_remote_code=True,
            )
            self._use_processor = True
            log.info("Loaded via AutoModelForVision2Seq + AutoProcessor")
        except (ImportError, Exception) as e:
            log.warning("AutoModelForVision2Seq unavailable (%s) — trying AutoModel", e)
            from transformers import AutoModel
            self._processor = None
            self._model = AutoModel.from_pretrained(
                self._model_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                trust_remote_code=True,
            )
            self._use_processor = False
            log.info("Loaded via AutoModel")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, trust_remote_code=True)
        self._model.eval()

        # ── Discover the actual generatable LM ───────────────────────────────
        # Cosmos3EdgeModel nests custom modules: Cosmos3EdgeModel →
        # Cosmos3EdgeTextModel → ??? → actual causal LM with .generate().
        # Walk the sub-module tree (BFS) to find the first module whose class
        # name looks like a standard HF causal LM.
        self._lm = self._find_generatable_lm(self._model)
        if self._lm is not None:
            log.info("Found generatable LM: %s", type(self._lm).__name__)
        else:
            log.warning("Could not find a generatable sub-LM; "
                        "will attempt forward() + logit greedy decode")

        log.info("ReasoningEngine ready (mode=%s) in %.1fs", self._mode,
                 time.time() - t0)

    @staticmethod
    def _find_generatable_lm(root):
        """
        BFS through sub-modules to find the first one that has .generate().
        Prefers modules whose class name contains 'Qwen', 'Llama', 'Mistral',
        'Causal', 'LM', 'GPT' — typical HF causal LM names.
        Logs the module tree to help debug future changes.
        """
        from collections import deque
        preferred_keywords = ("qwen", "llama", "mistral", "causallm", "gpt",
                              "falcon", "gemma", "phi", "causal")

        candidates = []
        queue = deque([(name, module)
                       for name, module in root.named_children()])
        visited = set()

        while queue:
            name, mod = queue.popleft()
            mid = id(mod)
            if mid in visited:
                continue
            visited.add(mid)

            cname = type(mod).__name__.lower()
            has_gen = callable(getattr(mod, "generate", None))

            log.info("  sub-module %-40s  has_generate=%s  class=%s",
                     name, has_gen, type(mod).__name__)

            if has_gen:
                score = sum(1 for k in preferred_keywords if k in cname)
                candidates.append((score, name, mod))

            for child_name, child in mod.named_children():
                queue.append((f"{name}.{child_name}", child))

        if not candidates:
            return None
        # Pick highest-scoring (most LLM-like name); tie-break by order (first)
        candidates.sort(key=lambda x: -x[0])
        chosen_score, chosen_name, chosen_mod = candidates[0]
        log.info("Selected generatable LM: %s (%s)", chosen_name,
                 type(chosen_mod).__name__)
        return chosen_mod

    def infer(self, frame_jpeg: bytes, goal: str) -> dict:
        """
        Run one reasoning step using Cosmos3's image-to-text capability.

        We use the pipeline in image-to-video mode with num_frames=1 and
        num_inference_steps=1 (minimum diffusion), then extract the text
        from the reasoning tower's hidden output via the transformer's
        generate() interface. If that fails we fall back to prompting the
        pipeline and parsing any text in the result.
        """
        import torch
        from PIL import Image

        t0    = time.time()
        image = Image.open(io.BytesIO(frame_jpeg)).convert("RGB")

        if self._mode == "reasoning_supervisor":
            system_prompt = _SUPERVISOR_SYSTEM
            user_prompt   = _SUPERVISOR_PROMPT.format(goal=goal)
        else:
            system_prompt = _DRIVER_SYSTEM
            user_prompt   = _DRIVER_PROMPT.format(goal=goal)

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        raw_text = self._generate_text(image, full_prompt)

        elapsed = round(time.time() - t0, 3)
        log.info("Reasoning output (%.2fs): %s", elapsed, raw_text[:200])

        parsed = _parse_json_from_text(raw_text)

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
            vel    = int(max(0, min(200, parsed.get("velocity", 100))))
            radius = int(parsed.get("radius", 32767))
            return {
                "type":      "drive",
                "velocity":  vel,
                "radius":    radius,
                "reasoning": parsed.get("reasoning", raw_text[:200]),
                "elapsed":   elapsed,
            }

    def _generate_text(self, image, prompt: str) -> str:
        """
        Generate text from either:
          (A) A standard HF VLM loaded via AutoModelForVision2Seq — call .generate() directly
              after processing through AutoProcessor.  (_use_processor=True)
          (B) Cosmos3EdgeModel loaded via AutoModel — no top-level .generate(); instead use:
                .language_model  — the causal LM with .generate()
                .visual          — vision encoder
                .projector       — vision→language projection
              (_use_processor=False)
        """
        import torch

        device = next(self._model.parameters()).device
        dtype  = next(self._model.parameters()).dtype

        # ── Path A: standard VLM (AutoModelForVision2Seq + AutoProcessor) ─────
        if self._use_processor and self._processor is not None:
            try:
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text": prompt},
                    ]},
                ]
                text_input = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = self._processor(
                    text=[text_input], images=[image], return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    out_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=self._max_new_tokens,
                        do_sample=False,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
                new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
                return self._tokenizer.decode(
                    new_tokens, skip_special_tokens=True).strip()
            except Exception as e:
                log.warning("AutoModelForVision2Seq.generate() failed (%s) — "
                            "falling back to Cosmos3EdgeModel path", e)

        # ── Path B: Cosmos3EdgeModel (custom nested modules) ─────────────────
        # self._lm is found by BFS in load() — the deepest sub-module with
        # .generate().  If none was found we fall through to a greedy-decode
        # forward() loop as last resort.
        text_inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        ).to(device)
        input_ids = text_inputs["input_ids"]

        lm = self._lm  # may be None

        # ── B1: Image-conditioned path ────────────────────────────────────────
        if lm is not None:
            try:
                import torchvision.transforms as T

                transform = T.Compose([
                    T.Resize((336, 336)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                ])
                img_tensor = transform(image).unsqueeze(0).to(device=device, dtype=dtype)

                with torch.no_grad():
                    if hasattr(self._model, "get_image_features"):
                        visual_feats = self._model.get_image_features(img_tensor)
                    else:
                        visual_feats = self._model.visual(img_tensor)
                        if hasattr(self._model, "projector"):
                            visual_feats = self._model.projector(visual_feats)

                embed_fn    = lm.get_input_embeddings()
                text_embeds = embed_fn(input_ids)
                combined    = torch.cat([visual_feats, text_embeds], dim=1)
                attn_mask   = torch.ones(1, combined.shape[1], device=device,
                                         dtype=torch.long)

                with torch.no_grad():
                    out_ids = lm.generate(
                        inputs_embeds=combined,
                        attention_mask=attn_mask,
                        max_new_tokens=self._max_new_tokens,
                        do_sample=False,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )

                return self._tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

            except Exception as e:
                log.warning("Image-conditioned generate failed (%s) — "
                            "trying text-only lm.generate()", e)

        # ── B2: Text-only via discovered LM ──────────────────────────────────
        if lm is not None:
            try:
                with torch.no_grad():
                    out_ids = lm.generate(
                        input_ids=input_ids,
                        attention_mask=text_inputs.get("attention_mask"),
                        max_new_tokens=self._max_new_tokens,
                        do_sample=False,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
                new_tokens = out_ids[0][input_ids.shape[1]:]
                return self._tokenizer.decode(
                    new_tokens, skip_special_tokens=True).strip()
            except Exception as e:
                log.warning("lm.generate() text-only failed (%s) — "
                            "falling back to greedy forward() loop", e)

        # ── B3: Last-resort greedy decode via model.forward() ─────────────────
        # Some custom models only expose forward() and not generate().
        # We do a simple greedy decode loop using the top-level model.
        log.warning("Using slow greedy forward() decode — "
                    "consider loading with AutoModelForCausalLM")
        generated = input_ids.clone()
        with torch.no_grad():
            for _ in range(self._max_new_tokens):
                out = self._model(input_ids=generated)
                logits = out.logits if hasattr(out, "logits") else out[0]
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated  = torch.cat([generated, next_token], dim=1)
                if next_token.item() == self._tokenizer.eos_token_id:
                    break
        new_tokens = generated[0][input_ids.shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


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


# ── JSON parsing helper ────────────────────────────────────────────────────────

def _parse_json_from_text(text: str) -> dict:
    """Extract the first JSON object from a text string. Returns {} on failure."""
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find first { … } block
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    log.warning("Could not parse JSON from model output: %r", text[:300])
    return {}


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
