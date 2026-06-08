#!/usr/bin/env python3
"""
qwen_cloud_server.py — WebSocket inference server for Qwen2.5-VL-7B-Instruct.

Loads the Qwen2.5-VL model and serves text responses to rover strategies over
a persistent WebSocket connection.  Accepts an image + instruction and returns
the model's free-text output (e.g. navigation decisions, scene descriptions,
end-of-row detection).

Setup (on the cloud instance)
──────────────────────────────
    pip install torch torchvision transformers accelerate websockets pillow
    # optional but recommended for full image preprocessing support:
    pip install qwen-vl-utils

    python qwen_cloud_server.py \\
        --model-path /path/to/Qwen2.5-VL-7B-Instruct \\
        --host 0.0.0.0 --port 8766

Protocol (UTF-8 JSON over WebSocket)
─────────────────────────────────────
Client → Server

  {"type": "system", "prompt": "<text>"}
      Set (or replace) the system-level prompt.  Optional — if omitted the
      server uses a default rover-assistant prompt.

  {"type": "infer", "instruction": "<text>", "frame_b64": "<base64 JPEG>"}
      Run one inference step.  Returns a "response" message.

Server → Client

  {"type": "ready"}
      Model fully loaded; accepting "infer" requests.

  {"type": "response",
   "text":    "<model output text>",
   "elapsed": <seconds as float>}

  {"type": "error", "message": "<text>"}

On the rover:
    python rover_agent.py --strategy row_end_uturn \\
        --qwen-server ws://<cloud-ip>:8766 \\
        --rover atlas --atlas-port /dev/ttyACM0
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import time
from pathlib import Path

log = logging.getLogger("qwen_cloud_server")

_DEFAULT_SYSTEM_PROMPT = (
    "You are a visual assistant for an agricultural rover. "
    "Look at the image and answer the question concisely. "
    "Do not write code. Do not explain your reasoning. "
    "Reply with only the exact word or words requested."
)

_DEFAULT_MAX_NEW_TOKENS = 256
_DEFAULT_TEMPERATURE    = 0.0    # 0 = greedy decoding (do_sample=False)


# ── Inference engine ──────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Wraps Qwen2.5-VL-7B-Instruct for image + text → text inference.

    load() must be called once before the server starts accepting connections.
    infer() is blocking and intended to be called from a thread executor.
    """

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
        temperature: float  = _DEFAULT_TEMPERATURE,
        device_map: str     = "auto",
    ):
        self._model_path     = Path(model_path)
        self._max_new_tokens = max_new_tokens
        self._temperature    = temperature
        self._device_map     = device_map
        self._model          = None
        self._processor      = None
        self._system_prompt  = _DEFAULT_SYSTEM_PROMPT

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt
        log.info("System prompt updated (%d chars)", len(prompt))

    def load(self) -> None:
        """Load model and processor.  Blocks until ready (~30 s on GPU)."""
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq

        log.info("Loading Qwen2.5-VL from %s …", self._model_path)
        log.info("device_map=%s", self._device_map)

        self._processor = AutoProcessor.from_pretrained(
            str(self._model_path),
            trust_remote_code=True,
        )

        self._model = AutoModelForVision2Seq.from_pretrained(
            str(self._model_path),
            torch_dtype=torch.float16,
            device_map=self._device_map,
            trust_remote_code=True,
        )
        self._model.eval()
        log.info("Qwen2.5-VL loaded — %d parameters",
                 sum(p.numel() for p in self._model.parameters()))

    def infer(self, frames: "list[bytes]", instruction: str) -> dict:
        """
        Run one inference step (blocking).

        frames      : list of JPEG bytes, oldest→newest.  Usually 1–3 frames.
        instruction : task-specific text instruction.

        Returns {"text": str, "elapsed": float}.
        """
        import torch
        from PIL import Image as PIL_Image

        t0 = time.time()

        pil_imgs = [PIL_Image.open(io.BytesIO(f)).convert("RGB") for f in frames]

        # Build multi-image content block
        content = []
        for idx, img in enumerate(pil_imgs):
            if len(pil_imgs) > 1:
                label = f"Frame {idx + 1} of {len(pil_imgs)}" + (
                    " (oldest)" if idx == 0
                    else " (most recent)" if idx == len(pil_imgs) - 1
                    else "")
                content.append({"type": "text", "text": label})
            content.append({"type": "image", "image": img})

        # Add context hint + instruction after the images
        if len(pil_imgs) > 1:
            content.append({"type": "text", "text": (
                f"The {len(pil_imgs)} frames above are consecutive camera images "
                f"from the rover (oldest to most recent). "
                f"Use the sequence for context. {instruction}"
            )})
        else:
            content.append({"type": "text", "text": instruction})

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": content},
        ]

        # Build prompt text from the chat template
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Encode inputs — pass all PIL images
        inputs = self._processor(
            text=[text],
            images=pil_imgs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                temperature=None,  # suppress warning from model's generation_config.json
            )

        # Trim prompt tokens — only decode the new tokens
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        response_text = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        # Strip markdown code fences the model sometimes adds (```python\nYES\n```)
        response_text = _strip_markdown(response_text)

        return {"text": response_text, "elapsed": round(time.time() - t0, 3)}


def _strip_markdown(text: str) -> str:
    """Remove markdown code fences and extra whitespace from model output."""
    import re
    # Remove ```lang ... ``` blocks, keeping only the inner content
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = text.replace("```", "")
    return text.strip()


# ── Per-connection session ────────────────────────────────────────────────────

class ConnectionSession:
    """Manages one WebSocket client connection."""

    def __init__(self, engine: InferenceEngine, loop: asyncio.AbstractEventLoop,
                 context_frames: int = 3):
        self._engine  = engine
        self._loop    = loop
        from collections import deque
        self._frame_buffer: "deque[bytes]" = deque(maxlen=context_frames)
        log.info("Frame context buffer: %d frames per query", context_frames)

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

        if mtype == "system":
            prompt = msg.get("prompt", "")
            if prompt:
                self._engine.set_system_prompt(prompt)
            return

        if mtype == "infer":
            frame_b64   = msg.get("frame_b64", "")
            instruction = msg.get("instruction", "")

            if not frame_b64:
                await websocket.send(json.dumps(
                    {"type": "error", "message": "missing frame_b64"}))
                return
            if not instruction:
                await websocket.send(json.dumps(
                    {"type": "error", "message": "missing instruction"}))
                return

            frame_bytes = base64.b64decode(frame_b64)
            self._frame_buffer.append(frame_bytes)
            frames = list(self._frame_buffer)   # oldest → newest
            try:
                result = await self._loop.run_in_executor(
                    None, self._engine.infer, frames, instruction
                )
                await websocket.send(json.dumps({"type": "response", **result}))
                log.info("Infer OK  elapsed=%.2fs  frames=%d", result["elapsed"], len(frames))
                log.info("  PROMPT  : %s", instruction)
                log.info("  RESPONSE: %s", result["text"])
            except Exception as e:
                log.error("Inference error: %s", e, exc_info=True)
                await websocket.send(json.dumps(
                    {"type": "error", "message": str(e)}))
            return

        await websocket.send(json.dumps(
            {"type": "error", "message": f"unknown type: {mtype!r}"}))


# ── Server ────────────────────────────────────────────────────────────────────

async def _serve(engine: InferenceEngine, host: str, port: int,
                 context_frames: int = 3) -> None:
    import websockets

    loop = asyncio.get_running_loop()

    async def _handler(ws):
        session = ConnectionSession(engine, loop, context_frames=context_frames)
        await session.handle(ws)

    async with websockets.serve(_handler, host, port,
                                max_size=20 * 1024 * 1024):  # 20 MB max message
        log.info("Qwen2.5-VL WebSocket server listening on ws://%s:%d", host, port)
        log.info("Rover command:  --qwen-server ws://%s:%d", host, port)
        await asyncio.Future()   # run until cancelled


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL-7B-Instruct WebSocket inference server")
    parser.add_argument("--model-path", required=True, metavar="DIR",
                        help="Path to downloaded Qwen2.5-VL-7B-Instruct directory")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", default=8766, type=int,
                        help="WebSocket port (default: 8766)")
    parser.add_argument("--max-tokens", default=256, type=int,
                        help="Max new tokens per response (default: 256)")
    parser.add_argument("--temperature", default=0.1, type=float,
                        help="Sampling temperature — 0 for greedy (default: 0.1)")
    parser.add_argument("--device-map", default="auto",
                        help="HuggingFace device_map ('auto', 'cuda:0', 'cpu'; "
                             "default: auto)")
    parser.add_argument("--system-prompt", default=None, metavar="TEXT",
                        help="Override the default system prompt")
    parser.add_argument("--context-frames", default=3, type=int, metavar="N",
                        help="Number of consecutive frames to buffer and send to the "
                             "model per query for temporal context (default: 3). "
                             "Use 1 to disable buffering.")
    args = parser.parse_args()

    engine = InferenceEngine(
        model_path     = args.model_path,
        max_new_tokens = args.max_tokens,
        temperature    = args.temperature,
        device_map     = args.device_map,
    )

    if args.system_prompt:
        engine.set_system_prompt(args.system_prompt)

    engine.load()

    asyncio.run(_serve(engine, args.host, args.port,
                       context_frames=args.context_frames))


if __name__ == "__main__":
    main()
