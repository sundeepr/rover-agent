#!/usr/bin/env python3
"""
moondream_cloud_server.py — WebSocket inference server for Moondream2.

Same protocol as qwen_cloud_server.py — row_change_strategy.py works with
both servers, just point --qwen-server at this one instead.

Moondream2 processes one frame per query (no temporal buffering).

Setup
─────
    pip install torch transformers websockets pillow einops

    python moondream_cloud_server.py \\
        --model-path /path/to/moondream2 \\
        --host 0.0.0.0 --port 8767

Protocol (identical to qwen_cloud_server.py)
────────────────────────────────────────────
Client → Server
  {"type": "infer", "instruction": "<text>", "frame_b64": "<base64 JPEG>"}

Server → Client
  {"type": "ready"}
  {"type": "response", "text": "<answer>", "elapsed": <float>}
  {"type": "error",    "message": "<text>"}
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import time
from pathlib import Path

log = logging.getLogger("moondream_cloud_server")

_INFER_W, _INFER_H = 320, 240   # resize before inference


# ── Inference engine ──────────────────────────────────────────────────────────

class InferenceEngine:

    def __init__(self, model_path: str, device_map: str = "auto"):
        self._model_path = Path(model_path)
        self._device_map = device_map
        self._model      = None
        self._tokenizer  = None

    def load(self) -> None:
        import shutil
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("Loading Moondream2 from %s …", self._model_path)

        # transformers looks for the model's custom Python files in its modules
        # cache (~/.cache/huggingface/modules/transformers_modules/<name>/).
        # When loading from a local path the cache is not auto-populated, so
        # copy the .py files there before loading.
        cache_dir = (Path.home() / ".cache" / "huggingface" / "modules"
                     / "transformers_modules" / self._model_path.name)
        cache_dir.mkdir(parents=True, exist_ok=True)
        _FUTURE = "from __future__ import annotations\n"
        copied = []
        for py_file in self._model_path.glob("*.py"):
            dest = cache_dir / py_file.name
            if not dest.exists() or py_file.stat().st_mtime > dest.stat().st_mtime:
                src_text = py_file.read_text(encoding="utf-8")
                # Inject __future__ annotation import for Python 3.8 compatibility.
                # tuple[x, y] / list[x] / dict[x, y] syntax requires 3.9+ without it.
                if _FUTURE not in src_text:
                    src_text = _FUTURE + src_text
                dest.write_text(src_text, encoding="utf-8")
                copied.append(py_file.name)
        if copied:
            log.info("Copied+patched model code to HF cache: %s", ", ".join(copied))

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self._model_path), trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self._model_path),
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=self._device_map,
        )
        self._model.eval()
        log.info("Moondream2 loaded")

    def infer(self, frame_jpeg: bytes, instruction: str) -> dict:
        from PIL import Image as PIL_Image
        import torch

        t0 = time.time()

        pil_img = (PIL_Image.open(io.BytesIO(frame_jpeg))
                   .convert("RGB")
                   .resize((_INFER_W, _INFER_H), PIL_Image.LANCZOS))

        with torch.no_grad():
            enc = self._model.encode_image(pil_img)
            answer = self._model.answer_question(enc, instruction, self._tokenizer)

        return {"text": answer.strip(), "elapsed": round(time.time() - t0, 3)}


# ── Per-connection session ────────────────────────────────────────────────────

class ConnectionSession:

    def __init__(self, engine: InferenceEngine, loop: asyncio.AbstractEventLoop):
        self._engine = engine
        self._loop   = loop

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
            await websocket.send(json.dumps({"type": "error", "message": "invalid JSON"}))
            return

        mtype = msg.get("type")

        if mtype == "infer":
            frame_b64   = msg.get("frame_b64", "")
            instruction = msg.get("instruction", "")
            if not frame_b64:
                await websocket.send(json.dumps({"type": "error", "message": "missing frame_b64"}))
                return
            if not instruction:
                await websocket.send(json.dumps({"type": "error", "message": "missing instruction"}))
                return

            frame_bytes = base64.b64decode(frame_b64)
            try:
                result = await self._loop.run_in_executor(
                    None, self._engine.infer, frame_bytes, instruction)
                await websocket.send(json.dumps({"type": "response", **result}))
                log.info("Infer OK  elapsed=%.2fs", result["elapsed"])
                log.info("  PROMPT  : %s", instruction)
                log.info("  RESPONSE: %s", result["text"])
            except Exception as e:
                log.error("Inference error: %s", e, exc_info=True)
                await websocket.send(json.dumps({"type": "error", "message": str(e)}))
            return

        await websocket.send(json.dumps({"type": "error", "message": f"unknown type: {mtype!r}"}))


# ── Server ────────────────────────────────────────────────────────────────────

async def _serve(engine: InferenceEngine, host: str, port: int) -> None:
    import websockets

    loop = asyncio.get_running_loop()

    async def _handler(ws):
        session = ConnectionSession(engine, loop)
        await session.handle(ws)

    async with websockets.serve(_handler, host, port, max_size=20 * 1024 * 1024):
        log.info("Moondream2 WebSocket server listening on ws://%s:%d", host, port)
        log.info("Rover command:  --qwen-server ws://%s:%d", host, port)
        await asyncio.Future()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Moondream2 WebSocket inference server")
    parser.add_argument("--model-path", required=True, metavar="DIR",
                        help="Path to downloaded Moondream2 directory")
    parser.add_argument("--host",       default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port",       default=8767, type=int,
                        help="WebSocket port (default: 8767)")
    parser.add_argument("--device-map", default="auto",
                        help="HuggingFace device_map (default: auto)")
    args = parser.parse_args()

    engine = InferenceEngine(model_path=args.model_path, device_map=args.device_map)
    engine.load()

    asyncio.run(_serve(engine, args.host, args.port))


if __name__ == "__main__":
    main()
