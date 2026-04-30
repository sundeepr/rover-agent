#!/usr/bin/env python3
"""
paligemma_cloud_server.py — WebSocket proxy to PaliGemma on Google Cloud Vertex AI.

Runs on any machine (no GPU needed).  Receives camera frames from rover_agent
(via paligemma_strategy.py) over WebSocket, calls the PaliGemma endpoint on
Vertex AI, parses the waypoint output, and returns waypoints to the rover.

Setup
─────
1. Deploy PaliGemma 2 from Vertex AI Model Garden:
       console.cloud.google.com → Vertex AI → Model Garden → PaliGemma 2
       Click "Deploy" → note the ENDPOINT_ID.

2. Authenticate:
       gcloud auth application-default login
   or set GOOGLE_APPLICATION_CREDENTIALS to a service-account key file.

3. Install deps:
       pip install google-cloud-aiplatform websockets

4. Start:
       python paligemma_cloud_server.py \\
           --project my-gcp-project \\
           --endpoint 1234567890123456789 \\
           --host 0.0.0.0 --port 8766

   Optional: --location (default us-central1), --model-version (for logging).

On the rover:
    python rover_agent.py --strategy paligemma \\
        --cloud-server ws://<server-ip>:8766 \\
        --goal "Follow the crop row" --interval 1.0

Vertex AI endpoint input format
────────────────────────────────
The request format depends on the serving container used when deploying
from Model Garden.  The default PaliGemma 2 container expects:

    instances = [{"prompt": "<prompt>", "image": "<base64 JPEG>"}]

If your deployment uses a different format (e.g. HuggingFace TGI), set
--request-format hf-tgi  (see RequestFormat below).

Protocol (identical to omnivla_cloud_server.py)
────────────────────────────────────────────────
Client → Server
  {"type": "goal",  "goal": "<text>"}
  {"type": "infer", "goal": "<text>", "frame_b64": "<base64 JPEG>"}

Server → Client
  {"type": "ready"}
  {"type": "waypoints", "waypoints": [[dx,dy,cos_h,sin_h]×8], "elapsed": <s>}
  {"type": "error",     "message": "<text>"}
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import re
import time

import websockets

log = logging.getLogger("paligemma_cloud_server")

NUM_WAYPOINTS = 8
WAYPOINT_DIM  = 4   # [dx, dy, cos_heading, sin_heading]


# ── Prompt ────────────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
You are the navigation controller of a wheeled outdoor robot.
The image is from the robot's forward-facing camera.

Goal: {goal}

Predict the robot's next 8 navigation waypoints in its local coordinate frame:
  dx  — forward distance in 0.1 m units (positive = ahead)
  dy  — lateral distance in 0.1 m units (positive = left, negative = right)
  cos_h, sin_h — cosine and sine of the robot heading at that waypoint

Output ONLY a JSON array of exactly 8 waypoints with 4 floats each.
No explanation, no markdown, just the array.

Example (going straight):
[[1.0,0.0,1.0,0.0],[2.0,0.0,1.0,0.0],[3.0,0.0,1.0,0.0],[4.0,0.0,1.0,0.0],[5.0,0.0,1.0,0.0],[6.0,0.0,1.0,0.0],[7.0,0.0,1.0,0.0],[8.0,0.0,1.0,0.0]]

Waypoints:"""


def _build_prompt(goal: str) -> str:
    return _PROMPT_TEMPLATE.format(goal=goal)


# ── Waypoint parser ────────────────────────────────────────────────────────────

def _parse_waypoints(text: str) -> list[list[float]] | None:
    """Extract a valid 8×4 waypoint array from PaliGemma's text output."""
    text = text.strip()

    # 1. Whole string as JSON
    try:
        data = json.loads(text)
        if _valid(data):
            return data
    except json.JSONDecodeError:
        pass

    # 2. First [...] block
    m = re.search(r'(\[\s*\[[\s\S]*?\]\s*\])', text)
    if m:
        try:
            data = json.loads(m.group(1))
            if _valid(data):
                return data
        except json.JSONDecodeError:
            pass

    # 3. Extract all numbers and reshape 8×4
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    if len(nums) >= NUM_WAYPOINTS * WAYPOINT_DIM:
        flat = [float(n) for n in nums[:NUM_WAYPOINTS * WAYPOINT_DIM]]
        data = [flat[i * WAYPOINT_DIM:(i + 1) * WAYPOINT_DIM]
                for i in range(NUM_WAYPOINTS)]
        if _valid(data):
            return data

    return None


def _valid(data) -> bool:
    return (isinstance(data, list) and len(data) == NUM_WAYPOINTS
            and all(isinstance(wp, list) and len(wp) == WAYPOINT_DIM
                    for wp in data))


def _fallback_waypoints() -> list[list[float]]:
    """Straight-ahead waypoints used when parsing fails."""
    return [[float(i + 1), 0.0, 1.0, 0.0] for i in range(NUM_WAYPOINTS)]


# ── Vertex AI client ──────────────────────────────────────────────────────────

class VertexAIClient:
    """
    Calls a deployed PaliGemma endpoint on Google Cloud Vertex AI.

    Two request formats are supported (set with --request-format):
      default  — {"prompt": "...", "image": "<base64>"}
                 Used by the standard PaliGemma Model Garden container.
      hf-tgi   — {"inputs": "<prompt>", "parameters": {...}, "image": "<b64>"}
                 Used when deploying with a HuggingFace TGI serving container.
    """

    def __init__(
        self,
        project: str,
        endpoint_id: str,
        location: str = "us-central1",
        request_format: str = "default",
        max_tokens: int = 128,
    ):
        self._project        = project
        self._endpoint_id    = endpoint_id
        self._location       = location
        self._request_format = request_format
        self._max_tokens     = max_tokens
        self._endpoint       = None

    def connect(self) -> None:
        """Initialise Vertex AI SDK and resolve the endpoint.  Call once at startup."""
        from google.cloud import aiplatform

        aiplatform.init(project=self._project, location=self._location)
        self._endpoint = aiplatform.Endpoint(self._endpoint_id)
        log.info("Vertex AI endpoint ready: %s (project=%s, location=%s)",
                 self._endpoint_id, self._project, self._location)

    def predict(self, jpeg_bytes: bytes, prompt: str) -> str:
        """Call the endpoint and return the raw text response."""
        b64 = base64.b64encode(jpeg_bytes).decode()

        if self._request_format == "hf-tgi":
            instance = {
                "inputs":     prompt,
                "parameters": {"max_new_tokens": self._max_tokens},
                "image":      b64,
            }
        else:  # default Model Garden container format
            instance = {
                "prompt":     prompt,
                "image":      b64,
                "max_tokens": self._max_tokens,
            }

        response = self._endpoint.predict(instances=[instance])

        # Vertex AI wraps predictions in a list; extract first item
        pred = response.predictions[0]
        if isinstance(pred, dict):
            # Some containers return {"generated_text": "..."}
            return pred.get("generated_text") or pred.get("output") or str(pred)
        return str(pred)


# ── Inference engine ──────────────────────────────────────────────────────────

class InferenceEngine:
    """Formats requests, calls Vertex AI, parses waypoints."""

    def __init__(self, client: VertexAIClient):
        self._client = client

    def infer(self, frame_jpeg: bytes, goal: str) -> dict:
        """Blocking — call from a thread executor."""
        t0 = time.time()

        prompt   = _build_prompt(goal)
        raw_text = self._client.predict(frame_jpeg, prompt)

        log.info("PaliGemma raw: %r", raw_text[:120])
        waypoints = _parse_waypoints(raw_text)
        if waypoints is None:
            log.warning("Waypoint parse failed — using straight-ahead fallback")
            waypoints = _fallback_waypoints()

        return {
            "waypoints": waypoints,
            "elapsed":   round(time.time() - t0, 3),
        }


# ── Per-connection session (unchanged protocol) ───────────────────────────────

class ConnectionSession:
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
            await websocket.send(json.dumps({"type": "error", "message": "invalid JSON"}))
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
                    {"type": "error", "message": "no goal — send a 'goal' message first"}))
                return
            frame_b64 = msg.get("frame_b64", "")
            if not frame_b64:
                await websocket.send(json.dumps(
                    {"type": "error", "message": "missing frame_b64"}))
                return

            goal = self._goal
            frame_bytes = base64.b64decode(frame_b64)
            try:
                result = await self._loop.run_in_executor(
                    None, self._engine.infer, frame_bytes, goal
                )
                await websocket.send(json.dumps({"type": "waypoints", **result}))
                log.info("Infer OK  elapsed=%.2fs", result["elapsed"])
            except Exception as e:
                log.error("Inference error: %s", e, exc_info=True)
                await websocket.send(json.dumps({"type": "error", "message": str(e)}))

        else:
            await websocket.send(json.dumps(
                {"type": "error", "message": f"unknown type: {mtype!r}"}))


# ── Server entry point ────────────────────────────────────────────────────────

async def _serve(engine: InferenceEngine, host: str, port: int) -> None:
    loop = asyncio.get_running_loop()

    async def _handler(ws):
        session = ConnectionSession(engine, loop)
        await session.handle(ws)

    async with websockets.serve(_handler, host, port):
        log.info("PaliGemma proxy WebSocket server on ws://%s:%d", host, port)
        await asyncio.Future()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="WebSocket proxy to PaliGemma on Google Cloud Vertex AI")
    parser.add_argument("--project",  required=True,
                        help="Google Cloud project ID")
    parser.add_argument("--endpoint", required=True,
                        help="Vertex AI endpoint ID or full resource name")
    parser.add_argument("--location", default="us-central1",
                        help="Vertex AI region (default: us-central1)")
    parser.add_argument("--request-format", default="default",
                        choices=["default", "hf-tgi"],
                        help="Endpoint request format: 'default' for the "
                             "standard Model Garden container, 'hf-tgi' for "
                             "HuggingFace TGI serving container (default: default)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max tokens to generate (default: 128)")
    parser.add_argument("--host",     default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port",     default=8766, type=int,
                        help="WebSocket port (default: 8766)")
    args = parser.parse_args()

    client = VertexAIClient(
        project=args.project,
        endpoint_id=args.endpoint,
        location=args.location,
        request_format=args.request_format,
        max_tokens=args.max_tokens,
    )
    client.connect()

    engine = InferenceEngine(client)
    asyncio.run(_serve(engine, args.host, args.port))


if __name__ == "__main__":
    main()
