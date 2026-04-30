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
import time

import websockets

log = logging.getLogger("paligemma_cloud_server")

NUM_WAYPOINTS = 8
WAYPOINT_DIM  = 4   # [dx, dy, cos_heading, sin_heading]


# ── Prompt ────────────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = "To {goal}, should the robot go straight, turn left, or turn right?"

_turn_dy = 0.1   # set by main() from --turn-dy; lateral offset in 0.1 m units


def _build_prompt(goal: str) -> str:
    return _PROMPT_TEMPLATE.format(goal=goal)


# ── Direction parser → waypoints ───────────────────────────────────────────────

def _parse_waypoints(text: str) -> list[list[float]] | None:
    t = text.strip().lower()
    log.info("Direction answer: %r", t)

    if "left" in t:
        dy = _turn_dy
    elif "right" in t:
        dy = -_turn_dy
    elif "straight" in t or "forward" in t or "ahead" in t:
        dy = 0.0
    else:
        return None

    return [[float(i + 1), dy, 1.0, 0.0] for i in range(NUM_WAYPOINTS)]


def _fallback_waypoints() -> list[list[float]]:
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
    ):
        self._project     = project
        self._endpoint_id = endpoint_id
        self._location    = location
        self._endpoint    = None

    def connect(self) -> None:
        """Initialise Vertex AI SDK and resolve the endpoint.  Call once at startup."""
        from google.cloud import aiplatform

        aiplatform.init(project=self._project, location=self._location)

        if self._endpoint_id.isdigit():
            # Numeric ID — use directly
            self._endpoint = aiplatform.Endpoint(self._endpoint_id)
            log.info("Vertex AI endpoint ready: %s", self._endpoint_id)
        else:
            # Display name or partial name — list all endpoints and match
            log.info("Looking up endpoint by name: '%s' …", self._endpoint_id)
            endpoints = aiplatform.Endpoint.list(
                filter=f'display_name="{self._endpoint_id}"',
                project=self._project,
                location=self._location,
            )
            if not endpoints:
                # Fallback: list all and do substring match
                all_endpoints = aiplatform.Endpoint.list(
                    project=self._project, location=self._location
                )
                endpoints = [e for e in all_endpoints
                             if self._endpoint_id.lower() in e.display_name.lower()]
            if not endpoints:
                all_endpoints = aiplatform.Endpoint.list(
                    project=self._project, location=self._location
                )
                names = [e.display_name for e in all_endpoints]
                raise ValueError(
                    f"No endpoint matching '{self._endpoint_id}' found.\n"
                    f"Available endpoints: {names}\n"
                    f"Pass the numeric ID with --endpoint <number>."
                )
            if len(endpoints) > 1:
                log.warning("Multiple endpoints match '%s' — using first: %s",
                            self._endpoint_id, endpoints[0].display_name)
            self._endpoint = endpoints[0]
            log.info("Vertex AI endpoint resolved: %s  (id=%s)",
                     self._endpoint.display_name,
                     self._endpoint.name.split("/")[-1])

    def predict(self, jpeg_bytes: bytes, prompt: str) -> str:
        """Call the endpoint and return the raw text response."""
        from PIL import Image as PIL_Image

        img = PIL_Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        img = img.resize((224, 224), PIL_Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()

        instance = {
            "prompt": prompt,
            "image":  b64,
        }

        response = self._endpoint.predict(instances=[instance])

        # Vertex AI wraps predictions in a list; extract first item
        pred = response.predictions[0]
        log.info("Vertex AI raw prediction: %s", pred)
        if isinstance(pred, dict):
            return (pred.get("response") or pred.get("output")
                    or pred.get("generated_text") or str(pred))
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
    parser.add_argument("--host",    default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port",    default=8766, type=int,
                        help="WebSocket port (default: 8766)")
    parser.add_argument("--turn-dy", default=0.1, type=float,
                        help="Lateral offset per waypoint step when turning "
                             "left or right, in 0.1 m units (default: 0.1 = 1 cm)")
    args = parser.parse_args()

    global _turn_dy
    _turn_dy = args.turn_dy
    log.info("Turn dy: %.3f (%.1f cm per step)", args.turn_dy, args.turn_dy * 10)

    client = VertexAIClient(
        project=args.project,
        endpoint_id=args.endpoint,
        location=args.location,
    )
    client.connect()

    engine = InferenceEngine(client)
    asyncio.run(_serve(engine, args.host, args.port))


if __name__ == "__main__":
    main()
