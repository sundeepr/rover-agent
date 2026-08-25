#!/usr/bin/env python3
"""
moondream_client.py — persistent WebSocket client for moondream_cloud_server.py's
"detect" grounding protocol.

Mirrors row_change_strategy.py's _QwenClient exactly (same persistent-connection,
background-thread, blocking-query shape) but for the "detect"/"detections"
message pair instead of "infer"/"response". Kept as its own module (rather
than embedded in a strategy file, the way _QwenClient is) so any strategy or
test script can import MoondreamClient without pulling in a strategy's other
dependencies — solar_dock_strategy.py's eventual client is meant to reuse
this directly.

Protocol (see moondream_cloud_server.py)
─────────────────────────────────────────
  Client → Server
    {"type": "detect", "object": "<text>", "frame_b64": "<base64 JPEG>"}
  Server → Client
    {"type": "detections", "objects": [{"x_min":.., "y_min":.., "x_max":.., "y_max":..}, ...], "elapsed": <float>}
    {"type": "error", "message": "<text>"}

Usage
─────
    client = MoondreamClient("ws://192.168.1.100:8767")
    ...
    objects = client.detect(jpeg_bytes, "solar panel")   # blocks, thread-safe
    if objects is None:
        ...  # timeout / error / not connected
    for obj in objects:
        x1, y1, x2, y2 = obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"]
"""

import asyncio
import base64
import json
import logging
import threading

log = logging.getLogger("rover.moondream_client")


class MoondreamClient:
    """
    Persistent WebSocket connection to moondream_cloud_server.py's "detect" endpoint.

    detect() is blocking and safe to call from any thread. The asyncio event
    loop runs on a dedicated daemon thread; the connection auto-reconnects
    with backoff on drop, same as row_change_strategy.py's _QwenClient.
    """

    _RECONNECT_DELAY = 3.0

    def __init__(self, url: str):
        self._url   = url
        self._ws    = None
        self._ready = False

        self._response_event = threading.Event()
        self._pending: dict | None = None
        self._lock = threading.Lock()

        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._run_loop, daemon=True,
                         name="moondream-ws").start()

    @property
    def ready(self) -> bool:
        return self._ready

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_loop())

    async def _connect_loop(self) -> None:
        import websockets
        delay = self._RECONNECT_DELAY
        while True:
            try:
                log.info("Connecting to Moondream server: %s", self._url)
                async with websockets.connect(self._url,
                                              ping_interval=30,
                                              ping_timeout=120) as ws:
                    self._ws    = ws
                    self._ready = True
                    delay       = self._RECONNECT_DELAY
                    log.info("Moondream server connected")
                    await self._recv_loop(ws)
            except Exception as e:
                log.warning("Moondream server disconnected (%s) — retry in %.0fs", e, delay)
            finally:
                self._ws    = None
                self._ready = False
                with self._lock:
                    self._pending = None
                self._response_event.set()   # unblock any waiting detect()
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 30.0)

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") in ("detections", "error", "ready"):
                with self._lock:
                    self._pending = msg
                self._response_event.set()

    def detect(self, frame_jpeg: bytes, object_name: str,
              timeout: float = 30.0) -> "list[dict] | None":
        """
        Send one frame + object name, block until the server replies.

        Returns a list of {"x_min","y_min","x_max","y_max"} pixel-coordinate
        dicts (empty list if nothing found), or None on timeout / error /
        disconnect.
        """
        if not self._ready or self._ws is None:
            log.warning("Moondream server not connected — skipping detect")
            return None

        frame_b64 = base64.b64encode(frame_jpeg).decode()

        self._response_event.clear()
        with self._lock:
            self._pending = None

        try:
            asyncio.run_coroutine_threadsafe(
                self._ws.send(json.dumps({
                    "type":       "detect",
                    "object":     object_name,
                    "frame_b64":  frame_b64,
                })),
                self._loop,
            ).result(timeout=5.0)
        except Exception as e:
            log.warning("Moondream send failed: %s", e)
            return None

        if not self._response_event.wait(timeout=timeout):
            log.warning("Moondream detect timed out after %.0fs", timeout)
            return None

        with self._lock:
            resp = self._pending

        if resp is None:
            return None
        if resp.get("type") == "error":
            log.warning("Moondream error: %s", resp.get("message"))
            return None
        return resp.get("objects", [])
