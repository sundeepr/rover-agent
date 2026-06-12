"""
RtspCamStreamer — re-publishes an RTSP camera stream over WebSocket as JPEG.

Reads frames from an RTSP source (e.g. a CP Plus IP camera) and broadcasts
them to all connected WebSocket clients as raw JPEG bytes — the same protocol
used by sensors/webcam_stream.py and consumed by frame_source.WebSocketFrameSource.

This lets any rover-agent strategy use an IP camera via --device or --left-cam:

    # 1. Start the RTSP bridge (in a separate terminal or as a background service)
    python -m sensors.rtsp_cam

    # 2. Point the rover agent at it
    python rover_agent.py --strategy crop_row \\
        --rover atlas --atlas-port /dev/ttyACM0 \\
        --left-cam ws://localhost:5011

CP Plus default streams (confirmed working):
    Main  (1280x720): rtsp://admin:pass@192.168.1.100:554/video/live?channel=1&subtype=0
    Sub 1  (640x480): rtsp://admin:pass@192.168.1.100:554/video/live?channel=1&subtype=1

Protocol:
    server → client:  raw JPEG bytes (one message per frame, no JSON wrapper)

Multiple clients can connect simultaneously — every client receives every frame.

Usage (standalone):
    python -m sensors.rtsp_cam
    python -m sensors.rtsp_cam --url "rtsp://admin:pass@ip:554/video/live?channel=1&subtype=1"
    python -m sensors.rtsp_cam --port 5011 --fps 15 --quality 75

Usage (embedded in another process):
    from sensors.rtsp_cam import RtspCamStreamer
    streamer = RtspCamStreamer(url="rtsp://...", port=5011)
    streamer.start()   # non-blocking daemon threads
    ...
    streamer.stop()
"""

import asyncio
import logging
import os
import threading
import time

import cv2
import numpy as np

log = logging.getLogger("rover.sensors.rtsp_cam")

_DEFAULT_PORT    = 5011
_DEFAULT_FPS     = 15
_DEFAULT_QUALITY = 80
_RECONNECT_DELAY = 3.0    # seconds between reconnect attempts

# CP Plus sub-stream — 640×480 is ideal for vision analysis
_DEFAULT_URL = (
    "rtsp://admin:Cam3ra_1234@192.168.1.100:554"
    "/video/live?channel=1&subtype=1"
)

# Low-latency RTSP capture flags — reduce buffering so frames arrive fresh
_FFMPEG_OPTS = (
    "rtsp_transport;tcp"
    "|timeout;5000000"
    "|fflags;nobuffer"
    "|flags;low_delay"
    "|probesize;32768"
    "|analyzeduration;0"
)


class RtspCamStreamer:
    """
    Reads frames from an RTSP stream and pushes them as JPEG over WebSocket.

    Parameters
    ----------
    url : str
        Full RTSP URL including credentials.
    port : int
        WebSocket port to listen on (default 5011).
    fps : int
        Target broadcast rate.  The reader always grabs the latest frame;
        this controls how fast it is sent to clients.
    quality : int
        JPEG encode quality 0-100 (default 80).
    """

    def __init__(
        self,
        url:     str = _DEFAULT_URL,
        port:    int = _DEFAULT_PORT,
        fps:     int = _DEFAULT_FPS,
        quality: int = _DEFAULT_QUALITY,
    ) -> None:
        self._url     = url
        self._port    = port
        self._fps     = fps
        self._quality = quality

        self._latest_jpeg:  bytes | None = None
        self._jpeg_lock     = threading.Lock()
        self._clients: set  = set()
        self._clients_lock  = threading.Lock()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event = threading.Event()

        # Diagnostics
        self._frames_captured = 0
        self._reconnects      = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start capture and WebSocket server in background daemon threads."""
        threading.Thread(target=self._capture_loop, daemon=True,
                         name="rtsp-capture").start()
        threading.Thread(target=self._ws_loop, daemon=True,
                         name="rtsp-ws").start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ── RTSP capture thread ───────────────────────────────────────────────────

    def _open_capture(self) -> cv2.VideoCapture | None:
        """Open the RTSP stream with low-latency FFmpeg options."""
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _FFMPEG_OPTS
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return None
        # Warmup — discard first few frames (exposure / reference frame settle)
        for _ in range(3):
            cap.read()
        return cap

    def _capture_loop(self) -> None:
        interval      = 1.0 / self._fps
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._quality]
        safe_url      = self._url.split("@")[-1] if "@" in self._url else self._url

        log.info("rtsp_cam: connecting to %s", safe_url)
        cap = self._open_capture()
        if cap is None:
            log.error("rtsp_cam: could not open %s", safe_url)

        fail_streak = 0

        while not self._stop_event.is_set():
            t0 = time.monotonic()

            if cap is None or not cap.isOpened():
                log.warning("rtsp_cam: reconnecting in %.0fs…", _RECONNECT_DELAY)
                time.sleep(_RECONNECT_DELAY)
                cap = self._open_capture()
                if cap is None:
                    self._reconnects += 1
                    continue
                log.info("rtsp_cam: reconnected (attempt %d)", self._reconnects)
                fail_streak = 0

            ret, frame = cap.read()
            if not ret or frame is None:
                fail_streak += 1
                if fail_streak > 20:
                    log.warning("rtsp_cam: too many read failures — reopening")
                    cap.release()
                    cap = None
                    fail_streak = 0
                continue
            fail_streak = 0

            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue

            jpeg = buf.tobytes()
            self._frames_captured += 1

            with self._jpeg_lock:
                self._latest_jpeg = jpeg

            # Notify the WS loop to broadcast
            if self._loop is not None and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(
                    self._loop.create_task,
                    self._broadcast(jpeg),
                )

            elapsed = time.monotonic() - t0
            wait    = interval - elapsed
            if wait > 0:
                time.sleep(wait)

        if cap is not None:
            cap.release()
        log.info("rtsp_cam: capture loop stopped (%d frames captured)",
                 self._frames_captured)

    # ── WebSocket server thread ───────────────────────────────────────────────

    def _ws_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as exc:
            log.error("rtsp_cam WS server error: %s", exc)

    async def _serve(self) -> None:
        try:
            import websockets
        except ImportError:
            log.error("'websockets' not installed — rtsp_cam disabled. "
                      "Run: pip install 'websockets>=12.0'")
            return

        log.info("rtsp_cam: WebSocket server on ws://0.0.0.0:%d", self._port)
        async with websockets.serve(self._handle, "0.0.0.0", self._port,
                                    ping_interval=10, ping_timeout=20):
            await asyncio.get_event_loop().create_future()  # run forever

    async def _handle(self, ws) -> None:
        client = ws.remote_address
        log.info("rtsp_cam: client connected %s", client)
        with self._clients_lock:
            self._clients.add(ws)

        # Send latest frame immediately so client isn't blank on connect
        with self._jpeg_lock:
            initial = self._latest_jpeg
        if initial:
            try:
                await ws.send(initial)
            except Exception:
                pass

        try:
            await ws.wait_closed()
        finally:
            with self._clients_lock:
                self._clients.discard(ws)
            log.info("rtsp_cam: client disconnected %s", client)

    async def _broadcast(self, jpeg: bytes) -> None:
        with self._clients_lock:
            targets = list(self._clients)
        if not targets:
            return
        for ws in targets:
            try:
                await ws.send(jpeg)
            except Exception:
                pass


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Publish an RTSP camera stream over WebSocket as JPEG"
    )
    parser.add_argument("--url", default=_DEFAULT_URL,
                        help="RTSP source URL (default: CP Plus sub-stream at 192.168.1.100)")
    parser.add_argument("--port",    type=int, default=_DEFAULT_PORT,
                        help=f"WebSocket port (default: {_DEFAULT_PORT})")
    parser.add_argument("--fps",     type=int, default=_DEFAULT_FPS,
                        help=f"Broadcast frame rate (default: {_DEFAULT_FPS})")
    parser.add_argument("--quality", type=int, default=_DEFAULT_QUALITY,
                        help=f"JPEG quality 0-100 (default: {_DEFAULT_QUALITY})")
    parser.add_argument("--main",    action="store_true",
                        help="Use main stream (1280x720) instead of sub-stream (640x480)")
    args = parser.parse_args()

    url = args.url
    if args.main and url == _DEFAULT_URL:
        url = url.replace("subtype=1", "subtype=0")

    safe = url.split("@")[-1] if "@" in url else url
    streamer = RtspCamStreamer(url=url, port=args.port,
                               fps=args.fps, quality=args.quality)
    streamer.start()

    log.info("Streaming rtsp://%s → ws://0.0.0.0:%d  (Ctrl-C to stop)",
             safe, args.port)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        log.info("Stopping")
        streamer.stop()
