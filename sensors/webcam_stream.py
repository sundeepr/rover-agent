"""
WebcamStreamer — streams camera frames to WebSocket clients as JPEG.

Captures frames from a V4L2 camera and broadcasts them to all connected
WebSocket clients at a configurable frame rate.  Clients receive raw JPEG
bytes; no JSON envelope so overhead is minimal.

Protocol:
    server → client:  raw JPEG bytes (one message per frame)

Multiple clients can connect; every client gets every frame.
Late-joining clients receive the next frame captured after they connect.

Usage (standalone):
    python -m sensors.webcam_stream --device 0 --port 5010 --fps 15

Usage (from rover_agent or another process):
    from sensors.webcam_stream import WebcamStreamer
    streamer = WebcamStreamer(device=0, port=5010, fps=15)
    streamer.start()   # non-blocking — runs in daemon threads
    ...
    streamer.stop()
"""

import asyncio
import logging
import threading
import time

import cv2

log = logging.getLogger("rover.sensors.webcam_stream")

_DEFAULT_PORT   = 5010
_DEFAULT_FPS    = 15
_DEFAULT_DEVICE = 0
_JPEG_QUALITY   = 80   # 0-100; lower = smaller messages, faster on slow Wi-Fi


class WebcamStreamer:
    """
    Captures frames from a webcam and pushes them as JPEG over WebSocket.

    Parameters
    ----------
    device : int | str
        V4L2 device index or path (e.g. 0 or '/dev/video0').
    port : int
        WebSocket port to listen on.
    fps : int
        Target capture and broadcast rate.
    width, height : int
        Requested camera resolution (camera may choose nearest supported size).
    """

    def __init__(
        self,
        device: int | str = _DEFAULT_DEVICE,
        port: int = _DEFAULT_PORT,
        fps: int = _DEFAULT_FPS,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self._device  = device
        self._port    = port
        self._fps     = fps
        self._width   = width
        self._height  = height

        self._latest_jpeg: bytes | None = None
        self._jpeg_lock   = threading.Lock()
        self._clients: set = set()
        self._clients_lock = threading.Lock()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event = threading.Event()

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start capture and WebSocket server in background daemon threads."""
        threading.Thread(target=self._capture_loop, daemon=True,
                         name="webcam-capture").start()
        threading.Thread(target=self._ws_loop, daemon=True,
                         name="webcam-ws").start()

    def stop(self) -> None:
        """Signal both threads to shut down."""
        self._stop_event.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ── Camera capture thread ──────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
        if not cap.isOpened():
            log.error("webcam_stream: could not open device %s — trying default backend",
                      self._device)
            cap = cv2.VideoCapture(self._device)
        if not cap.isOpened():
            log.error("webcam_stream: failed to open device %s", self._device)
            return

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("webcam_stream: opened device %s at %dx%d, target %d fps",
                 self._device, actual_w, actual_h, self._fps)

        # Warmup — discard first few frames (Jetson USB cameras need this)
        for _ in range(5):
            cap.read()

        interval = 1.0 / self._fps
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]

        while not self._stop_event.is_set():
            t0 = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                log.warning("webcam_stream: frame read failed — skipping")
                time.sleep(interval)
                continue

            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue

            jpeg = buf.tobytes()
            with self._jpeg_lock:
                self._latest_jpeg = jpeg

            # Notify the WS loop to broadcast
            if self._loop is not None and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(
                    self._loop.create_task,
                    self._broadcast(jpeg),
                )

            elapsed = time.monotonic() - t0
            wait = interval - elapsed
            if wait > 0:
                time.sleep(wait)

        cap.release()
        log.info("webcam_stream: capture loop stopped")

    # ── WebSocket server thread ────────────────────────────────────────────────

    def _ws_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as exc:
            log.error("webcam_stream WS server error: %s", exc)

    async def _serve(self) -> None:
        try:
            import websockets
        except ImportError:
            log.error("'websockets' not installed — webcam_stream disabled. "
                      "Run: pip install 'websockets>=12.0'")
            return

        log.info("webcam_stream: WebSocket server listening on ws://0.0.0.0:%d",
                 self._port)
        async with websockets.serve(self._handle, "0.0.0.0", self._port,
                                    ping_interval=10, ping_timeout=20):
            await asyncio.get_event_loop().create_future()  # run forever

    async def _handle(self, ws) -> None:
        client = ws.remote_address
        log.info("webcam_stream: client connected %s", client)
        with self._clients_lock:
            self._clients.add(ws)

        # Send the last captured frame immediately so the client isn't blank
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
            log.info("webcam_stream: client disconnected %s", client)

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

    parser = argparse.ArgumentParser(description="Stream a webcam over WebSocket")
    parser.add_argument("--device", type=int, default=_DEFAULT_DEVICE,
                        help="Camera device index (default: 0)")
    parser.add_argument("--port",   type=int, default=_DEFAULT_PORT,
                        help=f"WebSocket port (default: {_DEFAULT_PORT})")
    parser.add_argument("--fps",    type=int, default=_DEFAULT_FPS,
                        help=f"Target frame rate (default: {_DEFAULT_FPS})")
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    streamer = WebcamStreamer(
        device=args.device,
        port=args.port,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    streamer.start()

    log.info("Streaming ws://0.0.0.0:%d — Ctrl-C to stop", args.port)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        log.info("Interrupted — stopping")
        streamer.stop()
