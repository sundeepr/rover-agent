#!/usr/bin/env python3
"""
camera_ws_server.py — Stream a V4L2 camera as JPEG frames over WebSocket.

Compatible with WebSocketFrameSource in frame_source.py: each WebSocket
message is a raw JPEG byte string (no base64, no envelope).

Multiple clients can connect simultaneously — each gets its own copy of
the latest frame.  Frames are broadcast at the requested FPS regardless
of how many clients are connected.

Usage
─────
    python camera_ws_server.py --device /dev/cam-left --port 9001
    python camera_ws_server.py --device 0 --port 9001 --fps 10 --width 640 --height 480

    # On the rover:
    python rover_agent.py --strategy wheel_guard \\
        --left-cam ws://192.168.2.10:9001 \\
        --right-cam ws://192.168.2.10:9002

Dependencies
────────────
    pip install opencv-python websockets
"""

import argparse
import asyncio
import logging
import signal
import sys
import threading
import time

import cv2

log = logging.getLogger("camera_ws_server")


# ── Camera capture thread ─────────────────────────────────────────────────────

class CameraCapture:
    """
    Reads frames from a V4L2 camera on a background thread.
    The latest JPEG-encoded frame is always available via latest_jpeg.
    """

    def __init__(self, device, width: int, height: int, fps: int, quality: int):
        self._device  = device
        self._width   = width
        self._height  = height
        self._fps     = fps
        self._quality = quality

        self._lock        = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._running     = True
        self._ready       = threading.Event()

        threading.Thread(target=self._capture_loop, daemon=True,
                         name="camera-capture").start()

    @property
    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def wait_ready(self, timeout: float = 30.0) -> bool:
        return self._ready.wait(timeout)

    def stop(self) -> None:
        self._running = False

    def _capture_loop(self) -> None:
        device = self._device
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            log.error("Cannot open camera: %s", device)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS,          self._fps)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        log.info("Camera opened: %s  %dx%d  fps=%.0f", device, actual_w, actual_h, actual_fps)

        # Warmup
        for _ in range(30):
            ret, _ = cap.read()
            if ret:
                break
            time.sleep(0.05)

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._quality]
        frame_interval = 1.0 / self._fps
        consecutive_failures = 0

        while self._running:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= 30:
                    log.error("Camera %s: 30 consecutive read failures — giving up", device)
                    break
                time.sleep(0.05)
                continue

            consecutive_failures = 0
            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if ok:
                jpeg = buf.tobytes()
                with self._lock:
                    self._latest_jpeg = jpeg
                self._ready.set()

            elapsed = time.time() - t0
            sleep_s = frame_interval - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

        cap.release()
        log.info("Camera released")


# ── WebSocket broadcast server ────────────────────────────────────────────────

class StreamServer:
    """Broadcasts the latest JPEG frame to all connected WebSocket clients."""

    def __init__(self, camera: CameraCapture, fps: int):
        self._camera   = camera
        self._interval = 1.0 / fps
        self._clients: set = set()
        self._clients_lock = asyncio.Lock()

    async def handler(self, websocket) -> None:
        addr = getattr(websocket, "remote_address", "?")
        log.info("Client connected: %s  (total: %d)", addr, len(self._clients) + 1)
        async with self._clients_lock:
            self._clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            async with self._clients_lock:
                self._clients.discard(websocket)
            log.info("Client disconnected: %s  (total: %d)", addr, len(self._clients))

    async def broadcast_loop(self) -> None:
        """Send the latest frame to every connected client at the target FPS."""
        while True:
            t0 = asyncio.get_event_loop().time()
            jpeg = self._camera.latest_jpeg
            if jpeg:
                async with self._clients_lock:
                    clients = list(self._clients)
                if clients:
                    # Send to all clients, ignore individual send errors
                    results = await asyncio.gather(
                        *[_safe_send(ws, jpeg) for ws in clients],
                        return_exceptions=True,
                    )
                    dropped = sum(1 for r in results if isinstance(r, Exception))
                    if dropped:
                        log.debug("Dropped %d client(s) this frame", dropped)

            elapsed = asyncio.get_event_loop().time() - t0
            sleep_s = self._interval - elapsed
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)


async def _safe_send(websocket, data: bytes) -> None:
    try:
        await websocket.send(data)
    except Exception as e:
        raise e   # re-raise so gather can count it


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Stream a V4L2 camera as JPEG frames over WebSocket")
    parser.add_argument("--device", default="/dev/video0",
                        help="Camera device path or index (default: /dev/video0)")
    parser.add_argument("--port",   default=9001, type=int,
                        help="WebSocket port to listen on (default: 9001)")
    parser.add_argument("--host",   default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--fps",    default=10, type=int,
                        help="Capture and stream frame rate (default: 10)")
    parser.add_argument("--width",  default=640, type=int,
                        help="Camera frame width (default: 640)")
    parser.add_argument("--height", default=480, type=int,
                        help="Camera frame height (default: 480)")
    parser.add_argument("--quality", default=85, type=int,
                        help="JPEG quality 1-100 (default: 85)")
    args = parser.parse_args()

    # Accept integer device index
    device = args.device
    try:
        device = int(device)
    except (ValueError, TypeError):
        pass

    camera = CameraCapture(
        device  = device,
        width   = args.width,
        height  = args.height,
        fps     = args.fps,
        quality = args.quality,
    )

    log.info("Waiting for camera to produce first frame…")
    if not camera.wait_ready(timeout=30.0):
        log.error("Camera did not produce a frame within 30 s — exiting")
        sys.exit(1)
    log.info("Camera ready")

    async def _serve() -> None:
        import websockets

        server = StreamServer(camera, fps=args.fps)
        async with websockets.serve(server.handler, args.host, args.port,
                                    max_size=None):
            log.info("Streaming on ws://%s:%d  (%dx%d @ %d fps  quality=%d)",
                     args.host, args.port,
                     args.width, args.height, args.fps, args.quality)
            log.info("Connect with:  --left-cam ws://<this-ip>:%d", args.port)
            await server.broadcast_loop()

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        log.info("Server stopped")


if __name__ == "__main__":
    main()
