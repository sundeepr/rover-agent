"""
FrameSource — unified interface for camera frames regardless of origin.

Two implementations:
  DeviceFrameSource    — wraps cv2.VideoCapture (USB / V4L2 camera)
  WebSocketFrameSource — receives JPEG frames from a WebSocket server

Factory
-------
  open_frame_source(device_or_url, label, cam_controls, fps) → FrameSource | None

If device_or_url starts with "ws://" or "wss://" a WebSocketFrameSource is
returned; otherwise a DeviceFrameSource is opened via V4L2.

WebSocket frame protocol
------------------------
The server sends one frame per WebSocket message, either as:
  - raw binary JPEG bytes  (preferred — lowest latency)
  - base64-encoded string  (fallback, auto-detected)

Usage
-----
  src = open_frame_source("/dev/cam-left", "left")
  src = open_frame_source("ws://192.168.1.10:9000/left", "left")

  frame = src.read()      # np.ndarray BGR | None
  src.is_open()           # bool
  src.release()           # cleanup
"""

import base64
import logging
import threading
import time

import cv2
import numpy as np

log = logging.getLogger("rover.frame_source")


# ── Abstract base ─────────────────────────────────────────────────────────────

class FrameSource:
    def read(self) -> np.ndarray | None:
        raise NotImplementedError

    def is_open(self) -> bool:
        raise NotImplementedError

    def release(self) -> None:
        pass


# ── USB / V4L2 camera ─────────────────────────────────────────────────────────

class DeviceFrameSource(FrameSource):
    """Wraps a cv2.VideoCapture opened on a local device."""

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap

    def read(self) -> np.ndarray | None:
        if self._cap is None or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None


# ── WebSocket camera ──────────────────────────────────────────────────────────

class WebSocketFrameSource(FrameSource):
    """
    Receives JPEG frames from a WebSocket server on a background thread.

    read() always returns the most recent decoded frame (or None while
    connecting).  The background thread reconnects automatically on disconnect.
    """

    _RECONNECT_DELAY_S = 2.0

    def __init__(self, url: str):
        self._url     = url
        self._frame: np.ndarray | None = None
        self._lock    = threading.Lock()
        self._open    = False
        self._running = True
        threading.Thread(target=self._recv_thread, daemon=True,
                         name=f"ws-cam-{url}").start()
        log.info("WebSocketFrameSource: connecting to %s", url)

    def read(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def is_open(self) -> bool:
        return self._open

    def release(self) -> None:
        self._running = False

    # ── Background receive loop ───────────────────────────────────────────────

    def _recv_thread(self) -> None:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._connect_loop())

    async def _connect_loop(self) -> None:
        try:
            import websockets
        except ImportError:
            log.error("websockets package not installed — WebSocketFrameSource unavailable")
            return

        while self._running:
            try:
                async with websockets.connect(self._url,
                                              ping_interval=20,
                                              ping_timeout=10) as ws:
                    self._open = True
                    log.info("WebSocketFrameSource connected: %s", self._url)
                    async for msg in ws:
                        if not self._running:
                            break
                        frame = self._decode(msg)
                        if frame is not None:
                            with self._lock:
                                self._frame = frame
            except Exception as e:
                self._open = False
                if self._running:
                    log.warning("WebSocketFrameSource %s disconnected (%s) — retry in %.0fs",
                                self._url, e, self._RECONNECT_DELAY_S)
                    import asyncio as _a
                    await _a.sleep(self._RECONNECT_DELAY_S)

    @staticmethod
    def _decode(msg) -> np.ndarray | None:
        try:
            data = msg if isinstance(msg, (bytes, bytearray)) else base64.b64decode(msg)
            buf  = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:
            return None


# ── Factory ───────────────────────────────────────────────────────────────────

def open_frame_source(device_or_url,
                      label: str,
                      cam_controls: dict | None = None,
                      fps: int = 10) -> FrameSource | None:
    """
    Open a FrameSource for the given device or WebSocket URL.

    device_or_url:
      int or "/dev/..." path → DeviceFrameSource (V4L2)
      "ws://..." or "wss://..." → WebSocketFrameSource

    Returns None if the device cannot be opened.
    """
    if isinstance(device_or_url, str) and device_or_url.startswith(("ws://", "wss://")):
        return WebSocketFrameSource(device_or_url)

    # ── Local device ──────────────────────────────────────────────────────────
    path = device_or_url if isinstance(device_or_url, str) else f"/dev/video{device_or_url}"
    cap  = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        log.warning("%s camera NOT FOUND at %s", label, path)
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Apply v4l2 controls (exposure, saturation, backlight, …)
    if cam_controls:
        import subprocess, shlex
        ctrl_str = ",".join(f"{k}={v}" for k, v in cam_controls.items())
        if "exposure_time_absolute" in cam_controls:
            subprocess.run(
                shlex.split(f"v4l2-ctl --device {path} --set-ctrl=auto_exposure=1"),
                check=False, capture_output=True)
        subprocess.run(
            shlex.split(f"v4l2-ctl --device {path} --set-ctrl={ctrl_str}"),
            check=False, capture_output=True)
        log.info("%s camera %s: applied controls %s", label, path, ctrl_str)
    else:
        log.info("%s camera %s: using auto exposure", label, path)

    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str    = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))
    actual_fps    = cap.get(cv2.CAP_PROP_FPS)

    # Warmup reads
    for _ in range(60):
        ret, _ = cap.read()
        if ret:
            log.info("%s camera ready at %s  (%dx%d)  fourcc=%s  fps=%.0f",
                     label, path,
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     fourcc_str, actual_fps)
            return DeviceFrameSource(cap)
        time.sleep(0.05)

    log.warning("%s camera %s opened but delivers no frames", label, path)
    cap.release()
    return None
