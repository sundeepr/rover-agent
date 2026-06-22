"""
FrameSource — unified interface for camera frames regardless of origin.

Three implementations:
  DeviceFrameSource    — wraps cv2.VideoCapture (USB / V4L2 camera)
  WebSocketFrameSource — receives JPEG frames from a WebSocket server
  RtspFrameSource      — reads from an RTSP stream with auto-reconnect

Factory
-------
  open_frame_source(device_or_url, label, cam_controls, fps) → FrameSource | None

Dispatch:
  "ws://" or "wss://"   → WebSocketFrameSource
  "rtsp://" or "rtsps://" → RtspFrameSource
  int or "/dev/..."      → DeviceFrameSource (V4L2)

WebSocket frame protocol
------------------------
The server sends one frame per WebSocket message, either as:
  - raw binary JPEG bytes  (preferred — lowest latency)
  - base64-encoded string  (fallback, auto-detected)

Usage
-----
  src = open_frame_source("/dev/cam-left", "left")
  src = open_frame_source("ws://192.168.1.10:9000/left", "left")
  src = open_frame_source("rtsp://user:pass@192.168.1.100:554/video/live?channel=1&subtype=1", "left")

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
    """
    Base class for all camera/sensor frame sources.

    Any FrameSource can have a recorder attached after construction:
        src.attach_recorder(session_recorder, "left_wheel")

    Once attached, every new frame is automatically written to the session
    via recorder.record(stream_name, frame).  Subclasses with background
    threads record in the thread (one write per captured frame).
    DeviceFrameSource records on each read() call.
    """

    def __init__(self) -> None:
        self._recorder   = None
        self._rec_name:  str | None = None
        self._rec_fps:   float = 10.0

    def attach_recorder(self, recorder, stream_name: str,
                        fps: float = 10.0) -> None:
        """Attach a SessionRecorder so every captured frame is auto-recorded."""
        self._recorder  = recorder
        self._rec_name  = stream_name
        self._rec_fps   = fps

    def _auto_record(self, frame: np.ndarray) -> None:
        """Call from subclasses whenever a new frame is captured."""
        if self._recorder is not None and frame is not None:
            self._recorder.record(self._rec_name, frame, fps=self._rec_fps)

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
        super().__init__()
        self._cap = cap

    def read(self) -> np.ndarray | None:
        if self._cap is None or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        if ret and frame is not None:
            self._auto_record(frame)
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
        super().__init__()
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
                            self._auto_record(frame)
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


# ── RTSP camera ───────────────────────────────────────────────────────────────

# Low-latency FFmpeg options for RTSP — set as env var before VideoCapture
_RTSP_FFMPEG_OPTS = (
    "rtsp_transport;tcp"
    "|timeout;5000000"
    "|fflags;nobuffer"
    "|flags;low_delay"
    "|probesize;32768"
    "|analyzeduration;0"
)
_RTSP_RECONNECT_S = 3.0


class RtspFrameSource(FrameSource):
    """
    Reads frames from an RTSP stream on a background thread.

    A dedicated reader thread continuously pulls frames from the stream and
    stores only the latest one.  read() always returns immediately with the
    freshest available frame — no buffering, no growing latency.

    Auto-reconnects on stream drop after _RTSP_RECONNECT_S seconds.
    """

    def __init__(self, url: str, label: str = "rtsp") -> None:
        super().__init__()
        self._url     = url
        self._label   = label
        self._frame:  np.ndarray | None = None
        self._lock    = threading.Lock()
        self._open    = False
        self._running = True
        threading.Thread(target=self._reader, daemon=True,
                         name=f"rtsp-{label}").start()
        safe = url.split("@")[-1] if "@" in url else url
        log.info("RtspFrameSource: connecting to %s", safe)

    def read(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def is_open(self) -> bool:
        return self._open

    def release(self) -> None:
        self._running = False

    def _open_cap(self) -> cv2.VideoCapture | None:
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _RTSP_FFMPEG_OPTS
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return None
        # Warmup — let the decoder settle reference frames
        for _ in range(5):
            cap.read()
        return cap

    def _reader(self) -> None:
        safe = self._url.split("@")[-1] if "@" in self._url else self._url
        cap  = None
        fail = 0

        while self._running:
            if cap is None or not cap.isOpened():
                self._open = False
                cap = self._open_cap()
                if cap is None:
                    log.warning("RtspFrameSource %s: open failed — retry in %.0fs",
                                safe, _RTSP_RECONNECT_S)
                    time.sleep(_RTSP_RECONNECT_S)
                    continue
                log.info("RtspFrameSource %s: connected  %dx%d @ %.0f fps",
                         safe,
                         int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                         cap.get(cv2.CAP_PROP_FPS) or 0)
                self._open = True
                fail = 0

            ret, frame = cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
                self._auto_record(frame)
                fail = 0
            else:
                fail += 1
                if fail > 20:
                    log.warning("RtspFrameSource %s: too many read failures — reconnecting",
                                safe)
                    cap.release()
                    cap = None

        if cap is not None:
            cap.release()
        log.info("RtspFrameSource %s: stopped", safe)


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

    if isinstance(device_or_url, str) and device_or_url.startswith(("rtsp://", "rtsps://")):
        return RtspFrameSource(device_or_url, label=label)

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
