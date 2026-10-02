#!/usr/bin/env python3
"""
record_mcap.py — records the 3 RTSP cameras, RC/host telemetry, and onboard
GPS into a single timestamped .mcap file (playable in Foxglove Studio).

Sources
───────
  Cameras        : same 3 RTSP URLs as record_all_cameras.py -> topics
                   /cam/<name>, foxglove.CompressedImage (JPEG)
  RC telemetry   : ST-LINK virtual COM port, 115200 8N1 -> topic /rc_telemetry
                   Shared ASCII stream carrying 20Hz "TLM,..." lines plus
                   unrelated Debug_Printf() boot/status/diagnostic text.
                   Only lines starting with "TLM," are parsed; everything
                   else on the wire is discarded.
  GPS            : NEO-6/8-style module on the Jetson UART pins (8/10),
                   9600 baud NMEA -> topic /gps, foxglove.LocationFix.
                   Only $GPGGA sentences are parsed.

Timestamping
────────────
Every message is stamped with the *host's* wall-clock time on arrival
(time.time_ns()), not the board's `t` field or the GPS's UTC field:
  - The board's `t` is HAL_GetTick() — ms since that board's last boot/reset,
    not wall-clock, and resets across power cycles. It's kept in the message
    payload for gap/reorder sanity checks only.
  - The GPS's UTC time-of-day is meaningless before a fix and carries no date.
This keeps video, telemetry, and GPS on one common clock for alignment.

Robustness
──────────
  - Cameras and serial ports reconnect on failure/disconnect independently;
    one dropping out doesn't stop the others or crash the recording.
  - RC telemetry lines can be dropped (non-blocking UART TX) or garbled
    (partial lines on connect, or two message types interleaved) — any line
    that doesn't match the expected TLM,... field format is skipped.
  - GPS sentences with no fix (gps_qual == 0) or that fail to parse are
    skipped.

Camera credentials
──────────────────
RTSP username/password are read from the RTSP_USER / RTSP_PASSWORD env vars
(not hardcoded) — set them in a local, gitignored .env or export them before
running. See .env.example. Camera hosts/paths are not secret and stay below.

Usage
─────
    python record_mcap.py
    python record_mcap.py --out sessions/20260930_120000/recording.mcap
    python record_mcap.py --rc-port /dev/ttyACM0 --gps-port /dev/ttyTHS1
    python record_mcap.py --no-gps --no-rc      # cameras only, e.g. on a desk
    # Ctrl-C to stop cleanly.
"""

import argparse
import base64
import json
import logging
import os
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import serial

from mcap.writer import Writer

log = logging.getLogger("rover.record_mcap")

# ── Camera sources ───────────────────────────────────────────────────────────
# Same 3 camera hosts as record_all_cameras.py. Credentials come from the
# environment (RTSP_USER / RTSP_PASSWORD) rather than being hardcoded here —
# built lazily in main() so --no-cameras can skip the requirement.
_CAMERA_HOSTS = ["10.0.1.101", "10.0.1.102", "10.0.1.103"]


def _build_camera_sources() -> dict:
    user = os.environ.get("RTSP_USER", "admin")
    password = os.environ.get("RTSP_PASSWORD")
    if not password:
        raise SystemExit(
            "RTSP_PASSWORD is not set. Export RTSP_USER/RTSP_PASSWORD (see "
            ".env.example) before running, or pass --no-cameras to skip camera capture."
        )
    return {
        f"cam_{host.replace('.', '_')}":
            f"rtsp://{user}:{password}@{host}:554/video/live?channel=1&subtype=1"
        for host in _CAMERA_HOSTS
    }

# Low-latency RTSP capture flags (same as sensors/rtsp_cam.py)
_FFMPEG_OPTS = (
    "rtsp_transport;tcp"
    "|timeout;5000000"
    "|fflags;nobuffer"
    "|flags;low_delay"
    "|probesize;32768"
    "|analyzeduration;0"
)

_DEFAULT_RC_PORT   = "/dev/ttyACM0"
_DEFAULT_RC_BAUD   = 115200
_DEFAULT_GPS_PORT  = "/dev/ttyTHS1"
_DEFAULT_GPS_BAUD  = 9600

_RECONNECT_S  = 3.0
_JPEG_QUALITY = 85

_TLM_RE = re.compile(
    r"^TLM,t=(?P<t>-?\d+),armed=(?P<armed>[01]),autonomy=(?P<autonomy>[01]),"
    r"host_stale=(?P<host_stale>[01]),rc_thr=(?P<rc_thr>-?\d+),rc_str=(?P<rc_str>-?\d+),"
    r"host_L=(?P<host_L>-?\d+),host_R=(?P<host_R>-?\d+),host_aux=(?P<host_aux>-?\d+),"
    r"cmd_L=(?P<cmd_L>-?\d+),cmd_R=(?P<cmd_R>-?\d+),pump_cmd=(?P<pump_cmd>-?\d+)\s*$"
)

_TLM_INT_FIELDS = (
    "t", "armed", "autonomy", "host_stale", "rc_thr", "rc_str",
    "host_L", "host_R", "host_aux", "cmd_L", "cmd_R", "pump_cmd",
)

# ── MCAP schemas ─────────────────────────────────────────────────────────────

_COMPRESSED_IMAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {
            "type": "object",
            "properties": {
                "sec":  {"type": "integer"},
                "nsec": {"type": "integer"},
            },
        },
        "frame_id": {"type": "string"},
        "data":     {"type": "string", "contentEncoding": "base64"},
        "format":   {"type": "string"},
    },
}

_LOCATION_FIX_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {
            "type": "object",
            "properties": {
                "sec":  {"type": "integer"},
                "nsec": {"type": "integer"},
            },
        },
        "frame_id":  {"type": "string"},
        "latitude":  {"type": "number"},
        "longitude": {"type": "number"},
        "altitude":  {"type": "number"},
        "num_satellites": {"type": "integer"},
        "fix_quality":    {"type": "integer"},
        "nmea_utc_time":  {"type": "string"},
    },
}

_RC_TELEMETRY_SCHEMA = {
    "type": "object",
    "properties": {
        "board_t_ms": {"type": "integer"},
        "armed":      {"type": "integer"},
        "autonomy":   {"type": "integer"},
        "host_stale": {"type": "integer"},
        "rc_thr":     {"type": "integer"},
        "rc_str":     {"type": "integer"},
        "host_L":     {"type": "integer"},
        "host_R":     {"type": "integer"},
        "host_aux":   {"type": "integer"},
        "cmd_L":      {"type": "integer"},
        "cmd_R":      {"type": "integer"},
        "pump_cmd":   {"type": "integer"},
    },
}


def _stamp(ts_ns: int) -> dict:
    return {"sec": ts_ns // 1_000_000_000, "nsec": ts_ns % 1_000_000_000}


class McapRecorder:
    """
    Fans in camera / RC telemetry / GPS messages from separate threads into
    one mcap file via a single writer thread (mcap.writer.Writer isn't
    safe to call concurrently).
    """

    def __init__(self, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(out_path, "wb")
        self._writer = Writer(self._file)
        self._writer.start(profile="", library="rover-agent-record_mcap")

        self._image_schema_id = self._writer.register_schema(
            name="foxglove.CompressedImage",
            encoding="jsonschema",
            data=json.dumps(_COMPRESSED_IMAGE_SCHEMA).encode("utf-8"),
        )
        self._gps_schema_id = self._writer.register_schema(
            name="foxglove.LocationFix",
            encoding="jsonschema",
            data=json.dumps(_LOCATION_FIX_SCHEMA).encode("utf-8"),
        )
        self._rc_schema_id = self._writer.register_schema(
            name="rover.RcTelemetry",
            encoding="jsonschema",
            data=json.dumps(_RC_TELEMETRY_SCHEMA).encode("utf-8"),
        )

        self._channel_ids: dict[str, int] = {}
        self._queue: "queue.Queue[tuple[str, int, bytes]]" = queue.Queue(maxsize=2000)
        self._stop_event = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._drain_loop, daemon=True, name="mcap-writer"
        )
        self._writer_thread.start()

    def _channel_for(self, topic: str, schema_id: int, message_encoding: str = "json") -> int:
        if topic not in self._channel_ids:
            self._channel_ids[topic] = self._writer.register_channel(
                schema_id=schema_id, topic=topic, message_encoding=message_encoding,
            )
        return self._channel_ids[topic]

    def publish_image(self, cam_name: str, ts_ns: int, jpeg: bytes) -> None:
        msg = {
            "timestamp": _stamp(ts_ns),
            "frame_id":  cam_name,
            "data":      base64.b64encode(jpeg).decode("ascii"),
            "format":    "jpeg",
        }
        self._enqueue(f"/cam/{cam_name}", self._image_schema_id, ts_ns, msg)

    def publish_gps(self, ts_ns: int, fields: dict) -> None:
        msg = {"timestamp": _stamp(ts_ns), "frame_id": "gps", **fields}
        self._enqueue("/gps", self._gps_schema_id, ts_ns, msg)

    def publish_rc_telemetry(self, ts_ns: int, fields: dict) -> None:
        self._enqueue("/rc_telemetry", self._rc_schema_id, ts_ns, fields)

    def _enqueue(self, topic: str, schema_id: int, ts_ns: int, msg: dict) -> None:
        try:
            self._queue.put_nowait((topic, schema_id, ts_ns, json.dumps(msg).encode("utf-8")))
        except queue.Full:
            log.warning("writer queue full — dropping message on %s", topic)

    def _drain_loop(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                topic, schema_id, ts_ns, data = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            channel_id = self._channel_for(topic, schema_id)
            self._writer.add_message(
                channel_id=channel_id, log_time=ts_ns, data=data, publish_time=ts_ns,
            )

    def close(self) -> None:
        self._stop_event.set()
        self._writer_thread.join(timeout=10.0)
        self._writer.finish()
        self._file.close()


# ── Camera capture ───────────────────────────────────────────────────────────

_FAIL_STREAK_LIMIT = 20  # consecutive failed reads tolerated before reconnecting (matches sensors/rtsp_cam.py)


def _open_rtsp_capture(url: str) -> "cv2.VideoCapture | None":
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _FFMPEG_OPTS
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # best-effort; not all backends honor this
    except Exception:
        pass
    for _ in range(3):
        cap.read()
    return cap


def _camera_loop(name: str, url: str, recorder: McapRecorder, running: threading.Event) -> None:
    # No artificial fps throttle here: read as fast as the source delivers.  A
    # sleep-then-read loop lets each camera's own network/ffmpeg buffer build
    # up a backlog while we sleep, and cap.read() hands back the OLDEST
    # buffered frame first — how much backlog accumulates differs per camera
    # (bitrate, RTT, encoder latency), which is what made 101/102/103 drift
    # out of sync even though every frame is stamped on arrival.
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
    safe_url = url.split("@")[-1] if "@" in url else url
    cap = None
    fail_streak = 0

    while running.is_set():
        if cap is None:
            log.info("[%s] connecting to %s …", name, safe_url)
            cap = _open_rtsp_capture(url)
            if cap is None:
                log.warning("[%s] failed to open — retrying in %.0fs", name, _RECONNECT_S)
                time.sleep(_RECONNECT_S)
                continue
            log.info("[%s] connected", name)
            fail_streak = 0

        ret, frame = cap.read()
        if not ret or frame is None:
            fail_streak += 1
            if fail_streak > _FAIL_STREAK_LIMIT:
                # Sustained failure, not a one-off hiccup -- actually reconnect.
                log.warning("[%s] %d consecutive read failures — reconnecting",
                            name, fail_streak)
                cap.release()
                cap = None
                time.sleep(_RECONNECT_S)
                fail_streak = 0
            continue
        fail_streak = 0

        ok, buf = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            recorder.publish_image(name, time.time_ns(), buf.tobytes())

    if cap is not None:
        cap.release()
    log.info("[%s] camera thread stopped", name)


# ── RC telemetry ─────────────────────────────────────────────────────────────

def _rc_telemetry_loop(port: str, baud: int, recorder: McapRecorder, running: threading.Event) -> None:
    ser: Optional[serial.Serial] = None

    while running.is_set():
        if ser is None:
            try:
                ser = serial.Serial(port, baudrate=baud, timeout=1.0)
                log.info("rc_telemetry: connected to %s @ %d", port, baud)
            except Exception as exc:
                log.warning("rc_telemetry: failed to open %s (%s) — retrying in %.0fs",
                            port, exc, _RECONNECT_S)
                time.sleep(_RECONNECT_S)
                continue

        try:
            raw = ser.readline()
        except Exception as exc:
            log.warning("rc_telemetry: read error (%s) — reconnecting", exc)
            ser.close()
            ser = None
            time.sleep(_RECONNECT_S)
            continue

        if not raw:
            continue  # read timeout, no line yet

        line = raw.decode("ascii", errors="replace").strip()
        if not line.startswith("TLM,"):
            continue  # boot banner / debug printf / arming diagnostics etc.

        m = _TLM_RE.match(line)
        if not m:
            continue  # partial or garbled line — skip rather than crash

        ts_ns = time.time_ns()
        fields = {"board_t_ms": int(m.group("t"))}
        fields.update({k: int(m.group(k)) for k in _TLM_INT_FIELDS if k != "t"})
        recorder.publish_rc_telemetry(ts_ns, fields)

    if ser is not None:
        ser.close()
    log.info("rc_telemetry: thread stopped")


# ── GPS ───────────────────────────────────────────────────────────────────────

def _gps_loop(port: str, baud: int, recorder: McapRecorder, running: threading.Event) -> None:
    import pynmea2

    ser: Optional[serial.Serial] = None

    while running.is_set():
        if ser is None:
            try:
                ser = serial.Serial(port, baudrate=baud, timeout=1.0)
                log.info("gps: connected to %s @ %d", port, baud)
            except Exception as exc:
                log.warning("gps: failed to open %s (%s) — retrying in %.0fs",
                            port, exc, _RECONNECT_S)
                time.sleep(_RECONNECT_S)
                continue

        try:
            raw = ser.readline()
        except Exception as exc:
            log.warning("gps: read error (%s) — reconnecting", exc)
            ser.close()
            ser = None
            time.sleep(_RECONNECT_S)
            continue

        if not raw:
            continue

        line = raw.decode("ascii", errors="replace").strip()
        if "$GPGGA" not in line:
            continue

        try:
            msg = pynmea2.parse(line)
        except Exception:
            continue  # malformed sentence — skip

        if msg.gps_qual is None or msg.gps_qual == 0:
            continue  # no fix yet

        ts_ns = time.time_ns()
        recorder.publish_gps(ts_ns, {
            "latitude":       float(msg.latitude),
            "longitude":      float(msg.longitude),
            "altitude":       float(msg.altitude) if msg.altitude is not None else 0.0,
            "num_satellites": int(msg.num_sats) if msg.num_sats else 0,
            "fix_quality":    int(msg.gps_qual),
            "nmea_utc_time":  str(msg.timestamp) if msg.timestamp else "",
        })

    if ser is not None:
        ser.close()
    log.info("gps: thread stopped")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=None,
                        help="Output .mcap path (default: sessions/<timestamp>/recording.mcap)")
    parser.add_argument("--rc-port", default=_DEFAULT_RC_PORT)
    parser.add_argument("--rc-baud", type=int, default=_DEFAULT_RC_BAUD)
    parser.add_argument("--gps-port", default=_DEFAULT_GPS_PORT)
    parser.add_argument("--gps-baud", type=int, default=_DEFAULT_GPS_BAUD)
    parser.add_argument("--no-rc", action="store_true", help="Disable RC telemetry capture")
    parser.add_argument("--no-gps", action="store_true", help="Disable GPS capture")
    parser.add_argument("--no-cameras", action="store_true", help="Disable camera capture")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    out_path = args.out
    if out_path is None:
        session_dir = Path("sessions") / datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = session_dir / "recording.mcap"

    log.info("Recording to %s", out_path.resolve())
    recorder = McapRecorder(out_path)

    running = threading.Event()
    running.set()
    threads = []

    if not args.no_cameras:
        for name, url in _build_camera_sources().items():
            t = threading.Thread(target=_camera_loop, args=(name, url, recorder, running),
                                 daemon=True, name=name)
            threads.append(t)

    if not args.no_rc:
        threads.append(threading.Thread(
            target=_rc_telemetry_loop, args=(args.rc_port, args.rc_baud, recorder, running),
            daemon=True, name="rc_telemetry"))

    if not args.no_gps:
        threads.append(threading.Thread(
            target=_gps_loop, args=(args.gps_port, args.gps_baud, recorder, running),
            daemon=True, name="gps"))

    for t in threads:
        t.start()

    log.info("Recording %d source(s). Ctrl-C to stop.", len(threads))
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        log.info("Stopping…")
    finally:
        running.clear()
        for t in threads:
            t.join(timeout=5.0)
        recorder.close()
        log.info("Recording closed: %s", out_path.resolve())


if __name__ == "__main__":
    main()
