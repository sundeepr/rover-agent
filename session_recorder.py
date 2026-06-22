"""
SessionRecorder — persists raw video, annotated video, and LLM/VLM decisions
for one agent run.

The sessions directory and minimum free-disk threshold are read from config.json
in the working directory.  If config.json is absent the built-in defaults are
used silently.

On startup SessionRecorder:
  1. Reads config.json (falls back to defaults if missing or malformed).
  2. Resolves the sessions directory:
       - Uses the path from config.json if it exists on disk.
       - Falls back to ./sessions/ with a WARNING if it does not.
  3. Checks free disk space on the target drive and logs a WARNING if it is
     below min_free_gb (default 1 GB).

Creates a new timestamped directory under the resolved sessions root each run:

    <sessions_dir>/
      YYYYMMDD_HHMMSS/
        raw.avi        — front camera raw feed (MJPG; crash-safe)
        down.avi       — downward camera raw feed (MJPG; when --down-device set)
        data.jsonl     — one JSON record per inference step containing:
                           waypoints, drive command (vel, radius, L%, R%),
                           vegetation detection (walls, gap, blobs),
                           trajectory arc pixel points, intersection side,
                           ICR offset, goal string
        decisions.jsonl — one JSON record per LLM/VLM decision step (legacy)
        events.jsonl   — every drive command, goal change, pause/resume, and
                         operator joystick input; each record has ts (epoch
                         float) and frame_idx for video correlation

write_frames() is called from the agent_loop thread on every camera tick.
write_decision() is called from strategy threads (thread-safe via lock).
write_data() is called from strategy threads with per-step computed values.
close() is called once on agent shutdown.
"""

import json
import logging
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("rover.session_recorder")

_CONFIG_PATH = Path("config.json")
_DEFAULTS = {"sessions_dir": "sessions", "min_free_gb": 1.0}


# ── Config helpers ─────────────────────────────────────────────────────────────

def _load_config() -> dict:
    """Read config.json; return merged defaults on any error."""
    if not _CONFIG_PATH.exists():
        return dict(_DEFAULTS)
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        return {**_DEFAULTS, **data}
    except Exception as exc:
        log.warning("Could not read %s (%s) — using defaults", _CONFIG_PATH, exc)
        return dict(_DEFAULTS)


def _resolve_sessions_dir(configured: Path) -> Path:
    """
    Return the sessions root to use.

    If the configured path exists (or can be created), use it.
    Otherwise fall back to ./sessions/ and emit a WARNING.
    """
    if configured.exists():
        return configured

    # Try to create it — the user may have specified a valid-but-not-yet-created path
    try:
        configured.mkdir(parents=True, exist_ok=True)
        return configured
    except OSError:
        pass

    fallback = Path("sessions")
    log.warning(
        "Configured sessions_dir %r does not exist and could not be created — "
        "falling back to %r",
        str(configured),
        str(fallback.resolve()),
    )
    return fallback


def _check_disk_space(path: Path, min_free_gb: float) -> None:
    """
    Warn if free space on the drive containing *path* is below min_free_gb.

    Uses the nearest existing ancestor so the check works even when the target
    directory has not been created yet.
    """
    check_path = path
    while not check_path.exists():
        check_path = check_path.parent

    try:
        usage = shutil.disk_usage(check_path)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < min_free_gb:
            log.warning(
                "LOW DISK SPACE: only %.2f GB free on %r "
                "(threshold: %.1f GB) — recording may fail or be truncated",
                free_gb,
                str(check_path.resolve()),
                min_free_gb,
            )
        else:
            log.info("Disk space OK: %.1f GB free", free_gb)
    except Exception as exc:
        log.warning("Could not check disk space: %s", exc)


# ── Recorder ───────────────────────────────────────────────────────────────────

class SessionRecorder:
    """Records raw video, annotated video, and LLM/VLM decisions for one run."""

    def __init__(self, fps: float = 30.0,
                 stale_threshold_s: float = 10.0,
                 watchdog_interval_s: float = 5.0) -> None:
        cfg = _load_config()

        sessions_dir = _resolve_sessions_dir(Path(cfg["sessions_dir"]))
        _check_disk_space(sessions_dir, float(cfg["min_free_gb"]))

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = sessions_dir / ts
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self._fps = fps

        # VideoWriters are lazy-initialised on the first write call so the
        # frame dimensions are known before the writers are created.
        # Only raw video is saved — annotated frames are not written to disk.
        self._raw_writer:  Optional[cv2.VideoWriter] = None
        self._down_writer: Optional[cv2.VideoWriter] = None

        # Generic named sensor streams — any camera or data source can call
        # record(name, data). Data type is detected automatically:
        #   np.ndarray → video file  (<name>.avi, lazy VideoWriter init)
        #   dict / scalar → JSONL file (<name>.jsonl, one record per call)
        self._stream_writers: dict[str, cv2.VideoWriter]  = {}   # name → VideoWriter
        self._stream_files:   dict[str, object]           = {}   # name → file handle
        self._stream_lock = threading.Lock()

        # Frame counter — incremented on every write_frames() call so that
        # decisions and events can be correlated with a specific video frame.
        self._frame_idx = 0

        # ── Video watchdog ────────────────────────────────────────────────────
        # Tracks the last time each active video stream was written to.
        # The watchdog thread checks periodically and calls _stale_callback
        # for any stream that has gone silent longer than _stale_threshold_s.
        self._video_last_write: dict[str, float] = {}   # stream_name → epoch
        self._stale_threshold_s   = stale_threshold_s
        self._watchdog_interval_s = watchdog_interval_s
        self._stale_callback = None        # callable(stream_name) or None
        self._watchdog_running = True
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="recorder-watchdog"
        )
        self._watchdog_thread.start()

        self._decisions_path = self.session_dir / "decisions.jsonl"
        self._decisions_lock = threading.Lock()
        self._decisions_fh = open(self._decisions_path, "a", encoding="utf-8")

        # Unified event log: joystick, goal changes, pause/resume, model steps.
        self._events_path = self.session_dir / "events.jsonl"
        self._events_lock = threading.Lock()
        self._events_fh = open(self._events_path, "a", encoding="utf-8")

        # Per-step data log: waypoints, drive params, vegetation, arc, intersection
        self._data_path = self.session_dir / "data.jsonl"
        self._data_lock = threading.Lock()
        self._data_fh = open(self._data_path, "a", encoding="utf-8")

        log.info("Session recording started: %s", self.session_dir.resolve())

    # ── Watchdog ───────────────────────────────────────────────────────────────

    def set_stale_callback(self, fn) -> None:
        """
        Register a callable invoked when a video stream stops being written.

        fn(stream_name: str) is called once per stale stream per watchdog cycle.
        Typical use: stop the rover and terminate the session.

        Example:
            recorder.set_stale_callback(lambda name: agent_stop(f"stream {name} stalled"))
        """
        self._stale_callback = fn

    def _watchdog_loop(self) -> None:
        """Background thread: checks video streams every watchdog_interval_s."""
        # Allow a generous startup window before first check so streams have
        # time to initialise (lazy VideoWriter creation on first frame).
        time.sleep(self._stale_threshold_s + self._watchdog_interval_s)

        while self._watchdog_running:
            now = time.time()
            with self._stream_lock:
                active = dict(self._video_last_write)   # snapshot

            for name, last in active.items():
                age = now - last
                if age > self._stale_threshold_s:
                    log.error(
                        "WATCHDOG: video stream %r has not been written for %.1f s "
                        "(threshold %.0f s) — recording may be broken",
                        name, age, self._stale_threshold_s,
                    )
                    self.write_event({
                        "type":   "watchdog_stale",
                        "stream": name,
                        "age_s":  round(age, 1),
                    })
                    if self._stale_callback:
                        try:
                            self._stale_callback(name)
                        except Exception as exc:
                            log.error("Stale callback raised: %s", exc)

            time.sleep(self._watchdog_interval_s)

    # ── Video ──────────────────────────────────────────────────────────────────
    # Called from the agent_loop thread — VideoWriters need no locking.

    def write_frames(self, raw: np.ndarray) -> None:
        """Write one raw camera frame to raw.avi (lazy-init, no annotated copy)."""
        h, w = raw.shape[:2]
        if self._raw_writer is None:
            # MJPG in .avi writes the index incrementally — files are playable
            # even if the process is killed before close() is called.
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self._raw_writer = cv2.VideoWriter(
                str(self.session_dir / "raw.avi"), fourcc, self._fps, (w, h)
            )
            if not self._raw_writer.isOpened():
                log.error(
                    "raw.avi VideoWriter failed to open in %s — "
                    "check disk space, permissions, and OpenCV MJPG codec. "
                    "Exiting — please fix the issue and restart the agent.",
                    self.session_dir,
                )
                import os
                os._exit(1)
        if self._raw_writer:
            self._raw_writer.write(raw)
            self._video_last_write["raw"] = time.time()
        self._frame_idx += 1

    def record(self, name: str, data, fps: float | None = None) -> None:
        """
        Generic sensor recorder — thread-safe, lazy-initialised.

        data types:
          np.ndarray          → <name>.avi  (MJPG video, lazy VideoWriter init)
          dict                → <name>.jsonl (one JSON line per call, ts injected)
          int | float | str   → <name>.jsonl (wrapped as {"value": data, "ts": ...})

        fps is only used when data is an ndarray and the writer hasn't been
        created yet; defaults to self._fps.
        """
        with self._stream_lock:
            if isinstance(data, np.ndarray):
                self._record_video(name, data, fps or self._fps)
            else:
                self._record_jsonl(name, data)

    def _record_video(self, name: str, frame: np.ndarray, fps: float) -> None:
        """Write one video frame to <name>.avi (must be called under _stream_lock)."""
        if name not in self._stream_writers:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(
                str(self.session_dir / f"{name}.avi"), fourcc, fps, (w, h)
            )
            if not writer.isOpened():
                log.error("VideoWriter failed for stream %r — frames will not be saved", name)
                self._stream_writers[name] = None
            else:
                log.info("Recording stream %r → %s.avi  (%.0f fps)", name, name, fps)
                self._stream_writers[name] = writer
        writer = self._stream_writers.get(name)
        if writer:
            writer.write(frame)
            self._video_last_write[name] = time.time()

    def _record_jsonl(self, name: str, data) -> None:
        """Append one JSON record to <name>.jsonl (must be called under _stream_lock)."""
        if name not in self._stream_files:
            fh = open(self.session_dir / f"{name}.jsonl", "a", encoding="utf-8")
            self._stream_files[name] = fh
            log.info("Recording stream %r → %s.jsonl", name, name)
        fh = self._stream_files[name]
        if isinstance(data, dict):
            record = {"ts": time.time(), **data}
        else:
            record = {"ts": time.time(), "value": data}
        fh.write(json.dumps(record) + "\n")
        fh.flush()

    def write_down_frame(self, frame: np.ndarray) -> None:
        """Write one downward-camera frame to down.avi (lazy-init, thread-safe)."""
        h, w = frame.shape[:2]
        if self._down_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self._down_writer = cv2.VideoWriter(
                str(self.session_dir / "down.avi"), fourcc, self._fps, (w, h)
            )
            if not self._down_writer.isOpened():
                log.error("down.avi VideoWriter failed to open — down-camera video will not be saved")
                self._down_writer = None
        if self._down_writer:
            self._down_writer.write(frame)
            self._video_last_write["down"] = time.time()

    def write_wheel_frame(self, frame: np.ndarray, side: str) -> None:
        """Write one wheel camera frame — delegates to the generic record() API."""
        self.record(f"{side}_wheel", frame)

    # ── Per-step data log ──────────────────────────────────────────────────────
    # Called from strategy threads — guarded by a lock.

    def write_data(self, record: dict) -> None:
        """Append one JSON record to data.jsonl (thread-safe).

        Intended for per-inference-step computed values: waypoints, drive
        command, wheel powers, vegetation detection, arc points, intersection.
        Automatically injects 'ts' and 'frame_idx'.
        """
        record = {"ts": time.time(), "frame_idx": self._frame_idx, **record}
        with self._data_lock:
            self._data_fh.write(json.dumps(record) + "\n")
            self._data_fh.flush()

    # ── Decisions ──────────────────────────────────────────────────────────────
    # Called from strategy threads — guarded by a lock.

    def write_decision(self, record: dict) -> None:
        """Append one JSON record to decisions.jsonl (thread-safe).

        Also mirrors the record to events.jsonl so the event log is a
        complete single-file reconstruction source.
        """
        record = {"frame_idx": self._frame_idx, **record}
        with self._decisions_lock:
            self._decisions_fh.write(json.dumps(record) + "\n")
            self._decisions_fh.flush()
        self.write_event({"type": "omnivla_step", **record})

    # ── Events ─────────────────────────────────────────────────────────────────
    # Called from any thread — guarded by a lock.

    def write_event(self, record: dict) -> None:
        """Append one JSON record to events.jsonl (thread-safe).

        Automatically injects 'ts' (epoch float) and 'frame_idx' unless
        the caller has already set them (e.g. write_decision mirror).
        """
        if "ts" not in record:
            record = {"ts": time.time(), "frame_idx": self._frame_idx, **record}
        with self._events_lock:
            self._events_fh.write(json.dumps(record) + "\n")
            self._events_fh.flush()

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release video writers and close all log files."""
        self._watchdog_running = False
        if self._raw_writer:
            self._raw_writer.release()
            self._raw_writer = None
        if self._down_writer:
            self._down_writer.release()
            self._down_writer = None
        with self._stream_lock:
            for name, writer in self._stream_writers.items():
                if writer:
                    writer.release()
            self._stream_writers.clear()
            for name, fh in self._stream_files.items():
                try:
                    fh.close()
                except Exception:
                    pass
            self._stream_files.clear()
        with self._decisions_lock:
            self._decisions_fh.close()
        with self._events_lock:
            self._events_fh.close()
        with self._data_lock:
            self._data_fh.close()
        log.info("Session recording saved: %s", self.session_dir.resolve())
