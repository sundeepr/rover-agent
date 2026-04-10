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
        raw.mp4           — continuous raw camera feed (camera fps)
        annotated.mp4     — LLM-annotated frames (same fps; last decision held
                            until the next one arrives)
        decisions.jsonl   — one JSON record per LLM/VLM decision step
        events.jsonl      — every drive command, goal change, pause/resume, and
                            operator joystick input; each record has ts (epoch
                            float) and frame_idx for video correlation

write_frames() is called from the agent_loop thread on every camera tick.
write_decision() is called from strategy threads (thread-safe via lock).
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

    def __init__(self, fps: float = 30.0) -> None:
        cfg = _load_config()

        sessions_dir = _resolve_sessions_dir(Path(cfg["sessions_dir"]))
        _check_disk_space(sessions_dir, float(cfg["min_free_gb"]))

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = sessions_dir / ts
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self._fps = fps

        # VideoWriters are lazy-initialised on the first write_frames() call so
        # the frame dimensions are known before the writers are created.
        self._raw_writer: Optional[cv2.VideoWriter] = None
        self._ann_writer: Optional[cv2.VideoWriter] = None

        # Frame counter — incremented on every write_frames() call so that
        # decisions and events can be correlated with a specific video frame.
        self._frame_idx = 0

        self._decisions_path = self.session_dir / "decisions.jsonl"
        self._decisions_lock = threading.Lock()
        self._decisions_fh = open(self._decisions_path, "a", encoding="utf-8")

        # Unified event log: joystick, goal changes, pause/resume, model steps.
        self._events_path = self.session_dir / "events.jsonl"
        self._events_lock = threading.Lock()
        self._events_fh = open(self._events_path, "a", encoding="utf-8")

        log.info("Session recording started: %s", self.session_dir.resolve())

    # ── Video ──────────────────────────────────────────────────────────────────
    # Called from the agent_loop thread — VideoWriters need no locking.

    def write_frames(
        self,
        raw: np.ndarray,
        annotated: Optional[np.ndarray],
    ) -> None:
        """
        Write one camera-rate frame pair.

        raw       — the unmodified camera frame.
        annotated — the latest LLM-annotated frame, or None to repeat raw
                    (used before the first decision arrives).
        """
        h, w = raw.shape[:2]

        if self._raw_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._raw_writer = cv2.VideoWriter(
                str(self.session_dir / "raw.mp4"), fourcc, self._fps, (w, h)
            )

        if self._ann_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._ann_writer = cv2.VideoWriter(
                str(self.session_dir / "annotated.mp4"), fourcc, self._fps, (w, h)
            )

        self._raw_writer.write(raw)
        self._ann_writer.write(annotated if annotated is not None else raw)
        self._frame_idx += 1

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
        """Release video writers and close the decisions file."""
        if self._raw_writer:
            self._raw_writer.release()
            self._raw_writer = None
        if self._ann_writer:
            self._ann_writer.release()
            self._ann_writer = None
        with self._decisions_lock:
            self._decisions_fh.close()
        with self._events_lock:
            self._events_fh.close()
        log.info("Session recording saved: %s", self.session_dir.resolve())
