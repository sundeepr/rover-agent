"""
dataset_recorder.py — Writes teleoperation episodes to the training dataset format.

Directory layout
────────────────
dataset/
  episodes/
    episode_000001/
      episode_meta.json
      camera_front/
        000000.jpg
        000001.jpg
      frames.jsonl

Episode numbering auto-increments by scanning existing directories.
"""

import json
import math
import shutil
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_JPEG_QUALITY = 90


class DatasetRecorder:
    """Records one teleoperation episode to disk."""

    def __init__(self, base_dir: str, episode_meta: dict):
        self._base = Path(base_dir) / "episodes"
        self._base.mkdir(parents=True, exist_ok=True)

        episode_id = self._next_episode_id()
        self._episode_dir = self._base / episode_id
        self._img_dir     = self._episode_dir / "camera_front"
        self._episode_dir.mkdir()
        self._img_dir.mkdir()

        self._episode_id  = episode_id
        self._step        = 0
        self._start_time  = time.time()

        # Write episode_meta.json
        meta = {
            "episode_id": episode_id,
            **episode_meta,
        }
        (self._episode_dir / "episode_meta.json").write_text(
            json.dumps(meta, indent=2))

        self._jsonl = open(self._episode_dir / "frames.jsonl", "w")

    # ── Public interface ──────────────────────────────────────────────────────

    def write_frame(
        self,
        frame: np.ndarray,
        instruction: str,
        vel_mm_s: int,
        radius_mm: int,
        joy_fwd: float,
        joy_turn: float,
        waypoints_norm: list,
    ) -> int:
        """
        Save one frame + metadata.

        Parameters
        ----------
        frame          : BGR camera frame
        instruction    : language goal for this episode
        vel_mm_s       : last commanded linear velocity (mm/s)
        radius_mm      : last commanded radius (mm); 0x8000 = straight
        joy_fwd        : joystick forward [-1, 1]
        joy_turn       : joystick turn [-1, 1]
        waypoints_norm : list of [nx, ny] image-normalised waypoints

        Returns step index.
        """
        step      = self._step
        timestamp = time.time() - self._start_time
        img_path  = f"camera_front/{step:06d}.jpg"

        # Save image
        cv2.imwrite(
            str(self._img_dir / f"{step:06d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
        )

        h, w = frame.shape[:2]

        # Convert vel/radius → m/s and rad/s
        lin_mps = vel_mm_s / 1000.0
        if radius_mm == 0x8000 or radius_mm == 0:
            ang_radps = 0.0
        elif radius_mm == 1:        # spin left
            ang_radps = math.pi
        elif radius_mm == -1:       # spin right
            ang_radps = -math.pi
        else:
            ang_radps = lin_mps / (radius_mm / 1000.0)

        action: dict = {
            "linear_velocity_mps":  round(lin_mps, 4),
            "angular_velocity_radps": round(ang_radps, 4),
        }
        if waypoints_norm:
            action["waypoints_normalized"] = waypoints_norm

        record = {
            "episode_id":    self._episode_id,
            "step":          step,
            "timestamp_sec": round(timestamp, 4),
            "observation": {
                "front_rgb":    img_path,
                "image_width":  w,
                "image_height": h,
            },
            "instruction": instruction,
            "action":      action,
            "teleop_input": {
                "joystick_forward": round(joy_fwd, 3),
                "joystick_turn":    round(joy_turn, 3),
            },
        }
        self._jsonl.write(json.dumps(record) + "\n")
        self._jsonl.flush()

        self._step += 1
        return step

    @property
    def step(self) -> int:
        return self._step

    @property
    def episode_id(self) -> str:
        return self._episode_id

    def close(self) -> str:
        """Flush and close. Returns the episode directory path."""
        self._jsonl.close()
        return str(self._episode_dir)

    def discard(self) -> None:
        """Close and delete the episode directory."""
        self._jsonl.close()
        shutil.rmtree(self._episode_dir, ignore_errors=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _next_episode_id(self) -> str:
        existing = sorted(self._base.glob("episode_*"))
        if not existing:
            return "episode_000001"
        last = existing[-1].name
        try:
            n = int(last.split("_")[-1]) + 1
        except ValueError:
            n = len(existing) + 1
        return f"episode_{n:06d}"
