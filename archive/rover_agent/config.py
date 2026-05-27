"""
Central configuration for the rover navigation agent.

All values can be overridden via environment variables (loaded from .env).
"""

from __future__ import annotations

import os


class NavigationConfig:
    # ── Camera ────────────────────────────────────────────────────────────────
    # Horizontal field-of-view for RPi Camera Module v2 is ~62.2°.
    # Adjust for your lens if different.
    CAMERA_HFOV_DEGREES: float = float(os.getenv("CAMERA_HFOV_DEGREES", "62.2"))
    IMAGE_WIDTH: int = int(os.getenv("IMAGE_WIDTH", "640"))
    IMAGE_HEIGHT: int = int(os.getenv("IMAGE_HEIGHT", "480"))

    # ── Motion ────────────────────────────────────────────────────────────────
    # Fixed distance the rover drives toward each waypoint before reassessing.
    STEP_DISTANCE_METERS: float = float(os.getenv("STEP_DISTANCE_METERS", "0.5"))
    # Rotations smaller than this (in degrees) are skipped to avoid jitter.
    MIN_ROTATION_DEGREES: float = float(os.getenv("MIN_ROTATION_DEGREES", "2.0"))

    # ── LLM ───────────────────────────────────────────────────────────────────
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    # ── Navigation loop ───────────────────────────────────────────────────────
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "50"))
    # Seconds to pause between navigation steps (0 = as fast as possible).
    STEP_DELAY_SECONDS: float = float(os.getenv("STEP_DELAY_SECONDS", "0.5"))

    # ── Motors (linux_rover.py) ───────────────────────────────────────────────
    # Serial port for the motor controller (e.g. /dev/ttyUSB0, /dev/ttyACM0).
    # Leave unset to run in log-only mode (no physical motors).
    MOTOR_SERIAL_PORT: str | None = os.getenv("MOTOR_SERIAL_PORT", None)
    MOTOR_SERIAL_BAUD: int = int(os.getenv("MOTOR_SERIAL_BAUD", "115200"))
    # Camera device index for LinuxCamera (0 = /dev/video0).
    CAMERA_DEVICE_INDEX: int = int(os.getenv("CAMERA_DEVICE_INDEX", "0"))
    # Calibrated forward speed in meters per second (used for time-based distance control).
    ROVER_SPEED_MPS: float = float(os.getenv("ROVER_SPEED_MPS", "0.2"))
    # Calibrated rotation speed in degrees per second.
    ROVER_ROTATION_DPS: float = float(os.getenv("ROVER_ROTATION_DPS", "45.0"))


CONFIG = NavigationConfig()
