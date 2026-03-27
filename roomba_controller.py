"""
Roomba controller for the rover navigation agent.

Receives pixel-space waypoints from the Gemini vision model and drives the
Roomba toward the rank-1 waypoint using the iRobot Open Interface (OI).

Pixel-to-motion pipeline:
  1. Horizontal offset of waypoint from image centre → bearing angle (HFOV)
  2. Bearing angle → differential wheel velocities
  3. Wheel velocities → Roomba drive(velocity, radius) command
  4. Drive for STEP_DURATION seconds, then stop (agent re-queries on next frame)

Usage:
    ctrl = RoombaController(port="/dev/ttyUSB0")
    with ctrl.connect():
        ctrl.navigate_to_waypoint(waypoint_dict)
        ctrl.stop()
"""

import logging
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path

# Look for roomba_control in the same directory first, then ../roomba/
_HERE = Path(__file__).parent
_ROOMBA_DIR = _HERE.parent / "roomba"
for _p in (_HERE, _ROOMBA_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from roomba_control import Roomba  # type: ignore
    _ROOMBA_AVAILABLE = True
except ImportError:
    _ROOMBA_AVAILABLE = False

log = logging.getLogger("rover.roomba")

# ── Physical constants ────────────────────────────────────────────────────────

# Camera horizontal field of view (degrees). Adjust for your specific camera.
# Common values: Pi Camera v2 = 62.2°, generic USB wide-angle = ~70–90°.
CAMERA_HFOV_DEGREES = 62.2

# Roomba physical parameters
WHEEL_BASE_MM = 235          # Distance between left and right wheels (mm)
DRIVE_VELOCITY_MM_S = 150    # Forward drive speed (mm/s); Roomba max is 500
MIN_ROTATION_DEGREES = 3.0   # Ignore turns smaller than this (dead-band)

# How long to drive per step before stopping and re-querying the camera.
# Short nudge during alignment (lateral correction only).
STEP_DURATION_ALIGNING_S = 0.5   # ~7.5 cm nudge for lateral correction
# Longer step when on-path and aligned for forward progress.
STEP_DURATION_FOLLOWING_S = 2.0  # ~30 cm step when following path

# Image dimensions (must match prompts.py)
IMAGE_WIDTH = 640


class RoombaController:
    """
    Controls a Roomba to navigate toward pixel-space waypoints.

    Parameters
    ----------
    port : str
        Serial port the Roomba is connected to (e.g. ``/dev/ttyUSB0``).
    baud : int
        Baud rate; most modern Roombas use 115200.
    dry_run : bool
        If True, log the commands that *would* be sent but do not open the
        serial port.  Useful for testing without hardware.
    """

    def __init__(self, port: str, baud: int = 115200, dry_run: bool = False):
        self.port = port
        self.baud = baud
        self.dry_run = dry_run
        self._roomba: "Roomba | None" = None
        self._ctx = None

    # ── Connection management ─────────────────────────────────────────────────

    @contextmanager
    def connect(self):
        """Context manager: initialise the Roomba and enter SAFE mode."""
        if self.dry_run:
            log.info("Dry-run mode — Roomba commands will be logged, not sent")
            yield self
            return

        if not _ROOMBA_AVAILABLE:
            raise RuntimeError(
                "roomba_control module not found. "
                f"Copy roomba_control.py into {_HERE} or {_ROOMBA_DIR}"
            )

        self._roomba = Roomba(self.port, baud=self.baud)
        self._ctx = self._roomba.connect()
        self._ctx.__enter__()
        log.info("Roomba connected on %s", self.port)

        self._roomba.start()
        time.sleep(0.05)
        self._roomba.safe()
        log.info("Roomba in SAFE mode")

        try:
            yield self
        finally:
            self.stop()
            self._ctx.__exit__(None, None, None)
            log.info("Roomba disconnected")

    # ── High-level navigation ─────────────────────────────────────────────────

    def navigate_to_waypoint(self, waypoint: dict, navigation_mode: str = "following") -> None:
        """
        Execute one navigation step toward the given waypoint.

        The waypoint is a dict with at minimum ``x`` (pixel column) and ``y``
        (pixel row), as returned by the Gemini vision model.

        navigation_mode: "aligning" uses a short drive duration for lateral
        correction; "following" uses the full forward step duration.

        Steps:
          1. Compute bearing from the horizontal pixel offset.
          2. Turn in place if bearing exceeds dead-band.
          3. Drive forward for the mode-appropriate duration.
          4. Stop.
        """
        px = int(waypoint["x"])
        bearing = self._pixel_to_bearing(px)

        log.info(
            "Navigating to waypoint rank=%s (%d, %d) p=%.0f%% — %s [mode: %s]",
            waypoint.get("rank", "?"),
            px, int(waypoint.get("y", 0)),
            waypoint.get("probability", 0) * 100,
            waypoint.get("description", ""),
            navigation_mode,
        )
        log.info("Bearing: %.1f°", bearing)

        if abs(bearing) > MIN_ROTATION_DEGREES:
            self._turn(bearing)

        self._drive_forward(navigation_mode)

    def uturn(self) -> None:
        """Spin 180° in place to reverse direction."""
        self._turn(180.0)
        log.info("U-turn complete")

    def stop(self) -> None:
        """Stop all wheel motion."""
        self._send_drive(0, 0x8000)
        log.info("Roomba stopped")

    # ── Pixel → motion conversion ─────────────────────────────────────────────

    def _pixel_to_bearing(self, pixel_x: int) -> float:
        """
        Convert a horizontal pixel position to a bearing angle in degrees.

        Returns a value in [-HFOV/2, +HFOV/2]:
          - Negative = turn left
          - Positive = turn right
          - 0 = straight ahead
        """
        offset = pixel_x - IMAGE_WIDTH / 2          # pixels from centre
        fraction = offset / IMAGE_WIDTH              # -0.5 .. +0.5
        return fraction * CAMERA_HFOV_DEGREES

    def _bearing_to_drive(self, bearing_deg: float) -> tuple[int, int]:
        """
        Convert a bearing angle to a Roomba (velocity, radius) command.

        Uses the Roomba OI differential-drive formula:
          radius = (WHEEL_BASE / 2) * (v_r + v_l) / (v_r - v_l)

        For pure rotation in place we use radius = ±1 (OI special values).
        """
        bearing_rad = math.radians(bearing_deg)

        # Differential-drive inverse kinematics
        # v_r = v + omega * L/2,  v_l = v - omega * L/2
        omega = bearing_rad  # treat bearing as angular velocity (rad per step)
        v_r = DRIVE_VELOCITY_MM_S + omega * WHEEL_BASE_MM / 2
        v_l = DRIVE_VELOCITY_MM_S - omega * WHEEL_BASE_MM / 2

        # Clamp to Roomba limits (-500 .. 500 mm/s)
        v_r = int(max(-500, min(500, v_r)))
        v_l = int(max(-500, min(500, v_l)))

        velocity = (v_r + v_l) // 2
        diff = v_r - v_l

        if abs(diff) < 5:
            radius = 0x8000  # straight
        elif abs(velocity) < 5:
            # Spin in place
            radius = 1 if diff > 0 else -1
            velocity = abs(diff) // 2
        else:
            radius = int(WHEEL_BASE_MM * (v_r + v_l) / (2 * diff))
            radius = max(-2000, min(2000, radius))

        return velocity, radius

    # ── Low-level motion primitives ───────────────────────────────────────────

    def _turn(self, bearing_deg: float) -> None:
        """Spin in place toward the target bearing, then stop."""
        # Estimate rotation time based on angular velocity.
        # Roomba in-place spin at ~150 mm/s tangential ≈ ~73°/s for 235 mm base.
        tangential_mm_s = DRIVE_VELOCITY_MM_S
        degrees_per_second = math.degrees(2 * tangential_mm_s / WHEEL_BASE_MM)
        duration = abs(bearing_deg) / degrees_per_second

        if bearing_deg > 0:
            radius = -1    # CW (turn right)
        else:
            radius = 1     # CCW (turn left)

        log.info("Turning %.1f° (%.2fs @ %.1f°/s)", bearing_deg, duration, degrees_per_second)
        self._send_drive(DRIVE_VELOCITY_MM_S, radius)
        time.sleep(duration)
        self._send_drive(0, 0x8000)
        time.sleep(0.05)

    def _drive_forward(self, navigation_mode: str = "following") -> None:
        """Drive straight forward for the mode-appropriate duration, then stop."""
        duration = STEP_DURATION_ALIGNING_S if navigation_mode == "aligning" else STEP_DURATION_FOLLOWING_S
        log.info("Driving forward for %.2fs (mode: %s)", duration, navigation_mode)
        self._send_drive(DRIVE_VELOCITY_MM_S, 0x8000)
        time.sleep(duration)
        self._send_drive(0, 0x8000)

    def _send_drive(self, velocity: int, radius: int) -> None:
        """Send a DRIVE command, or log it in dry-run mode."""
        log.debug("drive(velocity=%d, radius=%d)", velocity, radius)
        if self.dry_run:
            log.info("[dry-run] drive(velocity=%d, radius=%d)", velocity, radius)
            return
        if self._roomba is not None:
            self._roomba.drive(velocity, radius)


# ── Convenience function ──────────────────────────────────────────────────────

def navigate_step(waypoints: list[dict], port: str, dry_run: bool = False) -> None:
    """
    Single-shot: connect, drive one step toward the rank-1 waypoint, disconnect.

    Intended for callers that manage their own Roomba connection externally.
    Use :class:`RoombaController` directly if you want to keep the connection
    open across multiple steps.
    """
    top = next((w for w in waypoints if w.get("rank") == 1), None)
    if not top:
        log.warning("No rank-1 waypoint found, skipping drive step")
        return

    ctrl = RoombaController(port=port, dry_run=dry_run)
    with ctrl.connect():
        ctrl.navigate_to_waypoint(top)
