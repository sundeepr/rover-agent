"""
Atlas-1 rover controller for the rover navigation agent.

Drives the four-wheeled STM32-based Atlas-1 rover over serial using the
command protocol:

    $CMD,L=<int>,R=<int>,AUX=<int>#

L, R  : left / right motor power  -100 .. 100  (negative = reverse)
AUX   : auxiliary output            0 .. 100

Default port : /dev/ttyACM0
Baud         : 115200

The interface mirrors RoombaController exactly — navigate_to_waypoint(),
uturn(), drive_raw(), stop(), connect() — so either controller can be
passed to any NavigationStrategy without changes.

Pixel-to-motion pipeline (same as RoombaController):
  1. Horizontal offset of waypoint from image centre → bearing angle (HFOV)
  2. Bearing angle → differential L/R motor percentages
  3. Turn in place if bearing > dead-band, then drive forward
  4. Drive for STEP_DURATION seconds, then stop

Usage:
    ctrl = AtlasController(port="/dev/ttyACM0")
    with ctrl.connect():
        ctrl.navigate_to_waypoint(waypoint_dict)
        ctrl.stop()
"""

import logging
import math
import time
from contextlib import contextmanager

import serial

log = logging.getLogger("rover.atlas")

# ── Physical constants ────────────────────────────────────────────────────────

# Camera horizontal field of view — same as roomba_controller.py
CAMERA_HFOV_DEGREES = 62.2

# Atlas-1 wheel base (mm). Measure centre-to-centre of left and right wheels.
WHEEL_BASE_MM = 300

# Forward cruising speed as % of full motor power (0–100).
# Keep this conservative until the rover's speed is characterised.
DRIVE_SPEED_PCT = 60

# Minimum bearing to bother turning (dead-band).
MIN_ROTATION_DEGREES = 3.0

# How long to drive per step before stopping and re-querying the camera.
STEP_DURATION_ALIGNING_S = 0.5   # short nudge for lateral correction
STEP_DURATION_FOLLOWING_S = 2.0  # longer step when following path

# Image width — must match prompts.py
IMAGE_WIDTH = 640

# Reference forward speed used to convert OmniVLA velocity_mm_s → %.
# Set to the maximum velocity the navigation strategies will ever command.
# OmniVLA caps at MAX_LIN_MM_S=50; Gemini uses DRIVE_VELOCITY_MM_S=150.
_MAX_VELOCITY_REF_MM_S = 200


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def _make_frame(L: int, R: int, AUX: int = 0) -> bytes:
    L   = _clamp(int(L),   -100, 100)
    R   = _clamp(int(R),   -100, 100)
    AUX = _clamp(int(AUX),    0, 100)
    return f"$CMD,L={L},R={R},AUX={AUX}#\n".encode("ascii")


# ── Controller ────────────────────────────────────────────────────────────────

class AtlasController:
    """
    Controls the Atlas-1 four-wheeled rover via STM32 serial protocol.

    Parameters
    ----------
    port : str
        Serial port (e.g. ``/dev/ttyACM0``).
    baud : int
        Baud rate; default 115200.
    dry_run : bool
        If True, log commands but do not open the serial port.
    """

    def __init__(self, port: str, baud: int = 115200, dry_run: bool = False):
        self.port    = port
        self.baud    = baud
        self.dry_run = dry_run
        self._ser: "serial.Serial | None" = None

    # ── Connection ────────────────────────────────────────────────────────────

    @contextmanager
    def connect(self):
        """Context manager: open serial port and yield self."""
        if self.dry_run:
            log.info("Dry-run mode — Atlas commands will be logged, not sent")
            yield self
            return

        self._ser = serial.Serial(
            port=self.port,
            baudrate=self.baud,
            timeout=0.2,
            write_timeout=0.2,
        )
        time.sleep(0.2)   # let the port settle after open
        log.info("Atlas connected on %s", self.port)

        try:
            yield self
        finally:
            self.stop()
            self._ser.close()
            self._ser = None
            log.info("Atlas disconnected")

    # ── High-level navigation (mirrors RoombaController) ─────────────────────

    def navigate_to_waypoint(self, waypoint: dict,
                             navigation_mode: str = "following") -> None:
        """
        Execute one navigation step toward the given pixel-space waypoint.

        Steps:
          1. Compute bearing from the horizontal pixel offset.
          2. Tank-turn in place if bearing exceeds the dead-band.
          3. Drive forward for the mode-appropriate duration.
          4. Stop.
        """
        px      = int(waypoint["x"])
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
        """Tank-spin 180° in place."""
        self._turn(180.0)
        log.info("U-turn complete")

    def drive_raw(self, velocity: int, radius: int) -> None:
        """
        Send a continuous drive command without sleeping.

        Converts Roomba-style (velocity_mm_s, radius_mm) to Atlas L/R %
        so OmniVLA can use AtlasController identically to RoombaController.
        """
        L, R = self._velocity_radius_to_lr(velocity, radius)
        self._send_cmd(L, R)

    def stop(self) -> None:
        """Stop all wheel motion."""
        self._send_cmd(0, 0)
        log.info("Atlas stopped")

    # ── Pixel → bearing ───────────────────────────────────────────────────────

    def _pixel_to_bearing(self, pixel_x: int) -> float:
        """
        Convert horizontal pixel position to bearing angle (degrees).

        Returns a value in [-HFOV/2, +HFOV/2]:
          negative = turn left, positive = turn right, 0 = straight ahead.
        """
        offset   = pixel_x - IMAGE_WIDTH / 2
        fraction = offset / IMAGE_WIDTH
        return fraction * CAMERA_HFOV_DEGREES

    # ── Motion primitives ─────────────────────────────────────────────────────

    def _turn(self, bearing_deg: float) -> None:
        """Tank-turn in place toward the target bearing, then stop."""
        # Estimate angular rate: tangential speed ≈ DRIVE_SPEED_PCT% of ref vel
        tangential_mm_s    = _MAX_VELOCITY_REF_MM_S * DRIVE_SPEED_PCT / 100
        degrees_per_second = math.degrees(2 * tangential_mm_s / WHEEL_BASE_MM)
        duration           = abs(bearing_deg) / degrees_per_second

        # Turn right: left fwd, right back.  Turn left: left back, right fwd.
        if bearing_deg > 0:
            L, R = DRIVE_SPEED_PCT, -DRIVE_SPEED_PCT
        else:
            L, R = -DRIVE_SPEED_PCT, DRIVE_SPEED_PCT

        log.info("Turning %.1f° (%.2fs @ %.1f°/s)",
                 bearing_deg, duration, degrees_per_second)
        self._send_cmd(L, R)
        time.sleep(duration)
        self._send_cmd(0, 0)
        time.sleep(0.05)

    def _drive_forward(self, navigation_mode: str = "following") -> None:
        """Drive straight forward for the mode-appropriate duration, then stop."""
        duration = (STEP_DURATION_ALIGNING_S
                    if navigation_mode == "aligning"
                    else STEP_DURATION_FOLLOWING_S)
        log.info("Driving forward for %.2fs (mode: %s)", duration, navigation_mode)
        self._send_cmd(DRIVE_SPEED_PCT, DRIVE_SPEED_PCT)
        time.sleep(duration)
        self._send_cmd(0, 0)

    def _velocity_radius_to_lr(self, velocity_mm_s: int,
                                radius_mm: int) -> tuple[int, int]:
        """
        Convert Roomba OI (velocity_mm_s, radius_mm) to Atlas (L%, R%).

        Special Roomba OI radius codes are handled explicitly:
          0x8000 → straight
          1      → spin clockwise (right)
          -1     → spin counter-clockwise (left)
        """
        if velocity_mm_s == 0:
            return 0, 0

        if radius_mm == 0x8000:          # straight
            pct = int(velocity_mm_s / _MAX_VELOCITY_REF_MM_S * 100)
            pct = _clamp(pct, -100, 100)
            return pct, pct

        if radius_mm == 1:               # spin right in place
            return DRIVE_SPEED_PCT, -DRIVE_SPEED_PCT

        if radius_mm == -1:              # spin left in place
            return -DRIVE_SPEED_PCT, DRIVE_SPEED_PCT

        # General curve: differential drive kinematics
        # v_r = v * (1 + W / (2R)),  v_l = v * (1 - W / (2R))
        ratio = WHEEL_BASE_MM / (2 * radius_mm)
        v_r   = velocity_mm_s * (1 + ratio)
        v_l   = velocity_mm_s * (1 - ratio)
        max_v = max(abs(v_r), abs(v_l), _MAX_VELOCITY_REF_MM_S)
        L     = _clamp(int(v_l / max_v * 100), -100, 100)
        R     = _clamp(int(v_r / max_v * 100), -100, 100)
        return L, R

    # ── Serial I/O ────────────────────────────────────────────────────────────

    def _send_cmd(self, L: int, R: int, AUX: int = 0) -> None:
        """Send one command frame, or log it in dry-run mode."""
        frame = _make_frame(L, R, AUX)
        log.debug("atlas: %s", frame.decode().strip())
        if self.dry_run:
            log.info("[dry-run] %s", frame.decode().strip())
            return
        if self._ser is not None:
            self._ser.write(frame)
            self._ser.flush()
