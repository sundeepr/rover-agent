"""
Raspberry Pi hardware implementation.

Requirements (install on the Pi):
    pip install "rover-agent[rpi]"
    # or: pip install picamera2 gpiozero RPi.GPIO

Motor wiring assumes a dual H-bridge driver (e.g. L298N or L9110S) connected to
four GPIO pins.  Pins are configured via environment variables or rover_agent/config.py.

Distance and rotation are time-based using calibrated speed constants.  Tune
ROVER_SPEED_MPS and ROVER_ROTATION_DPS in .env to match your hardware.
"""

from __future__ import annotations

import io
import time

from rover_agent.config import CONFIG
from rover_agent.hardware.base import Camera, RoverHardware, RoverMotors


class RpiCamera(Camera):
    """Capture JPEG images from the Raspberry Pi camera using picamera2."""

    def __init__(
        self,
        width: int = CONFIG.IMAGE_WIDTH,
        height: int = CONFIG.IMAGE_HEIGHT,
    ) -> None:
        # Import here so the module can be imported on non-Pi systems without error.
        from picamera2 import Picamera2  # type: ignore[import]

        self._cam = Picamera2()
        config = self._cam.create_still_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self._cam.configure(config)
        self._cam.start()
        # Allow the sensor to settle.
        time.sleep(0.5)

    def capture_image(self) -> bytes:
        buffer = io.BytesIO()
        self._cam.capture_file(buffer, format="jpeg")
        return buffer.getvalue()

    def close(self) -> None:
        self._cam.stop()
        self._cam.close()


class RpiMotors(RoverMotors):
    """
    Control two DC motors via a dual H-bridge driver.

    Each motor requires two GPIO pins: one for forward, one for backward.
    PWM at 100% duty cycle is used here; adjust speed_factor (0–1) to slow down.
    """

    def __init__(
        self,
        left_fwd_pin: int = CONFIG.MOTOR_LEFT_FORWARD_PIN,
        left_bwd_pin: int = CONFIG.MOTOR_LEFT_BACKWARD_PIN,
        right_fwd_pin: int = CONFIG.MOTOR_RIGHT_FORWARD_PIN,
        right_bwd_pin: int = CONFIG.MOTOR_RIGHT_BACKWARD_PIN,
        speed_mps: float = CONFIG.ROVER_SPEED_MPS,
        rotation_dps: float = CONFIG.ROVER_ROTATION_DPS,
    ) -> None:
        from gpiozero import Motor  # type: ignore[import]

        self._left = Motor(forward=left_fwd_pin, backward=left_bwd_pin)
        self._right = Motor(forward=right_fwd_pin, backward=right_bwd_pin)
        self._speed_mps = speed_mps
        self._rotation_dps = rotation_dps

    def drive_forward(self, distance_meters: float) -> None:
        duration = distance_meters / self._speed_mps
        self._left.forward()
        self._right.forward()
        time.sleep(duration)
        self.stop()

    def rotate(self, degrees: float) -> None:
        """
        Rotate in place by spinning motors in opposite directions.

        Positive degrees → clockwise (right motor backward, left motor forward).
        """
        duration = abs(degrees) / self._rotation_dps
        if degrees > 0:
            self._left.forward()
            self._right.backward()
        else:
            self._left.backward()
            self._right.forward()
        time.sleep(duration)
        self.stop()

    def stop(self) -> None:
        self._left.stop()
        self._right.stop()


def make_rpi_hardware() -> RoverHardware:
    """Convenience factory used by cli.py."""
    return RoverHardware(
        camera=RpiCamera(),
        motors=RpiMotors(),
    )
