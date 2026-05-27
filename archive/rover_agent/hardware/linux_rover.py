"""
Generic Linux hardware implementation.

Camera  : Any V4L2-compatible device (USB webcam, built-in) via OpenCV.
Motors  : Serial-based stub — fill in the serial commands for your motor controller
          (Arduino, Sabertooth, ODrive, etc.), or swap for CAN/USB as needed.

Install:
    pip install opencv-python pyserial
"""

from __future__ import annotations

import logging
import time

from rover_agent.config import CONFIG
from rover_agent.hardware.base import Camera, RoverHardware, RoverMotors

logger = logging.getLogger(__name__)


# ── Camera ─────────────────────────────────────────────────────────────────────

class LinuxCamera(Camera):
    """
    Capture frames from any V4L2 camera via OpenCV.

    Args:
        device_index: Camera device index (0 = /dev/video0, 1 = /dev/video1, …).
        width / height: Requested capture resolution.
        jpeg_quality: JPEG encoding quality (0-100).
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = CONFIG.IMAGE_WIDTH,
        height: int = CONFIG.IMAGE_HEIGHT,
        jpeg_quality: int = 90,
    ) -> None:
        try:
            import cv2  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "opencv-python is not installed. Run: pip install opencv-python"
            ) from exc

        self._cv2 = cv2
        self._quality = jpeg_quality

        self._cap = cv2.VideoCapture(device_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at device index {device_index} (/dev/video{device_index}). "
                "Check that the camera is connected and not in use by another process."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Camera opened: device %d, %dx%d", device_index, actual_w, actual_h)

    def capture_image(self) -> bytes:
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to capture frame from camera.")
        ok, buf = self._cv2.imencode(".jpg", frame, [self._cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            raise RuntimeError("Failed to encode frame as JPEG.")
        return buf.tobytes()

    def release(self) -> None:
        self._cap.release()


# ── Motors ─────────────────────────────────────────────────────────────────────

class LinuxMotors(RoverMotors):
    """
    Serial motor controller stub.

    Sends simple text commands over a serial port to a motor controller
    (e.g. an Arduino sketch that accepts "FWD 0.5", "ROT 15.0", "STOP").

    If no serial port is available the commands are just logged — useful for
    testing the full agent on a laptop without a physical rover attached.

    Args:
        port:         Serial device path, e.g. "/dev/ttyUSB0" or "/dev/ttyACM0".
                      Pass None to run in log-only mode (no serial connection).
        baud:         Baud rate (match your motor controller firmware).
        speed_mps:    Calibrated forward speed for time-based distance control.
        rotation_dps: Calibrated rotation speed (degrees per second).
    """

    def __init__(
        self,
        port: str | None = None,
        baud: int = 115200,
        speed_mps: float = CONFIG.ROVER_SPEED_MPS,
        rotation_dps: float = CONFIG.ROVER_ROTATION_DPS,
    ) -> None:
        self._speed_mps = speed_mps
        self._rotation_dps = rotation_dps
        self._serial = None

        if port is not None:
            try:
                import serial  # type: ignore[import]
                self._serial = serial.Serial(port, baud, timeout=1)
                time.sleep(2)  # allow controller to reset after connection
                logger.info("Serial motor controller connected: %s @ %d baud", port, baud)
            except ImportError as exc:
                raise ImportError(
                    "pyserial is not installed. Run: pip install pyserial"
                ) from exc
            except Exception as exc:
                raise RuntimeError(f"Could not open serial port {port}: {exc}") from exc
        else:
            logger.warning(
                "No serial port configured — motor commands will be logged only. "
                "Set MOTOR_SERIAL_PORT in .env to connect a real motor controller."
            )

    def rotate(self, degrees: float) -> None:
        duration = abs(degrees) / self._rotation_dps
        logger.info("Motor: rotate %.1f° (%.2fs)", degrees, duration)
        self._send(f"ROT {degrees:.2f}")
        time.sleep(duration)
        self._send("STOP")

    def drive_forward(self, distance_meters: float) -> None:
        duration = distance_meters / self._speed_mps
        logger.info("Motor: drive %.2fm (%.2fs)", distance_meters, duration)
        self._send(f"FWD {distance_meters:.2f}")
        time.sleep(duration)
        self._send("STOP")

    def stop(self) -> None:
        logger.info("Motor: STOP")
        self._send("STOP")

    def _send(self, command: str) -> None:
        if self._serial is not None:
            self._serial.write(f"{command}\n".encode())
            self._serial.flush()


# ── Factory ────────────────────────────────────────────────────────────────────

def make_linux_hardware(
    camera_device: int = 0,
    motor_serial_port: str | None = None,
) -> RoverHardware:
    """Convenience factory used by cli.py."""
    return RoverHardware(
        camera=LinuxCamera(device_index=camera_device),
        motors=LinuxMotors(port=motor_serial_port),
    )
