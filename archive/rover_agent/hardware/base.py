"""
Abstract base classes for rover hardware.

All hardware implementations (mock, RPi, ROS, …) must satisfy these
interfaces so the navigation loop can be tested and run without
touching real hardware.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Camera(ABC):
    @abstractmethod
    def capture_image(self) -> bytes:
        """Capture a frame and return it as raw JPEG bytes."""
        ...


class RoverMotors(ABC):
    @abstractmethod
    def rotate(self, degrees: float) -> None:
        """
        Rotate the rover in place.

        Positive degrees → clockwise (turn right).
        Negative degrees → counter-clockwise (turn left).
        Blocks until the rotation is complete.
        """
        ...

    @abstractmethod
    def drive_forward(self, distance_meters: float) -> None:
        """
        Drive the rover straight forward.

        Blocks until the rover has travelled *distance_meters*.
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Immediately halt all motor activity (emergency stop)."""
        ...


class RoverHardware:
    """Composite hardware bundle passed to the navigation loop."""

    def __init__(self, camera: Camera, motors: RoverMotors) -> None:
        self.camera = camera
        self.motors = motors
