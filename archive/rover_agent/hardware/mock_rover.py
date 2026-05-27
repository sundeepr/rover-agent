"""
Mock hardware implementations for development and testing.

MockCamera serves JPEG images from a directory (or a single file) in sequence,
cycling back to the start when the list is exhausted.

MockMotors records every command in move_log for assertion in tests.
"""

from __future__ import annotations

from pathlib import Path

from rover_agent.hardware.base import Camera, RoverHardware, RoverMotors


class MockCamera(Camera):
    """
    Serve images from disk in sequence.

    Args:
        source: Either a single JPEG file path, or a directory containing
                JPEG files (served in sorted order, cycling).
    """

    def __init__(self, source: Path | str) -> None:
        source = Path(source)
        if source.is_dir():
            self._images: list[Path] = sorted(
                p for p in source.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if not self._images:
                raise FileNotFoundError(f"No JPEG/PNG images found in {source}")
        elif source.is_file():
            self._images = [source]
        else:
            raise FileNotFoundError(f"Image source not found: {source}")

        self._index = 0

    def capture_image(self) -> bytes:
        path = self._images[self._index % len(self._images)]
        self._index += 1
        return path.read_bytes()

    @property
    def images_served(self) -> int:
        return self._index


class MockMotors(RoverMotors):
    """Record motor commands for inspection in tests."""

    def __init__(self) -> None:
        self.move_log: list[dict] = []

    def rotate(self, degrees: float) -> None:
        self.move_log.append({"type": "rotate", "degrees": degrees})

    def drive_forward(self, distance_meters: float) -> None:
        self.move_log.append({"type": "drive", "distance_meters": distance_meters})

    def stop(self) -> None:
        self.move_log.append({"type": "stop"})


def make_mock_hardware(source: Path | str) -> RoverHardware:
    """Convenience factory used by CLI and tests."""
    return RoverHardware(
        camera=MockCamera(source),
        motors=MockMotors(),
    )
