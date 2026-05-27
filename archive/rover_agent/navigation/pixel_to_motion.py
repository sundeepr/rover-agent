"""
Convert a pixel coordinate returned by the LLM into motor commands.

The horizontal pixel offset from the image center is mapped linearly to a
bearing angle using the camera's horizontal field-of-view.  The rover then
rotates by that bearing and drives forward a fixed step distance.
"""

from __future__ import annotations


def pixel_to_bearing(
    target_pixel_x: int,
    image_width: int,
    camera_hfov_degrees: float,
) -> float:
    """
    Return the bearing angle (degrees) the rover must turn to face the target pixel.

    - Positive bearing  → turn clockwise (right)
    - Negative bearing  → turn counter-clockwise (left)
    - Zero              → target is straight ahead

    The mapping is linear: the leftmost pixel maps to -HFOV/2 and the
    rightmost pixel maps to +HFOV/2.

    Args:
        target_pixel_x:      Horizontal pixel from the LLM response.
        image_width:         Total width of the image in pixels.
        camera_hfov_degrees: Horizontal field-of-view of the camera in degrees.
    """
    if image_width <= 0:
        raise ValueError(f"image_width must be positive, got {image_width}")

    # Offset of target from image centre, normalised to [-0.5, +0.5]
    normalised_offset = (target_pixel_x - image_width / 2.0) / image_width
    return normalised_offset * camera_hfov_degrees


def should_rotate(bearing_degrees: float, min_rotation_degrees: float) -> bool:
    """Return True when the bearing is large enough to warrant a physical rotation."""
    return abs(bearing_degrees) >= min_rotation_degrees
