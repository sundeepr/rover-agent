import pytest

from rover_agent.navigation.pixel_to_motion import pixel_to_bearing, should_rotate


IMAGE_WIDTH = 640
HFOV = 62.2  # RPi Camera Module v2


# ── pixel_to_bearing ──────────────────────────────────────────────────────────

def test_center_pixel_gives_zero_bearing():
    bearing = pixel_to_bearing(320, IMAGE_WIDTH, HFOV)
    assert bearing == pytest.approx(0.0, abs=1e-6)


def test_left_edge_gives_negative_bearing():
    bearing = pixel_to_bearing(0, IMAGE_WIDTH, HFOV)
    assert bearing == pytest.approx(-HFOV / 2, rel=1e-6)


def test_right_edge_gives_positive_bearing():
    bearing = pixel_to_bearing(IMAGE_WIDTH, IMAGE_WIDTH, HFOV)
    assert bearing == pytest.approx(HFOV / 2, rel=1e-6)


def test_quarter_right_of_center():
    # Pixel at 3/4 of width → quarter right of center → HFOV/4
    bearing = pixel_to_bearing(IMAGE_WIDTH * 3 // 4, IMAGE_WIDTH, HFOV)
    assert bearing == pytest.approx(HFOV / 4, rel=0.01)


def test_quarter_left_of_center():
    bearing = pixel_to_bearing(IMAGE_WIDTH // 4, IMAGE_WIDTH, HFOV)
    assert bearing == pytest.approx(-HFOV / 4, rel=0.01)


def test_invalid_image_width_raises():
    with pytest.raises(ValueError):
        pixel_to_bearing(320, 0, HFOV)


# ── should_rotate ─────────────────────────────────────────────────────────────

def test_small_bearing_no_rotate():
    assert should_rotate(1.0, min_rotation_degrees=2.0) is False


def test_exact_threshold_should_rotate():
    assert should_rotate(2.0, min_rotation_degrees=2.0) is True


def test_large_bearing_should_rotate():
    assert should_rotate(30.0, min_rotation_degrees=2.0) is True


def test_negative_bearing_should_rotate():
    assert should_rotate(-15.0, min_rotation_degrees=2.0) is True


def test_zero_bearing_no_rotate():
    assert should_rotate(0.0, min_rotation_degrees=2.0) is False
