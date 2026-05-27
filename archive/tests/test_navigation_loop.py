"""
Integration tests for NavigationLoop.

The LLM client is mocked so no real API calls are made.
A MockCamera serves a single test image; MockMotors records all commands.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rover_agent.hardware.mock_rover import MockMotors, make_mock_hardware
from rover_agent.llm.client import NavigationLLMClient
from rover_agent.models import GoalStatus, LLMNavigationResponse, Waypoint
from rover_agent.navigation.loop import NavigationLoop


# ── Fixtures ──────────────────────────────────────────────────────────────────

FIXTURE_IMAGE = Path(__file__).parent / "fixtures"


def _make_response(
    status: GoalStatus,
    px: int = 400,
    py: int = 300,
    description: str = "target",
) -> LLMNavigationResponse:
    wp = (
        Waypoint(target_pixel_x=px, target_pixel_y=py, description=description)
        if status == GoalStatus.IN_PROGRESS
        else None
    )
    return LLMNavigationResponse(
        goal_status=status,
        reasoning="test reasoning",
        waypoint=wp,
        obstacles_detected=[],
        confidence=0.9,
    )


@pytest.fixture()
def fixture_image(tmp_path: Path) -> Path:
    """Create a minimal valid JPEG for MockCamera."""
    # 1×1 white JPEG (smallest valid JPEG)
    jpeg_bytes = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
        b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
        b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04"
        b"\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa"
        b"\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br"
        b"\x82\t\n\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJ"
        b"STUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92"
        b"\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9"
        b"\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7"
        b"\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4"
        b"\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa"
        b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd4P\x00\x00\x00\x1f\xff\xd9"
    )
    img = tmp_path / "test.jpg"
    img.write_bytes(jpeg_bytes)
    return tmp_path


@pytest.fixture()
def mock_llm_client() -> MagicMock:
    return MagicMock(spec=NavigationLLMClient)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_loop_achieves_goal_in_two_steps(fixture_image, mock_llm_client):
    mock_llm_client.get_next_waypoint.side_effect = [
        _make_response(GoalStatus.IN_PROGRESS, px=400, py=300),
        _make_response(GoalStatus.ACHIEVED),
    ]
    hardware = make_mock_hardware(fixture_image)
    loop = NavigationLoop(
        hardware=hardware,
        llm_client=mock_llm_client,
        user_goal="reach the cone",
        step_distance_meters=0.5,
        step_delay_seconds=0,
        image_width=640,
        camera_hfov_degrees=62.2,
    )
    state = loop.run()

    assert state.goal_status == GoalStatus.ACHIEVED
    assert state.iteration_count == 2
    assert len(state.waypoints_visited) == 1


def test_motor_sequence_for_right_of_center(fixture_image, mock_llm_client):
    """pixel_x=400 on a 640-wide image → slight right turn then drive."""
    mock_llm_client.get_next_waypoint.side_effect = [
        _make_response(GoalStatus.IN_PROGRESS, px=400, py=300),
        _make_response(GoalStatus.ACHIEVED),
    ]
    hardware = make_mock_hardware(fixture_image)
    motors: MockMotors = hardware.motors  # type: ignore[assignment]

    loop = NavigationLoop(
        hardware=hardware,
        llm_client=mock_llm_client,
        user_goal="reach target",
        step_distance_meters=0.5,
        step_delay_seconds=0,
        image_width=640,
        camera_hfov_degrees=62.2,
        min_rotation_degrees=2.0,
    )
    loop.run()

    # px=400, center=320, offset=80/640 * 62.2 ≈ 7.8° → rotation expected
    rotate_cmds = [c for c in motors.move_log if c["type"] == "rotate"]
    drive_cmds = [c for c in motors.move_log if c["type"] == "drive"]

    assert len(rotate_cmds) == 1
    assert rotate_cmds[0]["degrees"] == pytest.approx(7.775, abs=0.01)
    assert len(drive_cmds) == 1
    assert drive_cmds[0]["distance_meters"] == pytest.approx(0.5)


def test_center_pixel_skips_rotation(fixture_image, mock_llm_client):
    """pixel_x=320 on a 640-wide image → 0° bearing → no rotation command."""
    mock_llm_client.get_next_waypoint.side_effect = [
        _make_response(GoalStatus.IN_PROGRESS, px=320, py=240),
        _make_response(GoalStatus.ACHIEVED),
    ]
    hardware = make_mock_hardware(fixture_image)
    motors: MockMotors = hardware.motors  # type: ignore[assignment]

    loop = NavigationLoop(
        hardware=hardware,
        llm_client=mock_llm_client,
        user_goal="straight ahead",
        step_distance_meters=0.5,
        step_delay_seconds=0,
        image_width=640,
        camera_hfov_degrees=62.2,
        min_rotation_degrees=2.0,
    )
    loop.run()

    rotate_cmds = [c for c in motors.move_log if c["type"] == "rotate"]
    assert rotate_cmds == []


def test_loop_exits_on_no_path(fixture_image, mock_llm_client):
    mock_llm_client.get_next_waypoint.return_value = _make_response(GoalStatus.NO_PATH)
    hardware = make_mock_hardware(fixture_image)
    loop = NavigationLoop(
        hardware=hardware,
        llm_client=mock_llm_client,
        user_goal="impossible goal",
        step_delay_seconds=0,
    )
    state = loop.run()

    assert state.goal_status == GoalStatus.NO_PATH
    assert state.iteration_count == 1


def test_loop_respects_max_iterations(fixture_image, mock_llm_client):
    mock_llm_client.get_next_waypoint.return_value = _make_response(GoalStatus.IN_PROGRESS)
    hardware = make_mock_hardware(fixture_image)
    loop = NavigationLoop(
        hardware=hardware,
        llm_client=mock_llm_client,
        user_goal="never-ending",
        max_iterations=3,
        step_delay_seconds=0,
    )
    state = loop.run()

    assert state.iteration_count == 3
    # Still IN_PROGRESS because LLM never said achieved — loop exited on cap
    assert state.goal_status == GoalStatus.IN_PROGRESS


def test_stop_called_on_keyboard_interrupt(fixture_image, mock_llm_client):
    mock_llm_client.get_next_waypoint.side_effect = KeyboardInterrupt()
    hardware = make_mock_hardware(fixture_image)
    motors: MockMotors = hardware.motors  # type: ignore[assignment]

    loop = NavigationLoop(
        hardware=hardware,
        llm_client=mock_llm_client,
        user_goal="interrupted",
        step_delay_seconds=0,
    )
    state = loop.run()

    assert state.goal_status == GoalStatus.FAILED
    stop_cmds = [c for c in motors.move_log if c["type"] == "stop"]
    assert len(stop_cmds) == 1
