import math
import pytest

from rover_agent.models import GoalStatus, LLMNavigationResponse, Waypoint
from rover_agent.navigation.state import NavigationStateManager


def make_response(status: GoalStatus, px: int = 320, py: int = 240) -> LLMNavigationResponse:
    waypoint = (
        Waypoint(target_pixel_x=px, target_pixel_y=py, description="target")
        if status == GoalStatus.IN_PROGRESS
        else None
    )
    return LLMNavigationResponse(
        goal_status=status,
        reasoning="test",
        waypoint=waypoint,
        obstacles_detected=[],
        confidence=0.9,
    )


def make_waypoint(px: int = 320, py: int = 240) -> Waypoint:
    return Waypoint(target_pixel_x=px, target_pixel_y=py, description="target")


# ── should_continue ───────────────────────────────────────────────────────────

def test_should_continue_initially():
    mgr = NavigationStateManager("find the cone", max_iterations=5)
    assert mgr.should_continue is True


def test_stops_when_max_iterations_reached():
    mgr = NavigationStateManager("find the cone", max_iterations=2)
    for _ in range(2):
        mgr.record_llm_response(make_response(GoalStatus.IN_PROGRESS))
    assert mgr.should_continue is False


def test_stops_when_achieved():
    mgr = NavigationStateManager("find the cone")
    mgr.record_llm_response(make_response(GoalStatus.ACHIEVED))
    assert mgr.should_continue is False


def test_stops_when_no_path():
    mgr = NavigationStateManager("find the cone")
    mgr.record_llm_response(make_response(GoalStatus.NO_PATH))
    assert mgr.should_continue is False


# ── record_llm_response ───────────────────────────────────────────────────────

def test_iteration_count_increments():
    mgr = NavigationStateManager("goal")
    mgr.record_llm_response(make_response(GoalStatus.IN_PROGRESS))
    mgr.record_llm_response(make_response(GoalStatus.IN_PROGRESS))
    assert mgr.state.iteration_count == 2


def test_goal_status_updated_from_response():
    mgr = NavigationStateManager("goal")
    mgr.record_llm_response(make_response(GoalStatus.ACHIEVED))
    assert mgr.state.goal_status == GoalStatus.ACHIEVED


# ── record_waypoint_executed ──────────────────────────────────────────────────

def test_straight_drive_updates_y():
    mgr = NavigationStateManager("goal")
    wp = make_waypoint()
    mgr.record_waypoint_executed(wp, bearing_degrees=0.0, step_distance_meters=1.0)
    pos = mgr.state.current_position
    assert pos.y_meters == pytest.approx(1.0, abs=1e-6)
    assert pos.x_meters == pytest.approx(0.0, abs=1e-6)
    assert pos.heading_degrees == pytest.approx(0.0, abs=1e-6)


def test_90_degree_right_then_forward():
    mgr = NavigationStateManager("goal")
    wp = make_waypoint()
    mgr.record_waypoint_executed(wp, bearing_degrees=90.0, step_distance_meters=1.0)
    pos = mgr.state.current_position
    assert pos.x_meters == pytest.approx(1.0, abs=1e-6)
    assert pos.y_meters == pytest.approx(0.0, abs=1e-6)
    assert pos.heading_degrees == pytest.approx(90.0, abs=1e-6)


def test_heading_wraps_at_360():
    mgr = NavigationStateManager("goal")
    wp = make_waypoint()
    mgr.record_waypoint_executed(wp, bearing_degrees=350.0, step_distance_meters=0.0)
    mgr.record_waypoint_executed(wp, bearing_degrees=20.0, step_distance_meters=0.0)
    assert mgr.state.current_position.heading_degrees == pytest.approx(10.0, abs=1e-6)


def test_waypoints_visited_appended():
    mgr = NavigationStateManager("goal")
    wp1 = make_waypoint(100, 200)
    wp2 = make_waypoint(300, 400)
    mgr.record_waypoint_executed(wp1, 0.0, 1.0)
    mgr.record_waypoint_executed(wp2, 0.0, 1.0)
    assert len(mgr.state.waypoints_visited) == 2
    assert mgr.state.waypoints_visited[0] is wp1


# ── history_summary / position_summary ───────────────────────────────────────

def test_history_summary_empty_initially():
    mgr = NavigationStateManager("goal")
    assert mgr.get_history_summary() == []


def test_history_summary_after_waypoint():
    mgr = NavigationStateManager("goal")
    wp = Waypoint(target_pixel_x=320, target_pixel_y=240, description="red cone")
    mgr.record_waypoint_executed(wp, 0.0, 0.5)
    summary = mgr.get_history_summary()
    assert len(summary) == 1
    assert "red cone" in summary[0]


def test_position_summary_format():
    mgr = NavigationStateManager("goal")
    summary = mgr.get_position_summary()
    assert "east" in summary
    assert "north" in summary
    assert "heading" in summary
