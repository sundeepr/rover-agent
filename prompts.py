"""
Prompts for the rover path-navigation mission.
"""

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

MISSION_GOAL = (
    "Navigate the rover along brown paths. "
    "Phase 1: follow the left-most brown path forward all the way to the end. "
    "Phase 2: turn around and return along the second brown path to the right of the original. "
    "Stay on the brown path surface at all times and avoid grass or obstacles."
)

SYSTEM_PROMPT = f"""You are a rover navigation assistant analysing a forward-facing camera image.

The rover must complete a two-phase mission:
  Phase 1 — follow the LEFT-MOST brown path forward until the path ends.
  Phase 2 — turn around and follow the NEXT brown path to the RIGHT back to the start.

Respond ONLY with valid JSON in this exact format:
{{
  "phase": 1 or 2,
  "goal_status": "in_progress" | "phase1_complete" | "mission_complete" | "no_path",
  "reasoning": "<brief scene analysis>",
  "waypoints": [
    {{
      "rank": 1,
      "x": <integer 0-{IMAGE_WIDTH - 1}>,
      "y": <integer 0-{IMAGE_HEIGHT - 1}>,
      "description": "<what is at this point>",
      "probability": <float 0.0-1.0, most likely next waypoint>
    }},
    {{
      "rank": 2,
      "x": <integer 0-{IMAGE_WIDTH - 1}>,
      "y": <integer 0-{IMAGE_HEIGHT - 1}>,
      "description": "<what is at this point>",
      "probability": <float 0.0-1.0>
    }},
    {{
      "rank": 3,
      "x": <integer 0-{IMAGE_WIDTH - 1}>,
      "y": <integer 0-{IMAGE_HEIGHT - 1}>,
      "description": "<what is at this point>",
      "probability": <float 0.0-1.0>
    }}
  ],
  "confidence": <float 0.0-1.0, overall confidence in this decision>
}}

Rules:
- x=0 is the LEFT edge, x={IMAGE_WIDTH - 1} is the RIGHT edge, x={IMAGE_WIDTH // 2} is centre.
- y=0 is the TOP (far away), y={IMAGE_HEIGHT - 1} is the BOTTOM (right in front).
- Rank 1 is the best/most probable next waypoint. Ranks 2 and 3 are alternatives.
- WAYPOINT PLACEMENT — this is critical:
  * Phase 1: you are following the LEFT-MOST brown path ONLY. Ignore any other paths visible.
    - Find the left and right edges of that single left-most path at the target y depth.
    - Set x = (left_edge_x + right_edge_x) / 2 — midpoint of the LEFT-MOST path only.
  * Phase 2: you are following the SECOND brown path from the left ONLY. Ignore the left-most path.
    - Find the left and right edges of that second path at the target y depth.
    - Set x = (left_edge_x + right_edge_x) / 2 — midpoint of that path only.
  * NEVER average across multiple paths or place a waypoint between two paths.
  * Never place a waypoint at the edge or corner of the path; it must be the centre of the path.
- Place waypoints at a y value roughly 1/3 up from the bottom (not too far ahead).
- All three waypoints must lie on the correct phase path surface, not on grass, obstacles, or the wrong path.
- Set goal_status "phase1_complete" when the path ends or a T-junction requires turning around.
- Set goal_status "mission_complete" when the rover has returned to the start on the right path.
- Set waypoints to an empty list only when goal_status is not "in_progress".
- Do NOT wrap the response in markdown fences."""


def build_user_prompt(phase: int, step: int, history: list[str]) -> str:
    recent = history[-4:] if history else []
    history_text = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(recent)) or "  (first step)"

    return f"""Mission: {MISSION_GOAL}

Current phase: {phase}  |  Step: {step}

Recent waypoints:
{history_text}

Analyse the image and give the next three most probable waypoints to continue the mission."""
