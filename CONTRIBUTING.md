# Rover Agent — Developer Guide

A quick reference for making changes to this codebase.

---

## File map

```
rover_agent.py          Thin orchestrator: camera loop, thread wiring, CLI
navigation_strategy.py  AgentState dataclass + NavigationStrategy ABC
gemini_strategy.py      GeminiStrategy — 3-frame Gemini vision approach
web_display.py          WebDisplay — Flask routes, MJPEG streams, HTML UI
gemini_client.py        Gemini API wrapper (model, schema, retry logic)
prompts.py              System prompt, user prompt builder, image dimensions
roomba_controller.py    Pixel → bearing → Roomba OI motion pipeline
roomba_control.py       Low-level Roomba serial driver (iRobot OI protocol)
```

---

## Running the agent

```bash
# Install deps
pip install -r requirements.txt

# Set Gemini API key
export GEMINI_API_KEY=your_key_here

# Run (camera device 0, query every 3s, web UI on :5000)
python rover_agent.py

# With Roomba
python rover_agent.py --roomba-port /dev/ttyUSB0

# Dry-run: logs Roomba commands without opening the serial port
python rover_agent.py --roomba-port /dev/ttyUSB0 --dry-run

# All options
python rover_agent.py --device 0 --interval 3.0 --port 5000 \
                      --roomba-port /dev/ttyUSB0 --dry-run \
                      --strategy gemini
```

Web UI: `http://<host>:5000`

---

## How a query step works

```
Camera loop (30 fps, daemon thread)
  │
  ├─ writes AgentState.raw_frame  →  /video/realtime MJPEG
  │
  └─ every `interval` seconds, if no query in-flight:
       increments AgentState.step
       spawns GeminiStrategy.run_query() on a new daemon thread
           │
           ├─ builds image list (up to 3 prior frames + current)
           ├─ calls gemini_client.get_waypoint()  ← retries up to 3× on bad JSON
           ├─ draws waypoint overlay → AgentState.llm_frame  →  /video/llm MJPEG
           ├─ writes result / trajectory → AgentState  (under result_lock)
           ├─ calls roomba_ctrl.navigate_to_waypoint(waypoint, nav_mode)
           └─ on phase1_complete: clears frame buffer, increments phase, U-turn

Flask (main thread)
  ├─ /              HTML page
  ├─ /video/realtime  MJPEG of AgentState.raw_frame
  ├─ /video/llm       MJPEG of AgentState.llm_frame
  ├─ /status          JSON snapshot of AgentState (polled every 1s by JS)
  └─ /logs            log file listing / download
```

---

## Adding a new navigation strategy

A strategy owns its own state (frame buffer, conversation history, etc.) and
implements two methods.

**1. Create the strategy file**

```python
# my_strategy.py
import threading
from pathlib import Path
import numpy as np
from navigation_strategy import AgentState, NavigationStrategy

class MyStrategy(NavigationStrategy):

    @property
    def name(self) -> str:
        return "mine"

    def on_reset(self) -> None:
        # clear any internal buffers
        pass

    def run_query(
        self,
        state: AgentState,
        frame: np.ndarray,
        captures_dir: Path,
        roomba_ctrl,
    ) -> None:
        try:
            # 1. Decide where to go (call an LLM, run CV, anything)
            result = {...}   # must match the response dict shape below

            # 2. Write results under the appropriate locks
            with state.llm_lock:
                state.llm_frame = annotated_frame.copy()

            with state.result_lock:
                state.latest_result = result
                state.llm_query_start = 0.0
                state.llm_response_s = elapsed
                if top_waypoint:
                    state.trajectory.append({...})

            # 3. Drive
            if roomba_ctrl and result["goal_status"] == "in_progress":
                roomba_ctrl.navigate_to_waypoint(top_waypoint, result["navigation_mode"])

            # 4. Phase transition
            if result["goal_status"] == "phase1_complete":
                with state.result_lock:
                    state.phase = 2
                if roomba_ctrl:
                    roomba_ctrl.uturn()

        except Exception as e:
            with state.result_lock:
                state.llm_query_start = 0.0
            log.error("Strategy error: %s", e, exc_info=True)
        finally:
            state.query_in_flight.clear()   # REQUIRED — never omit this
```

**2. Register it in `rover_agent.py`**

```python
def _build_strategy(name: str) -> NavigationStrategy:
    if name == "gemini":
        from gemini_strategy import GeminiStrategy
        return GeminiStrategy()
    if name == "mine":                         # ← add this
        from my_strategy import MyStrategy
        return MyStrategy()
    raise ValueError(f"Unknown strategy: {name!r}")
```

Also add `"mine"` to the `choices=` list in the `--strategy` argparse argument.

**Result dict shape** (must match for the web UI to display correctly):

```python
{
    "phase":           1 or 2,
    "navigation_mode": "aligning" | "following",
    "goal_status":     "in_progress" | "phase1_complete" | "mission_complete" | "no_path",
    "reasoning":       "...",
    "waypoints": [
        {"rank": 1, "x": int, "y": int, "description": "...", "probability": float},
        {"rank": 2, ...},
        {"rank": 3, ...},
    ],
    "confidence": float,
}
```

---

## Changing the Gemini prompt

Everything is in [prompts.py](prompts.py):

- `MISSION_GOAL` — one-line mission description included in every user prompt
- `SYSTEM_PROMPT` — full system instructions (alignment rules, waypoint depth
  logic, end-of-path detection, output schema). Edit this to change how the
  model reasons or what JSON fields it returns.
- `build_user_prompt(phase, step, trajectory)` — dynamic per-step context:
  current phase/step and the last 10 trajectory records.
- `IMAGE_WIDTH` / `IMAGE_HEIGHT` — must match the camera resolution set in
  `agent_loop()`. Change both together.

If you add a field to the JSON the model returns, also add it to
`gemini_client._RESPONSE_SCHEMA` so the structured-output enforcement stays
in sync.

---

## Changing the Gemini model

Edit the single constant at the top of [gemini_client.py](gemini_client.py):

```python
MODEL = "gemini-3-flash-preview"
```

---

## Tuning Roomba motion

All physical constants are at the top of [roomba_controller.py](roomba_controller.py):

| Constant | Default | Effect |
|---|---|---|
| `CAMERA_HFOV_DEGREES` | 62.2 | Bearing calculation — match your camera |
| `WHEEL_BASE_MM` | 235 | Turn accuracy — measure your Roomba |
| `DRIVE_VELOCITY_MM_S` | 150 | Forward speed (max 500) |
| `MIN_ROTATION_DEGREES` | 3.0 | Dead-band — ignore tiny bearing corrections |
| `STEP_DURATION_ALIGNING_S` | 0.5 | Drive time when `navigation_mode = "aligning"` |
| `STEP_DURATION_FOLLOWING_S` | 2.0 | Drive time when `navigation_mode = "following"` |

---

## AgentState — shared state contract

`AgentState` (in [navigation_strategy.py](navigation_strategy.py)) is the
single object shared between the camera loop, the active strategy, and the
web display. Lock discipline:

| Field | Protected by | Writers |
|---|---|---|
| `raw_frame` | `raw_lock` | agent_loop |
| `llm_frame` | `llm_lock` | strategy |
| `latest_result`, `trajectory`, `step`, `phase`, timing | `result_lock` | agent_loop (step), strategy (rest) |
| `query_in_flight` | threading.Event | agent_loop sets, strategy clears |

Never write to `AgentState` from a Flask request thread.

---

## Captured frames

Every Gemini query saves two files to `captures/`:

- `step_NNNN_raw.jpg` — the frame as sent to the model
- `step_NNNN_annotated.jpg` — the same frame with waypoint overlay

Useful for replaying a run or debugging model decisions.

---

## Logs

Each run creates a timestamped log at `logs/rover_YYYYMMDD_HHMMSS.log`.
Logs are also downloadable from the web UI. The file handler captures
DEBUG level; the console shows INFO and above.
