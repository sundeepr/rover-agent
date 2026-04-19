---
name: Project overview
description: Rover agent architecture, all key files, threading model, web UI
type: project
originSessionId: 74eb3b7b-c6c1-4d44-b30a-0225d3b8f403
---
## Architecture

Multi-threaded Python app. Main thread blocks; daemon threads handle camera, inference, publishing.

```
rover_agent.py          — CLI entry point, wires everything together
web_server.py           — Standalone Flask server (start once, leave running) on :5001
agent_publisher.py      — POSTs AgentState to web_server every 50ms
control_server.py       — WebSocket joystick server on :5002 (direct drive_raw, <20ms latency)
navigation_strategy.py  — AgentState dataclass + NavigationStrategy base class
session_recorder.py     — Writes raw.avi, annotated.avi (MJPG), decisions.jsonl, events.jsonl
```

## Key files by concern

| File | Purpose |
|------|---------|
| `rover_agent.py` | CLI args, strategy factory, agent loop, shutdown |
| `gemini_strategy.py` | Gemini Vision API strategy |
| `omnivla_strategy.py` | OmniVLA-edge local/server strategy |
| `clip_omnivla_strategy.py` | CLIP path detection + OmniVLA navigation |
| `crop_row_strategy.py` | YOLO-based crop row gap centering (no OmniVLA) |
| `row_centering_omnivla_strategy.py` | CLIP+OmniVLA front + YOLO down-camera centering |
| `hough_crop_row_strategy.py` | Old ExG+Hough approach (mostly unused) |
| `omnivla_server.py` | TCP server wrapping OmniVLA+CLIP for multi-session reuse |
| `prompt_generator.py` | Qwen3:4b via Ollama → CLIP prompts from goal text |
| `atlas_controller.py` | Atlas STM32 rover ($CMD protocol) |
| `roomba_controller.py` | iRobot Roomba OI protocol |
| `camera_resolutions.py` | Probe camera supported resolutions and FOV |

## AgentState threading
- `raw_lock` — raw_frame (camera thread writes, publisher reads)
- `llm_lock` — llm_frame (strategy thread writes, publisher reads)
- `result_lock` — latest_result, step, goal, trajectory
- `query_in_flight` — Event; prevents overlapping inference calls
- `goal_ready` — Event; blocks queries until goal is set
- `paused` — Event; stops drive commands
- `operator_control` / `operator_until` — joystick override window

## Session recording
Each run creates `sessions/YYYYMMDD_HHMMSS/`:
- `raw.avi` — raw camera at ~30fps (MJPG, crash-safe)
- `annotated.avi` — annotated frames at ~30fps
- `decisions.jsonl` — one record per inference step with frame_idx
- `events.jsonl` — all drive commands, goal changes, pause/resume, joystick; frame_idx for video sync

## Web UI
- `http://localhost:5001` — live video, HUD, chat/goal input, joystick
- Joystick: tries WebSocket :5002 first (low latency), falls back to HTTP POST /chat
