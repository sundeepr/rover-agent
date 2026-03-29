# Rover Agent

An autonomous rover navigation system that uses a camera, a vision model, and an iRobot Roomba as the drive platform. The agent captures frames at 30 fps, queries a navigation strategy at a configurable interval, and drives the Roomba toward the predicted waypoints. A live web UI streams both the raw camera feed and the annotated inference frame.

## Strategies

| Strategy | Backend | Best for |
|----------|---------|----------|
| `gemini` (default) | Google Gemini vision API | Rule-based missions with a natural-language prompt |
| `omnivla` | OmniVLA-edge (local neural network) | Open-ended language goals without prompt engineering |

---

## Hardware

- iRobot Roomba connected via USB serial (e.g. `/dev/ttyUSB0`)
- USB or CSI camera (OpenCV-compatible)
- Linux host (tested on Ubuntu 24.04)

---

## Installation

### Base dependencies

```bash
pip install opencv-python flask google-genai
```

The Gemini strategy also needs a free API key:

```bash
export GEMINI_API_KEY="your-key-here"
```

### OmniVLA dependencies (optional)

Only needed when running `--strategy omnivla`. Model weights (~200 MB) are downloaded automatically from HuggingFace on first run.

```bash
pip install -r requirements-omnivla.txt
```

---

## Running

### Gemini strategy (default)

```bash
python rover_agent.py
python rover_agent.py --device 1 --interval 5 --port 5000
python rover_agent.py --roomba-port /dev/ttyUSB0
python rover_agent.py --dry-run   # log Roomba commands, no hardware needed
```

### OmniVLA strategy

```bash
python rover_agent.py --strategy omnivla --goal "blue trash bin"
python rover_agent.py --strategy omnivla --goal "go forward" --interval 1.0 --dry-run
python rover_agent.py --strategy omnivla --goal "door" --roomba-port /dev/ttyUSB0
```

`--interval 1.0` is recommended for OmniVLA — it matches the model's 1 Hz control rate.

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `0` | Camera device index |
| `--interval` | `3.0` | Seconds between inference steps |
| `--port` | `5000` | Web UI port |
| `--roomba-port` | *(disabled)* | Roomba serial port (e.g. `/dev/ttyUSB0`) |
| `--dry-run` | `false` | Log drive commands without sending them |
| `--strategy` | `gemini` | Navigation strategy: `gemini` or `omnivla` |
| `--goal` | `navigate forward` | Language goal for OmniVLA |

---

## Web UI

Once running, open `http://localhost:5000` in a browser.

| URL | Description |
|-----|-------------|
| `/` | Mission dashboard |
| `/video/realtime` | Live MJPEG stream (30 fps) |
| `/video/llm` | Last annotated inference frame |
| `/status` | JSON — current step, phase, waypoints, trajectory |
| `/logs` | Log file browser |

---

## File Map

```
rover_agent.py          Thin orchestrator — agent loop, strategy factory, main()
navigation_strategy.py  NavigationStrategy ABC + AgentState dataclass
gemini_strategy.py      GeminiStrategy — Gemini vision API, 3-frame JPEG history
omnivla_strategy.py     OmniVLAStrategy — OmniVLA-edge local neural network
web_display.py          WebDisplay — Flask server, MJPEG streams, status API
gemini_client.py        Gemini API wrapper with retry logic
prompts.py              System prompt + user prompt builder for Gemini
roomba_controller.py    Pixel-to-motion conversion + Roomba OI serial driver

requirements-omnivla.txt  Extra deps for the omnivla strategy
```

---

## How it works

```
camera thread (30 fps)
    │  raw frame → state.raw_frame  ──────────────────► /video/realtime
    │
    └─ every --interval seconds, if no query in flight:
           spawn query thread
               │
               ▼
       strategy.run_query(state, frame, ...)
               │
         ┌─────┴──────┐
         │             │
      Gemini API    OmniVLA-edge
      (cloud)       (local GPU/CPU)
         │             │
         └─────┬───────┘
               │  waypoints / drive command
               ▼
       state.llm_frame  ────────────────────────────► /video/llm
       state.latest_result ─────────────────────────► /status
       roomba_ctrl.navigate_to_waypoint()  (Gemini)
       roomba_ctrl.drive_raw()             (OmniVLA)
```

**Gemini** drives step-and-stop: turn to align with the waypoint, drive forward 0.5 s (aligning mode) or 2.0 s (following mode), then stop and wait for the next query.

**OmniVLA** drives continuously: sends one `(velocity, radius)` command per step and the Roomba keeps moving at that speed until the next inference result arrives.

---

## Gemini mission

The default Gemini prompt navigates a two-phase brown-path mission:

1. **Phase 1** — follow the left-most brown path forward to its end.
2. **Phase 2** — turn around and return along the next brown path to the right.

Edit `prompts.py` to change the mission goal, alignment thresholds, or output schema.

The model used is `gemini-3-flash-preview`, configured in `gemini_client.py`.

---

## Logs and captured frames

- Logs are written to `logs/rover_YYYYMMDD_HHMMSS.log` and streamed to the console.
- Each frame sent to the Gemini API is saved as a JPEG under `captures/`.
