# Rover Agent

An autonomous rover navigation system that uses a camera, a vision model, and a wheeled robot as the drive platform. The agent captures frames at 30 fps, queries a navigation strategy at a configurable interval, and drives the rover toward the predicted waypoints. A live web UI streams both the raw camera feed and the annotated inference frame.

## Strategies

| Strategy | Backend | Best for |
|----------|---------|----------|
| `gemini` (default) | Google Gemini vision API | Rule-based missions with a natural-language prompt |
| `omnivla` | OmniVLA-edge (local neural network) | Open-ended language goals without prompt engineering |

---

## Hardware

| Rover | Connection | Default port |
|-------|-----------|--------------|
| iRobot Roomba | USB serial (iRobot OI) | `/dev/ttyUSB0` |
| Atlas-1 (four-wheel STM32) | USB serial (`$CMD` protocol) | `/dev/ttyACM0` |

Both rovers work with any navigation strategy. A USB or CSI camera (OpenCV-compatible) is required. Tested on Ubuntu 24.04.

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

### Roomba + Gemini (default)

```bash
python rover_agent.py
python rover_agent.py --roomba-port /dev/ttyUSB0
python rover_agent.py --dry-run   # no hardware needed
```

### Atlas-1 + Gemini

```bash
python rover_agent.py --rover atlas --atlas-port /dev/ttyACM0
python rover_agent.py --rover atlas --atlas-port /dev/ttyACM0 --dry-run
```

### Atlas-1 + OmniVLA

```bash
python rover_agent.py --rover atlas --atlas-port /dev/ttyACM0 \
    --strategy omnivla --goal "Follow the brown path" --interval 1.0
```

### Roomba + OmniVLA

```bash
python rover_agent.py --roomba-port /dev/ttyUSB0 \
    --strategy omnivla --goal "blue trash bin" --interval 1.0
```

With a goal image instead of a text goal:

```bash
python rover_agent.py --roomba-port /dev/ttyUSB0 \
    --strategy omnivla --goal "follow path" --goal-image /path/to/goal.jpg --interval 1.0
```

`--interval 1.0` is recommended for OmniVLA — it matches the model's 1 Hz control rate.

---

### OmniVLA model server (optional)

Loading OmniVLA-edge takes ~30 seconds on first start. The model server keeps the weights in memory between `rover_agent` runs, so restarting the agent is instant.

**Step 1 — start the server once** (leave it running):

```bash
python omnivla_server.py
# Optionally bind to a specific address:
python omnivla_server.py --host 127.0.0.1 --port 5100
```

The server prints `OmniVLA server listening on 127.0.0.1:5100` when ready.

**Step 2 — point `rover_agent` at the server**:

```bash
python rover_agent.py --roomba-port /dev/ttyUSB0 \
    --strategy omnivla --omnivla-server localhost:5100 \
    --goal "Follow the brown path" --interval 1.0

# With a goal image:
python rover_agent.py --roomba-port /dev/ttyUSB0 \
    --strategy omnivla --omnivla-server localhost:5100 \
    --goal "follow path" --goal-image /path/to/goal.jpg --interval 1.0
```

When `--omnivla-server` is set, `rover_agent` connects to the server instead of loading the model locally. The server caches CLIP text encodings per unique goal string, so changing goals between runs costs only a single CLIP encode (~200 ms).

**Server flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Address to bind |
| `--port` | `5100` | TCP port |
| `--authkey` | `omnivla-edge` | Shared secret between server and client |

> Without `--omnivla-server`, the agent loads the model locally on every start (original behaviour, unchanged).

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `0` | Camera device index |
| `--interval` | `3.0` | Seconds between inference steps |
| `--port` | `5000` | Web UI port |
| `--rover` | `roomba` | Rover hardware: `roomba` or `atlas` |
| `--roomba-port` | *(disabled)* | Roomba serial port (e.g. `/dev/ttyUSB0`) |
| `--atlas-port` | *(disabled)* | Atlas-1 serial port (e.g. `/dev/ttyACM0`) |
| `--dry-run` | `false` | Log drive commands without sending them |
| `--strategy` | `gemini` | Navigation strategy: `gemini` or `omnivla` |
| `--goal` | `navigate forward` | Language goal for OmniVLA |
| `--goal-image` | *(none)* | Path to a goal image for OmniVLA (modality 6: image+language) |
| `--omnivla-server` | *(none)* | `host:port` of a running `omnivla_server.py` — skips local model load |

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
rover_agent.py          Thin orchestrator — agent loop, rover factory, main()
navigation_strategy.py  NavigationStrategy ABC + AgentState dataclass
gemini_strategy.py      GeminiStrategy — Gemini vision API, 3-frame JPEG history
omnivla_strategy.py     OmniVLAStrategy — OmniVLA-edge local or server-mode inference
omnivla_server.py       Standalone model server — keeps weights loaded between runs
web_display.py          WebDisplay — Flask server, MJPEG streams, status API
gemini_client.py        Gemini API wrapper with retry logic
prompts.py              System prompt + user prompt builder for Gemini
roomba_controller.py    Pixel-to-motion conversion + Roomba OI serial driver
atlas_controller.py     Pixel-to-motion conversion + Atlas-1 $CMD serial driver

omnivla_model.py        OmniVLA-edge model architecture (copied from NHirose/OmniVLA)
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
       rover_ctrl.navigate_to_waypoint()   (Gemini)
       rover_ctrl.drive_raw()              (OmniVLA)
```

**Gemini** drives step-and-stop: turn to align with the waypoint, drive forward 0.5 s (aligning mode) or 2.0 s (following mode), then stop and wait for the next query.

**OmniVLA** drives continuously: sends one command per step and the rover keeps moving at that velocity until the next inference result arrives.

Both strategies work with either rover — the controllers share the same interface.

---

## Atlas-1 calibration

Before the first real drive, update these constants in [atlas_controller.py](atlas_controller.py):

| Constant | Default | What to do |
|----------|---------|------------|
| `WHEEL_BASE_MM` | 300 | Measure centre-to-centre distance between left and right wheel contact patches |
| `DRIVE_SPEED_PCT` | 60 | Lower if too fast, raise if the rover barely moves |
| `CAMERA_HFOV_DEGREES` | 62.2 | Match your camera spec (Pi Camera v2 = 62.2°, wide-angle USB = 70–90°) |
| `_MAX_VELOCITY_REF_MM_S` | 200 | OmniVLA only — increase if the rover feels slow with `--strategy omnivla` |

`WHEEL_BASE_MM` has the biggest impact on turn accuracy — get this right first.

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
