# Rover Agent

An autonomous rover navigation system that uses a camera, a vision model, and a wheeled robot as the drive platform. The agent captures frames at 30 fps, queries a navigation strategy at a configurable interval, and drives the rover toward the predicted waypoints. A live web UI streams both the raw camera feed and the annotated inference frame.

## Strategies

| Strategy | Detection | Navigation | Best for |
|----------|-----------|------------|----------|
| `gemini` (default) | Google Gemini vision API | Gemini waypoints | Rule-based missions with natural-language prompt |
| `omnivla` | — | OmniVLA-edge | Open-ended language goals |
| `clip_omnivla` | CLIP (prompts from Qwen3) | OmniVLA-edge | Fast path detection; Qwen generates CLIP prompts from goal at startup |
| `qwen_omnivla` | Qwen2.5-VL via Ollama | OmniVLA-edge | No CLIP needed; Qwen directly answers "is the path visible?" |

---

## Hardware

| Rover | Connection | Default port |
|-------|-----------|--------------|
| iRobot Roomba | USB serial (iRobot OI) | `/dev/ttyUSB0` |
| Atlas-1 (four-wheel STM32) | USB serial (`$CMD` protocol) | `/dev/ttyACM0` |

Both rovers work with any navigation strategy. A USB or CSI camera (OpenCV-compatible) is required. Tested on Ubuntu 24.04 and Jetson Orin Nano Super (JetPack 6).

---

## Installation

### Base dependencies

```bash
pip install opencv-python flask google-genai requests
```

The Gemini strategy also needs a free API key:

```bash
export GEMINI_API_KEY="your-key-here"
```

### OmniVLA dependencies

Required for `omnivla`, `clip_omnivla`, and `qwen_omnivla` strategies. Model weights (~200 MB) are downloaded automatically from HuggingFace on first run.

```bash
pip install -r requirements-omnivla.txt
```

### Ollama + Qwen models (Jetson Orin Nano Super)

Required for `clip_omnivla` (Qwen3 prompt generation at startup) and `qwen_omnivla` (Qwen2.5-VL path detection at runtime).

**1. Install Ollama:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Ollama installs as a systemd service and starts automatically. Verify it is running:

```bash
ollama list
```

**2. Pull the required models:**

```bash
# Qwen3 4B — used by clip_omnivla to generate CLIP prompts at startup
ollama pull qwen3:4b

# Qwen2.5-VL 3B — used by qwen_omnivla for per-frame path detection
ollama pull qwen2.5vl:3b
```

Download sizes: `qwen3:4b` ≈ 2.6 GB, `qwen2.5vl:3b` ≈ 2.3 GB.

**Memory on Jetson Orin Nano Super (8 GB unified):**

| Component | Memory |
|-----------|--------|
| OS + processes | ~1.5 GB |
| Qwen3:4b (startup only, then unloaded) | ~2.6 GB |
| Qwen2.5-VL:3b (`qwen_omnivla` runtime) | ~2.3 GB |
| CLIP ViT-B/32 | ~0.6 GB |
| OmniVLA-edge | ~0.3 GB |

For `clip_omnivla`: Qwen3 loads at startup to generate prompts, then Ollama unloads it. CLIP + OmniVLA run for the remainder of the session (~2 GB total).

For `qwen_omnivla`: Qwen2.5-VL stays loaded during navigation. OmniVLA runs alongside it (~2.6 GB total).

**Optional — unload models immediately after use** (frees memory faster):

```bash
# Set in your shell or systemd override
export OLLAMA_KEEP_ALIVE=0
```

---

## Web server (standalone)

The web UI runs as a separate process that stays alive across agent restarts and crashes. Start it once and leave it running:

```bash
python web_server.py                     # default: 0.0.0.0:5001
python web_server.py --port 5001
```

Open `http://localhost:5001` in a browser. The header shows a green/red indicator for agent connection status.

The agent connects to the web server when it starts and publishes frames + status over HTTP. If the agent stops or crashes the browser tab stays open and shows "Agent disconnected" until it restarts.

---

## Running

### Start the web server first (one terminal, leave running)

```bash
python web_server.py
```

### Roomba + Gemini (default)

```bash
python rover_agent.py --dry-run
python rover_agent.py --roomba-port /dev/ttyUSB0
```

### Atlas-1 + Gemini

```bash
python rover_agent.py --rover atlas --atlas-port /dev/ttyACM0
```

### OmniVLA (local model load)

```bash
python rover_agent.py --rover atlas --atlas-port /dev/ttyACM0 \
    --strategy omnivla --goal "Follow the brown path" --interval 1.0
```

### CLIP + OmniVLA with Qwen3 prompt generation

Qwen3 generates the CLIP positive/negative prompts from the goal at startup, then CLIP runs at ~50 ms per frame for path detection.

```bash
python rover_agent.py --rover atlas --atlas-port /dev/ttyACM0 \
    --strategy clip_omnivla \
    --goal "Follow the brown path" \
    --ollama-server http://localhost:11434 \
    --path-threshold 0.5 --interval 1.0
```

### Qwen2.5-VL + OmniVLA (no CLIP)

Qwen2.5-VL directly answers "is the path visible?" on every step. No CLIP image encoding. Use a longer interval to match Qwen's ~1.5–2.5 s inference time.

```bash
python rover_agent.py --rover atlas --atlas-port /dev/ttyACM0 \
    --strategy qwen_omnivla \
    --goal "Follow the brown path" \
    --ollama-server http://localhost:11434 \
    --path-threshold 0.6 --interval 3.0
```

---

## OmniVLA model server (optional, recommended)

Loading OmniVLA-edge takes ~30 seconds on first start. The model server keeps the weights in memory between `rover_agent` runs so restarting the agent is instant.

**Step 1 — start the server once** (leave it running):

```bash
python omnivla_server.py
python omnivla_server.py --host 127.0.0.1 --port 5100   # explicit address
```

**Step 2 — point `rover_agent` at the server:**

```bash
# omnivla
python rover_agent.py --rover atlas --atlas-port /dev/ttyACM0 \
    --strategy omnivla --omnivla-server localhost:5100 \
    --goal "Follow the brown path" --interval 1.0

# clip_omnivla — CLIP detection runs in server, OmniVLA inference in server
python rover_agent.py --rover atlas --atlas-port /dev/ttyACM0 \
    --strategy clip_omnivla --omnivla-server localhost:5100 \
    --goal "Follow the brown path" \
    --ollama-server http://localhost:11434 \
    --path-threshold 0.5 --interval 1.0

# qwen_omnivla — Qwen detection via Ollama, OmniVLA inference in server
python rover_agent.py --rover atlas --atlas-port /dev/ttyACM0 \
    --strategy qwen_omnivla --omnivla-server localhost:5100 \
    --goal "Follow the brown path" \
    --ollama-server http://localhost:11434 \
    --path-threshold 0.6 --interval 3.0
```

**OmniVLA server + clip_omnivla:** The server handles both CLIP path detection and OmniVLA inference. The agent generates CLIP prompts via Qwen3 at startup and passes them to the server on each detection call. Prompts are cached in the server after the first call.

**OmniVLA server + qwen_omnivla:** The server handles OmniVLA inference only. Path detection bypasses the server and goes directly to Ollama.

**Server flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Address to bind |
| `--port` | `5100` | TCP port |
| `--authkey` | `omnivla-edge` | Shared secret between server and client |

---

## All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `0` | Camera device index |
| `--interval` | `3.0` | Seconds between inference steps |
| `--web-server` | `http://localhost:5001` | URL of the running `web_server.py` |
| `--rover` | `roomba` | Rover hardware: `roomba` or `atlas` |
| `--roomba-port` | *(disabled)* | Roomba serial port (e.g. `/dev/ttyUSB0`) |
| `--atlas-port` | *(disabled)* | Atlas-1 serial port (e.g. `/dev/ttyACM0`) |
| `--dry-run` | `false` | Log drive commands without sending them |
| `--strategy` | `gemini` | `gemini`, `omnivla`, `clip_omnivla`, `qwen_omnivla` |
| `--goal` | `navigate forward` | Language goal for OmniVLA-based strategies |
| `--goal-image` | *(none)* | Path to a goal image for OmniVLA (modality 6) |
| `--omnivla-server` | *(none)* | `host:port` of a running `omnivla_server.py` |
| `--path-threshold` | `0.5` | Detection confidence threshold for `clip_omnivla` / `qwen_omnivla` |
| `--ollama-server` | `http://localhost:11434` | Ollama API URL for Qwen models |

---

## Web UI

Open `http://localhost:5001` (web server port).

| URL | Description |
|-----|-------------|
| `/` | Mission dashboard with agent connection indicator |
| `/video/realtime` | Live MJPEG stream (~20 fps from agent) |
| `/video/llm` | Last annotated inference frame |
| `/status` | JSON — step, phase, waypoints, goal status, agent connected |
| `/pause` | POST — toggle pause; rover stops immediately |
| `/logs` | Log file browser |

The Pause button stops the rover immediately and prevents the next drive command. Resuming re-enables the agent loop without restarting.

---

## File map

```
rover_agent.py            Thin orchestrator — agent loop, strategy factory, main()
navigation_strategy.py    NavigationStrategy ABC + AgentState dataclass
web_server.py             Standalone web server — survives agent restarts/crashes
agent_publisher.py        Background thread: reads AgentState, POSTs to web_server.py

gemini_strategy.py        GeminiStrategy — Gemini vision API, 3-frame JPEG history
omnivla_strategy.py       OmniVLAStrategy — OmniVLA-edge local or server-mode inference
clip_omnivla_strategy.py  CLIP path detection + OmniVLA navigation state machine
qwen_omnivla_strategy.py  Qwen2.5-VL path detection + OmniVLA navigation state machine
prompt_generator.py       Calls Qwen3 via Ollama to generate CLIP prompts from goal text

omnivla_server.py         Standalone model server — keeps OmniVLA + CLIP loaded between runs
omnivla_model.py          OmniVLA-edge model architecture (NHirose/OmniVLA)

gemini_client.py          Gemini API wrapper with retry logic
prompts.py                System prompt + user prompt builder for Gemini
roomba_controller.py      Pixel-to-motion conversion + Roomba OI serial driver
atlas_controller.py       Pixel-to-motion conversion + Atlas-1 $CMD serial driver

requirements.txt          Base Python dependencies
requirements-omnivla.txt  Extra deps for OmniVLA-based strategies
```

---

## How it works

```
web_server.py (always running)
    ├── /video/realtime  ◄── MJPEG to browser
    ├── /video/llm       ◄── MJPEG to browser
    └── /status          ◄── JSON to browser (1 s poll)

rover_agent.py
    ├── agent_loop (30 fps)
    │     ├── camera.read() → state.raw_frame
    │     └── every --interval s → spawn query thread
    │               └── strategy.run_query(state, frame, rover_ctrl)
    │                     ├── path detection  (CLIP / Qwen2.5-VL)
    │                     ├── OmniVLA infer   (local or omnivla_server.py)
    │                     └── rover_ctrl.drive_raw(vel, radius)
    │
    └── publisher loop (20 fps)
          ├── POST /agent/frame?stream=realtime  → web_server
          ├── POST /agent/frame?stream=llm       → web_server (on change)
          └── POST /agent/status                 → web_server (+ reads pause state)
```

**clip_omnivla state machine:**
```
INITIALIZING ──► PATH_LOST ◄──────────────────────────┐
                    │                                  │
          score ≥ threshold                   score < threshold
                    │                                  │
                    └──────────────► NAVIGATING ───────┘
                                   (OmniVLA driving)
```

Detection: CLIP zero-shot with positive/negative prompts generated by Qwen3 from the goal text at startup. Runtime detection is ~50 ms per frame.

**qwen_omnivla state machine:** same structure, but detection is a Qwen2.5-VL vision query (~1.5–2.5 s) instead of CLIP. No CLIP image encoding at runtime.

---

## Atlas-1 calibration

Before the first real drive, update these constants in [atlas_controller.py](atlas_controller.py):

| Constant | Default | What to do |
|----------|---------|------------|
| `WHEEL_BASE_MM` | 300 | Measure centre-to-centre distance between wheel contact patches |
| `DRIVE_SPEED_PCT` | 60 | Lower if too fast, raise if the rover barely moves |
| `CAMERA_HFOV_DEGREES` | 62.2 | Match your camera spec (Pi Camera v2 = 62.2°, wide-angle USB = 70–90°) |
| `_MAX_VELOCITY_REF_MM_S` | 200 | OmniVLA only — increase if the rover feels slow |

`WHEEL_BASE_MM` has the biggest impact on turn accuracy — get this right first.

---

## Gemini mission

The default Gemini prompt navigates a two-phase brown-path mission:

1. **Phase 1** — follow the left-most brown path forward to its end.
2. **Phase 2** — turn around and return along the next brown path to the right.

Edit `prompts.py` to change the mission goal, alignment thresholds, or output schema.
The model used is `gemini-2.5-flash-preview`, configured in `gemini_client.py`.

---

## Logs and captured frames

- Logs are written to `logs/rover_YYYYMMDD_HHMMSS.log` and streamed to the console.
- Gemini strategy saves each queried frame as a JPEG under `captures/`.
- Log files are downloadable from the web UI at `http://localhost:5001/logs`.
