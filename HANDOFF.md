# Rover Agent — Handoff Context

## 1. Files Changed (this session)

### New files
| File | Purpose |
|------|---------|
| `ollama_strategy.py` | **New primary strategy.** Sends rolling 5-frame history + down-camera frame to a local Ollama vision model (default: qwen2.5vl). Converts `next_point [x,y]` to drive command. Supports down camera via `update_down_frame()`. |
| `cloud_omnivla_strategy.py` | WebSocket client strategy — delegates all inference to `omnivla_cloud_server.py` on a cloud GPU. Rover only runs waypoint→drive conversion locally. State machine: CONNECTING / WAITING_GOAL / NAVIGATING. |
| `omnivla_cloud_server.py` | WebSocket server (cloud GPU). Loads full OmniVLA (VLA backbone + pose projector + action head). Accepts frames, returns 8×[dx,dy,cos_h,sin_h] waypoints. |
| `paligemma_strategy.py` | Thin subclass of `CloudOmniVLAStrategy`, name="paligemma". Rover-side client for `paligemma_cloud_server.py`. |
| `paligemma_cloud_server.py` | WebSocket proxy (no GPU needed) → Google Cloud Vertex AI PaliGemma endpoint. Asks "left/right/center?" and converts answer to waypoints. `--turn-dy` controls aggressiveness. |
| `row_centering_omnivla_strategy.py` | OmniVLA + downward ExG (Excess Green) crop-row centering. Replaced YOLO with pure OpenCV `2G-R-B` vegetation detection. Params: `--exg-threshold` (default 20), `--exg-min-area` (default 500). |
| `experimental/ollama_waypoint_viewer.py` | Standalone test tool. Sends rolling frame history to Ollama, draws `next_point` + `path_points` as green dots on image. Supports `--image`, `--video`, `--camera`. |
| `utils/exg_viewer.py` | Interactive ExG tuning tool. Takes a video file, shows ExG overlay per frame with live trackbars for threshold and min_area. Space=step, continuous playback toggle. |

### Modified files
| File | What changed |
|------|-------------|
| `rover_agent.py` | Added strategies: `ollama`, `cloud_omnivla`, `paligemma`. Added args: `--cloud-server`, `--ollama-model`, `--ollama-history`, `--ollama-server` (existing, reused), `--exg-threshold`, `--exg-min-area`. Changed `--down-device` default from `1` to `None` — down-camera loop now only starts if flag is explicitly passed. |
| `web_server.py` | Fixed MJPEG drop: removed `display:none` JS toggle on `down-cam-box` — browser was dropping the persistent MJPEG connection when the element was hidden. Box is now always visible. |

---

## 2. Architecture — Data Flow

```
Camera (cv2) ──► rover_agent.py agent_loop
                    │
                    ├─ state.raw_frame  ──► SessionRecorder → raw.avi
                    │
                    ▼
              strategy.run_query(state, frame, captures_dir, rover_ctrl)
                    │
          ┌─────────┴──────────────────────────────────────────┐
          │                                                      │
     OllamaStrategy                               CloudOmniVLAStrategy
     (ollama_strategy.py)                         (cloud_omnivla_strategy.py)
          │                                                      │
     rolls 5-frame history                    WebSocket to cloud server
     + down frame (optional)                  (omnivla_cloud_server.py)
     → POST /api/generate                     → full OmniVLA inference
     → parse next_point [x,y]                 → 8×4 waypoints
     → _next_point_to_drive()                 → _waypoint_to_drive()
          │                                                      │
          └─────────┬───────────────────────────────────────────┘
                    │
              rover_ctrl.drive_raw(vel, radius_mm)
                    │
              AtlasController / RoombaController
```

**Down camera loop** (daemon thread, only if `--down-device N` passed):
```
Down camera ──► _down_camera_loop() ──► strategy.update_down_frame(frame)
                                    └──► SessionRecorder → down.avi
```

**State shared between threads** (`AgentState`):
- `raw_frame` / `raw_lock` — latest forward camera frame
- `llm_frame` / `llm_lock` — annotated frame written by strategy
- `latest_result` / `result_lock` — JSON result for web UI
- `query_in_flight` — event preventing double-dispatch
- `paused` — operator pause flag

---

## 3. Strategy Registry

All strategies registered in `rover_agent._build_strategy()`:

| `--strategy` | Class | File | Backend |
|---|---|---|---|
| `gemini` | GeminiStrategy | gemini_strategy.py | Google Gemini API |
| `omnivla` | OmniVLAStrategy | omnivla_strategy.py | Local OmniVLA-edge model |
| `clip_omnivla` | ClipOmniVLAStrategy | clip_omnivla_strategy.py | CLIP + local OmniVLA |
| `qwen_omnivla` | QwenOmniVLAStrategy | qwen_omnivla_strategy.py | Qwen VLM + local OmniVLA |
| `hough_crop_row` | HoughCropRowStrategy | hough_crop_row_strategy.py | Hough lines + OmniVLA |
| `row_centering_omnivla` | RowCenteringOmniVLAStrategy | row_centering_omnivla_strategy.py | ExG downward centering + OmniVLA |
| `crop_row` | CropRowStrategy | crop_row_strategy.py | YOLO detection |
| `cloud_omnivla` | CloudOmniVLAStrategy | cloud_omnivla_strategy.py | Full OmniVLA over WebSocket |
| `paligemma` | PaliGemmaStrategy | paligemma_strategy.py | Google Vertex AI PaliGemma |
| `ollama` | OllamaStrategy | ollama_strategy.py | Local Ollama (qwen2.5vl) |

---

## 4. OllamaStrategy — Key Design Decisions

- **Frame history**: `deque(maxlen=5)` stores `(frame, result)` pairs. All frames sent as `images[]` in one Ollama API call (oldest→newest), past `next_point` values summarised in prompt text.
- **Down camera**: implemented `update_down_frame()` so `rover_agent` auto-starts the down-camera loop when `--down-device` is set. Down frame appended as last image. Prompt asks model for `wheel_on_crop` + `crop_contact_side`.
- **Resize**: `_letterbox()` — scales to fit 640×480 preserving aspect ratio, pads with black. Avoids distorting plant shapes on 16:9 cameras.
- **Drive conversion**: `next_point.x` → radius. Dead band ±0.05 around x=0.5 = straight. `radius = -400 / (error × 10)`, clamped ±2000 mm. Forward speed always 60 mm/s.
- **Prompt**: locked to crop-row navigation. Crops (uniform rows) = avoid. Weeds (scattered) = drive over freely.

---

## 5. PaliGemma — Key Design Decisions & Lessons

- Endpoint addressed by display name (e.g. `paligemma-final-endpoint`), not numeric ID. `connect()` lists all endpoints and substring-matches.
- Instance format: `{"prompt": "...", "image": "<base64>"}` — no nested `image_bytes` dict, no separate `parameters` dict.
- Image resized to 224×224 (PIL BILINEAR) before sending — PaliGemma's training resolution.
- Response key is `"response"` not `"output"` or `"generated_text"`.
- PaliGemma **cannot generate JSON waypoints** — it hallucinated repeating tokens. Switched to asking `"Where is the crop row gap: left, right, or center?"` and converting to waypoints locally with `--turn-dy` (default 0.1 units = 1 cm/step).
- `--turn-dy` is configurable; small values prevent plant trampling.

---

## 6. ExG Row Centering (row_centering_omnivla)

Replaced YOLO (0 detections on field crops from overhead) with:
```python
exg = 2*G - R - B          # Excess Green index
mask = threshold(exg, 20)  # --exg-threshold
# morphological close+open with 5×5 ellipse kernel
# findContours, filter by area (--exg-min-area 500)
# split blobs left/right of frame centre
# gap_cx = (rightmost_left_blob_x2 + leftmost_right_blob_x1) / 2
```

---

## 7. Known Issues / Incomplete Work

1. **`raw.avi` and `annotated.avi` 0 bytes on external drive** — session directory on `/Volumes/Samsung_T5/atlas-1/`. Root cause not fully resolved. `down.avi` recorded correctly (10 MB). Likely a VideoWriter codec/path issue on the external volume. Not fixed.

2. **PaliGemma "traffic_split not set" error** — occurs if the Vertex AI model deployment is still in progress or failed silently. Wait for deployment to complete in Cloud Console before running.

3. **OllamaStrategy `next_point.x` constant** — qwen2.5vl often returns the same x for all path points regardless of actual corridor curvature. The prompt tries to force variation but results are inconsistent. This is a model capability limitation. Workaround: use ExG or the downward camera for fine-grained centering.

4. **`ollama_waypoint_viewer.py` normalised coordinates** — model returns coordinates in its internal 224×224 space regardless of prompt instructions. Fixed by switching to normalised 0–1 fractions. May still be unreliable depending on model version.

5. **PaliGemma endpoint lookup** — if `paligemma-final-endpoint` is not deployed, fall back to `paligemma-endpoint`. Check with `gcloud ai endpoints list --region=us-central1`.

---

## 8. Calibration Constants & Hardware Values

| Constant | Value | Where | Notes |
|---|---|---|---|
| Forward camera | `--device 0` | rover_agent.py | Default; change with `--device` |
| Down camera | `--down-device N` | rover_agent.py | Default now `None` (disabled). Was 1. |
| Down camera resolution | max (9999×9999 request) | rover_agent.py | Camera picks its own max |
| Send resolution to Ollama | 640×480 letterboxed | ollama_strategy.py | `_SEND_W, _SEND_H` |
| OmniVLA send resolution | 640×480 | cloud_omnivla_strategy.py | `_SEND_W, _SEND_H` |
| PaliGemma send resolution | 224×224 | paligemma_cloud_server.py | Model training resolution |
| Ollama forward speed | 60 mm/s | ollama_strategy.py | `_FWD_VEL` |
| Ollama turn radius base | 400 mm | ollama_strategy.py | `_TURN_RADIUS` |
| Ollama dead band | ±0.05 x-units | ollama_strategy.py | `_DEAD_BAND` |
| PaliGemma turn step | 0.1 (1 cm/step) | paligemma_cloud_server.py | `--turn-dy` |
| ExG threshold | 20 | row_centering_omnivla | `--exg-threshold` |
| ExG min blob area | 500 px² | row_centering_omnivla | `--exg-min-area` |
| Atlas serial port | `/dev/ttyACM0` | CLI | `--atlas-port` |
| Ollama API port | 11434 | ollama_strategy.py | Standard Ollama default |
| PaliGemma WS port | 8766 | paligemma_cloud_server.py | `--port` |
| OmniVLA cloud WS port | 8765 | omnivla_cloud_server.py | `--port` |
| Vertex AI region | `us-central1` | paligemma_cloud_server.py | `--location` |
| Ollama history size | 5 frames | ollama_strategy.py | `--ollama-history` |
| JPEG quality (Ollama) | 85 | ollama_strategy.py | `_JPEG_QUALITY` |
| Query interval | 3.0 s | rover_agent.py | `--interval` |

---

## 9. Typical Run Commands

```bash
# Ollama strategy (primary new strategy)
python rover_agent.py --strategy ollama \
    --device 0 \
    --ollama-server http://192.168.1.x:11434 \
    --ollama-model qwen2.5vl \
    --rover atlas --atlas-port /dev/ttyACM0

# With down camera
python rover_agent.py --strategy ollama \
    --device 0 --down-device 1 \
    --ollama-server http://192.168.1.x:11434 \
    --rover atlas --atlas-port /dev/ttyACM0

# PaliGemma (Google Cloud)
python paligemma_cloud_server.py \
    --project YOUR_GCP_PROJECT \
    --endpoint paligemma-final-endpoint \
    --port 8766
python rover_agent.py --strategy paligemma \
    --cloud-server ws://<server-ip>:8766 \
    --rover atlas --atlas-port /dev/ttyACM0

# ExG row centering (no cloud needed)
python rover_agent.py --strategy row_centering_omnivla \
    --device 0 --down-device 1 \
    --rover atlas --atlas-port /dev/ttyACM0

# Experimental viewer (test Ollama on images/video without rover)
python experimental/ollama_waypoint_viewer.py --video clip.mp4
python experimental/ollama_waypoint_viewer.py --image frame.jpg
python experimental/ollama_waypoint_viewer.py --camera 0

# Verify Ollama reachable from rover
curl http://192.168.1.x:11434/api/tags
```

---

## 10. Ollama Server Setup (macOS)

```bash
# Open port to network (Ollama binds localhost by default)
launchctl setenv OLLAMA_HOST "0.0.0.0"
pkill ollama && ollama serve

# Verify listening on all interfaces
lsof -i :11434   # should show 0.0.0.0:11434

# Get Mac IP
ipconfig getifaddr en0   # WiFi
```
