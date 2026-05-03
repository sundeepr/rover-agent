# Rover Agent — Handoff Context

## 1. Files Changed (this session)

### New files
| File | Purpose |
|------|---------|
| `ollama_strategy.py` | Sends rolling 5-frame history + down-camera frame to local Ollama vision model (default: qwen2.5vl). Converts `next_point [x,y]` to drive command. |
| `cloud_omnivla_strategy.py` | WebSocket client strategy — delegates inference to `omnivla_cloud_server.py` on cloud GPU. State machine: CONNECTING / WAITING_GOAL / NAVIGATING. |
| `omnivla_cloud_server.py` | WebSocket server (cloud GPU). Loads full OmniVLA model. Accepts frames+goal, returns 8×[dx,dy,cos_h,sin_h] waypoints. Inference matches `run_omnivla.py` exactly. |
| `omnivla_full_strategy.py` | Thin subclass of `CloudOmniVLAStrategy` with `name="omnivla_full"`. |
| `paligemma_strategy.py` | Rover-side client for `paligemma_cloud_server.py`. |
| `paligemma_cloud_server.py` | WebSocket proxy → Google Cloud Vertex AI PaliGemma endpoint. Asks "left/right/center?" and converts to waypoints. |
| `row_centering_omnivla_strategy.py` | OmniVLA + ExG (Excess Green `2G-R-B`) crop-row centering. Params: `--exg-threshold` (default 20), `--exg-min-area` (default 500). |
| `experimental/ollama_waypoint_viewer.py` | Standalone test tool. Sends rolling frame history to Ollama, draws `next_point` + `path_points`. |
| `utils/exg_viewer.py` | Interactive ExG tuning tool with live trackbars. |

### Modified files
| File | What changed |
|------|-------------|
| `rover_agent.py` | Added strategies: `ollama`, `cloud_omnivla`, `omnivla_full`, `paligemma`. Added args: `--cloud-server`, `--ollama-model`, `--ollama-history`, `--exg-threshold`, `--exg-min-area`. `--down-device` default changed to `None`. |
| `omnivla_strategy.py` | `_waypoint_to_drive()` rewritten: proportional radius from `dx/dy` ratio, `MAX_LIN_MM_S=50`. `DT=1/3` matching run_omnivla.py tick_rate. |
| `web_server.py` | Fixed MJPEG drop: removed `display:none` JS toggle on `down-cam-box`. |

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
     OllamaStrategy                          CloudOmniVLAStrategy / OmniVLAFullStrategy
     (ollama_strategy.py)                    (cloud_omnivla_strategy.py)
          │                                                      │
     rolls 5-frame history                  WebSocket to cloud server
     → POST /api/generate                   (omnivla_cloud_server.py)
     → parse next_point [x,y]               → full OmniVLA inference
     → _next_point_to_drive()               → 8×4 waypoints
          │                                                      │
          └─────────┬───────────────────────────────────────────┘
                    │
              rover_ctrl.drive_raw(vel, radius_mm)
                    │
              AtlasController / RoombaController
```

---

## 3. Strategy Registry

| `--strategy` | Class | File | Backend |
|---|---|---|---|
| `gemini` | GeminiStrategy | gemini_strategy.py | Google Gemini API |
| `omnivla` | OmniVLAStrategy | omnivla_strategy.py | Local OmniVLA-edge |
| `clip_omnivla` | ClipOmniVLAStrategy | clip_omnivla_strategy.py | CLIP + local OmniVLA-edge |
| `qwen_omnivla` | QwenOmniVLAStrategy | qwen_omnivla_strategy.py | Qwen VLM + local OmniVLA-edge |
| `hough_crop_row` | HoughCropRowStrategy | hough_crop_row_strategy.py | Hough lines + OmniVLA-edge |
| `row_centering_omnivla` | RowCenteringOmniVLAStrategy | row_centering_omnivla_strategy.py | ExG centering + OmniVLA-edge |
| `crop_row` | CropRowStrategy | crop_row_strategy.py | YOLO detection |
| `cloud_omnivla` | CloudOmniVLAStrategy | cloud_omnivla_strategy.py | Full OmniVLA over WebSocket |
| `omnivla_full` | OmniVLAFullStrategy | omnivla_full_strategy.py | Full OmniVLA over WebSocket (same as cloud_omnivla, different name) |
| `paligemma` | PaliGemmaStrategy | paligemma_strategy.py | Google Vertex AI PaliGemma |
| `ollama` | OllamaStrategy | ollama_strategy.py | Local Ollama (qwen2.5vl) |

---

## 4. omnivla_cloud_server.py — Key Design

- Loads full OmniVLA (`omnivla-original` or `omnivla-finetuned-cast`) on cloud GPU
- Inference matches `run_omnivla.py` exactly:
  - Labels built with `IGNORE_INDEX` masking on non-action tokens
  - Uses `get_current_action_mask | get_next_actions_mask` on `labels[:, 1:]`
  - `text_hidden_states = last_hidden_states[:, num_patches:-1]`
  - `num_patches = get_num_patches() * num_images_in_input + 1` (the +1 is goal pose token)
  - `modality_id = tensor([7], float32) → bfloat16`
  - Calls `action_head.predict_action(actions_hidden_states, modality_id)` directly
- `--unnorm-key bridge_orig` — needed for un-normalization (passed but inference doesn't use it since we call action_head directly, not vla.predict_action)
- WebSocket protocol: `{"type":"infer", "goal":"...", "frame_b64":"..."}` → `{"type":"waypoints", "waypoints":[[...],...]}`

### Start command (cloud):
```bash
python omnivla_cloud_server.py \
    --model-path ../OmniVLA/omnivla-original \
    --omnivla-repo ../OmniVLA \
    --unnorm-key bridge_orig \
    --host 0.0.0.0 --port 8765
```

---

## 5. Waypoint → Drive Conversion (`omnivla_strategy._waypoint_to_drive`)

Uses proportional radius from `dx/dy` ratio:
```python
radius_mm = (dx / dy) * 100   # clipped to ±2000mm
vel = MAX_LIN_MM_S = 50 mm/s
```
- Large `dy` (big lateral offset) → small radius (sharp turn)
- Small `dy` (near-straight) → large radius (gentle turn)
- `dx/dy` sign determines left/right
- `DT = 1/3` (matches run_omnivla.py tick_rate=3) — used only in OllamaStrategy now

---

## 6. Known Issues / Incomplete Work

1. **Full OmniVLA turns in circles on straight carpet** — model produces small but consistent `dy` which causes turning. Proportional radius helps but the model is not fine-tuned for wheeled navigation. `omnivla-finetuned-cast` may work better — not yet tested.

2. **`raw.avi` and `annotated.avi` 0 bytes on external drive** — session directory on `/Volumes/Samsung_T5/atlas-1/`. Root cause not fully resolved. `down.avi` records correctly. Not fixed.

3. **PaliGemma "traffic_split not set" error** — occurs if Vertex AI model deployment is in progress. Wait for deployment to complete in Cloud Console.

4. **OllamaStrategy `next_point.x` constant** — qwen2.5vl often returns same x regardless of corridor curvature. Model capability limitation.

5. **`hough_crop_row_strategy.py` CLIP encoding bug** — encodes empty prompts (Ollama deferred but CLIP still runs first). Strategy unused so left as-is.

---

## 7. Calibration Constants & Hardware Values

| Constant | Value | Where | Notes |
|---|---|---|---|
| Forward camera | `--device 0` | rover_agent.py | Default |
| Down camera | `--down-device N` | rover_agent.py | Default `None` (disabled) |
| Send resolution to cloud | 640×480 | cloud_omnivla_strategy.py | `_SEND_W, _SEND_H` |
| OmniVLA full vel | 50 mm/s | omnivla_strategy.py | `MAX_LIN_MM_S` |
| OmniVLA full radius scale | ×100 | omnivla_strategy.py | `(dx/dy)*100` → mm |
| Ollama forward speed | 60 mm/s | ollama_strategy.py | `_FWD_VEL` |
| Atlas serial port | `/dev/ttyACM0` | CLI | `--atlas-port` |
| Roomba serial port | `/dev/ttyUSB0` | CLI | `--roomba-port` |
| Ollama API port | 11434 | ollama_strategy.py | Standard Ollama default |
| OmniVLA cloud WS port | 8765 | omnivla_cloud_server.py | `--port` |
| PaliGemma WS port | 8766 | paligemma_cloud_server.py | `--port` |
| Cloud server IP (Tailscale) | 100.116.45.19 | CLI | May change — check `tailscale ip` |
| ExG threshold | 20 | row_centering_omnivla | `--exg-threshold` |
| ExG min blob area | 500 px² | row_centering_omnivla | `--exg-min-area` |

---

## 8. Typical Run Commands

```bash
# Full OmniVLA on cloud (primary)
python rover_agent.py --strategy omnivla_full \
    --cloud-server ws://100.116.45.19:8765 \
    --goal "Drive forward along the center of the brown carpet" \
    --interval 1.0 --rover roomba --roomba-port /dev/ttyUSB0

# Ollama strategy
python rover_agent.py --strategy ollama \
    --device 0 \
    --ollama-server http://192.168.1.x:11434 \
    --ollama-model qwen2.5vl \
    --rover atlas --atlas-port /dev/ttyACM0

# ExG row centering (no cloud needed)
python rover_agent.py --strategy row_centering_omnivla \
    --device 0 --down-device 1 \
    --rover atlas --atlas-port /dev/ttyACM0

# Roomba straight-line test (mechanical calibration)
python roomba_control.py --port /dev/ttyUSB0 drive --velocity 50 --duration 5
```

---

## 9. Ollama Server Setup (macOS)

```bash
launchctl setenv OLLAMA_HOST "0.0.0.0"
pkill ollama && ollama serve
lsof -i :11434   # verify listening on 0.0.0.0
ipconfig getifaddr en0   # get Mac IP
```

---

## 10. Pending / Next Steps

- Test `omnivla-finetuned-cast` model on cloud — likely better for outdoor wheeled navigation
- Tune `(dx/dy)*100` radius scale factor empirically — may need adjustment for actual rover turning behaviour
- Investigate full OmniVLA `dy` bias — likely needs camera centring or prompt tuning
