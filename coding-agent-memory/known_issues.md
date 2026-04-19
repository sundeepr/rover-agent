---
name: Known issues and fixes
description: GPU OOM, numpy conflict, video crash, watchdog firing, and other solved problems
type: project
originSessionId: 74eb3b7b-c6c1-4d44-b30a-0225d3b8f403
---
## GPU OOM on Jetson (local mode)
**Root cause:** Ollama/Qwen3 called before OmniVLA+CLIP loaded → fragments unified memory
**Fix:** `clip_omnivla_strategy._load()` calls `generate_clip_prompts()` only AFTER OmniVLA+CLIP
are fully on GPU. `set_goal()` returns early if `not self._loaded.is_set()`.
`num_gpu: 0` was removed from `prompt_generator.py` — load ordering alone is sufficient.

## numpy<2.0 required for YOLO
**Root cause:** Ultralytics 8.x uses removed numpy aliases (`np.bool` etc.)
**Fix:** `pip install "numpy>=1.23.0,<2.0"` — pinned in requirements.txt
Jetson had numpy 2.4.3; PyTorch 2.11+cu130 works fine with numpy 1.x.

## Video unplayable after crash (raw.avi missing)
**Root cause:** mp4v/.mp4 writes moov atom only on clean close — crash = unplayable.
**Fix:** Switched to MJPG/.avi — writes index incrementally, crash-safe.
If raw.avi VideoWriter fails to open, `os._exit(1)` with a clear error message.

## WS watchdog firing during joystick use
**Root cause:** Joystick JS only sends messages when non-zero; dead zone silence >300ms triggers watchdog.
**Current state:** Not fixed — _WATCHDOG_S = 0.3s, user decided no change needed.

## OmniVLA θ (heading) was ignored
**Root cause:** Was using `atan2(dy, dx)` (geometric angle) instead of model's explicit θ.
**Fix:** `heading = atan2(wp[3], wp[2])` — uses cosθ/sinθ directly from waypoint.
Applied in both `omnivla_strategy._waypoint_to_drive()` and `clip_omnivla_strategy`.
Note: `omnivla_server.py` infer() still uses old `atan2(dy, dx)` — not yet fixed there.

## ultralyticsplus breaks pip on Python 3.10
Removed from requirements.txt — AssertionError in pip resolver. Use plain `ultralytics`.

## CLIP can't detect soil gap in crop rows
CLIP ViT-B/32 not trained on agricultural imagery — path detection unreliable.
Workaround: use `crop_row` or `row_centering_omnivla` strategies with YOLO instead.
AgriCLIP exists but was not integrated.
