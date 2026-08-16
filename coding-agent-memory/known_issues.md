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

## cosmos_cloud_server.py --mode av_policy fails to load (venv: /home/sundeep/claude-cosmos/venv)
**Symptom 1 — crash on import:** `RuntimeError: Detected that PyTorch and TorchAudio were
compiled with different CUDA versions` (torch `2.13.0+cu132` vs `torchaudio 2.11.0`, PyPI's
final torchaudio release — the project no longer tracks new torch versions). `transformers`/
`diffusers` only catch `ImportError` around the optional `import torchaudio`, so the `RuntimeError`
propagates and crashes startup, even though `av_policy` mode never touches audio.
**Fix:** `pip uninstall -y torchaudio` — the optional import then fails softly (ModuleNotFoundError)
and audio support is disabled, which is fine for image/video-only modes.

**Symptom 2 — crash after fixing #1:** `NotImplementedError: Cannot copy out of meta tensor;
no data!` when `Cosmos3OmniPipeline.from_pretrained(..., device_map="cuda")` tries to move the
transformer to GPU. Root cause: the Cosmos3-Edge checkpoint's `model_index.json` was exported
against `diffusers==0.40.0.dev0`, but the venv had the PyPI release `0.39.0`. The 0.39.0
`Cosmos3OmniTransformer` doesn't have matching `norm_q`/`norm_k`/MoE (`mlp_moe_gen.gate_proj`)
weight names, so those params never get loaded from the checkpoint and stay on the `meta`
device — `.to("cuda")` then has nothing to copy.
**Fix:** `pip install --upgrade "git+https://github.com/huggingface/diffusers.git"` (installs
main branch, which resolves to `0.40.0.dev0` — matches the checkpoint exactly, no more
missing/newly-initialized weight warnings).
**Side effect:** this venv also has `vllm-omni 0.26.0` installed, which pins
`diffusers==0.38.0` and `accelerate==1.12.0` — now conflicts with the upgraded versions
(`0.40.0.dev0`/`1.14.0`). Not an issue for `av_policy` mode itself, but if the `reasoning_*`
modes (which use vllm-omni) break in this same venv, this is why — may need a separate venv.
**Also noted (non-blocking):** PyTorch prints "No published PyTorch CUDA builds... support this
GPU" for the Jetson Orin's compute capability 8.7 — cosmetic in testing (dispatch to `cuda:0`
and inference load both worked fine), but worth checking first if numerical/perf oddities show
up later.
