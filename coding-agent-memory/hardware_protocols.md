---
name: Hardware and protocols
description: Roomba OI, Atlas STM32 protocol, kinematic formulas, Jetson memory constraints
type: project
originSessionId: 74eb3b7b-c6c1-4d44-b30a-0225d3b8f403
---
## Rovers

### iRobot Roomba (OI protocol)
- `drive_raw(velocity_mm_s, radius_mm)`
- velocity: ±500 mm/s
- radius: signed 16-bit mm; `0x8000` = straight; `+1` = CCW spin; `-1` = CW spin
- Positive radius = turn left (CCW); negative radius = turn right (CW)
- CLI: `--rover roomba --roomba-port /dev/ttyUSB0`

### Atlas STM32 (`$CMD` protocol)
- Same `drive_raw()` interface
- CLI: `--rover atlas --atlas-port /dev/ttyACM0`

## Kinematic formula (used everywhere)
```
radius = velocity / angular_rate
angular_rate = Kp * error_px        # crop_row / joystick
angular_rate = atan2(sinθ, cosθ)/DT # OmniVLA (θ from waypoint wp[4])
```
Positive error → gap right of centre → steer right → **negative radius**

## Joystick (control_server.py)
```python
MAX_VEL     = 150  # mm/s
MAX_ANG_RAD_S = 1.0
vel    = fwd * MAX_VEL // 100
ang    = (turn / 100) * MAX_ANG_RAD_S
radius = vel / ang  # clamped ±2000mm
```
Watchdog: 0.3s without message → stop rover

## Jetson unified memory constraints
- GPU and CPU share same physical RAM
- Load order matters: OmniVLA → CLIP → then Ollama/Qwen3
- `clip_omnivla_strategy._load()` defers `generate_clip_prompts()` call to AFTER OmniVLA+CLIP loaded
- `set_goal()` returns early if `not self._loaded.is_set()` to prevent Ollama racing OmniVLA
- `omnivla_server.py` runs OmniVLA+CLIP in a persistent server process; use `--omnivla-server localhost:5100`
  to avoid reloading models on every rover_agent restart

## numpy version constraint
- Ultralytics 8.x requires `numpy<2.0`
- Jetson had numpy 2.4.3 installed — causes YOLO to crash
- Fix: `pip install "numpy>=1.23.0,<2.0"` and pinned in requirements.txt
