#!/usr/bin/env python3
"""
Rover navigation agent — thin orchestrator.

Opens the camera, wires up the chosen navigation strategy and rover controller,
then runs the agent loop and a publisher thread in the background. The main
thread blocks until Ctrl-C or SIGTERM.

The web UI is served by the standalone web_server.py process. Start it once and
leave it running; the agent connects to it on startup and publishes frames and
status over HTTP. The browser tab survives agent restarts and crashes.

Supported rovers    : roomba (iRobot OI), atlas (STM32 $CMD protocol)
Supported strategies:
    omnivla_full   — full OmniVLA on cloud GPU (primary)
    bev_omnivla    — BEV safety layer + cloud OmniVLA
    omnivla        — OmniVLA-edge local inference
    cloud_omnivla  — full OmniVLA over WebSocket (alias for omnivla_full)
    line_follow    — pure-CV HSV line/pipe follower
    plant_center   — down-camera plant centering (no cloud)
    boundary_guard — down-camera crop row safety stop
    teleop         — human teleoperation + data collection

Usage:
    # Start the web server once (separate terminal):
    python web_server.py

    # Full OmniVLA on cloud (primary use case)
    python rover_agent.py --strategy omnivla_full \\
        --cloud-server ws://<cloud-ip>:8765 \\
        --goal "Follow the crop row" \\
        --rover roomba --roomba-port /dev/ttyUSB0

    # Line following (no cloud needed)
    python rover_agent.py --strategy line_follow \\
        --rover atlas --atlas-port /dev/ttyACM0 \\
        --line-color black

    # Camera only, no hardware
    python rover_agent.py --dry-run
"""

import argparse
import json
import logging
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

import roomba_controller
import atlas_controller
from navigation_strategy import AgentState, NavigationStrategy
from agent_publisher import AgentPublisher


# ── Logging ────────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"rover_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("rover")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                                       datefmt="%Y-%m-%d %H:%M:%S"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                                       datefmt="%Y-%m-%d %H:%M:%S"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Log file: %s", log_file.resolve())
    return logger


log = setup_logging()


# ── Rover geometry helpers ────────────────────────────────────────────────────

_GEOMETRY_DEFAULTS = {
    "icr_offset_mm":          480,
    "down_px_per_mm":         2.5,
    "rover_polygon_px":       [[120, 180], [520, 180], [520, 380], [120, 380]],
    "lookahead_s":            1.0,
    "arc_steps":              10,
    "exg_threshold":          60,
    "exg_min_area":           500,
    "exg_density_pct":        8.0,
    "correction_goal_suffix": "steer slightly {direction} to avoid vegetation",
}


def load_geometry(path: str | None) -> dict:
    """Load rover_geometry.json, merging with defaults for any missing keys."""
    cfg = _GEOMETRY_DEFAULTS.copy()
    if path:
        p = Path(path)
        if p.exists():
            try:
                cfg.update(json.loads(p.read_text()))
                log.info("Rover geometry loaded from %s", p)
            except Exception as e:
                log.warning("Could not read %s (%s) — using defaults", p, e)
        else:
            log.warning("rover-geometry file not found: %s — using defaults", p)
    return cfg


# ── Agent loop (runs on a daemon thread) ───────────────────────────────────────

def _build_rover_ctrl(rover: str, port: str | None, dry_run: bool):
    """Instantiate and return the appropriate rover controller, or None."""
    if not port:
        return None
    if rover == "roomba":
        return roomba_controller.RoombaController(port=port, dry_run=dry_run)
    if rover == "atlas":
        return atlas_controller.AtlasController(port=port, dry_run=dry_run)
    raise ValueError(f"Unknown rover: {rover!r}")


def agent_loop(
    state: AgentState,
    strategy: NavigationStrategy,
    device,
    interval: float,
    rover_ctrl=None,
    cam_controls: dict | None = None,
) -> None:
    """
    Camera capture loop — runs on a daemon thread at ~30 fps.

    Continuously reads frames from the camera and pushes them to
    state.raw_frame. Every `interval` seconds, if no query is already
    in-flight, increments state.step and spawns a new daemon thread to
    call strategy.run_query(). This keeps the camera loop completely
    non-blocking regardless of how long inference takes.

    rover_ctrl is an already-connected controller (or None). Connection
    lifecycle is managed by main() so the stop command is guaranteed to
    run on shutdown even when the program is killed.

    device may be an int index, a /dev/ path, or a ws:// WebSocket URL.
    WebSocket URLs are handled via WebSocketFrameSource (camera_ws_server.py).
    cam_controls is applied via v4l2-ctl for local devices (exposure, etc.).
    """
    from frame_source import open_frame_source

    src = open_frame_source(device, "front", cam_controls=cam_controls, fps=10)
    if src is None:
        log.error("Could not open front camera: %s", device)
        return

    # Wait up to 15 s for first frame (WebSocket streams need connect time)
    log.info("Waiting for front camera first frame…")
    for _ in range(150):
        if src.read() is not None:
            break
        time.sleep(0.1)
    else:
        log.error("Front camera %s: no frame after 15 s", device)
        src.release()
        return
    log.info("Front camera ready: %s", device)

    captures_dir = Path("captures")
    captures_dir.mkdir(exist_ok=True)
    log.info("Saving LLM frames to: %s", captures_dir.resolve())

    last_query_time       = 0.0
    _logged_in_flight     = False
    _consecutive_failures = 0

    # Prefer the strategy's own cycle_interval over the CLI --interval flag.
    # Pure-vision strategies (plant_center, boundary_guard) set this to ~0.1 s
    # so the rover gets a fresh drive command at 10 Hz instead of the default 3 s.
    # Cloud/LLM strategies leave it None and rely on the user-supplied --interval.
    effective_interval = getattr(strategy, "cycle_interval", None) or interval
    log.info("Query cycle interval: %.2f s  (strategy=%s, cli_interval=%.2f s)",
             effective_interval, strategy.name, interval)

    while True:
        frame = src.read()
        if frame is None:
            _consecutive_failures += 1
            if _consecutive_failures == 1:
                log.warning("Camera %s: failed to grab frame (will retry)", device)
            if _consecutive_failures >= 30:
                log.error("Camera %s: 30 consecutive failures — giving up", device)
                break
            time.sleep(0.033)
            continue
        _consecutive_failures = 0

        # Always push raw frame — never blocked by queries
        with state.raw_lock:
            state.raw_frame = frame.copy()

        # Record raw frame only (annotated frames are not saved to disk)
        if state.recorder:
            state.recorder.write_frames(frame)

        now = time.time()

        # Fire strategy query in a separate thread so camera loop never blocks.
        # Gate on goal_ready so no queries fire until a goal has been received,
        # unless the strategy declares requires_goal=False (e.g. plant_center).
        goal_ok = (state.goal_ready.is_set()
                   or not getattr(strategy, "requires_goal", True))
        if (goal_ok
                and now - last_query_time >= effective_interval
                and not state.paused.is_set()):
            if state.query_in_flight.is_set():
                if not _logged_in_flight:
                    log.info("Previous query still in-flight — skipping until complete")
                    _logged_in_flight = True
            else:
                _logged_in_flight = False
                last_query_time = now
                with state.result_lock:
                    state.step += 1
                state.query_in_flight.set()
                threading.Thread(
                    target=strategy.run_query,
                    args=(state, frame.copy(), captures_dir, rover_ctrl),
                    daemon=True,
                ).start()

        time.sleep(0.033)   # ~30 fps

    src.release()
    log.info("Camera released")


# ── Headless strategy loop (no front camera) ──────────────────────────────────

def _headless_loop(state, strategy, interval: float, rover_ctrl=None) -> None:
    """
    Dispatch strategy.run_query() at the strategy's cycle_interval without
    opening any camera.  Used when --device is omitted (e.g. wheel_guard).
    A blank frame is passed so run_query signatures still work.
    """
    import numpy as np
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    last_query_time   = 0.0
    _logged_in_flight = False

    while True:
        now               = time.time()
        effective_interval = getattr(strategy, "cycle_interval", interval)
        goal_ok = (state.goal_ready.is_set()
                   or not getattr(strategy, "requires_goal", True))

        if goal_ok and now - last_query_time >= effective_interval:
            if state.query_in_flight.is_set():
                if not _logged_in_flight:
                    log.info("Previous query still in-flight — skipping")
                    _logged_in_flight = True
            else:
                _logged_in_flight = False
                last_query_time   = now
                with state.result_lock:
                    state.step += 1
                state.query_in_flight.set()
                threading.Thread(
                    target=strategy.run_query,
                    args=(state, blank, None, rover_ctrl),
                    daemon=True,
                ).start()

        time.sleep(0.01)


# ── Down-camera loop ───────────────────────────────────────────────────────────

def _scan_cameras(max_index: int = 6) -> list[int]:
    """Return a list of camera device indices that can be opened."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
        cap.release()
    return available


def _down_camera_loop(strategy, device: int, state=None) -> None:
    """
    Capture loop for the downward-facing camera (row_centering_omnivla only).

    Opened and managed here — not inside the strategy — so all camera lifecycle
    stays in rover_agent.py and device indices are unambiguous.

    Retries indefinitely if the device is not yet available at startup.
    """
    cap = None
    while cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            available = _scan_cameras()
            log.error(
                "Down-camera device %s not available — available devices: %s  "
                "(use --down-device to select the correct one)",
                device, available or "none found",
            )
            time.sleep(3.0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    log.info("Down-camera opened: device %s  %dx%d (max)", device,
             int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    consecutive_failures = 0
    while True:
        ret, frame = cap.read()
        if ret:
            strategy.update_down_frame(frame)
            if state is not None and state.recorder:
                # Always write raw down-camera frame (no annotated overlays on disk)
                state.recorder.write_down_frame(frame)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= 30:
                log.warning("Down-camera device %s: 30 consecutive read failures — reopening", device)
                cap.release()
                cap = cv2.VideoCapture(device)
                consecutive_failures = 0
        time.sleep(0.033)
    cap.release()


# ── Strategy factory ───────────────────────────────────────────────────────────

def _build_cam_controls(args) -> dict:
    """Build v4l2 control dict from CLI args for wheel cameras."""
    ctrl = {}
    ctrl["backlight_compensation"] = args.cam_backlight
    if args.cam_saturation is not None:
        ctrl["saturation"] = args.cam_saturation
    if args.cam_exposure is not None:
        ctrl["exposure_time_absolute"] = args.cam_exposure
    return ctrl


def _build_strategy(name: str, args) -> NavigationStrategy:
    """
    Instantiate and return the requested NavigationStrategy.

    Strategies are imported lazily so their heavy dependencies (torch, etc.)
    are only loaded when actually needed.
    """
    if name == "omnivla":
        from omnivla_strategy import OmniVLAStrategy, load_camera_calibration
        return OmniVLAStrategy(goal=args.goal, goal_image_path=args.goal_image,
                               server_addr=args.omnivla_server,
                               camera_calibration=load_camera_calibration(
                                   args.camera_calibration))
    if name == "bev_omnivla":
        from bev_omnivla_strategy import BevOmniVLAStrategy
        return BevOmniVLAStrategy(
            server_url    = args.cloud_server,
            goal          = args.goal,
            max_lin_mm_s  = args.omnivla_velocity,
            geometry_path = args.rover_geometry,
        )
    if name == "boundary_guard":
        from boundary_guard_strategy import BoundaryGuardStrategy
        return BoundaryGuardStrategy(geometry_path=args.rover_geometry)
    if name == "plant_center":
        from plant_center_strategy import PlantCenterStrategy
        return PlantCenterStrategy(geometry_path=args.rover_geometry)
    if name == "cloud_omnivla":
        from cloud_omnivla_strategy import CloudOmniVLAStrategy
        _geo = load_geometry(getattr(args, "rover_geometry", None))
        return CloudOmniVLAStrategy(
            server_url=args.cloud_server,
            goal=args.goal,
            max_lin_mm_s=args.omnivla_velocity,
            icr_offset_m=_geo["icr_offset_mm"] / 1000.0,
        )
    if name == "crop_guard":
        from crop_guard_strategy import CropGuardStrategy
        from omnivla_strategy import load_camera_calibration
        _geo = load_geometry(getattr(args, "rover_geometry", None))
        return CropGuardStrategy(
            server_url          = args.cloud_server,
            goal                = args.goal,
            left_device         = args.left_cam,
            right_device        = args.right_cam,
            max_lin_mm_s        = args.omnivla_velocity,
            crop_guard_vel      = args.crop_guard_vel,
            icr_offset_m        = _geo["icr_offset_mm"] / 1000.0,
            exg_threshold       = args.exg_threshold,
            exg_min_area        = args.exg_min_area,
            exg_density_pct     = args.exg_density_pct,
            veg_index           = args.veg_index,
            clahe               = args.clahe,
            clahe_clip          = args.clahe_clip,
            cam_controls        = _build_cam_controls(args),
            camera_calibration  = load_camera_calibration(args.camera_calibration),
        )
    if name == "wheel_guard":
        from wheel_guard_strategy import WheelGuardStrategy
        return WheelGuardStrategy(
            left_device       = args.left_cam,
            right_device      = args.right_cam,
            forward_vel       = args.crop_guard_vel,
            exg_threshold     = args.exg_threshold,
            exg_min_area      = args.exg_min_area,
            exg_density_pct   = args.exg_density_pct,
            veg_index         = args.veg_index,
            clahe             = args.clahe,
            clahe_clip        = args.clahe_clip,
            cam_controls      = _build_cam_controls(args),
        )
    if name == "crop_spray":
        from crop_spray_strategy import CropSprayStrategy
        return CropSprayStrategy(
            left_device       = args.left_cam,
            right_device      = args.right_cam,
            forward_vel       = args.crop_guard_vel,
            guard_duration_s  = args.guard_duration,
            exg_threshold     = args.exg_threshold,
            exg_min_area      = args.exg_min_area,
            exg_density_pct   = args.exg_density_pct,
            veg_index         = args.veg_index,
            clahe             = args.clahe,
            clahe_clip        = args.clahe_clip,
            cam_controls      = _build_cam_controls(args),
            arm_port          = args.arm_port,
            arm_cam_device    = args.arm_cam,
            arm_config_path   = args.arm_config,
            arm_sweep_spd     = args.arm_spd,
            arm_aux_pct       = args.arm_aux,
        )
    if name == "omnivla_full":
        from omnivla_full_strategy import OmniVLAFullStrategy
        _geo = load_geometry(getattr(args, "rover_geometry", None))
        return OmniVLAFullStrategy(
            server_url=args.cloud_server,
            goal=args.goal,
            max_lin_mm_s=args.omnivla_velocity,
            icr_offset_m=_geo["icr_offset_mm"] / 1000.0,
        )
    if name == "crop_row":
        from crop_row_strategy import CropRowStrategy
        return CropRowStrategy(
            left_device       = args.left_cam,
            forward_vel       = args.crop_guard_vel,
            flow_threshold    = args.flow_threshold,
            row_end_frames    = args.row_end_frames,
            overshoot_s       = args.overshoot_s,
            turn_90_s         = args.turn_90_duration,
            inter_row_s       = args.inter_row_s,
            turn_direction    = args.turn_direction,
            rock_fwd_s        = args.rock_fwd_s,
            rock_bwd_s        = args.rock_bwd_s,
            rock_max_cycles   = args.rock_max_cycles,
            balance_threshold = args.balance_threshold,
            balance_frames    = args.balance_frames,
            align_fwd_vel     = args.nudge_vel,
            veg_index         = args.veg_index,
            exg_threshold     = args.exg_threshold,
            cam_controls      = _build_cam_controls(args),
        )
    if name == "row_change":
        from row_change_strategy import RowChangeStrategy
        return RowChangeStrategy(
            qwen_server_url    = args.qwen_server,
            forward_vel        = args.crop_guard_vel,
            turn_90_duration_s = args.turn_90_duration,
            nudge_mm           = args.nudge_mm,
            nudge_vel          = args.nudge_vel,
            qwen_interval_s    = args.qwen_interval,
            end_confirmations  = args.end_confirmations,
        )
    if name == "line_follow":
        from line_follow_strategy import LineFollowStrategy
        return LineFollowStrategy(
            vel_mm_s=args.line_vel,
            kp=args.line_kp,
            color=args.line_color,
        )
    if name == "teleop":
        from teleop.teleop_strategy import TeleopStrategy
        return TeleopStrategy(
            dataset_dir=args.dataset_dir,
            instruction=args.teleop_instruction,
            fps=args.teleop_fps,
        )
    raise ValueError(f"Unknown strategy: {name!r}")


# ── Entry point ────────────────────────────────────────────────────────────────

def _device(value: str):
    """Accept either an integer index (0, 2, 4) or a device path (/dev/cam-front)."""
    try:
        return int(value)
    except ValueError:
        return value   # pass the path string straight through to cv2.VideoCapture


def main():
    parser = argparse.ArgumentParser(description="Rover navigation agent")
    parser.add_argument("--device",      type=_device, default=None,
                        metavar="INDEX|PATH",
                        help="Camera device index or path (e.g. 0 or /dev/cam-front)")
    parser.add_argument("--interval",    type=float, default=3.0,
                        help="Seconds between LLM queries")
    parser.add_argument("--web-server",  type=str,   default="http://localhost:5001",
                        metavar="URL",
                        help="URL of the running web_server.py "
                             "(default: http://localhost:5001)")
    parser.add_argument("--rover",       type=str,   default="roomba",
                        choices=["roomba", "atlas"],
                        help="Rover hardware (default: roomba)")
    parser.add_argument("--roomba-port", type=str,   default=None,
                        help="Roomba serial port (e.g. /dev/ttyUSB0)")
    parser.add_argument("--atlas-port",  type=str,   default=None,
                        help="Atlas-1 serial port (e.g. /dev/ttyACM0)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Log rover commands but do not send them")
    parser.add_argument("--strategy",    type=str,   default="omnivla_full",
                        choices=["omnivla_full", "cloud_omnivla", "bev_omnivla",
                                 "omnivla", "line_follow", "plant_center",
                                 "boundary_guard", "teleop", "crop_guard",
                                 "wheel_guard", "crop_spray", "row_change",
                                 "crop_row"],
                        help="Navigation strategy (default: omnivla_full)")
    parser.add_argument("--left-cam",   type=_device, default=1,
                        metavar="INDEX|PATH",
                        help="Left wheel camera index or path (crop_guard, default 1)")
    parser.add_argument("--right-cam",  type=_device, default=2,
                        metavar="INDEX|PATH",
                        help="Right wheel camera index or path (crop_guard, default 2)")
    parser.add_argument("--crop-guard-vel", type=int, default=10,
                        metavar="MM_S",
                        help="Navigation speed (mm/s) in crop_guard mode (default 10). "
                             "Deadband compensation ensures motors receive at least "
                             "_MOTOR_DEADBAND_PCT%% power even at this low value.")
    parser.add_argument("--guard-duration", type=float, default=2.0,
                        metavar="SECS",
                        help="Seconds of clear-wheel driving before arm spray sweep (default 2.0)")
    parser.add_argument("--arm-port",   type=str, default="/dev/ttyUSB0",
                        help="Serial port for RoArm-M2-S (default /dev/ttyUSB0)")
    parser.add_argument("--arm-cam",    type=int, default=0,
                        help="Camera device index mounted on arm tip (default 0)")
    parser.add_argument("--arm-config", type=str,
                        default=str(Path(__file__).parent / "experimental" / "arm_scan_config.json"),
                        help="Path to arm_scan_config.json")
    parser.add_argument("--arm-spd",    type=int, default=5,
                        help="Arm continuous rotation speed 0-20 (default 5)")
    parser.add_argument("--arm-aux",    type=int, default=50,
                        metavar="PCT",
                        help="AUX output %% sent to rover when plant centred (default 50)")
    parser.add_argument("--exg-threshold", type=int, default=60,
                        metavar="N",
                        help="ExG vegetation threshold for wheel cameras (default 60)")
    parser.add_argument("--exg-min-area",  type=int, default=500,
                        metavar="PX",
                        help="Min vegetation blob area in pixels (default 500)")
    parser.add_argument("--cam-exposure", type=int, default=None,
                        metavar="N",
                        help="Manual exposure time (1–10000, C270 units ~100µs). "
                             "Sets auto_exposure=1 then exposure_time_absolute=N. "
                             "Try 50–80 in bright sun (default=None = camera auto)")
    parser.add_argument("--cam-backlight", type=int, default=0,
                        choices=[0, 1],
                        help="backlight_compensation (0=off, 1=on, default 0). "
                             "Off is better outdoors — on boosts brightness for "
                             "backlit subjects which washes out outdoor scenes")
    parser.add_argument("--cam-saturation", type=int, default=None,
                        metavar="N",
                        help="Saturation 0–255 (C270 default 32). Higher values make "
                             "greens more distinct from white/grey soil. Try 80–120")
    parser.add_argument("--clahe", action="store_true",
                        help="Apply CLAHE contrast enhancement to wheel camera frames "
                             "before vegetation detection — helps in bright sunlight")
    parser.add_argument("--clahe-clip", type=float, default=2.0,
                        metavar="CLIP",
                        help="CLAHE clip limit (default 2.0; higher = more contrast)")
    parser.add_argument("--veg-index", type=str, default="ngrdi",
                        choices=["exg", "exgnorm", "ngrdi", "vari"],
                        help="Vegetation index for wheel cameras. "
                             "ngrdi/vari are ratio-based and work better in "
                             "bright sunlight where exg fails (default: ngrdi)")
    parser.add_argument("--exg-density-pct", type=float, default=8.0,
                        metavar="PCT",
                        help="Min %% of wheel-zone pixels above ExG threshold to "
                             "declare trampling; filters out sparse soil noise "
                             "(default 8.0)")
    parser.add_argument("--cloud-server", type=str,  default="ws://localhost:8765",
                        metavar="URL",
                        help="WebSocket URL of omnivla_cloud_server.py "
                             "(cloud_omnivla strategy, default: ws://localhost:8765)")
    parser.add_argument("--qwen-server", type=str,  default="ws://localhost:8766",
                        metavar="URL",
                        help="WebSocket URL of qwen_cloud_server.py "
                             "(row_change strategy, default: ws://localhost:8766)")
    parser.add_argument("--turn-90-duration", type=float, default=4.5,
                        metavar="SECS",
                        help="Time in seconds for a 90° tank turn (row_change, default 4.5). "
                             "Calibrate on the actual hardware.")
    parser.add_argument("--nudge-mm", type=int, default=150,
                        metavar="MM",
                        help="Step size in mm for each forward nudge when finding/aligning "
                             "to next row (row_change, default 150)")
    parser.add_argument("--nudge-vel", type=int, default=30,
                        metavar="MM_S",
                        help="Speed in mm/s for nudge moves (row_change, default 30)")
    parser.add_argument("--qwen-interval", type=float, default=3.0,
                        metavar="SECS",
                        help="Seconds between Qwen end-of-row checks during row following "
                             "(row_change, default 3.0)")
    parser.add_argument("--end-confirmations", type=int, default=2,
                        metavar="N",
                        help="Consecutive YES answers from Qwen before triggering row change "
                             "(default 2)")
    # ── crop_row strategy args ─────────────────────────────────────────────
    parser.add_argument("--flow-threshold", type=float, default=1.5,
                        metavar="PX",
                        help="Residual optical flow magnitude (px/frame) above which "
                             "plant leaf motion is detected (crop_row, default 1.5). "
                             "Tune from LEFT CAM FLOW log: raise if false positives on "
                             "soil, lower if plants are missed.")
    parser.add_argument("--row-end-frames", type=int, default=10,
                        metavar="N",
                        help="Consecutive 20 Hz frames with zero left-cam EXG before "
                             "row-end is declared (crop_row, default 10 ≈ 0.5 s)")
    parser.add_argument("--overshoot-s", type=float, default=1.5,
                        metavar="SECS",
                        help="Seconds to drive forward past row-end before first 90° turn "
                             "(crop_row, default 1.5)")
    parser.add_argument("--inter-row-s", type=float, default=2.0,
                        metavar="SECS",
                        help="Seconds to drive across the inter-row gap between the two "
                             "90° headland turns (crop_row, default 2.0)")
    parser.add_argument("--turn-direction", type=str, default="right",
                        choices=["right", "left"],
                        help="Direction of both headland 90° turns (crop_row, default right)")
    parser.add_argument("--rock-fwd-s", type=float, default=0.4,
                        metavar="SECS",
                        help="Duration of each forward burst during row-alignment rocking "
                             "(crop_row, default 0.4)")
    parser.add_argument("--rock-bwd-s", type=float, default=0.4,
                        metavar="SECS",
                        help="Duration of each backward burst during row-alignment rocking "
                             "(crop_row, default 0.4)")
    parser.add_argument("--rock-max-cycles", type=int, default=20,
                        metavar="N",
                        help="Maximum rock cycles before proceeding regardless "
                             "(crop_row, default 20)")
    parser.add_argument("--balance-threshold", type=float, default=15.0,
                        metavar="EXG",
                        help="Front-cam EXG mean difference (0–255) below which "
                             "left/right alignment is declared (crop_row, default 15)")
    parser.add_argument("--balance-frames", type=int, default=5,
                        metavar="N",
                        help="Consecutive balanced readings required to exit rocking "
                             "(crop_row, default 5)")
    parser.add_argument("--omnivla-weights", type=str, default=None,
                        metavar="PATH",
                        help="Path to custom OmniVLA-edge weights (.pth). "
                             "Defaults to downloading from HuggingFace if not set.")
    parser.add_argument("--goal",        type=str,   default="",
                        help="Language goal for omnivla strategies. "
                             "If omitted, wait for goal via web chat UI.")
    parser.add_argument("--goal-image",  type=str,   default=None,
                        help="Path to a goal image for omnivla strategy (optional)")
    parser.add_argument("--omnivla-server", type=str, default=None,
                        metavar="HOST:PORT",
                        help="Address of a running omnivla_server.py "
                             "(e.g. localhost:5100)")
    parser.add_argument("--camera-calibration", type=str, default=None,
                        metavar="FILE",
                        help="Path to camera_calibration.json produced by "
                             "calibration/camera_calibrate.py. Enables perspective "
                             "projection of OmniVLA waypoints onto the camera feed.")
    parser.add_argument("--rover-geometry", type=str, default="rover_geometry.json",
                        metavar="FILE",
                        help="Path to rover_geometry.json for tunable measurements "
                             "(ICR offset, down-camera scale, rover polygon, etc.). "
                             "Defaults to rover_geometry.json in the working directory. "
                             "Hot-reloaded each inference cycle by bev_omnivla.")
    parser.add_argument("--omnivla-velocity", type=int, default=55,
                        metavar="MM_S",
                        help="Forward velocity for OmniVLA strategies in mm/s "
                             "(default: 55, ~27%% wheel power). MIN_RADIUS_MM "
                             "is auto-computed as vel / MAX_ANG_RAD_S.")
    parser.add_argument("--line-vel", type=int, default=80, metavar="MM_S",
                        help="Forward speed for line_follow strategy (default: 80)")
    parser.add_argument("--line-kp", type=float, default=2000.0, metavar="GAIN",
                        help="Proportional steering gain for line_follow (default: 2000)")
    parser.add_argument("--line-color", type=str, default="black",
                        choices=["black", "grey", "blue", "orange", "red"],
                        help="Target line colour for line_follow (default: black)")
    parser.add_argument("--line-threshold", type=int, default=80, metavar="0-255",
                        help="(unused) kept for compatibility")
    parser.add_argument("--line-roi-frac", type=float, default=0.4, metavar="FRAC",
                        help="(unused) kept for compatibility")
    parser.add_argument("--line-edge-margin", type=float, default=0.15, metavar="FRAC",
                        help="(unused) kept for compatibility")
    parser.add_argument("--dataset-dir", type=str, default="./dataset",
                        help="Base directory for teleop dataset (default: ./dataset)")
    parser.add_argument("--teleop-instruction", type=str, default="",
                        help="Default navigation instruction for teleop episodes")
    parser.add_argument("--teleop-fps", type=int, default=10,
                        help="Frame recording rate for teleop strategy (default: 10)")
    parser.add_argument("--down-device",     type=_device, default=None,
                        metavar="INDEX|PATH",
                        help="Camera device index or path for downward-facing camera. "
                             "Omit to disable the down camera entirely.")
    parser.add_argument("--control-port",  type=int, default=5002,
                        metavar="PORT",
                        help="WebSocket joystick control port for browser/Android "
                             "(default: 5002, 0 = disabled)")
    args = parser.parse_args()

    rover_port = args.atlas_port if args.rover == "atlas" else args.roomba_port

    log.info("=== Rover agent starting ===")
    log.info("Camera device : %s", args.device)
    log.info("Query interval: %.1fs", args.interval)
    log.info("Strategy      : %s", args.strategy)
    if args.strategy == "omnivla":
        log.info("Goal          : %s", args.goal)
        if args.goal_image:
            log.info("Goal image    : %s", args.goal_image)
        if args.omnivla_server:
            log.info("OmniVLA server: %s", args.omnivla_server)
        else:
            log.info("OmniVLA server: (loading locally)")
    if args.strategy in ("cloud_omnivla", "omnivla_full", "bev_omnivla"):
        log.info("Cloud server  : %s", args.cloud_server)
        log.info("Goal          : %s", args.goal)
    log.info("Web server    : %s", args.web_server)
    log.info("Rover         : %s", args.rover)
    if rover_port:
        log.info("Rover port    : %s%s", rover_port, " (dry-run)" if args.dry_run else "")
    else:
        log.info("Rover         : disabled (pass --roomba-port or --atlas-port to enable)")

    from session_recorder import SessionRecorder
    state                = AgentState()
    state.recorder       = SessionRecorder()
    state.query_interval = args.interval
    strategy             = _build_strategy(args.strategy, args)
    if hasattr(strategy, "set_recorder"):
        strategy.set_recorder(state.recorder)

    # If a goal was given on the CLI, apply it immediately so the agent
    # starts navigating without waiting for web chat input.
    _NO_GOAL_STRATEGIES = ("line_follow", "plant_center", "boundary_guard", "teleop",
                           "wheel_guard", "crop_spray", "row_change", "crop_row")
    if args.goal and args.strategy not in _NO_GOAL_STRATEGIES:
        strategy.set_goal(args.goal)
        with state.result_lock:
            state.goal = args.goal
        log.info("CLI goal applied: '%s'", args.goal)
        # goal_ready is set below after camera check + user confirmation
    elif args.strategy in _NO_GOAL_STRATEGIES:
        # These strategies don't need a language goal to start.
        state.goal_ready.set()

    # Open rover connection on the main thread so stop() is guaranteed to
    # run on shutdown — daemon threads are killed hard and cannot clean up.
    rover_ctrl = _build_rover_ctrl(args.rover, rover_port, args.dry_run)
    rover_ctx  = None
    if rover_ctrl:
        rover_ctx  = rover_ctrl.connect()
        rover_ctrl = rover_ctx.__enter__()
        log.info("%s controller active on %s%s",
                 args.rover.capitalize(), rover_port,
                 " (dry-run)" if args.dry_run else "")
        # Wire recorder so every drive_raw / stop is captured in events.jsonl
        if hasattr(rover_ctrl, "set_recorder"):
            rover_ctrl.set_recorder(state.recorder)

    # SIGTERM handler — ensures the finally block runs on `kill <pid>`
    def _on_sigterm(signum, frame):
        sys.exit(0)
    signal.signal(signal.SIGTERM, _on_sigterm)

    # WebSocket control server — direct joystick channel for browser / Android
    if args.control_port:
        from control_server import ControlServer
        ControlServer(state, rover_ctrl, port=args.control_port).start()
        log.info("WS control     : ws://0.0.0.0:%d", args.control_port)

    # Agent loop + publisher start immediately so all camera feeds appear in
    # the browser before the user confirms navigation.  goal_ready gates motor
    # commands — these threads are safe to run before Enter is pressed.
    if args.device is not None:
        log.info("Camera device : %s", args.device)
        threading.Thread(
            target=agent_loop,
            args=(state, strategy, args.device, args.interval, rover_ctrl),
            kwargs={"cam_controls": _build_cam_controls(args)},
            daemon=True,
        ).start()
    else:
        log.info("Camera device : none (headless mode)")
        threading.Thread(
            target=_headless_loop,
            args=(state, strategy, args.interval, rover_ctrl),
            daemon=True,
        ).start()

    if hasattr(strategy, "update_down_frame") and args.down_device is not None:
        log.info("Down-camera    : device %s", args.down_device)
        threading.Thread(
            target=_down_camera_loop,
            args=(strategy, args.down_device, state),
            daemon=True,
        ).start()

    publisher = AgentPublisher(args.web_server)
    threading.Thread(
        target=publisher.run,
        args=(state, rover_ctrl, strategy),
        daemon=True,
    ).start()
    log.info("Agent running — publishing to %s", args.web_server)

    # ── Wait for all cameras, then ask user to confirm before navigating ──────
    if hasattr(strategy, "cameras_ready") and args.goal:
        log.info("Waiting for wheel cameras to initialise (up to 60 s)…")
        deadline = time.time() + 60
        while time.time() < deadline:
            front_ok, left_ok, right_ok = strategy.cameras_ready()
            status = (f"  front={'OK' if front_ok else 'WAIT'}  "
                      f"left={'OK' if left_ok else 'WAIT'}  "
                      f"right={'OK' if right_ok else 'WAIT'}")
            if left_ok and right_ok:
                log.info("All wheel cameras ready: %s", status)
                break
            log.info("Camera status: %s", status)
            time.sleep(1.0)
        else:
            front_ok, left_ok, right_ok = strategy.cameras_ready()
            log.warning("Camera init timeout — left=%s right=%s — continuing anyway",
                        "OK" if left_ok else "MISSING",
                        "OK" if right_ok else "MISSING")

        print("\n" + "=" * 60)
        front_ok, left_ok, right_ok = strategy.cameras_ready()
        print(f"  Front camera : {'✓ ready' if front_ok else '✗ missing'}")
        print(f"  Left  wheel  : {'✓ ready' if left_ok  else '✗ missing'}")
        print(f"  Right wheel  : {'✓ ready' if right_ok else '✗ missing'}")
        print(f"  Goal         : {args.goal}")
        print("=" * 60)
        try:
            import termios
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            pass
        try:
            input("  Press Enter to start navigation (Ctrl-C to abort)… ")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return
        print("Starting navigation.\n")
        state.goal_ready.set()
    elif args.goal and args.strategy not in _NO_GOAL_STRATEGIES:
        state.goal_ready.set()

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        log.info("Shutting down — stopping rover")
        if rover_ctx:
            rover_ctx.__exit__(None, None, None)  # calls rover_ctrl.stop()
        if state.recorder:
            state.recorder.close()


if __name__ == "__main__":
    main()
