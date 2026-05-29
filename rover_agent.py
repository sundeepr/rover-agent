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
    ch.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))

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
    device: int,
    interval: float,
    rover_ctrl=None,
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
    """
    # On Jetson, each USB camera creates two /dev/videoN nodes (capture +
    # metadata).  The metadata node opens successfully but cap.read() always
    # returns False.  We try the V4L2 backend explicitly and use warmup reads
    # to let the driver stabilise before entering the main loop.
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        log.error("Could not open camera at device %s — trying default backend", device)
        cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        log.error("Could not open camera at device %s", device)
        return

    # Force MJPEG so the camera compresses on-chip — dramatically reduces USB
    # bandwidth vs the default YUYV (uncompressed), which matters when 3 cameras
    # share the same USB controller on Jetson.  FOURCC must be set first.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 10)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    log.info("Camera opened: %dx%d  fourcc=%s  fps=%.0f",
             int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
             fourcc_str, actual_fps)

    # Warmup — Jetson USB cameras often need several reads before the first
    # valid frame arrives.  Discard up to 30 frames silently.
    log.info("Camera warmup (device %s)…", device)
    for _w in range(30):
        ret, _ = cap.read()
        if ret:
            break
        time.sleep(0.05)
    else:
        log.warning(
            "Camera device %s: no frame after 30 warmup reads.\n"
            "  On Jetson, even-numbered /dev/videoN nodes are capture nodes;\n"
            "  odd-numbered ones are metadata-only and never produce frames.",
            device,
        )
        cap.release()
        return

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
        ret, frame = cap.read()
        if not ret:
            _consecutive_failures += 1
            if _consecutive_failures == 1:
                log.warning("Camera device %s: failed to grab frame (will retry)", device)
            if _consecutive_failures >= 30:
                log.error("Camera device %s: 30 consecutive failures — giving up", device)
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

    cap.release()
    log.info("Camera released")


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
            icr_offset_m        = _geo["icr_offset_mm"] / 1000.0,
            exg_threshold       = args.exg_threshold,
            exg_min_area        = args.exg_min_area,
            exg_density_pct     = args.exg_density_pct,
            camera_calibration  = load_camera_calibration(args.camera_calibration),
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
    parser.add_argument("--device",      type=_device, default=0,
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
                                 "boundary_guard", "teleop", "crop_guard"],
                        help="Navigation strategy (default: omnivla_full)")
    parser.add_argument("--left-cam",   type=_device, default=1,
                        metavar="INDEX|PATH",
                        help="Left wheel camera index or path (crop_guard, default 1)")
    parser.add_argument("--right-cam",  type=_device, default=2,
                        metavar="INDEX|PATH",
                        help="Right wheel camera index or path (crop_guard, default 2)")
    parser.add_argument("--exg-threshold", type=int, default=60,
                        metavar="N",
                        help="ExG vegetation threshold for wheel cameras (default 60)")
    parser.add_argument("--exg-min-area",  type=int, default=500,
                        metavar="PX",
                        help="Min vegetation blob area in pixels (default 500)")
    parser.add_argument("--exg-density-pct", type=float, default=8.0,
                        metavar="PCT",
                        help="Min %% of wheel-zone pixels above ExG threshold to "
                             "declare trampling; filters out sparse soil noise "
                             "(default 8.0)")
    parser.add_argument("--cloud-server", type=str,  default="ws://localhost:8765",
                        metavar="URL",
                        help="WebSocket URL of omnivla_cloud_server.py "
                             "(cloud_omnivla strategy, default: ws://localhost:8765)")
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
    state               = AgentState()
    state.recorder      = SessionRecorder()
    state.query_interval = args.interval
    strategy             = _build_strategy(args.strategy, args)

    # If a goal was given on the CLI, apply it immediately so the agent
    # starts navigating without waiting for web chat input.
    _NO_GOAL_STRATEGIES = ("line_follow", "plant_center", "boundary_guard", "teleop")
    if args.goal and args.strategy not in _NO_GOAL_STRATEGIES:
        strategy.set_goal(args.goal)
        with state.result_lock:
            state.goal = args.goal
        state.goal_ready.set()
        log.info("CLI goal applied: '%s'", args.goal)
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

    # SIGTERM handler — ensures the finally block runs on `kill <pid>`
    def _on_sigterm(signum, frame):
        sys.exit(0)
    signal.signal(signal.SIGTERM, _on_sigterm)

    # WebSocket control server — direct joystick channel for browser / Android
    if args.control_port:
        from control_server import ControlServer
        ControlServer(state, rover_ctrl, port=args.control_port).start()
        log.info("WS control     : ws://0.0.0.0:%d", args.control_port)

    # Agent loop (front camera + inference dispatch)
    threading.Thread(
        target=agent_loop,
        args=(state, strategy, args.device, args.interval, rover_ctrl),
        daemon=True,
    ).start()

    # Down-camera loop — only when strategy supports it AND device was specified
    if hasattr(strategy, "update_down_frame") and args.down_device is not None:
        log.info("Down-camera    : device %s", args.down_device)
        threading.Thread(
            target=_down_camera_loop,
            args=(strategy, args.down_device, state),
            daemon=True,
        ).start()

    # Publisher loop (reads AgentState, POSTs to web server)
    publisher = AgentPublisher(args.web_server)
    threading.Thread(
        target=publisher.run,
        args=(state, rover_ctrl, strategy),
        daemon=True,
    ).start()

    log.info("Agent running — publishing to %s", args.web_server)

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
