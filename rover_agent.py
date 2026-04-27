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
Supported strategies: gemini (Gemini vision API), omnivla (local neural network)

Usage:
    # Start the web server once (separate terminal):
    python web_server.py

    # Camera only, no hardware
    python rover_agent.py --dry-run

    # Roomba + Gemini (default strategy)
    python rover_agent.py --roomba-port /dev/ttyUSB0

    # Atlas-1 + Gemini
    python rover_agent.py --rover atlas --atlas-port /dev/ttyACM0

    # Atlas-1 + OmniVLA
    python rover_agent.py --rover atlas --atlas-port /dev/ttyACM0 \\
        --strategy omnivla --goal "Follow the brown path" --interval 1.0
"""

import argparse
import logging
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

import gemini_client
import prompts
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
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        log.error("Could not open camera at device %d", device)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
    log.info("Camera opened: %dx%d (max)",
             int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    captures_dir = Path("captures")
    captures_dir.mkdir(exist_ok=True)
    log.info("Saving LLM frames to: %s", captures_dir.resolve())

    last_query_time    = 0.0
    _logged_in_flight  = False

    while True:
        ret, frame = cap.read()
        if not ret:
            log.error("Failed to grab frame")
            break

        # Always push raw frame — never blocked by queries
        with state.raw_lock:
            state.raw_frame = frame.copy()

        # Record raw + latest annotated frame at camera rate
        if state.recorder:
            with state.llm_lock:
                llm = state.llm_frame.copy() if state.llm_frame is not None else None
            state.recorder.write_frames(frame, llm)

        now = time.time()

        # Fire strategy query in a separate thread so camera loop never blocks.
        # Gate on goal_ready so no queries fire until a goal has been received.
        if (state.goal_ready.is_set()
                and now - last_query_time >= interval
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
                "Down-camera device %d not available — available devices: %s  "
                "(use --down-device to select the correct one)",
                device, available or "none found",
            )
            time.sleep(3.0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
    log.info("Down-camera opened: device %d  %dx%d (max)", device,
             int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    consecutive_failures = 0
    while True:
        ret, frame = cap.read()
        if ret:
            strategy.update_down_frame(frame)
            if state is not None and state.recorder:
                # Prefer annotated frame (YOLO + gap overlay, updated at inference rate).
                # Falls back to raw until the first inference completes.
                ann = (strategy.get_down_annotated_frame()
                       if hasattr(strategy, "get_down_annotated_frame") else None)
                state.recorder.write_down_frame(ann if ann is not None else frame)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= 30:
                log.warning("Down-camera device %d: 30 consecutive read failures — reopening", device)
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
    if name == "gemini":
        from gemini_strategy import GeminiStrategy
        return GeminiStrategy()
    if name == "omnivla":
        from omnivla_strategy import OmniVLAStrategy
        return OmniVLAStrategy(goal=args.goal, goal_image_path=args.goal_image,
                               server_addr=args.omnivla_server)
    if name == "clip_omnivla":
        from clip_omnivla_strategy import ClipOmniVLAStrategy
        return ClipOmniVLAStrategy(goal=args.goal, goal_image_path=args.goal_image,
                                   server_addr=args.omnivla_server,
                                   path_threshold=args.path_threshold,
                                   ollama_url=args.ollama_server,
                                   weights_path=args.omnivla_weights)
    if name == "qwen_omnivla":
        from qwen_omnivla_strategy import QwenOmniVLAStrategy
        return QwenOmniVLAStrategy(goal=args.goal, goal_image_path=args.goal_image,
                                   server_addr=args.omnivla_server,
                                   path_threshold=args.path_threshold,
                                   ollama_url=args.ollama_server)
    if name == "hough_crop_row":
        from hough_crop_row_strategy import HoughCropRowStrategy
        return HoughCropRowStrategy(goal=args.goal, goal_image_path=args.goal_image,
                                    server_addr=args.omnivla_server,
                                    path_threshold=args.path_threshold,
                                    ollama_url=args.ollama_server)
    if name == "row_centering_omnivla":
        from row_centering_omnivla_strategy import RowCenteringOmniVLAStrategy
        return RowCenteringOmniVLAStrategy(
            goal=args.goal,
            goal_image_path=args.goal_image,
            server_addr=args.omnivla_server,
            path_threshold=args.path_threshold,
            ollama_url=args.ollama_server,
            weights_path=args.omnivla_weights,
            centering_gain=args.centering_gain,
            centering_alpha=args.centering_alpha,
            yolo_model_path=args.yolo_model or f"{args.crop_type}.pt",
            yolo_class_ids=None,
            yolo_conf=args.yolo_conf,
        )
    if name == "crop_row":
        from crop_row_strategy import CropRowStrategy
        if args.lab_mode:
            log.info("crop_row: LAB MODE — yolov8n.pt, class 58 (potted plant)")
            return CropRowStrategy(
                crop_type="lab-plant",
                model_path="yolov8n.pt",
                class_ids=[58],
                fwd_vel=args.fwd_vel,
                kp=args.steering_kp,
                conf=args.yolo_conf,
            )
        return CropRowStrategy(
            crop_type=args.crop_type,
            model_path=args.yolo_model or None,
            fwd_vel=args.fwd_vel,
            kp=args.steering_kp,
            conf=args.yolo_conf,
        )
    raise ValueError(f"Unknown strategy: {name!r}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rover navigation agent")
    parser.add_argument("--device",      type=int,   default=0,
                        help="Camera device index")
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
    parser.add_argument("--strategy",    type=str,   default="gemini",
                        choices=["gemini", "omnivla", "clip_omnivla", "qwen_omnivla",
                                 "hough_crop_row", "crop_row", "row_centering_omnivla"],
                        help="Navigation strategy (default: gemini)")
    parser.add_argument("--omnivla-weights", type=str, default=None,
                        metavar="PATH",
                        help="Path to custom OmniVLA-edge weights (.pth). "
                             "Defaults to downloading from HuggingFace if not set.")
    parser.add_argument("--crop-type",   type=str,   default="plant",
                        metavar="NAME",
                        help="Crop type for crop_row strategy, e.g. chilli, tomato, corn "
                             "(default: plant). Also used as default model filename: <crop>.pt")
    parser.add_argument("--yolo-model",  type=str,   default="",
                        metavar="PATH",
                        help="YOLOv8/v11 weights file for crop_row strategy. "
                             "Defaults to <crop-type>.pt if not set.")
    parser.add_argument("--yolo-conf",   type=float, default=0.35,
                        metavar="FLOAT",
                        help="YOLO detection confidence threshold (default: 0.35)")
    parser.add_argument("--fwd-vel",     type=int,   default=80,
                        metavar="MM_S",
                        help="Forward velocity mm/s for crop_row (default: 80)")
    parser.add_argument("--steering-kp", type=float, default=0.003,
                        metavar="FLOAT",
                        help="Proportional steering gain for crop_row (default: 0.003)")
    parser.add_argument("--lab-mode",    action="store_true",
                        help="crop_row lab testing mode: use yolov8n.pt (COCO) and detect "
                             "'potted plant' (class 58) — works with plastic/fake plants "
                             "on the floor. Overrides --yolo-model and --crop-type.")
    parser.add_argument("--goal",        type=str,   default="",
                        help="Language goal for omnivla strategies. "
                             "If omitted, wait for goal via web chat UI.")
    parser.add_argument("--goal-image",  type=str,   default=None,
                        help="Path to a goal image for omnivla strategy (optional)")
    parser.add_argument("--omnivla-server", type=str, default=None,
                        metavar="HOST:PORT",
                        help="Address of a running omnivla_server.py "
                             "(e.g. localhost:5100)")
    parser.add_argument("--path-threshold", type=float, default=0.5,
                        metavar="FLOAT",
                        help="Path detection confidence threshold for "
                             "clip_omnivla / qwen_omnivla (default: 0.5)")
    parser.add_argument("--ollama-server",  type=str,
                        default="http://localhost:11434",
                        metavar="URL",
                        help="Ollama API URL for Qwen models "
                             "(default: http://localhost:11434)")
    parser.add_argument("--down-device",     type=int,   default=1,
                        metavar="INDEX",
                        help="Camera device index for downward-facing row-centering camera "
                             "(row_centering_omnivla strategy, default: 1)")
    parser.add_argument("--centering-gain",  type=float, default=0.001,
                        metavar="FLOAT",
                        help="Proportional gain (rad/s per pixel) for row-centering correction "
                             "(row_centering_omnivla strategy, default: 0.001)")
    parser.add_argument("--centering-alpha", type=float, default=0.4,
                        metavar="FLOAT",
                        help="Centering correction blend weight: 0=off, 1=full override "
                             "(row_centering_omnivla strategy, default: 0.4)")
    parser.add_argument("--control-port",  type=int, default=5002,
                        metavar="PORT",
                        help="WebSocket joystick control port for browser/Android "
                             "(default: 5002, 0 = disabled)")
    args = parser.parse_args()

    rover_port = args.atlas_port if args.rover == "atlas" else args.roomba_port

    log.info("=== Rover agent starting ===")
    log.info("Camera device : %d", args.device)
    log.info("Query interval: %.1fs", args.interval)
    log.info("Strategy      : %s", args.strategy)
    _omnivla_strategies = ("omnivla", "clip_omnivla", "qwen_omnivla",
                           "hough_crop_row", "row_centering_omnivla")
    if args.strategy in _omnivla_strategies:
        log.info("Goal          : %s", args.goal)
        if args.goal_image:
            log.info("Goal image    : %s", args.goal_image)
        if args.omnivla_server:
            log.info("OmniVLA server: %s", args.omnivla_server)
        else:
            log.info("OmniVLA server: (loading locally)")
        if args.strategy in ("clip_omnivla", "qwen_omnivla",
                              "hough_crop_row", "row_centering_omnivla"):
            log.info("Path threshold: %.2f", args.path_threshold)
            log.info("Ollama server : %s", args.ollama_server)
        if args.strategy == "row_centering_omnivla":
            log.info("Down device   : %d", args.down_device)
            log.info("Center gain   : %.4f  alpha: %.2f",
                     args.centering_gain, args.centering_alpha)
            log.info("YOLO model    : %s  conf=%.2f",
                     args.yolo_model or f"{args.crop_type}.pt", args.yolo_conf)
    else:
        log.info("Model         : %s", gemini_client.MODEL)
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
    if args.goal and args.strategy not in ("gemini",):
        strategy.set_goal(args.goal)
        with state.result_lock:
            state.goal = args.goal
        state.goal_ready.set()
        log.info("CLI goal applied: '%s'", args.goal)
    elif args.strategy == "gemini":
        # Gemini strategy manages its own goal; always ready to start.
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

    # Down-camera loop (row_centering_omnivla only)
    if hasattr(strategy, "update_down_frame"):
        log.info("Down-camera    : device %d", args.down_device)
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
