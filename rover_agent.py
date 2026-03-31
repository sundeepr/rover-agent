#!/usr/bin/env python3
"""
Rover navigation agent — thin orchestrator.

Opens the camera, wires up the chosen navigation strategy and rover controller,
then runs the agent loop in a background thread while Flask serves the UI
on the main thread.

Supported rovers    : roomba (iRobot OI), atlas (STM32 $CMD protocol)
Supported strategies: gemini (Gemini vision API), omnivla (local neural network)

Usage:
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
from web_display import WebDisplay


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
    state.raw_frame (for the live web stream). Every `interval` seconds,
    if no query is already in-flight, increments state.step and spawns a
    new daemon thread to call strategy.run_query(). This keeps the camera
    loop completely non-blocking regardless of how long inference takes.

    rover_ctrl is an already-connected controller (or None). Connection
    lifecycle is managed by main() so the stop command is guaranteed to
    run on shutdown even when the program is killed.
    """
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        log.error("Could not open camera at device %d", device)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, prompts.IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, prompts.IMAGE_HEIGHT)
    log.info("Camera opened: %dx%d",
             int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    captures_dir = Path("captures")
    captures_dir.mkdir(exist_ok=True)
    log.info("Saving LLM frames to: %s", captures_dir.resolve())

    last_query_time = 0.0
    _logged_in_flight = False

    while True:
        ret, frame = cap.read()
        if not ret:
            log.error("Failed to grab frame")
            break

        # Always push raw frame to realtime stream — never blocked by queries
        with state.raw_lock:
            state.raw_frame = frame.copy()

        now = time.time()

        # Fire strategy query in a separate thread so camera loop never blocks
        if now - last_query_time >= interval and not state.paused.is_set():
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


# ── Strategy factory ───────────────────────────────────────────────────────────

def _build_strategy(name: str, args) -> NavigationStrategy:
    """
    Instantiate and return the requested NavigationStrategy.

    Strategies are imported lazily so their heavy dependencies (torch, etc.)
    are only loaded when actually needed. To add a new strategy, import and
    return it here, and add its name to the --strategy choices in main().
    """
    if name == "gemini":
        from gemini_strategy import GeminiStrategy
        return GeminiStrategy()
    if name == "omnivla":
        from omnivla_strategy import OmniVLAStrategy
        return OmniVLAStrategy(goal=args.goal, goal_image_path=args.goal_image,
                               server_addr=args.omnivla_server)
    raise ValueError(f"Unknown strategy: {name!r}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rover navigation agent")
    parser.add_argument("--device",      type=int,   default=0,
                        help="Camera device index")
    parser.add_argument("--interval",    type=float, default=3.0,
                        help="Seconds between LLM queries")
    parser.add_argument("--port",        type=int,   default=5000,
                        help="Web server port")
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
                        choices=["gemini", "omnivla"],
                        help="Navigation strategy (default: gemini)")
    parser.add_argument("--goal",        type=str,   default="navigate forward",
                        help="Language goal for omnivla strategy (e.g. 'blue trash bin')")
    parser.add_argument("--goal-image",  type=str,   default=None,
                        help="Path to a goal image for omnivla strategy (optional)")
    parser.add_argument("--omnivla-server", type=str, default=None,
                        metavar="HOST:PORT",
                        help="Address of a running omnivla_server.py "
                             "(e.g. localhost:5100). When set, inference is "
                             "delegated to the server instead of loading "
                             "the model locally.")
    args = parser.parse_args()

    # Resolve rover port: explicit flag takes priority, else auto-detect from rover type
    rover_port = args.atlas_port if args.rover == "atlas" else args.roomba_port

    log.info("=== Rover agent starting ===")
    log.info("Camera device : %d", args.device)
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
    else:
        log.info("Model         : %s", gemini_client.MODEL)
    log.info("Web UI        : http://localhost:%d", args.port)
    log.info("Rover         : %s", args.rover)
    if rover_port:
        log.info("Rover port    : %s%s", rover_port, " (dry-run)" if args.dry_run else "")
    else:
        log.info("Rover         : disabled (pass --roomba-port or --atlas-port to enable)")

    state    = AgentState()
    strategy = _build_strategy(args.strategy, args)

    # Open rover connection on the main thread so stop() is guaranteed to
    # run on shutdown — daemon threads are killed hard and cannot clean up.
    rover_ctrl = _build_rover_ctrl(args.rover, rover_port, args.dry_run)
    rover_ctx  = None
    if rover_ctrl:
        rover_ctx = rover_ctrl.connect()
        rover_ctrl = rover_ctx.__enter__()
        log.info("%s controller active on %s%s",
                 args.rover.capitalize(), rover_port,
                 " (dry-run)" if args.dry_run else "")

    # SIGTERM handler (e.g. `kill <pid>`) — Flask catches SIGINT itself,
    # but SIGTERM would otherwise skip the finally block below.
    def _on_sigterm(signum, frame):
        sys.exit(0)
    signal.signal(signal.SIGTERM, _on_sigterm)

    threading.Thread(
        target=agent_loop,
        args=(state, strategy, args.device, args.interval, rover_ctrl),
        daemon=True,
    ).start()

    display = WebDisplay(state, log_dir=Path("logs"), rover_ctrl=rover_ctrl)
    try:
        display.run(port=args.port)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        log.info("Shutting down — stopping rover")
        if rover_ctx:
            rover_ctx.__exit__(None, None, None)  # calls rover_ctrl.stop()


if __name__ == "__main__":
    main()
