#!/usr/bin/env python3
"""
Rover navigation agent — thin orchestrator.

Opens the camera, wires up the chosen navigation strategy and web display,
then runs the agent loop in a background thread while Flask serves the UI
on the main thread.

Usage:
    python rover_agent.py
    python rover_agent.py --device 1 --interval 5 --port 5000
    python rover_agent.py --roomba-port /dev/ttyUSB0
    python rover_agent.py --strategy gemini   # default
"""

import argparse
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

import gemini_client
import prompts
import roomba_controller
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

def agent_loop(
    state: AgentState,
    strategy: NavigationStrategy,
    device: int,
    interval: float,
    roomba_port: str | None = None,
    dry_run: bool = False,
) -> None:
    """Camera capture loop — never blocked by LLM queries or Roomba motion."""

    # Optionally connect to the Roomba
    roomba_ctrl: roomba_controller.RoombaController | None = None
    roomba_ctx = None
    if roomba_port:
        roomba_ctrl = roomba_controller.RoombaController(port=roomba_port, dry_run=dry_run)
        roomba_ctx = roomba_ctrl.connect()
        roomba_ctx.__enter__()
        log.info("Roomba controller active on %s%s", roomba_port, " (dry-run)" if dry_run else "")

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        log.error("Could not open camera at device %d", device)
        if roomba_ctx:
            roomba_ctx.__exit__(None, None, None)
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
        if now - last_query_time >= interval:
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
                    args=(state, frame.copy(), captures_dir, roomba_ctrl),
                    daemon=True,
                ).start()

        time.sleep(0.033)   # ~30 fps

    cap.release()
    log.info("Camera released")

    if roomba_ctx:
        roomba_ctx.__exit__(None, None, None)


# ── Strategy factory ───────────────────────────────────────────────────────────

def _build_strategy(name: str, args) -> NavigationStrategy:
    if name == "gemini":
        from gemini_strategy import GeminiStrategy
        return GeminiStrategy()
    if name == "omnivla":
        from omnivla_strategy import OmniVLAStrategy
        return OmniVLAStrategy(goal=args.goal)
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
    parser.add_argument("--roomba-port", type=str,   default=None,
                        help="Roomba serial port (e.g. /dev/ttyUSB0); omit to disable")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Log Roomba commands but do not send them")
    parser.add_argument("--strategy",    type=str,   default="gemini",
                        choices=["gemini", "omnivla"],
                        help="Navigation strategy (default: gemini)")
    parser.add_argument("--goal",        type=str,   default="navigate forward",
                        help="Language goal for omnivla strategy (e.g. 'blue trash bin')")
    args = parser.parse_args()

    log.info("=== Rover agent starting ===")
    log.info("Camera device : %d", args.device)
    log.info("Query interval: %.1fs", args.interval)
    log.info("Strategy      : %s", args.strategy)
    if args.strategy == "omnivla":
        log.info("Goal          : %s", args.goal)
    else:
        log.info("Model         : %s", gemini_client.MODEL)
    log.info("Web UI        : http://localhost:%d", args.port)
    if args.roomba_port:
        log.info("Roomba port   : %s%s", args.roomba_port, " (dry-run)" if args.dry_run else "")
    else:
        log.info("Roomba        : disabled (pass --roomba-port to enable)")

    state    = AgentState()
    strategy = _build_strategy(args.strategy, args)

    threading.Thread(
        target=agent_loop,
        args=(state, strategy, args.device, args.interval,
              args.roomba_port, args.dry_run),
        daemon=True,
    ).start()

    display = WebDisplay(state, log_dir=Path("logs"))
    display.run(port=args.port)


if __name__ == "__main__":
    main()
