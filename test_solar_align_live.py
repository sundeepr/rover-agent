#!/usr/bin/env python3
"""
test_solar_align_live.py — drive the rover so the LEFTMOST detected solar
panel is centered in the camera AND the rover is perpendicular to its face.

Standalone rover-driving tuning script (see /Users/sailaja/.claude/plans/
zesty-discovering-crayon.md) — proves the alignment control loop out live
before it becomes a phase inside a real NavigationStrategy.

Why this needs more than "turn to center it"
──────────────────────────────────────────────
A single bounding box only gives bearing (which way to turn) — nothing
about the panel's face orientation. This script gets an orientation signal
by arcing laterally in small steps and re-detecting: a fronto-planar
panel's apparent WIDTH is maximized when viewed face-on and foreshortens
off-axis, while its apparent HEIGHT stays roughly constant across
yaw-only viewpoint changes. So width/height (aspect ratio), not raw
width, is the hill-climb signal — it cancels out the confound of the arc
also changing distance to the panel slightly.

State machine
─────────────
  SEARCH  Spin in place in small increments until >=1 "solar panel" is
          detected. Picks the leftmost box (smallest x_min) as the target.
  CENTER  Blocking iterative turn until the target's bearing is within
          --center-tolerance-deg. Reused after SEARCH and after every
          PROBE arc step (arcing changes heading too).
  PROBE   Coordinate-descent (Hooke-Jeeves style) hill-climb on aspect
          ratio: try a step left, undo, try a step right, undo, commit to
          whichever improved (re-centering after every move), or halve the
          probe duration if neither did. Terminates when the duration
          shrinks below --probe-min-duration-s (converged) or
          --max-probe-iterations is hit. Probe arcs are driven by
          (duration, %power) — the Atlas has no wheel encoders, so
          distance/mm is not a real controllable quantity here.
  DONE    Stop and hold. Ctrl-C / any exception also stops the rover.

Usage
─────
    # Dry run first — zero hardware risk, confirms the state machine logs
    # sensible bearings/aspect ratios against real detections:
    python test_solar_align_live.py \\
        --moondream-server ws://<gpu-box>:8767 \\
        --device "rtsp://10.0.1.103:554/video/live?channel=1&subtype=1" \\
        --rover atlas --atlas-port /dev/ttyACM0 --dry-run \\
        --web-server http://localhost:5001

    # Real run, once the math checks out and gains are tuned:
    python test_solar_align_live.py \\
        --moondream-server ws://<gpu-box>:8767 \\
        --device "rtsp://10.0.1.103:554/video/live?channel=1&subtype=1" \\
        --rover atlas --atlas-port /dev/ttyACM0 \\
        --web-server http://localhost:5001

Then open http://<web-ui-box>:5001/ in a browser for the live HUD.
"""

import argparse
import logging
import threading
import time
from enum import Enum, auto

import cv2

import atlas_controller
import roomba_controller
from agent_publisher import AgentPublisher
from frame_source import open_frame_source
from moondream_client import MoondreamClient
from navigation_strategy import AgentState

log = logging.getLogger("test_solar_align_live")


def _build_rover_ctrl(rover: str, port: str, dry_run: bool):
    """
    Same dispatch as rover_agent._build_rover_ctrl — inlined rather than
    imported because importing rover_agent.py runs setup_logging() at
    module scope, which creates a logs/ directory and a timestamped log
    file as a side effect of the import alone. Not worth pulling in for
    an 8-line dispatch.
    """
    if rover == "roomba":
        return roomba_controller.RoombaController(port=port, dry_run=dry_run)
    if rover == "atlas":
        return atlas_controller.AtlasController(port=port, dry_run=dry_run)
    raise ValueError(f"Unknown rover: {rover!r}")

# Radius-sign convention, matching plant_center_strategy.py:
#   positive radius = steer left, negative radius = steer right.
# Used for both spin sentinels (radius=1/-1) and real arc radii.
_LEFT  = 1
_RIGHT = -1

# The Atlas board has no wheel encoders — it only accepts %power per wheel
# (see atlas_controller.py's $CMD,L=,R=,AUX=# protocol). drive_raw() still
# takes a nominal velocity_mm_s + radius_mm (to share one interface with
# RoombaController, which DOES take real mm/s), but internally that
# velocity is only ever used as velocity_mm_s / _MAX_VELOCITY_REF_MM_S to
# produce a %power — there is no distance feedback of any kind. So this
# script's probe motion is parameterized directly as (duration_s,
# power_pct) instead of a fictional "step_mm" — power_pct is converted to
# the vel_mm_s drive_raw() expects using Atlas's own reference constant,
# so the number you pass is close to the real wheel %power.
_ATLAS_POWER_REF_MM_S = atlas_controller._MAX_VELOCITY_REF_MM_S


def _power_pct_to_vel_mm_s(power_pct: float) -> float:
    return power_pct / 100.0 * _ATLAS_POWER_REF_MM_S

_BOX_COLOR     = (0, 255, 255)   # bright yellow (BGR)
_BOX_THICKNESS = 4
_CORNER_LEN    = 24


def _device(value: str):
    """Accept an int index, a /dev path, or a ws://|rtsp:// URL (same as rover_agent.py)."""
    try:
        return int(value)
    except ValueError:
        return value


class _Phase(Enum):
    SEARCH = auto()
    CENTER = auto()
    PROBE  = auto()
    DONE   = auto()
    FAILED = auto()


# ── Geometry helpers ────────────────────────────────────────────────────────

def _box_center_x(box: dict) -> float:
    return (box["x_min"] + box["x_max"]) / 2.0


def _aspect_ratio(box: dict) -> float:
    w = box["x_max"] - box["x_min"]
    h = box["y_max"] - box["y_min"]
    return w / h if h > 1e-6 else 0.0


def _bearing_deg(box_cx: float, img_w: int, hfov_deg: float) -> float:
    """Positive = target is to the right (turn right); negative = turn left."""
    offset   = box_cx - img_w / 2.0
    fraction = offset / img_w
    return fraction * hfov_deg


def _leftmost_box(objects: "list[dict]") -> "dict | None":
    if not objects:
        return None
    return min(objects, key=lambda o: o["x_min"])


# ── Motion primitives (blocking, mirrors row_change_strategy's shape) ───────

def _spin(rover_ctrl, direction: int, duration_s: float, settle_s: float) -> None:
    """direction: _LEFT or _RIGHT. Spin sentinel: radius=1 -> right, radius=-1 -> left."""
    radius = 1 if direction == _RIGHT else -1
    if rover_ctrl:
        rover_ctrl.drive_raw(1, radius)
    time.sleep(duration_s)
    if rover_ctrl:
        rover_ctrl.stop()
    time.sleep(settle_s)


def _arc(rover_ctrl, direction: int, duration_s: float, radius_mm: float,
        power_pct: float, settle_s: float) -> None:
    """
    direction: _LEFT (+radius) or _RIGHT (-radius). Drives for duration_s —
    the actual controllable quantity on a board with no distance feedback —
    at approximately power_pct on the faster (outer) wheel. radius_mm only
    controls the L/R wheel-speed RATIO (how tight the curve is), which is
    real geometry independent of any distance/speed calibration.
    """
    signed_radius = radius_mm if direction == _LEFT else -radius_mm
    vel_mm_s = _power_pct_to_vel_mm_s(power_pct)
    if rover_ctrl:
        rover_ctrl.drive_raw(int(vel_mm_s), int(signed_radius))
    time.sleep(duration_s)
    if rover_ctrl:
        rover_ctrl.stop()
    time.sleep(settle_s)


# ── Detection helper ─────────────────────────────────────────────────────────

def _measure(state: AgentState, client: MoondreamClient, object_name: str,
            timeout: float):
    """Grab the latest frame, detect(), return (leftmost_box_or_None, frame_or_None)."""
    with state.raw_lock:
        frame = state.raw_frame.copy() if state.raw_frame is not None else None
    if frame is None:
        return None, None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None, frame
    objects = client.detect(buf.tobytes(), object_name, timeout=timeout)
    return _leftmost_box(objects or []), frame


# ── HUD annotation ────────────────────────────────────────────────────────────

def _annotate(frame, box, phase: _Phase, bearing_deg: float,
              aspect_ratio: float, best_ar: float, probe_duration_s: float) -> "any":
    out = frame.copy()
    h, w = out.shape[:2]

    if box is not None:
        x1 = max(2, min(int(box["x_min"]), w - 3))
        y1 = max(2, min(int(box["y_min"]), h - 3))
        x2 = max(x1 + 1, min(int(box["x_max"]), w - 2))
        y2 = max(y1 + 1, min(int(box["y_max"]), h - 2))
        cv2.rectangle(out, (x1, y1), (x2, y2), _BOX_COLOR, _BOX_THICKNESS)
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                               (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(out, (cx, cy), (cx + dx * _CORNER_LEN, cy), (0, 0, 0), _BOX_THICKNESS + 2)
            cv2.line(out, (cx, cy), (cx, cy + dy * _CORNER_LEN), (0, 0, 0), _BOX_THICKNESS + 2)
            cv2.line(out, (cx, cy), (cx + dx * _CORNER_LEN, cy), _BOX_COLOR, _BOX_THICKNESS)
            cv2.line(out, (cx, cy), (cx, cy + dy * _CORNER_LEN), _BOX_COLOR, _BOX_THICKNESS)

    # Image-center reference line
    cx = w // 2
    cv2.line(out, (cx, 0), (cx, h), (255, 255, 255), 1)

    lines = [
        f"phase={phase.name}",
        f"bearing={bearing_deg:+.1f}deg" if box is not None else "bearing=--",
        f"AR={aspect_ratio:.3f}  best={best_ar:.3f}" if box is not None else "AR=--",
        f"probe_dur={probe_duration_s:.2f}s",
    ]
    cv2.rectangle(out, (0, 0), (w, 22 * len(lines) + 10), (30, 30, 30), -1)
    for i, text in enumerate(lines):
        cv2.putText(out, text, (8, 20 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, _BOX_COLOR, 2, cv2.LINE_AA)
    return out


def _publish(state: AgentState, frame, box, phase: _Phase, bearing_deg: float,
            aspect_ratio: float, best_ar: float, probe_duration_s: float) -> None:
    annotated = _annotate(frame, box, phase, bearing_deg, aspect_ratio, best_ar, probe_duration_s)
    with state.llm_lock:
        state.llm_frame = annotated
    with state.result_lock:
        state.latest_result = {
            "strategy":     "solar_align_smoketest",
            "step":         state.step,
            "phase":        phase.name,
            "bearing_deg":  bearing_deg,
            "aspect_ratio": aspect_ratio,
            "best_ar":      best_ar,
            "probe_duration_s": probe_duration_s,
            "found":        box is not None,
        }
        state.step += 1


# ── Camera thread ─────────────────────────────────────────────────────────────

def _camera_thread(state: AgentState, device, running: threading.Event) -> None:
    src = open_frame_source(device, "solar-align-test")
    if src is None:
        log.error("Cannot open camera/stream %s", device)
        running.clear()
        return
    log.info("Camera/stream %s opened", device)
    while running.is_set():
        frame = src.read()
        if frame is None:
            time.sleep(0.1)
            continue
        with state.raw_lock:
            state.raw_frame = frame
        time.sleep(0.03)
    src.release()


# ── Alignment state machine ───────────────────────────────────────────────────

def _align_loop(state: AgentState, client: MoondreamClient, rover_ctrl,
                args, running: threading.Event) -> None:
    phase = _Phase.SEARCH
    bearing = 0.0
    ar = best_ar = 0.0

    # ── SEARCH ────────────────────────────────────────────────────────────
    log.info("[SEARCH] spinning to find '%s' …", args.object)
    box = frame = None
    for i in range(args.max_search_spins):
        if not running.is_set():
            return
        box, frame = _measure(state, client, args.object, timeout=30.0)
        if frame is not None:
            _publish(state, frame, box, phase, bearing, ar, best_ar, args.probe_duration_s)
        if box is not None:
            log.info("[SEARCH] found leftmost box after %d spin(s): %s", i, box)
            break
        _spin(rover_ctrl, _RIGHT, args.search_spin_s, args.settle_s)
    else:
        phase = _Phase.FAILED
        log.error("[SEARCH] no '%s' found after %d spins — giving up",
                 args.object, args.max_search_spins)
        if frame is not None:
            _publish(state, frame, None, phase, 0.0, 0.0, 0.0, args.probe_duration_s)
        return

    def _center(label: str) -> "dict | None":
        """Blocking iterative turn until bearing <= tolerance. Returns the final box."""
        nonlocal phase, bearing
        phase = _Phase.CENTER
        last_box = box
        for i in range(args.max_center_attempts):
            if not running.is_set():
                return last_box
            b, f = _measure(state, client, args.object, timeout=30.0)
            if b is None:
                log.warning("[CENTER/%s] lost target — holding", label)
                return last_box
            last_box = b
            img_w = f.shape[1]
            bearing = _bearing_deg(_box_center_x(b), img_w, args.camera_hfov_deg)
            _publish(state, f, b, phase, bearing, ar, best_ar, args.probe_duration_s)
            log.info("[CENTER/%s] attempt %d/%d  bearing=%+.1fdeg",
                     label, i + 1, args.max_center_attempts, bearing)
            if abs(bearing) <= args.center_tolerance_deg:
                return last_box
            turn_duration = abs(bearing) / args.turn_deg_per_sec
            direction = _RIGHT if bearing > 0 else _LEFT
            _spin(rover_ctrl, direction, turn_duration, args.settle_s)
        log.warning("[CENTER/%s] did not converge after %d attempts",
                   label, args.max_center_attempts)
        return last_box

    box = _center("initial")
    if box is None:
        phase = _Phase.FAILED
        log.error("[CENTER] lost target before PROBE could start")
        return

    b, f = _measure(state, client, args.object, timeout=30.0)
    if b is None or f is None:
        phase = _Phase.FAILED
        log.error("[PROBE] target lost right before starting hill-climb")
        return
    best_ar = ar = _aspect_ratio(b)
    log.info("[PROBE] baseline aspect ratio = %.3f", best_ar)

    # ── PROBE ─────────────────────────────────────────────────────────────
    phase = _Phase.PROBE
    step = args.probe_duration_s   # "step" = how long each probe arc runs, in seconds
    iterations = 0
    while (step >= args.probe_min_duration_s
          and iterations < args.max_probe_iterations
          and running.is_set()):
        iterations += 1
        log.info("[PROBE] iteration %d  duration=%.2fs  power=%.0f%%  best_AR=%.3f",
                 iterations, step, args.probe_power_pct, best_ar)

        _arc(rover_ctrl, _LEFT, step, args.probe_radius_mm, args.probe_power_pct, args.settle_s)
        _center("probe-left")
        b, f = _measure(state, client, args.object, timeout=30.0)
        ar_left = _aspect_ratio(b) if b is not None else -1.0
        if f is not None:
            _publish(state, f, b, phase, bearing, ar_left, best_ar, step)
        log.info("[PROBE] left  AR=%.3f", ar_left)

        _arc(rover_ctrl, _RIGHT, step, args.probe_radius_mm, args.probe_power_pct, args.settle_s)
        _center("undo-left")

        _arc(rover_ctrl, _RIGHT, step, args.probe_radius_mm, args.probe_power_pct, args.settle_s)
        _center("probe-right")
        b, f = _measure(state, client, args.object, timeout=30.0)
        ar_right = _aspect_ratio(b) if b is not None else -1.0
        if f is not None:
            _publish(state, f, b, phase, bearing, ar_right, best_ar, step)
        log.info("[PROBE] right AR=%.3f", ar_right)

        _arc(rover_ctrl, _LEFT, step, args.probe_radius_mm, args.probe_power_pct, args.settle_s)
        _center("undo-right")

        if (ar_left > best_ar + args.aspect_improve_threshold
              and ar_left >= ar_right):
            log.info("[PROBE] committing LEFT (AR %.3f > best %.3f)", ar_left, best_ar)
            _arc(rover_ctrl, _LEFT, step, args.probe_radius_mm, args.probe_power_pct, args.settle_s)
            _center("commit-left")
            best_ar = ar_left
        elif ar_right > best_ar + args.aspect_improve_threshold:
            log.info("[PROBE] committing RIGHT (AR %.3f > best %.3f)", ar_right, best_ar)
            _arc(rover_ctrl, _RIGHT, step, args.probe_radius_mm, args.probe_power_pct, args.settle_s)
            _center("commit-right")
            best_ar = ar_right
        else:
            step /= 2.0
            log.info("[PROBE] neither side improved — halving duration to %.2fs", step)

    # ── DONE ──────────────────────────────────────────────────────────────
    phase = _Phase.DONE
    if rover_ctrl:
        rover_ctrl.stop()
    b, f = _measure(state, client, args.object, timeout=30.0)
    final_bearing = (_bearing_deg(_box_center_x(b), f.shape[1], args.camera_hfov_deg)
                     if (b is not None and f is not None) else bearing)
    final_ar = _aspect_ratio(b) if b is not None else best_ar
    if f is not None:
        _publish(state, f, b, phase, final_bearing, final_ar, best_ar, step)
    log.info("[DONE] aligned — bearing=%+.1fdeg  aspect_ratio=%.3f  "
            "iterations=%d  final_duration=%.2fs",
            final_bearing, final_ar, iterations, step)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--moondream-server", type=str, required=True, metavar="URL",
                        help="WebSocket URL of moondream_cloud_server.py")
    parser.add_argument("--device", type=_device, default=0, metavar="INDEX|PATH|RTSP_URL",
                        help="Camera device index, /dev path, or rtsp:// URL (default 0)")
    parser.add_argument("--object", type=str, default="solar panel",
                        help="Object name to ground/detect (default 'solar panel')")
    parser.add_argument("--web-server", type=str, default="http://localhost:5001", metavar="URL",
                        help="URL of the running web_server.py")

    parser.add_argument("--rover", type=str, required=True, choices=["roomba", "atlas"],
                        help="Rover controller type")
    parser.add_argument("--atlas-port", type=str, default=None,
                        help="Serial port for Atlas controller")
    parser.add_argument("--roomba-port", type=str, default=None,
                        help="Serial port for Roomba controller")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log rover commands but do not send them — use this first")

    parser.add_argument("--camera-hfov-deg", type=float, default=138.0, metavar="DEG",
                        help="Camera horizontal field of view in degrees (default 138, "
                             "matches atlas_controller's default — recalibrate per camera)")
    parser.add_argument("--turn-deg-per-sec", type=float, default=20.0, metavar="DEG/S",
                        help="Empirical in-place spin rate — NEEDS CALIBRATION per rover "
                             "(default 20.0)")
    parser.add_argument("--center-tolerance-deg", type=float, default=3.0, metavar="DEG",
                        help="Bearing considered 'centered' (default 3.0)")
    parser.add_argument("--max-center-attempts", type=int, default=6, metavar="N")
    parser.add_argument("--max-search-spins", type=int, default=24, metavar="N")
    parser.add_argument("--search-spin-s", type=float, default=0.3, metavar="SECS",
                        help="Spin duration per SEARCH increment (default 0.3s)")

    parser.add_argument("--probe-duration-s", type=float, default=2.0, metavar="SECS",
                        help="Initial lateral probe arc duration — the Atlas has no "
                             "distance feedback, so time is the real controllable "
                             "quantity, not distance. NEEDS TUNING (default 2.0)")
    parser.add_argument("--probe-min-duration-s", type=float, default=0.3, metavar="SECS",
                        help="Stop refining once the probe duration shrinks below this "
                             "(default 0.3)")
    parser.add_argument("--probe-radius-mm", type=float, default=800.0, metavar="MM",
                        help="Arc radius used for probe steps — a real geometric "
                             "quantity (controls the L/R wheel-speed ratio), independent "
                             "of any speed/distance calibration. NEEDS TUNING (default 800)")
    parser.add_argument("--probe-power-pct", type=float, default=15.0, metavar="PCT",
                        help="Approx. %% power on the faster wheel during probe arcs "
                             "(default 15.0 — conservative; motor deadband is ~8%%, so "
                             "below that the rover won't move at all)")
    parser.add_argument("--aspect-improve-threshold", type=float, default=0.03, metavar="RATIO",
                        help="Minimum aspect-ratio delta to count as real improvement, "
                             "not detection jitter (default 0.03)")
    parser.add_argument("--max-probe-iterations", type=int, default=8, metavar="N")
    parser.add_argument("--settle-s", type=float, default=0.4, metavar="SECS",
                        help="Pause after each motion before the next detect (default 0.4)")
    args = parser.parse_args()

    if args.rover == "atlas" and not args.atlas_port:
        parser.error("--atlas-port is required for --rover atlas")
    if args.rover == "roomba" and not args.roomba_port:
        parser.error("--roomba-port is required for --rover roomba")

    port = args.atlas_port if args.rover == "atlas" else args.roomba_port
    rover_ctrl = _build_rover_ctrl(args.rover, port, args.dry_run)
    log.info("Rover controller: %s  dry_run=%s", args.rover, args.dry_run)

    client = MoondreamClient(args.moondream_server)

    state = AgentState()
    running = threading.Event()
    running.set()

    threading.Thread(target=_camera_thread, args=(state, args.device, running),
                     daemon=True, name="camera").start()

    log.info("Publishing to %s — open that URL in a browser to view", args.web_server)
    publisher_thread = threading.Thread(
        target=AgentPublisher(args.web_server).run,
        args=(state, rover_ctrl, None), daemon=True, name="publisher")
    publisher_thread.start()

    try:
        _align_loop(state, client, rover_ctrl, args, running)
        log.info("Alignment finished — holding. Ctrl-C to exit.")
        while running.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        running.clear()
        if rover_ctrl:
            rover_ctrl.stop()


if __name__ == "__main__":
    main()
