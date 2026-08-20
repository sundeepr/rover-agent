#!/usr/bin/env python3
"""
arm_trace_y_surface.py — Trace a line along Y while holding constant contact
pressure against a surface, using the RoArm-M2-S's own servo torque feedback
as the pressure signal (no external force sensor needed).

Background — feedback signal
─────────────────────────────
T:105 (CMD_SERVO_RAD_FEEDBACK) replies with T:1051, which includes per-joint
torque: torB (base), torS (shoulder), torE (elbow) — see the firmware source
(json_cmd.h in https://github.com/waveshareteam/roarm_m2). There is no
wrist/EoAT torque field. This script combines torS + torE (abs sum) into a
single "pressure" signal — shoulder and elbow are the joints that load up as
X (depth toward the surface) increases for this geometry.

These torque units are raw servo load values (Waveshare's docs mention a
default max-torque limit of 1000), not calibrated newtons — --target-torque
is a unitless setpoint you'll need to find empirically for your setup. A
reasonable way to find one: run arm_read_position.py --watch --json while
manually pressing the EoAT against the surface at roughly the pressure you
want, and read off the torS/torE values it reports.

Control loop
────────────
At each Y waypoint, Y is held fixed while X is corrected repeatedly until
torque *settles*, only then does Y advance to the next waypoint:

    x += kp * (target_torque - measured_torque)

Too much measured torque (pressing too hard) -> shrink x (retract).
Too little (losing contact) -> grow x (press in). "Settled" means
--stabilize-consecutive torque readings in a row landed inside --deadband;
--stabilize-tries caps how many correction attempts we'll make at one Y
position before giving up and advancing anyway (with a warning), so a
Y position the controller can't converge at doesn't stall the whole run.
--max-step caps how far x can move in one correction so a bad reading can't
slam the arm into the surface.

Because each correction needs a full move + settle + feedback round-trip,
and Y won't advance until pressure has actually stabilized, this is NOT the
fire-and-forget streaming used by arm_trace_y_line.py — expect a slower,
uneven cadence (some Y positions settle in one read, others take many),
not smooth continuous motion.

Usage
─────
    # No hardware needed — simulate a wavy surface and see if the
    # controller tracks it (prints + --plot the X/torque response):
    python experimental/arm_trace_y_surface.py --simulate --plot

    # On hardware, once you've picked a --target-torque:
    python experimental/arm_trace_y_surface.py --port /dev/ttyUSB0 \\
        --target-torque 150 --kp 0.05

    # Live X/Y/Z + torque dashboard instead of scrolling log lines
    # (works with --simulate too):
    python experimental/arm_trace_y_surface.py --simulate --curses
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

ARM_X_MIN, ARM_X_MAX =   50, 410
ARM_Y_MIN, ARM_Y_MAX = -250, 250
ARM_Z_MIN, ARM_Z_MAX =   30, 300

SERVO_STEPS_PER_REV = 4096   # 12-bit encoder: one revolution = 4096 steps
WRIST_ANGLE = 0

FEEDBACK_COMMAND = {"T": 105}
FEEDBACK_REPLY_T = 1051
FEEDBACK_TIMEOUT_S = 1.0


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _ease(t: float, mode: str) -> float:
    if mode == "cosine":
        return 0.5 * (1.0 - math.cos(math.pi * t))
    return t


def auto_speed(x: float, y_start: float, y_end: float,
                steps: int, delay: float, ease: str) -> int:
    """Size spd off base rotation swept per step — see arm_trace_y_line.py."""
    if delay <= 0 or steps < 1:
        return 0
    a_start = math.atan2(y_start, x)
    a_end   = math.atan2(y_end, x)
    total_rad = abs(a_end - a_start)
    if total_rad > math.pi:
        total_rad = 2 * math.pi - total_rad
    if total_rad <= 0:
        return 0
    mean_rad_per_step = total_rad / steps
    if ease == "cosine":
        mean_rad_per_step *= math.pi / 2
    rad_per_s = mean_rad_per_step / delay
    spd = rad_per_s / (2 * math.pi) * SERVO_STEPS_PER_REV
    return max(10, int(spd * 1.2))


def build_command(x: float, y: float, z: float, speed: int) -> str:
    return json.dumps({
        "T":   1041,
        "x":   round(float(x), 2),
        "y":   round(float(y), 2),
        "z":   round(float(z), 2),
        "t":   WRIST_ANGLE,
        "spd": speed,
    })


def request_feedback(ser, timeout: float = FEEDBACK_TIMEOUT_S) -> dict | None:
    """Send T:105 and read back the T:1051 feedback line. None on timeout."""
    ser.write((json.dumps(FEEDBACK_COMMAND) + "\n").encode())
    ser.flush()
    deadline = time.time() + timeout
    buf = b""
    while time.time() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if not chunk:
            continue
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and data.get("T") == FEEDBACK_REPLY_T:
                return data
    return None


def combine_torque(fb: dict) -> float | None:
    """abs(torS) + abs(torE) — see module docstring for why these two."""
    tor_s = fb.get("torS")
    tor_e = fb.get("torE")
    if not isinstance(tor_s, (int, float)) or not isinstance(tor_e, (int, float)):
        return None
    return abs(tor_s) + abs(tor_e)


def simulate_torque(x_cmd: float, y: float,
                    surface_x: float, surface_amp: float, surface_freq: float,
                    y_start: float, y_end: float,
                    stiffness: float, idle_torque: float,
                    noise: float) -> tuple[float, float]:
    """
    Fake (torS, torE) for --simulate: a surface at surface_x(y), wavy by
    surface_amp/surface_freq across the sweep. Penetration past the surface
    (x_cmd beyond surface_x) generates torque proportional to `stiffness`,
    split evenly between the two joints; otherwise just idle_torque each
    (no contact).
    """
    span = (y_end - y_start) or 1.0
    phase = (y - y_start) / span * 2 * math.pi * surface_freq
    surf_x_here = surface_x + surface_amp * math.sin(phase)
    penetration = max(0.0, x_cmd - surf_x_here)
    torque = idle_torque + stiffness * penetration
    tor_s = max(0.0, torque / 2)
    tor_e = max(0.0, torque / 2)
    if noise:
        import random
        tor_s = max(0.0, tor_s + random.uniform(-noise, noise))
        tor_e = max(0.0, tor_e + random.uniform(-noise, noise))
    return tor_s, tor_e


def generate_steps(args, ser, clamp: bool, x0: float, z: float,
                   y_start: float, y_end: float, speed: int):
    """
    Shared stepper for both the plain and curses front-ends: yields one dict
    per waypoint with the commanded/measured state, and does the actual
    move + settle + feedback + control-update work. Pacing (--delay) and
    Ctrl-C both happen inside this generator.
    """
    def measure(x, y):
        if args.simulate:
            tor_s, tor_e = simulate_torque(
                x, y, args.sim_surface_x, args.sim_surface_amp,
                args.sim_surface_freq, y_start, y_end,
                args.sim_stiffness, args.sim_idle_torque, args.sim_noise)
            return tor_s, tor_e, abs(tor_s) + abs(tor_e)

        cmd = build_command(x, y, z, speed)
        ser.write((cmd + "\n").encode())
        ser.flush()
        time.sleep(args.settle)
        fb = request_feedback(ser, args.feedback_timeout)
        if fb:
            return fb.get("torS"), fb.get("torE"), combine_torque(fb)
        return None, None, None

    x = x0
    start_t = time.monotonic()
    for i in range(args.steps + 1):
        t = i / args.steps
        y = y_start + (y_end - y_start) * _ease(t, args.ease)

        # Hold Y fixed at this waypoint and keep correcting X until torque
        # settles (>= --stabilize-consecutive readings inside the deadband
        # in a row), instead of advancing Y every iteration regardless of
        # whether the pressure loop has converged. --stabilize-tries caps
        # how long we'll wait before giving up and moving on anyway.
        consecutive_stable = 0
        for settle_iter in range(args.stabilize_tries):
            tor_s, tor_e, measured = measure(x, y)

            state = {
                "i": i, "steps": args.steps, "x": x, "y": y, "z": z,
                "tor_s": tor_s, "tor_e": tor_e, "torque": measured,
                "target": args.target_torque, "deadband": args.deadband,
                "miss": measured is None, "error": None,
                "delta": 0.0, "in_deadband": False, "x_next": x,
                "elapsed": time.monotonic() - start_t,
                "settle_iter": settle_iter, "stable": False,
                "gave_up": False,
            }

            if measured is not None:
                error = args.target_torque - measured
                state["error"] = error
                if abs(error) <= args.deadband:
                    state["in_deadband"] = True
                    consecutive_stable += 1
                    if consecutive_stable >= args.stabilize_consecutive:
                        state["stable"] = True
                else:
                    consecutive_stable = 0
                    delta = _clamp(args.kp * error, -args.max_step, args.max_step)
                    x_next = x + delta
                    if clamp:
                        x_next = _clamp(x_next, ARM_X_MIN, ARM_X_MAX)
                    state["delta"] = x_next - x
                    state["x_next"] = x_next
                    x = x_next
            else:
                consecutive_stable = 0

            gave_up = (not state["stable"]
                      and settle_iter == args.stabilize_tries - 1)
            state["gave_up"] = gave_up

            yield state

            if state["stable"] or gave_up:
                break

            time.sleep(args.delay)


def plot_run(steps_log: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed — skipping plot")
        return

    # Use a sequential read-index rather than waypoint `i` for the X axis:
    # each Y waypoint can take several settle iterations (same `i`, `y`)
    # while X hunts for the target torque, so `i` alone repeats.
    idx = list(range(len(steps_log)))
    xs = [s["x"] for s in steps_log]
    ys = [s["y"] for s in steps_log]
    torques = [s["torque"] for s in steps_log if s["torque"] is not None]
    torque_idx = [n for n, s in enumerate(steps_log) if s["torque"] is not None]
    target = steps_log[0]["target"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(idx, ys, 's-', color='tomato', markersize=4)
    axes[0].set_xlabel('Read #'); axes[0].set_ylabel('Y (mm)')
    axes[0].set_title('Y profile (moving axis)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(idx, xs, 'o-', color='steelblue', markersize=4)
    axes[1].set_xlabel('Read #'); axes[1].set_ylabel('X (mm)')
    axes[1].set_title('X — controller output (depth)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(torque_idx, torques, '^-', color='darkorange', markersize=4,
                label='measured')
    axes[2].axhline(target, color='grey', ls='--', lw=1.5, label='target')
    axes[2].set_xlabel('Read #'); axes[2].set_ylabel('Torque (|torS|+|torE|)')
    axes[2].set_title('Pressure signal vs. target')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('arm_trace_y_surface — contact-pressure control loop', fontsize=12)
    plt.tight_layout()
    plt.show()


def _print_step(state: dict) -> None:
    i, steps = state["i"], state["steps"]
    tag = f"Step {i:03d}/{steps}.{state['settle_iter']:02d}"
    if state["miss"]:
        print(f"{tag}  y={state['y']:7.1f}  x={state['x']:7.1f}  "
              f"[warn] no/incomplete feedback")
        return
    if state["stable"]:
        tail = "STABLE -> advancing Y"
    elif state["gave_up"]:
        tail = "[warn] gave up waiting to stabilize -> advancing Y anyway"
    elif state["in_deadband"]:
        tail = "in deadband, confirming..."
    else:
        tail = f"-> x_next={state['x_next']:7.1f}  (waiting for Y to advance)"
    print(f"{tag}  y={state['y']:7.1f}  x={state['x']:7.1f}  "
          f"torque={state['torque']:7.1f}  err={state['error']:+7.1f}  {tail}")


def run_plain(args, ser, clamp, x0, z, y_start, y_end, speed) -> list[dict]:
    steps_log = []
    misses = 0
    try:
        for state in generate_steps(args, ser, clamp, x0, z, y_start, y_end, speed):
            _print_step(state)
            steps_log.append(state)
            if state["miss"]:
                misses += 1
    except KeyboardInterrupt:
        print("\nInterrupted.")

    if not args.simulate and misses:
        print(f"\n[warn] {misses}/{len(steps_log)} reads had no feedback — "
              f"increase --feedback-timeout or --settle if this is frequent.")
    print("\nDone.")
    return steps_log


def _bar(value: float, target: float, scale: float, width: int) -> str:
    """ASCII bar for `value`, with a `|` marker showing where `target` sits."""
    value_n = _clamp(value / scale, 0.0, 1.0) if scale else 0.0
    target_n = _clamp(target / scale, 0.0, 1.0) if scale else 0.0
    fill = int(round(value_n * width))
    mark = int(round(target_n * width))
    chars = list("#" * fill + "-" * (width - fill))
    if 0 <= mark < width:
        chars[mark] = "|"
    return "".join(chars)


def run_curses(args, ser, clamp, x0, z, y_start, y_end, speed) -> list[dict]:
    import curses

    steps_log = []
    mode = "SIMULATE" if args.simulate else "HARDWARE"
    bar_scale = max(args.target_torque * 2, 50.0)

    def body(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        for state in generate_steps(args, ser, clamp, x0, z, y_start, y_end, speed):
            steps_log.append(state)

            key = stdscr.getch()
            if key in (ord('q'), ord('Q'), 27):
                break

            stdscr.erase()
            row = 0

            def line(text="", attr=0):
                nonlocal row
                try:
                    stdscr.addstr(row, 0, text, attr)
                except curses.error:
                    pass   # window too small for this line — skip, don't crash
                row += 1

            line(f" ARM SURFACE TRACE — [{mode}]  ('q' to quit)", curses.A_BOLD)
            line("─" * 60)
            line(f" Step:      {state['i']:4d} / {state['steps']}"
                f"   Settle try: {state['settle_iter']+1:3d} / {args.stabilize_tries}"
                f"   Elapsed: {state['elapsed']:6.1f}s")
            line()
            line(" Position (mm)")
            line(f"   X = {state['x']:8.2f}    Y = {state['y']:8.2f}"
                f"    Z = {state['z']:8.2f}")
            line()
            line(" Torque (servo load units)")
            ts = state["tor_s"]
            te = state["tor_e"]
            line(f"   torS = {ts:8.2f}    torE = {te:8.2f}"
                if ts is not None and te is not None else
                "   torS =      n/a    torE =      n/a")
            if state["torque"] is not None:
                line(f"   combined = {state['torque']:8.2f}"
                    f"    target = {state['target']:8.2f}"
                    f"    error = {state['error']:+8.2f}")
                line()
                line(f"   [{_bar(state['torque'], state['target'], bar_scale, 40)}]"
                    f"  (| = target, scale 0–{bar_scale:.0f})")
            else:
                line("   [warn] no/incomplete feedback this step", curses.A_REVERSE)
            line()
            if state["miss"]:
                status = "NO FEEDBACK"
            elif state["stable"]:
                status = "STABLE — advancing Y"
            elif state["gave_up"]:
                status = "[warn] gave up waiting to stabilize — advancing Y anyway"
            elif state["in_deadband"]:
                status = "in deadband, confirming... (Y held)"
            elif state["delta"] > 0:
                status = f"LOSING CONTACT — advancing X by {state['delta']:+.2f}mm (Y held)"
            elif state["delta"] < 0:
                status = f"PRESSING TOO HARD — retracting X by {state['delta']:+.2f}mm (Y held)"
            else:
                status = "—"
            line(f" Status: {status}", curses.A_BOLD)

            stdscr.refresh()

    try:
        curses.wrapper(body)
    except KeyboardInterrupt:
        pass
    return steps_log


def run(args) -> None:
    clamp = not args.no_clamp
    x0 = _clamp(args.x_start, ARM_X_MIN, ARM_X_MAX) if clamp else args.x_start
    z = _clamp(args.z, ARM_Z_MIN, ARM_Z_MAX) if clamp else args.z
    y_start = _clamp(args.y_start, ARM_Y_MIN, ARM_Y_MAX) if clamp else args.y_start
    y_end   = _clamp(args.y_end,   ARM_Y_MIN, ARM_Y_MAX) if clamp else args.y_end

    speed = args.speed
    if speed is None:
        speed = auto_speed(x0, y_start, y_end, args.steps, args.delay, args.ease)
        if not args.curses:
            print(f"[auto-speed] spd={speed}")

    ser = None
    if not args.simulate:
        try:
            import serial
            ser = serial.Serial(args.port, 115200, timeout=0.1)
            if not args.curses:
                print(f"Serial opened: {args.port}")
            time.sleep(2)   # allow ESP32 to boot after serial open
            ser.reset_input_buffer()
        except Exception as e:
            sys.exit(f"Cannot open {args.port}: {e}")

    if not args.curses:
        mode = "SIMULATE" if args.simulate else "HARDWARE"
        print(f"\n[{mode}] Surface-following Y trace  target_torque={args.target_torque} "
              f"kp={args.kp} max_step={args.max_step}mm deadband={args.deadband}")
        print(f"  X start: {x0:.1f} mm   Z: {z:.1f} mm (fixed)")
        print(f"  Y: {y_start:.1f} → {y_end:.1f} mm\n")

    try:
        if args.curses:
            steps_log = run_curses(args, ser, clamp, x0, z, y_start, y_end, speed)
        else:
            steps_log = run_plain(args, ser, clamp, x0, z, y_start, y_end, speed)
    finally:
        if ser:
            ser.close()

    if args.plot:
        plot_run(steps_log)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace Y while holding constant contact pressure via "
                    "servo torque feedback (RoArm-M2-S)")
    parser.add_argument("--x-start", type=float, default=380,
                        help="Initial X in mm — a rough guess of the surface "
                             "distance; the controller corrects from there "
                             "(default: 380)")
    parser.add_argument("--z",       type=float, default=155,
                        help="Fixed Z in mm (default: 155)")
    parser.add_argument("--y-start", type=float, default=250,
                        help="Y start in mm (default: 250)")
    parser.add_argument("--y-end",   type=float, default=-250,
                        help="Y end in mm (default: -250)")
    parser.add_argument("--steps",   type=int,   default=100,
                        help="Steps along the sweep (default: 100)")
    parser.add_argument("--ease",    choices=["linear", "cosine"], default="cosine",
                        help="Velocity profile for Y (default: cosine)")

    parser.add_argument("--target-torque", type=float, default=150,
                        help="Setpoint for |torS|+|torE| — unitless servo "
                             "load, NOT calibrated to real force. Find a "
                             "starting value empirically (see module "
                             "docstring). (default: 150)")
    parser.add_argument("--kp",      type=float, default=0.05,
                        help="Proportional gain: mm of X correction per unit "
                             "of torque error (default: 0.05)")
    parser.add_argument("--max-step", type=float, default=5.0,
                        help="Max X change allowed per step, mm — safety cap "
                             "so a bad reading can't slam the arm into the "
                             "surface (default: 5.0)")
    parser.add_argument("--deadband", type=float, default=10.0,
                        help="Ignore torque error smaller than this — avoids "
                             "hunting/oscillation on measurement noise "
                             "(default: 10.0)")
    parser.add_argument("--stabilize-consecutive", type=int, default=2,
                        help="Torque readings in a row that must land inside "
                             "the deadband before Y is allowed to advance to "
                             "the next waypoint (default: 2)")
    parser.add_argument("--stabilize-tries", type=int, default=20,
                        help="Max X-correction attempts at one Y position "
                             "before giving up and advancing anyway (with a "
                             "warning) — a safety cap against stalling "
                             "forever on a Y position the controller can't "
                             "settle at (default: 20)")

    parser.add_argument("--speed",   type=int,   default=None,
                        help="Servo steps/sec for the Y move (default: "
                             "auto-computed, see arm_trace_y_line.py)")
    parser.add_argument("--delay",   type=float, default=0.15,
                        help="Seconds between X-correction attempts while "
                             "waiting for torque to settle at a Y position "
                             "(default: 0.15)")
    parser.add_argument("--settle",  type=float, default=0.08,
                        help="Seconds to wait after each move before reading "
                             "feedback, so the servo has actually loaded up "
                             "against the surface (default: 0.08)")
    parser.add_argument("--feedback-timeout", type=float, default=FEEDBACK_TIMEOUT_S,
                        help=f"Seconds to wait for T:1051 (default: {FEEDBACK_TIMEOUT_S})")

    parser.add_argument("--port",    type=str,   default="/dev/ttyUSB0",
                        help="Serial port (default: /dev/ttyUSB0)")
    parser.add_argument("--no-clamp", action="store_true",
                        help="Send X/Y/Z raw instead of clamping to workspace limits")
    parser.add_argument("--plot",    action="store_true",
                        help="Show matplotlib summary of X/torque vs. step "
                             "(after the run finishes)")
    parser.add_argument("--curses",  action="store_true",
                        help="Live ncurses-style dashboard (X/Y/Z, torS/torE, "
                             "target/error, status) updated in place each "
                             "step instead of scrolling log lines. Press 'q' "
                             "to stop early.")

    parser.add_argument("--simulate", action="store_true",
                        help="No hardware — fake a wavy surface and torque "
                             "response so you can sanity-check the "
                             "controller (gains, deadband, max-step) first")
    parser.add_argument("--sim-surface-x",    type=float, default=380,
                        help="[--simulate] baseline surface X, mm (default: 380)")
    parser.add_argument("--sim-surface-amp",  type=float, default=15,
                        help="[--simulate] surface waviness amplitude, mm (default: 15)")
    parser.add_argument("--sim-surface-freq", type=float, default=1.0,
                        help="[--simulate] waviness cycles across the sweep (default: 1.0)")
    parser.add_argument("--sim-stiffness",    type=float, default=8.0,
                        help="[--simulate] torque units per mm of penetration (default: 8.0)")
    parser.add_argument("--sim-idle-torque",  type=float, default=20.0,
                        help="[--simulate] torque with no contact, mm (default: 20.0)")
    parser.add_argument("--sim-noise",        type=float, default=3.0,
                        help="[--simulate] uniform +/- measurement noise (default: 3.0)")

    args = parser.parse_args()

    if args.steps < 1:
        sys.exit("--steps must be >= 1")

    run(args)


if __name__ == "__main__":
    main()
