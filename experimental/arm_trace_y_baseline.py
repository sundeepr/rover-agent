#!/usr/bin/env python3
"""
arm_trace_y_baseline.py — Calibrate the pose-dependent "no-contact" torque
baseline across a Y sweep, for use as a correction in arm_trace_y_surface.py.

Why this exists
────────────────
arm_trace_y_surface.py's contact-pressure controller uses raw torS+torE as
its error signal against a single flat --target-torque. But torS/torE
aren't a calibrated force sensor — they're the servo's own effort to hold
its commanded position, which also includes the arm's gravity load and
joint friction. Gravity load isn't constant: as Y sweeps (rotating the
base) and the arm's shoulder/elbow angles change, the "idle" (no-contact)
torque drifts across the sweep. A flat target-torque is measuring against
a moving baseline without knowing it.

This script isolates that baseline: it sweeps Y at the SAME X/Z (and the
same step grid/easing) as the real trace — not a retracted/offset pose —
and records torS/torE at each Y, averaged over --samples reads to reduce
noise. Using the exact same X means the arm's joint angles/gravity loading
during calibration match the real run exactly, with no offset-geometry
error to worry about. It's on you to make sure there's actually no contact
during this pass — move the surface/workpiece out of the way (or move the
arm to a clear area) before running it, since the script itself doesn't
retract anything. The result is a y -> baseline-torque profile written to
JSON. arm_trace_y_surface.py can then subtract baseline(y) from measured
torque before computing error, so the controller reacts to *contact*
effort, not pose-dependent gravity drift.

Usage
─────
    # No hardware — simulate a gravity-load curve to see the file format
    # and sanity-check the calibration pass itself:
    python experimental/arm_trace_y_baseline.py --simulate --plot

    # On hardware: move the surface/workpiece out of the arm's way first,
    # then match --x/--z/--y-start/--y-end/--steps/--ease to what you'll
    # run in arm_trace_y_surface.py:
    python experimental/arm_trace_y_baseline.py --port /dev/ttyUSB0 \\
        --x 400 --z 155 --y-start 250 --y-end -250 --steps 50 \\
        --output experimental/arm_surface_baseline.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone

from arm_trace_y_surface import (
    ARM_X_MIN, ARM_X_MAX, ARM_Y_MIN, ARM_Y_MAX, ARM_Z_MIN, ARM_Z_MAX,
    _clamp, _ease, auto_speed, build_command, request_feedback,
    combine_torque, go_home,
)

FEEDBACK_TIMEOUT_S = 1.0


def simulate_baseline(x: float, y: float, y_start: float, y_end: float,
                      idle_torque: float, gravity_amp: float,
                      gravity_freq: float, noise: float) -> tuple[float, float]:
    """
    Fake (torS, torE) with NO contact term — models how gravity loading on
    the shoulder/elbow shifts with pose as Y sweeps, so --simulate can
    exercise this script without a real gravity-dependent torque curve.
    """
    span = (y_end - y_start) or 1.0
    phase = (y - y_start) / span * 2 * math.pi * gravity_freq
    base = idle_torque + gravity_amp * math.sin(phase)
    tor_s = max(0.0, base / 2)
    tor_e = max(0.0, base / 2)
    if noise:
        import random
        tor_s = max(0.0, tor_s + random.uniform(-noise, noise))
        tor_e = max(0.0, tor_e + random.uniform(-noise, noise))
    return tor_s, tor_e


def sample_point(args, ser, x: float, y: float, z: float, speed: int,
                 y_start: float, y_end: float) -> dict:
    """Move to (x, y, z), settle, then take --samples readings and average."""
    if not args.simulate:
        ser.write((build_command(x, y, z, speed) + "\n").encode())
        ser.flush()
        time.sleep(args.settle)

    tor_s_vals, tor_e_vals = [], []
    misses = 0
    for s in range(args.samples):
        if args.simulate:
            tor_s, tor_e = simulate_baseline(
                x, y, y_start, y_end, args.sim_idle_torque,
                args.sim_gravity_amp, args.sim_gravity_freq, args.sim_noise)
        else:
            fb = request_feedback(ser, args.feedback_timeout)
            if fb:
                tor_s, tor_e = fb.get("torS"), fb.get("torE")
            else:
                tor_s = tor_e = None

        if tor_s is None or tor_e is None:
            misses += 1
        else:
            tor_s_vals.append(tor_s)
            tor_e_vals.append(tor_e)

        if s < args.samples - 1:
            time.sleep(args.sample_delay)

    if not tor_s_vals:
        return {"y": round(y, 2), "tor_s": None, "tor_e": None,
                "tor_combined": None, "samples_ok": 0, "samples_miss": misses}

    tor_s_avg = statistics.mean(tor_s_vals)
    tor_e_avg = statistics.mean(tor_e_vals)
    return {
        "y": round(y, 2),
        "tor_s": round(tor_s_avg, 3),
        "tor_e": round(tor_e_avg, 3),
        "tor_combined": round(abs(tor_s_avg) + abs(tor_e_avg), 3),
        "samples_ok": len(tor_s_vals),
        "samples_miss": misses,
    }


def run(args) -> list[dict]:
    clamp = not args.no_clamp
    x = _clamp(args.x, ARM_X_MIN, ARM_X_MAX) if clamp else args.x
    z = _clamp(args.z, ARM_Z_MIN, ARM_Z_MAX) if clamp else args.z
    y_start = _clamp(args.y_start, ARM_Y_MIN, ARM_Y_MAX) if clamp else args.y_start
    y_end   = _clamp(args.y_end,   ARM_Y_MIN, ARM_Y_MAX) if clamp else args.y_end

    speed = args.speed
    if speed is None:
        speed = auto_speed(x, y_start, y_end, args.steps, args.settle, args.ease)
        print(f"[auto-speed] spd={speed}")

    ser = None
    if not args.simulate:
        try:
            import serial
            ser = serial.Serial(args.port, 115200, timeout=0.1)
            print(f"Serial opened: {args.port}")
            time.sleep(2)   # allow ESP32 to boot after serial open
            ser.reset_input_buffer()
        except Exception as e:
            sys.exit(f"Cannot open {args.port}: {e}")

        if not args.skip_home:
            go_home(ser, args.home_speed, args.home_accel, args.home_dwell)

    mode = "SIMULATE" if args.simulate else "HARDWARE"
    print(f"\n[{mode}] Calibrating no-contact torque baseline")
    print(f"  X: {x:.1f} mm  (must NOT be touching the surface — move it "
          f"out of the way before running this)")
    print(f"  Z: {z:.1f} mm (fixed)   Y: {y_start:.1f} → {y_end:.1f} mm")
    print(f"  {args.steps + 1} points, {args.samples} samples each\n")

    points = []
    misses_total = 0
    try:
        for i in range(args.steps + 1):
            t = i / args.steps
            y = y_start + (y_end - y_start) * _ease(t, args.ease)
            pt = sample_point(args, ser, x, y, z, speed, y_start, y_end)
            points.append(pt)
            if pt["tor_combined"] is None:
                misses_total += 1
                print(f"Point {i:03d}/{args.steps}  y={y:7.1f}  [warn] no feedback")
            else:
                print(f"Point {i:03d}/{args.steps}  y={y:7.1f}  "
                      f"torS={pt['tor_s']:7.2f}  torE={pt['tor_e']:7.2f}  "
                      f"combined={pt['tor_combined']:7.2f}"
                      + (f"  ({pt['samples_miss']} dropped reads)"
                         if pt["samples_miss"] else ""))
    except KeyboardInterrupt:
        print("\nInterrupted — writing what was collected so far.")
    finally:
        if ser:
            ser.close()

    if misses_total:
        print(f"\n[warn] {misses_total}/{len(points)} points had no usable "
              f"feedback at all.")

    return points


def write_baseline(path: str, points: list[dict], args) -> None:
    payload = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "arm_trace_y_baseline.py",
        "x": args.x,
        "z": args.z,
        "y_start": args.y_start,
        "y_end": args.y_end,
        "steps": args.steps,
        "ease": args.ease,
        "samples_per_point": args.samples,
        "simulate": args.simulate,
        "points": points,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {len(points)} points to {path}")


def plot_baseline(points: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed — skipping plot")
        return

    ys = [p["y"] for p in points if p["tor_combined"] is not None]
    tor_s = [p["tor_s"] for p in points if p["tor_combined"] is not None]
    tor_e = [p["tor_e"] for p in points if p["tor_combined"] is not None]
    combined = [p["tor_combined"] for p in points if p["tor_combined"] is not None]

    plt.figure(figsize=(9, 5))
    plt.plot(ys, tor_s, 'o-', label='torS', markersize=4)
    plt.plot(ys, tor_e, 's-', label='torE', markersize=4)
    plt.plot(ys, combined, '^-', label='combined', markersize=4, color='black')
    plt.xlabel('Y (mm)')
    plt.ylabel('No-contact baseline torque')
    plt.title('arm_trace_y_baseline — pose-dependent torque profile')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the no-contact torque baseline across a Y "
                    "sweep for arm_trace_y_surface.py's contact correction")
    parser.add_argument("--x",       type=float, required=True,
                        help="X in mm — should match what you'll use in "
                             "arm_trace_y_surface.py exactly. You're "
                             "responsible for making sure the arm doesn't "
                             "actually touch anything during this pass "
                             "(move the surface/workpiece away first)")
    parser.add_argument("--z",       type=float, default=155,
                        help="Fixed Z in mm — match what you'll use in "
                             "arm_trace_y_surface.py (default: 155)")
    parser.add_argument("--y-start", type=float, default=250,
                        help="Y start in mm (default: 250)")
    parser.add_argument("--y-end",   type=float, default=-250,
                        help="Y end in mm (default: -250)")
    parser.add_argument("--steps",   type=int,   default=50,
                        help="Points across the sweep — doesn't need to "
                             "match arm_trace_y_surface.py's --steps since "
                             "the correction interpolates between points "
                             "(default: 50)")
    parser.add_argument("--ease",    choices=["linear", "cosine"], default="cosine",
                        help="Should match arm_trace_y_surface.py's --ease "
                             "(default: cosine)")

    parser.add_argument("--samples", type=int, default=5,
                        help="Torque reads averaged per Y point, to reduce "
                             "noise (default: 5)")
    parser.add_argument("--sample-delay", type=float, default=0.05,
                        help="Seconds between repeated reads at one point "
                             "(default: 0.05)")
    parser.add_argument("--settle",  type=float, default=0.3,
                        help="Seconds to wait after each move before the "
                             "first read — longer than the contact script's "
                             "since there's no resistance to indicate "
                             "arrival (default: 0.3)")
    parser.add_argument("--feedback-timeout", type=float, default=FEEDBACK_TIMEOUT_S,
                        help=f"Seconds to wait for T:1051 (default: {FEEDBACK_TIMEOUT_S})")
    parser.add_argument("--speed",   type=int,   default=None,
                        help="Servo steps/sec for the Y move (default: "
                             "auto-computed)")

    parser.add_argument("--port",    type=str,   default="/dev/ttyUSB0",
                        help="Serial port (default: /dev/ttyUSB0)")
    parser.add_argument("--no-clamp", action="store_true",
                        help="Send X/Y/Z raw instead of clamping to workspace limits")
    parser.add_argument("--skip-home", action="store_true",
                        help="Skip homing (T:122) before starting")
    parser.add_argument("--home-speed", type=int, default=30,
                        help="Homing speed in deg/s (default: 30)")
    parser.add_argument("--home-accel", type=int, default=10,
                        help="Homing acceleration in deg/s^2 (default: 10)")
    parser.add_argument("--home-dwell", type=float, default=3.0,
                        help="Seconds to wait after homing (default: 3.0)")

    parser.add_argument("--output",  type=str,
                        default="experimental/arm_surface_baseline.json",
                        help="Where to write the baseline JSON (default: "
                             "experimental/arm_surface_baseline.json)")
    parser.add_argument("--plot",    action="store_true",
                        help="Show the baseline curve after calibrating")

    parser.add_argument("--simulate", action="store_true",
                        help="No hardware — fake a gravity-load curve to "
                             "test the calibration pass and file format")
    parser.add_argument("--sim-idle-torque", type=float, default=20.0,
                        help="[--simulate] baseline torque with the arm "
                             "centred, mm (default: 20.0)")
    parser.add_argument("--sim-gravity-amp", type=float, default=15.0,
                        help="[--simulate] how much the baseline swings "
                             "across the sweep as pose changes (default: 15.0)")
    parser.add_argument("--sim-gravity-freq", type=float, default=1.0,
                        help="[--simulate] cycles across the sweep (default: 1.0)")
    parser.add_argument("--sim-noise", type=float, default=2.0,
                        help="[--simulate] uniform +/- measurement noise (default: 2.0)")

    args = parser.parse_args()

    if args.steps < 1:
        sys.exit("--steps must be >= 1")
    if args.samples < 1:
        sys.exit("--samples must be >= 1")

    points = run(args)
    write_baseline(args.output, points, args)

    if args.plot:
        plot_baseline(points)


if __name__ == "__main__":
    main()
