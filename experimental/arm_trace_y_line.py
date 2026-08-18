#!/usr/bin/env python3
"""
arm_trace_y_line.py — Trace a line parallel to the Y axis at fixed X and Z.

Defaults to X = 400 mm, Z = 100 mm, sweeping Y across the arm's lateral bounds
from positive to negative (450 → -450 mm). Note that X=400 and Y=±450 both sit
outside the workspace limits declared in arm_controller.py (X max 300,
Y max ±250) — by default those values are clamped. Pass --no-clamp to send them
raw and let the arm firmware decide.

Motion is streamed rather than stepped: many small waypoints are sent at a
fixed rate, so the arm is always already moving toward the next target instead
of decelerating to a stop at each one.

`spd` must be sized to the step rate. It is servo steps/sec (4096 = one servo
revolution), and `spd: 0` means *maximum* speed, not "unlimited/smooth" — with
spd:0 the arm bolts to each target, arrives early, idles, and reads as jumping
between extremes rather than tracing the line. Too slow and it lags behind the
stream; too fast and it jumps. See arm_rover_trace_y.py's auto_speed() for
computing this from --steps and --delay.

Usage
─────
    # Dry-run the default line (X=400, Z=100, Y from 450 to -450):
    python experimental/arm_trace_y_line.py --dry-run

    # Preview the trajectory:
    python experimental/arm_trace_y_line.py --dry-run --plot

    # Run on hardware, unclamped:
    python experimental/arm_trace_y_line.py --port /dev/ttyUSB0 --no-clamp

    # Smoother still — 400 steps at 50 Hz:
    python experimental/arm_trace_y_line.py --port /dev/ttyUSB0 \\
        --steps 400 --delay 0.02

    # Sweep back and forth twice:
    python experimental/arm_trace_y_line.py --port /dev/ttyUSB0 --passes 2
"""

import argparse
import json
import math
import sys
import time

# Arm workspace limits (mm) — mirrors arm_controller.py
ARM_X_MIN, ARM_X_MAX =   50, 300
ARM_Y_MIN, ARM_Y_MAX = -250, 250
ARM_Z_MIN, ARM_Z_MAX =   30, 300

WRIST_ANGLE = 0


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def build_command(x: float, y: float, z: float, speed: int) -> str:
    return json.dumps({
        "T":   1041,
        "x":   round(float(x), 2),
        "y":   round(float(y), 2),
        "z":   round(float(z), 2),
        "t":   WRIST_ANGLE,
        "spd": speed,
    })


def _ease(t: float, mode: str) -> float:
    """Map linear progress t∈[0,1] to eased progress, for accel/decel at the ends."""
    if mode == "cosine":
        # Smooth accel out of the start and decel into the end — removes the
        # lurch that a linear ramp produces at direction reversals.
        return 0.5 * (1.0 - math.cos(math.pi * t))
    return t


def generate_waypoints(x: float, z: float,
                       y_start: float, y_end: float,
                       steps: int, passes: int,
                       clamp: bool,
                       ease: str = "linear") -> list[tuple[float, float, float]]:
    """Y sweep at constant X/Z. Each extra pass reverses direction."""
    if clamp:
        x = _clamp(x, ARM_X_MIN, ARM_X_MAX)
        z = _clamp(z, ARM_Z_MIN, ARM_Z_MAX)
        y_start = _clamp(y_start, ARM_Y_MIN, ARM_Y_MAX)
        y_end   = _clamp(y_end,   ARM_Y_MIN, ARM_Y_MAX)

    pts = []
    for p in range(passes):
        lo, hi = (y_start, y_end) if p % 2 == 0 else (y_end, y_start)
        # Skip the duplicated endpoint when reversing
        first = 1 if p > 0 else 0
        for i in range(first, steps + 1):
            y = lo + (hi - lo) * _ease(i / steps, ease)
            pts.append((x, y, z))
    return pts


def plot_trajectory(pts: list[tuple[float, float, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed — skipping plot")
        return

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    steps = list(range(len(pts)))

    fig = plt.figure(figsize=(14, 5))

    ax3 = fig.add_subplot(1, 3, 1, projection='3d')
    ax3.plot(xs, ys, zs, 'o-', color='steelblue', markersize=6)
    ax3.scatter([xs[0]], [ys[0]], [zs[0]], color='green', s=80, label='start')
    ax3.scatter([xs[-1]], [ys[-1]], [zs[-1]], color='red', s=80, label='end')
    ax3.set_xlabel('X (mm)'); ax3.set_ylabel('Y (mm)'); ax3.set_zlabel('Z (mm)')
    ax3.set_title('3-D view')
    ax3.legend(fontsize=8)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(steps, ys, 's-', color='tomato', markersize=6)
    ax2.axhline(ARM_Y_MIN, color='grey', ls='--', lw=1)
    ax2.axhline(ARM_Y_MAX, color='grey', ls='--', lw=1)
    ax2.set_xlabel('Step'); ax2.set_ylabel('Y (mm)')
    ax2.set_title('Y profile (moving axis)')
    ax2.grid(True, alpha=0.3)

    ax1 = fig.add_subplot(1, 3, 3)
    ax1.plot(xs, ys, 'o-', color='mediumseagreen', markersize=6)
    ax1.scatter([xs[0]], [ys[0]], color='green', s=80)
    ax1.scatter([xs[-1]], [ys[-1]], color='red', s=80)
    ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)')
    ax1.set_title('Top-down view (X–Y)')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    plt.suptitle(f'arm_trace_y_line — X={pts[0][0]:.1f}mm  Z={pts[0][2]:.1f}mm',
                 fontsize=12)
    plt.tight_layout()
    plt.show()


def run(pts, speed: int, delay: float, port: str, dry_run: bool) -> None:
    ser = None
    if not dry_run:
        try:
            import serial
            ser = serial.Serial(port, 115200, timeout=1)
            print(f"Serial opened: {port}")
            time.sleep(2)   # allow ESP32 to boot after serial open
        except Exception as e:
            sys.exit(f"Cannot open {port}: {e}")

    ys = [p[1] for p in pts]
    print(f"\nTracing {len(pts)} points along Y  speed={speed}  delay={delay}s")
    print(f"  X: {pts[0][0]:.1f} mm  (fixed)")
    print(f"  Z: {pts[0][2]:.1f} mm  (fixed)")
    print(f"  Y: {ys[0]:.1f} → {ys[-1]:.1f} mm  ← moving\n")

    # Stream on a fixed wall-clock schedule rather than sleeping a fixed amount
    # after each write. Serial writes take non-zero and variable time; sleeping
    # `delay` on top of that lets timing drift and stutter. Anchoring each step
    # to start_t + i*delay keeps the command rate metronomic, which is what the
    # arm needs to blend one move into the next.
    start_t = time.monotonic()
    try:
        for i, (x, y, z) in enumerate(pts):
            cmd   = build_command(x, y, z, speed)
            label = f"Step {i:03d}/{len(pts)-1}  y={y:7.1f}mm"
            if dry_run:
                print(f"[dry-run] {label}  →  {cmd}")
            else:
                ser.write((cmd + "\n").encode())
                ser.flush()
                # Discard any feedback the firmware emitted, so the input buffer
                # can't fill and stall writes partway through a long sweep.
                if ser.in_waiting:
                    ser.reset_input_buffer()
                print(f"Sent     {label}")

            if i < len(pts) - 1:
                sleep_for = (start_t + (i + 1) * delay) - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if ser:
            ser.close()

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace a line parallel to Y at fixed X and Z (RoArm-M2-S)")
    parser.add_argument("--x",       type=float, default=400,
                        help="Fixed X in mm (default: 400)")
    parser.add_argument("--z",       type=float, default=100,
                        help="Fixed Z in mm (default: 100)")
    parser.add_argument("--y-start", type=float, default=450,
                        help="Y start in mm (default: 450)")
    parser.add_argument("--y-end",   type=float, default=-450,
                        help="Y end in mm (default: -450)")
    parser.add_argument("--steps",   type=int,   default=200,
                        help="Steps per pass — more, smaller steps move more "
                             "smoothly (default: 200)")
    parser.add_argument("--passes",  type=int,   default=1,
                        help="Sweeps along the line, alternating direction (default: 1)")
    parser.add_argument("--speed",   type=int,   default=300,
                        help="Arm speed in servo steps/sec (4096 = one servo "
                             "revolution). Do NOT use 0 — per the Waveshare docs "
                             "that means MAXIMUM speed, so the arm jumps between "
                             "targets instead of tracing the line (default: 300)")
    parser.add_argument("--delay",   type=float, default=0.05,
                        help="Seconds between steps — the streaming period "
                             "(default: 0.05, i.e. 20 Hz)")
    parser.add_argument("--ease",    choices=["linear", "cosine"], default="cosine",
                        help="Velocity profile along the line: cosine accelerates "
                             "and decelerates at the ends (default: cosine)")
    parser.add_argument("--port",    type=str,   default="/dev/ttyUSB0",
                        help="Serial port (default: /dev/ttyUSB0)")
    parser.add_argument("--no-clamp", action="store_true",
                        help="Send X/Y/Z raw instead of clamping to workspace limits")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without sending to hardware")
    parser.add_argument("--plot",    action="store_true",
                        help="Show matplotlib trajectory preview")
    args = parser.parse_args()

    if args.steps < 1:
        sys.exit("--steps must be >= 1")
    if args.passes < 1:
        sys.exit("--passes must be >= 1")

    clamp = not args.no_clamp
    if clamp and (args.x > ARM_X_MAX or args.z > ARM_Z_MAX
                  or args.y_start > ARM_Y_MAX or args.y_start < ARM_Y_MIN
                  or args.y_end   > ARM_Y_MAX or args.y_end   < ARM_Y_MIN):
        print(f"[warn] requested X={args.x:.0f} Z={args.z:.0f} "
              f"Y={args.y_start:.0f}→{args.y_end:.0f} exceed workspace limits "
              f"(X<={ARM_X_MAX}, Z<={ARM_Z_MAX}, {ARM_Y_MIN}<=Y<={ARM_Y_MAX}) "
              f"— clamping. Use --no-clamp to send raw.")

    pts = generate_waypoints(args.x, args.z, args.y_start, args.y_end,
                             args.steps, args.passes, clamp, args.ease)

    if args.plot:
        plot_trajectory(pts)

    run(pts, args.speed, args.delay, args.port, args.dry_run)


if __name__ == "__main__":
    main()
