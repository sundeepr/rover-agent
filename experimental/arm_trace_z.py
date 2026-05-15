#!/usr/bin/env python3
"""
arm_trace_z.py — Trace a straight line along X, Y, or Z with the RoArm-M2-S.

Keeps the other two axes constant so you can visualise the arm's coordinate
system one axis at a time.

Usage
─────
    # Dry-run along Z (up/down):
    python experimental/arm_trace_z.py --axis z --dry-run

    # Trace along X (forward/back) on hardware:
    python experimental/arm_trace_z.py --axis x --port /dev/ttyUSB0 \\
        --x-start 80 --x-end 250 --steps 10

    # Trace along Y (left/right):
    python experimental/arm_trace_z.py --axis y --port /dev/ttyUSB0 \\
        --y-start -100 --y-end 100 --steps 10

    # Trace along Z (up/down):
    python experimental/arm_trace_z.py --axis z --port /dev/ttyUSB0 \\
        --z-start 80 --z-end 200 --steps 10

    # Show matplotlib preview:
    python experimental/arm_trace_z.py --axis z --plot --dry-run
"""

import argparse
import json
import sys
import time

# Arm workspace limits (mm)
ARM_X_MIN, ARM_X_MAX =  50, 300
ARM_Y_MIN, ARM_Y_MAX = -200, 200
ARM_Z_MIN, ARM_Z_MAX =  30, 300

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


def generate_waypoints(axis: str,
                        x: float, y: float, z: float,
                        start: float, end: float,
                        steps: int) -> list[tuple[float, float, float]]:
    pts = []
    for i in range(steps + 1):
        t  = i / steps
        v  = start + (end - start) * t
        if axis == "x":
            xc, yc, zc = _clamp(v, ARM_X_MIN, ARM_X_MAX), \
                          _clamp(y, ARM_Y_MIN, ARM_Y_MAX), \
                          _clamp(z, ARM_Z_MIN, ARM_Z_MAX)
        elif axis == "y":
            xc, yc, zc = _clamp(x, ARM_X_MIN, ARM_X_MAX), \
                          _clamp(v, ARM_Y_MIN, ARM_Y_MAX), \
                          _clamp(z, ARM_Z_MIN, ARM_Z_MAX)
        else:  # z
            xc, yc, zc = _clamp(x, ARM_X_MIN, ARM_X_MAX), \
                          _clamp(y, ARM_Y_MIN, ARM_Y_MAX), \
                          _clamp(v, ARM_Z_MIN, ARM_Z_MAX)
        pts.append((xc, yc, zc))
    return pts


def plot_trajectory(pts: list[tuple[float, float, float]], axis: str) -> None:
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

    # 3-D view
    ax3 = fig.add_subplot(1, 3, 1, projection='3d')
    ax3.plot(xs, ys, zs, 'o-', color='steelblue', markersize=6)
    ax3.scatter([xs[0]], [ys[0]], [zs[0]], color='green',  s=80, zorder=5, label='start')
    ax3.scatter([xs[-1]], [ys[-1]], [zs[-1]], color='red', s=80, zorder=5, label='end')
    ax3.set_xlabel('X (mm)'); ax3.set_ylabel('Y (mm)'); ax3.set_zlabel('Z (mm)')
    ax3.set_title('3-D view')
    ax3.legend(fontsize=8)

    # Axis profile
    ax_vals = {'x': xs, 'y': ys, 'z': zs}[axis]
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(steps, ax_vals, 's-', color='tomato', markersize=8)
    ax2.set_xlabel('Step'); ax2.set_ylabel(f'{axis.upper()} (mm)')
    ax2.set_title(f'{axis.upper()} profile (moving axis)')
    ax2.grid(True, alpha=0.3)

    # Top-down X–Y view
    ax3b = fig.add_subplot(1, 3, 3)
    ax3b.plot(xs, ys, 'o-', color='mediumseagreen', markersize=8)
    ax3b.scatter([xs[0]], [ys[0]], color='green', s=80, zorder=5)
    ax3b.scatter([xs[-1]], [ys[-1]], color='red', s=80, zorder=5)
    ax3b.set_xlabel('X (mm)'); ax3b.set_ylabel('Y (mm)')
    ax3b.set_title('Top-down view (X–Y)')
    ax3b.grid(True, alpha=0.3)
    ax3b.set_aspect('equal')

    fixed = {k: round(v, 1) for k, v in
             zip(['x', 'y', 'z'], [pts[0][0], pts[0][1], pts[0][2]])
             if k != axis}
    plt.suptitle(
        f'arm_trace — axis={axis.upper()}  '
        f'fixed: {", ".join(f"{k}={v}" for k, v in fixed.items())}',
        fontsize=12)
    plt.tight_layout()
    plt.show()


def run(pts, axis: str, speed: int, delay: float, port: str, dry_run: bool) -> None:
    ser = None
    if not dry_run:
        try:
            import serial
            ser = serial.Serial(port, 115200, timeout=1)
            print(f"Serial opened: {port}")
            time.sleep(2)
        except Exception as e:
            sys.exit(f"Cannot open {port}: {e}")

    moving_vals = [p[{'x': 0, 'y': 1, 'z': 2}[axis]] for p in pts]
    print(f"\nTracing {len(pts)} points along {axis.upper()}-axis  "
          f"speed={speed}  delay={delay}s")
    print(f"  X: {pts[0][0]:.1f} mm  {'← moving' if axis == 'x' else '(fixed)'}")
    print(f"  Y: {pts[0][1]:.1f} mm  {'← moving' if axis == 'y' else '(fixed)'}")
    print(f"  Z: {pts[0][2]:.1f} mm  {'← moving' if axis == 'z' else '(fixed)'}")
    print(f"  Range: {moving_vals[0]:.1f} → {moving_vals[-1]:.1f} mm\n")

    for i, (x, y, z) in enumerate(pts):
        cmd   = build_command(x, y, z, speed)
        val   = moving_vals[i]
        label = f"Step {i:02d}/{len(pts)-1}  {axis}={val:6.1f}mm"
        if dry_run:
            print(f"[dry-run] {label}  →  {cmd}")
        else:
            ser.write((cmd + "\n").encode())
            ser.flush()
            print(f"Sent     {label}")

        if i < len(pts) - 1:
            time.sleep(delay)

    print("\nDone.")
    if ser:
        ser.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace a straight line along X, Y, or Z with the RoArm-M2-S")
    parser.add_argument("--axis",    choices=["x", "y", "z"], default="z",
                        help="Axis to move along (default: z)")
    parser.add_argument("--x",       type=float, default=150,
                        help="Fixed X in mm, or start of X trace (default: 150)")
    parser.add_argument("--y",       type=float, default=0,
                        help="Fixed Y in mm, or start of Y trace (default: 0)")
    parser.add_argument("--z",       type=float, default=80,
                        help="Fixed Z in mm, or start of Z trace (default: 80)")
    parser.add_argument("--x-start", type=float, default=None,
                        help="X start when --axis x (default: --x value)")
    parser.add_argument("--x-end",   type=float, default=250,
                        help="X end when --axis x (default: 250)")
    parser.add_argument("--y-start", type=float, default=None,
                        help="Y start when --axis y (default: --y value)")
    parser.add_argument("--y-end",   type=float, default=100,
                        help="Y end when --axis y (default: 100)")
    parser.add_argument("--z-start", type=float, default=None,
                        help="Z start when --axis z (default: --z value)")
    parser.add_argument("--z-end",   type=float, default=200,
                        help="Z end when --axis z (default: 200)")
    parser.add_argument("--steps",   type=int,   default=10,
                        help="Number of steps (default: 10)")
    parser.add_argument("--speed",   type=int,   default=300,
                        help="Arm speed (default: 300)")
    parser.add_argument("--delay",   type=float, default=1.0,
                        help="Seconds between steps (default: 1.0)")
    parser.add_argument("--port",    type=str,   default="/dev/ttyUSB0",
                        help="Serial port (default: /dev/ttyUSB0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without sending to hardware")
    parser.add_argument("--plot",    action="store_true",
                        help="Show matplotlib trajectory preview")
    args = parser.parse_args()

    # Resolve start/end for the chosen axis
    starts = {"x": args.x_start if args.x_start is not None else args.x,
              "y": args.y_start if args.y_start is not None else args.y,
              "z": args.z_start if args.z_start is not None else args.z}
    ends   = {"x": args.x_end, "y": args.y_end, "z": args.z_end}

    pts = generate_waypoints(args.axis, args.x, args.y, args.z,
                             starts[args.axis], ends[args.axis], args.steps)

    if args.plot:
        plot_trajectory(pts, args.axis)

    run(pts, args.axis, args.speed, args.delay, args.port, args.dry_run)


if __name__ == "__main__":
    main()
