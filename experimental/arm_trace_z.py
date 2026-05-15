#!/usr/bin/env python3
"""
arm_trace_z.py — Trace a straight vertical line (Z-axis) with the RoArm-M2-S.

Keeps X and Y constant (no base rotation).  Only the shoulder and elbow
joints move as the arm steps from z_start to z_end.

Sends the Waveshare T=1041 Cartesian move command for each step.

Usage
─────
    # Dry-run: print commands only, no serial
    python experimental/arm_trace_z.py --dry-run

    # Move from z=80 to z=200 in 10 steps at x=150 y=0
    python experimental/arm_trace_z.py --port /dev/ttyUSB0 \\
        --x 150 --y 0 --z-start 80 --z-end 200 --steps 10

    # Go back down after reaching the top
    python experimental/arm_trace_z.py --port /dev/ttyUSB0 \\
        --x 150 --y 0 --z-start 200 --z-end 80 --steps 10

Options
───────
    --x          Fixed X position in mm (default: 150)
    --y          Fixed Y position in mm (default: 0)
    --z-start    Starting Z in mm (default: 80)
    --z-end      Ending Z in mm (default: 200)
    --steps      Number of steps along the line (default: 10)
    --speed      Arm movement speed (default: 300)
    --delay      Seconds to wait between steps (default: 1.0)
    --port       Serial port (default: /dev/ttyUSB0)
    --dry-run    Print commands without opening serial port
    --plot       Show matplotlib preview of the trajectory
"""

import argparse
import json
import sys
import time

# Arm workspace limits (mm)
ARM_X_MIN, ARM_X_MAX =  50, 300
ARM_Y_MIN, ARM_Y_MAX = -200, 200
ARM_Z_MIN, ARM_Z_MAX =  30, 300

WRIST_ANGLE = 0   # keep wrist flat throughout


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


def generate_waypoints(x: float, y: float,
                        z_start: float, z_end: float,
                        steps: int) -> list[tuple[float, float, float]]:
    """Return a list of (x, y, z) positions along the vertical line."""
    pts = []
    for i in range(steps + 1):
        t = i / steps
        z = z_start + (z_end - z_start) * t
        # Clamp to workspace
        xc = _clamp(x,  ARM_X_MIN, ARM_X_MAX)
        yc = _clamp(y,  ARM_Y_MIN, ARM_Y_MAX)
        zc = _clamp(z,  ARM_Z_MIN, ARM_Z_MAX)
        if xc != x or yc != y or zc != z:
            print(f"[warn] step {i}: clamped ({x:.1f},{y:.1f},{z:.1f}) "
                  f"→ ({xc:.1f},{yc:.1f},{zc:.1f})")
        pts.append((xc, yc, zc))
    return pts


def plot_trajectory(pts: list[tuple[float, float, float]],
                    x_fixed: float, y_fixed: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed — skipping plot")
        return

    zs = [p[2] for p in pts]
    xs = [p[0] for p in pts]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Side view: X vs Z
    axes[0].plot(xs, zs, 'o-', color='steelblue', markersize=8)
    axes[0].axvline(x_fixed, color='gray', linestyle='--', alpha=0.5, label=f'x={x_fixed}')
    axes[0].set_xlabel('X (mm)')
    axes[0].set_ylabel('Z (mm)')
    axes[0].set_title('Side view (X–Z plane)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    for i, (x, _, z) in enumerate(pts):
        axes[0].annotate(str(i), (x, z), textcoords='offset points',
                         xytext=(6, 4), fontsize=8)

    # Z profile over steps
    axes[1].plot(range(len(zs)), zs, 's-', color='tomato', markersize=8)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Z (mm)')
    axes[1].set_title(f'Z profile  (x={x_fixed}, y={y_fixed} constant)')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('arm_trace_z — vertical trajectory preview', fontsize=12)
    plt.tight_layout()
    plt.show()


def run(pts: list[tuple[float, float, float]],
        speed: int, delay: float, port: str, dry_run: bool) -> None:
    ser = None
    if not dry_run:
        try:
            import serial
            ser = serial.Serial(port, 115200, timeout=1)
            print(f"Serial opened: {port}")
            time.sleep(2)   # let ESP32 boot
        except Exception as e:
            sys.exit(f"Cannot open serial port {port}: {e}")

    print(f"\nTracing {len(pts)} points  speed={speed}  delay={delay}s")
    print(f"  X fixed: {pts[0][0]:.1f} mm")
    print(f"  Y fixed: {pts[0][1]:.1f} mm")
    print(f"  Z range: {pts[0][2]:.1f} → {pts[-1][2]:.1f} mm\n")

    for i, (x, y, z) in enumerate(pts):
        cmd = build_command(x, y, z, speed)
        label = f"Step {i:02d}/{len(pts)-1}  z={z:6.1f}mm"
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
        description="Trace a straight Z-axis line with the RoArm-M2-S")
    parser.add_argument("--x",       type=float, default=150,
                        help="Fixed X position in mm (default: 150)")
    parser.add_argument("--y",       type=float, default=0,
                        help="Fixed Y position in mm (default: 0)")
    parser.add_argument("--z-start", type=float, default=80,
                        help="Starting Z in mm (default: 80)")
    parser.add_argument("--z-end",   type=float, default=200,
                        help="Ending Z in mm (default: 200)")
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

    pts = generate_waypoints(args.x, args.y, args.z_start, args.z_end, args.steps)

    if args.plot:
        plot_trajectory(pts, args.x, args.y)

    run(pts, args.speed, args.delay, args.port, args.dry_run)


if __name__ == "__main__":
    main()
