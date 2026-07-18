#!/usr/bin/env python3
"""
Sweep the RoArm-M2-S end-effector through a grid of XYZ positions.

Steps:
  1. Move to home XYZ position
  2. Sweep X then Y then Z axes independently, or run a full XY grid

Sweep modes (--mode):
  x     Sweep X from X_MIN to X_MAX at fixed Y, Z
  y     Sweep Y from Y_MIN to Y_MAX at fixed X, Z
  z     Sweep Z from Z_MIN to Z_MAX at fixed X, Y
  xy    Sweep a 2-D grid: for each X step, sweep full Y range

Usage:
    python experimental/arm_xyz_sweep.py
    python experimental/arm_xyz_sweep.py --mode y
    python experimental/arm_xyz_sweep.py --mode xy --x-step 20 --y-step 15
    python experimental/arm_xyz_sweep.py --mode z --z-min 40 --z-max 150
    python experimental/arm_xyz_sweep.py --port /dev/ttyUSB1 --spd 300
"""

import argparse
import json
import time
import serial

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE   = 115200

# Workspace bounds (mm) — matches arm_controller.py
X_MIN, X_MAX =  50, 300
Y_MIN, Y_MAX = -150, 150
Z_MIN, Z_MAX =  30, 300

# Home position — centred, mid-height
HOME_X =  0
HOME_Y =  0
HOME_Z =  0

WRIST_ANGLE = 0   # keep wrist flat throughout


def send(ser: serial.Serial, cmd: dict) -> None:
    line = json.dumps(cmd) + "\n"
    ser.write(line.encode())
    print(f"TX: {line.strip()}")
    time.sleep(0.05)


def move_xyz(ser: serial.Serial, x: float, y: float, z: float, spd: int) -> None:
    send(ser, {"T": 1041, "x": x, "y": y, "z": z, "t": WRIST_ANGLE, "spd": spd})


def home(ser: serial.Serial, spd: int, dwell: float = 3.0) -> None:
    print(f"\n-- Homing to ({HOME_X}, {HOME_Y}, {HOME_Z}) --")
    move_xyz(ser, HOME_X, HOME_Y, HOME_Z, spd)
    time.sleep(dwell)


def _dwell(spd: int, distance_mm: float, step_mm: float) -> float:
    """Estimate how long to wait for the arm to reach the next point."""
    if spd <= 0:
        return 1.5
    # spd units from Waveshare docs are mm/s for T:1041
    travel_s = step_mm / spd
    return max(0.5, travel_s * 1.5)


# ── Sweep functions ───────────────────────────────────────────────────────────

def sweep_x(ser, args) -> None:
    print(f"\n-- X sweep {args.x_min} → {args.x_max} mm (step {args.x_step}) --")
    x = float(args.x_min)
    while x <= args.x_max:
        move_xyz(ser, x, args.y_fixed, args.z_fixed, args.spd)
        time.sleep(_dwell(args.spd, abs(args.x_max - args.x_min), args.x_step))
        x += args.x_step


def sweep_y(ser, args) -> None:
    print(f"\n-- Y sweep {args.y_min} → {args.y_max} mm (step {args.y_step}) --")
    y = float(args.y_min)
    while y <= args.y_max:
        move_xyz(ser, args.x_fixed, y, args.z_fixed, args.spd)
        time.sleep(_dwell(args.spd, abs(args.y_max - args.y_min), args.y_step))
        y += args.y_step


def sweep_z(ser, args) -> None:
    print(f"\n-- Z sweep {args.z_min} → {args.z_max} mm (step {args.z_step}) --")
    z = float(args.z_min)
    while z <= args.z_max:
        move_xyz(ser, args.x_fixed, args.y_fixed, z, args.spd)
        time.sleep(_dwell(args.spd, abs(args.z_max - args.z_min), args.z_step))
        z += args.z_step


def sweep_xy(ser, args) -> None:
    print(f"\n-- XY grid sweep X:[{args.x_min}→{args.x_max}] Y:[{args.y_min}→{args.y_max}] --")
    x = float(args.x_min)
    while x <= args.x_max:
        y = float(args.y_min)
        while y <= args.y_max:
            move_xyz(ser, x, y, args.z_fixed, args.spd)
            time.sleep(_dwell(args.spd, args.y_step * 2, args.y_step))
            y += args.y_step
        x += args.x_step


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sweep RoArm-M2-S end-effector in XYZ")
    parser.add_argument("--port",    default=SERIAL_PORT)
    parser.add_argument("--mode",    choices=["x", "y", "z", "xy"], default="x",
                        help="Sweep axis/mode (default: x)")
    parser.add_argument("--spd",     type=int,   default=200,
                        help="Speed in mm/s for T:1041 (default 200)")

    # Axis ranges
    parser.add_argument("--x-min",   type=float, default=X_MIN)
    parser.add_argument("--x-max",   type=float, default=X_MAX)
    parser.add_argument("--x-step",  type=float, default=20)
    parser.add_argument("--y-min",   type=float, default=Y_MIN)
    parser.add_argument("--y-max",   type=float, default=Y_MAX)
    parser.add_argument("--y-step",  type=float, default=20)
    parser.add_argument("--z-min",   type=float, default=Z_MIN)
    parser.add_argument("--z-max",   type=float, default=Z_MAX)
    parser.add_argument("--z-step",  type=float, default=20)

    # Fixed-axis values used when that axis is not being swept
    parser.add_argument("--x-fixed", type=float, default=HOME_X,
                        help="X position held fixed in y/z sweeps")
    parser.add_argument("--y-fixed", type=float, default=HOME_Y,
                        help="Y position held fixed in x/z sweeps")
    parser.add_argument("--z-fixed", type=float, default=HOME_Z,
                        help="Z position held fixed in x/y sweeps")

    args = parser.parse_args()

    print(f"Opening serial {args.port} @ {BAUD_RATE}")
    ser = serial.Serial(args.port, BAUD_RATE, timeout=1)
    time.sleep(2)   # let ESP32 boot after serial open

    try:
        home(ser, args.spd)

        if args.mode == "x":
            sweep_x(ser, args)
        elif args.mode == "y":
            sweep_y(ser, args)
        elif args.mode == "z":
            sweep_z(ser, args)
        elif args.mode == "xy":
            sweep_xy(ser, args)

        print("\n-- Sweep complete. Returning to home --")
        home(ser, args.spd)
    finally:
        ser.close()
        print("Serial port closed.")


if __name__ == "__main__":
    main()
