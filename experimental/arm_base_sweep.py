#!/usr/bin/env python3
"""
Sweep the RoArm-M2-S base joint from -180° to +180°.

Steps:
  1. Move all joints to default/home position (T:122  b=0 s=0 e=90 h=180)
  2. Rotate only the base from -180° to +180° in configurable increments

Usage:
    python arm_base_sweep.py [--port /dev/ttyUSB0] [--step 10] [--spd 30] [--acc 10]
"""

import argparse
import json
import time
import serial

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE   = 115200

# Default joint angles (degrees) — from CMD_JOINTS_ANGLE_CTRL docs
DEFAULT_BASE     =   0
DEFAULT_SHOULDER =   0
DEFAULT_ELBOW    =  90
DEFAULT_EOAT     = 180   # closed (max = 180°, open = 45°)


def send(ser: serial.Serial, cmd: dict) -> None:
    line = json.dumps(cmd) + "\n"
    ser.write(line.encode())
    print(f"TX: {line.strip()}")
    time.sleep(0.05)   # give the controller a moment to parse


def home(ser: serial.Serial, spd: int, acc: int) -> None:
    """Move all joints to default position using CMD_JOINTS_ANGLE_CTRL (T:122)."""
    print("\n-- Homing all joints --")
    send(ser, {
        "T": 122,
        "b": DEFAULT_BASE,
        "s": DEFAULT_SHOULDER,
        "e": DEFAULT_ELBOW,
        "h": DEFAULT_EOAT,
        "spd": spd,
        "acc": acc,
    })
    # Wait for the arm to reach home before starting sweep
    time.sleep(3)


def sweep_base(ser: serial.Serial, step: int, spd: int, acc: int) -> None:
    """
    Rotate only the base joint from -180° to +180° in `step`-degree increments
    using CMD_SINGLE_JOINT_ANGLE (T:121, joint=1).
    """
    print("\n-- Starting base sweep -180° → +180° --")
    angle = -180
    while angle <= 180:
        send(ser, {
            "T": 121,
            "joint": 1,        # BASE_JOINT
            "angle": angle,
            "spd": spd,
            "acc": acc,
        })
        # Wait long enough for the servo to reach the target before next step
        time.sleep(max(1.0, step / spd * 1.5) if spd > 0 else 1.5)
        angle += step

    print("\n-- Sweep complete. Returning to home --")
    home(ser, spd, acc)


def main():
    parser = argparse.ArgumentParser(description="Sweep RoArm-M2-S base joint -180° to +180°")
    parser.add_argument("--port", default=SERIAL_PORT, help="Serial port")
    parser.add_argument("--step", type=int,   default=10,  help="Degrees per step (default 10)")
    parser.add_argument("--spd",  type=int,   default=30,  help="Speed in °/s (default 30)")
    parser.add_argument("--acc",  type=int,   default=10,  help="Acceleration in °/s² (default 10)")
    args = parser.parse_args()

    print(f"Opening serial port {args.port} at {BAUD_RATE} baud")
    ser = serial.Serial(args.port, BAUD_RATE, timeout=1)
    time.sleep(2)   # allow ESP32 to boot after serial open

    try:
        home(ser, args.spd, args.acc)
        sweep_base(ser, args.step, args.spd, args.acc)
    finally:
        ser.close()
        print("Serial port closed.")


if __name__ == "__main__":
    main()
