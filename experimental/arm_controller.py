#!/usr/bin/env python3
"""
RoArm-M2-S controller — sends JSON commands over serial.

Pixel-to-arm coordinate mapping:
    The camera sits at the arm base level, pointing forward/down at the floor.
    A detected red dot at pixel (px, py) is mapped to arm XYZ using a simple
    pinhole projection — requires CAMERA_HEIGHT_MM (camera above floor) and
    FOCAL_LENGTH_PX (camera focal length in pixels).

    These constants need to be measured/calibrated for your setup.

Usage (standalone test):
    python arm_controller.py --x 150 --y 0 --z 50

Usage (from arm_validator.py):
    from arm_controller import ArmController
    arm = ArmController()
    arm.move_to(x, y, z)
"""

import argparse
import json
import time
import serial
from serial import Serial

# ── Serial config ─────────────────────────────────────────────────────────────
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE   = 115200

# ── Arm workspace (mm) — adjust from Waveshare spec ──────────────────────────
ARM_X_MIN, ARM_X_MAX =   50, 410   # forward reach from base
ARM_Y_MIN, ARM_Y_MAX = -200, 200   # lateral (left/right)
ARM_Z_MIN, ARM_Z_MAX =   30, 300   # height above base

# ── Camera calibration — measure these for your setup ────────────────────────
CAMERA_HEIGHT_MM = 533   # camera height above the floor in mm
# Focal lengths derived empirically from known arm position vs pixel offset:
#   dot4 pixel=(654,560) centre=(1280,720) → dx=-626 dy=-160 → arm x=356 y=-272
#   scale_x = 356/160 = 2.225  (dy_px → arm x)
#   scale_y = 272/626 = 0.435  (dx_px → arm y, negated)
FOCAL_LENGTH_X_PX = 239   # 533 / 2.225  (controls dy_px → arm x)
FOCAL_LENGTH_Y_PX = 1224  # 533 / 0.435  (controls dx_px → arm y)
IMAGE_W           = 2560
IMAGE_H           = 1440

# Wrist angle held flat while moving to a position
WRIST_ANGLE = 0


class ArmController:
    def __init__(self, port: str = SERIAL_PORT, baud: int = BAUD_RATE):
        self._ser = Serial(port, baud, timeout=1)
        time.sleep(2)   # allow ESP32 to boot after serial open

    def move_to(self, x: float, y: float, z: float, speed: int = 500) -> None:
        """
        Move arm end-effector to (x, y, z) in mm relative to arm base.
        speed: movement speed (check Waveshare docs for units/range).
        """
        x = float(_clamp(x, ARM_X_MIN, ARM_X_MAX))
        y = float(_clamp(y, ARM_Y_MIN, ARM_Y_MAX))
        z = float(_clamp(z, ARM_Z_MIN, ARM_Z_MAX))

        cmd = json.dumps({
            "T": 1041,
            "x": x,
            "y": y,
            "z": z,
            "t": WRIST_ANGLE,
            "spd": speed,
        })
        self._ser.write((cmd + "\n").encode())
        print(f"ARM → x={x:.1f} y={y:.1f} z={z:.1f}")

    def close(self):
        self._ser.close()


def pixel_to_arm_xyz(px: int, py: int,
                     image_w: int = IMAGE_W,
                     image_h: int = IMAGE_H) -> tuple[float, float, float]:
    dx_px = px - image_w / 2
    dy_px = py - image_h / 2

    x_mm = -dy_px * (CAMERA_HEIGHT_MM / FOCAL_LENGTH_X_PX)
    y_mm =  dx_px * (CAMERA_HEIGHT_MM / FOCAL_LENGTH_Y_PX)
    z_mm = ARM_Z_MIN

    return x_mm, y_mm, z_mm


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    parser = argparse.ArgumentParser(description="Send a single move command to RoArm-M2-S")
    parser.add_argument("--x",    type=float, default=150)
    parser.add_argument("--y",    type=float, default=0)
    parser.add_argument("--z",    type=float, default=80)
    parser.add_argument("--port", type=str,   default=SERIAL_PORT)
    args = parser.parse_args()

    arm = ArmController(port=args.port)
    arm.move_to(args.x, args.y, args.z)
    time.sleep(2)
    arm.close()


if __name__ == "__main__":
    main()
