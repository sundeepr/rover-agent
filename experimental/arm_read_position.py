#!/usr/bin/env python3
"""
arm_read_position.py — Report the RoArm-M2-S's current position.

Sends CMD_SERVO_RAD_FEEDBACK (T:105) and prints the T:1051 response: the
end-effector XYZ in mm plus the base/shoulder/elbow/wrist joint angles.

Usage
─────
    # One reading:
    python experimental/arm_read_position.py

    # Live watch, refreshing 5x a second until Ctrl-C:
    python experimental/arm_read_position.py --watch --interval 0.2

    # Machine-readable, for piping into other tools:
    python experimental/arm_read_position.py --json
"""

import argparse
import json
import math
import sys
import time

import serial

SERIAL_PORT      = "/dev/ttyUSB0"
BAUD_RATE        = 115200
FEEDBACK_COMMAND = {"T": 105}
FEEDBACK_REPLY_T = 1051
FEEDBACK_TIMEOUT_S = 1.0

# Joint keys in the T:1051 reply, in kinematic order, with display names.
JOINTS = [("b", "base"), ("s", "shoulder"), ("e", "elbow"), ("t", "wrist")]


def request_feedback(ser: serial.Serial,
                     timeout: float = FEEDBACK_TIMEOUT_S) -> dict | None:
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


def format_feedback(fb: dict) -> str:
    def mm(key):
        v = fb.get(key)
        return f"{float(v):8.1f}" if isinstance(v, (int, float)) else "     n/a"

    lines = [
        "End-effector (mm)",
        f"  X: {mm('x')}",
        f"  Y: {mm('y')}",
        f"  Z: {mm('z')}",
        "Joints (rad / deg)",
    ]
    for key, name in JOINTS:
        v = fb.get(key)
        if isinstance(v, (int, float)):
            lines.append(f"  {name:<9} {float(v):7.4f} / {math.degrees(float(v)):7.2f}°")
        else:
            lines.append(f"  {name:<9}     n/a")

    extras = {k: v for k, v in fb.items()
              if k not in {"T", "x", "y", "z"} and k not in dict(JOINTS)}
    if extras:
        lines.append("Other")
        for k, v in sorted(extras.items()):
            lines.append(f"  {k:<9} {v}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read the current RoArm-M2-S position over serial")
    parser.add_argument("--port",     type=str,   default=SERIAL_PORT,
                        help=f"Serial port (default: {SERIAL_PORT})")
    parser.add_argument("--baud",     type=int,   default=BAUD_RATE,
                        help=f"Baud rate (default: {BAUD_RATE})")
    parser.add_argument("--watch",    action="store_true",
                        help="Poll continuously until Ctrl-C")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Seconds between polls when --watch (default: 0.5)")
    parser.add_argument("--timeout",  type=float, default=FEEDBACK_TIMEOUT_S,
                        help=f"Feedback wait in seconds (default: {FEEDBACK_TIMEOUT_S})")
    parser.add_argument("--json",     action="store_true",
                        help="Print the raw T:1051 reply as JSON, one per line")
    parser.add_argument("--no-wait",  action="store_true",
                        help="Skip the 2s ESP32 boot delay after opening the port")
    args = parser.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except Exception as e:
        sys.exit(f"Cannot open {args.port}: {e}")

    try:
        if not args.no_wait:
            time.sleep(2)   # allow ESP32 to boot after serial open
        ser.reset_input_buffer()

        while True:
            fb = request_feedback(ser, args.timeout)
            if fb is None:
                print("[warn] no feedback within timeout", file=sys.stderr)
            elif args.json:
                print(json.dumps(fb), flush=True)
            else:
                if args.watch:
                    print(f"\n─── {time.strftime('%H:%M:%S')} ───")
                print(format_feedback(fb), flush=True)

            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


if __name__ == "__main__":
    main()
