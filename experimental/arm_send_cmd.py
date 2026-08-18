#!/usr/bin/env python3
"""
Send a raw JSON command to the Waveshare RoArm over serial and print any
response.

Defaults to {"T":301,"mode":0} (EOAT type / gripper mode select — mode 0).

Usage:
    python experimental/arm_send_cmd.py
    python experimental/arm_send_cmd.py --cmd '{"T":301,"mode":0}'
    python experimental/arm_send_cmd.py --port /dev/ttyUSB1 --cmd '{"T":105}'
"""

import argparse
import json
import time
import serial

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
DEFAULT_CMD = {"T": 301, "mode": 0}


def send_cmd(port: str, baud: int, cmd: dict, read_seconds: float = 2.0) -> None:
    print(f"Opening {port} @ {baud}…")
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2)  # allow ESP32 to boot after serial open

    payload = json.dumps(cmd) + "\n"
    print(f"Sending: {payload.strip()}")
    ser.write(payload.encode())

    print(f"Reading response ({read_seconds:.0f}s)…")
    deadline = time.time() + read_seconds
    got_any = False
    while time.time() < deadline:
        line = ser.readline()
        if line:
            got_any = True
            print(f"  RX: {line!r}")

    if not got_any:
        print("  No response — command was sent regardless")

    ser.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=SERIAL_PORT)
    parser.add_argument("--baud", type=int, default=BAUD_RATE)
    parser.add_argument(
        "--cmd",
        type=str,
        default=json.dumps(DEFAULT_CMD),
        help='JSON command to send, e.g. \'{"T":301,"mode":0}\'',
    )
    args = parser.parse_args()

    cmd = json.loads(args.cmd)
    send_cmd(args.port, args.baud, cmd)


if __name__ == "__main__":
    main()
