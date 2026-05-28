#!/usr/bin/env python3
"""
Quick serial port sniffer — just prints everything the Atlas STM32 sends.

Run this BEFORE updating calibrate.py so we can see the exact IMU
data format coming from the rover.

Usage:
    python calibration/read_serial.py --port /dev/ttyACM0
    python calibration/read_serial.py --port /dev/ttyACM0 --baud 115200
"""

import argparse
import sys
import time
import serial

def main():
    parser = argparse.ArgumentParser(description="Print raw serial data from Atlas")
    parser.add_argument("--port",  default="/dev/ttyACM0")
    parser.add_argument("--baud",  type=int, default=115200)
    parser.add_argument("--lines", type=int, default=50,
                        help="Number of lines to capture (default 50, 0 = infinite)")
    args = parser.parse_args()

    print(f"Opening {args.port} @ {args.baud} baud…")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1.0)
    except serial.SerialException as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    time.sleep(0.3)   # let port settle
    ser.reset_input_buffer()
    print(f"Connected. Printing incoming data (Ctrl-C to stop):\n{'─'*60}")

    count = 0
    try:
        while True:
            raw = ser.readline()
            if raw:
                # Print both hex and decoded string so we can see the format
                try:
                    decoded = raw.decode("ascii", errors="replace").rstrip()
                except Exception:
                    decoded = repr(raw)
                print(f"[{count:04d}] {decoded}")
                print(f"       hex: {raw.hex()}")
                count += 1
                if args.lines > 0 and count >= args.lines:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print(f"\n{'─'*60}")
        print(f"Captured {count} lines.")

if __name__ == "__main__":
    main()
