#!/usr/bin/env python3
"""
Simple Roomba controller using the iRobot Open Interface (OI),
inspired by https://github.com/martinschaef/roomba

Requirements:
- Python 3.7+
- pyserial (`pip install pyserial`)

Examples:
- Start a clean:        python roomba_control.py --port /dev/ttyUSB0 clean
- Spot clean:           python roomba_control.py --port /dev/ttyUSB0 spot
- Seek dock:            python roomba_control.py --port /dev/ttyUSB0 dock
- Beep:                 python roomba_control.py --port /dev/ttyUSB0 beep
- Drive forward:        python roomba_control.py --port /dev/ttyUSB0 drive --velocity 200 --duration 2
- Rotate in place left: python roomba_control.py --port /dev/ttyUSB0 drive --velocity 100 --radius left --duration 1.5

Notes:
- Default baud is 115200 (many models). If your Roomba uses 57600, pass `--baud 57600`.
- Some commands require SAFE or FULL mode. This tool enters SAFE by default.
"""

import argparse
import sys
import time
from contextlib import contextmanager

try:
    import serial  # type: ignore
except Exception as e:  # pragma: no cover
    serial = None


class Roomba:
    # OI opcodes
    START = 128
    BAUD = 129
    CONTROL = 130  # legacy; prefer SAFE/FULL
    SAFE = 131
    FULL = 132
    POWER = 133
    SPOT = 134
    CLEAN = 135
    MAX = 136
    DRIVE = 137
    MOTORS = 138
    LEDS = 139
    SONG = 140
    PLAY = 141
    SEEK_DOCK = 143
    DRIVE_DIRECT = 145
    DRIVE_PWM = 146

    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.1):
        if serial is None:
            raise RuntimeError("pyserial is required. Install with: pip install pyserial")
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = None

    @contextmanager
    def connect(self):
        self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        # Give the serial interface a moment to settle
        time.sleep(0.1)
        try:
            yield self
        finally:
            try:
                if self.ser:
                    self.ser.flush()
                    self.ser.close()
            except Exception:
                pass

    def _send(self, *bytes_seq: int):
        if not self.ser:
            raise RuntimeError("Serial port not open. Use connect().")
        data = bytes(bytes_seq)
        self.ser.write(data)
        self.ser.flush()
        # Small delay to allow the Roomba to process commands
        time.sleep(0.01)

    # High-level helpers
    def start(self):
        self._send(self.START)

    def safe(self):
        self._send(self.SAFE)

    def full(self):
        self._send(self.FULL)

    def clean(self):
        self._send(self.CLEAN)

    def spot(self):
        self._send(self.SPOT)

    def dock(self):
        self._send(self.SEEK_DOCK)

    def power_off(self):
        self._send(self.POWER)

    def motors(self, main: bool = False, vacuum: bool = False, side: bool = False):
        # See OI spec for bit layout: main brush (bit 2), vacuum (bit 1), side brush (bit 0)
        bits = (4 if main else 0) | (2 if vacuum else 0) | (1 if side else 0)
        self._send(self.MOTORS, bits)

    def leds(self, status_color: int = 0, status_intensity: int = 0, check_robot: bool = False, dock: bool = False, spot: bool = False, debris: bool = False):
        # LEDs opcode layout varies by model. This sets the 4 status LEDs and the built-in status color/intensity.
        led_bits = (8 if check_robot else 0) | (4 if dock else 0) | (2 if spot else 0) | (1 if debris else 0)
        color = max(0, min(255, status_color))
        intensity = max(0, min(255, status_intensity))
        self._send(self.LEDS, led_bits, color, intensity)

    def song(self, song_number: int, notes: list[tuple[int, int]]):
        # notes: list of (note, duration_1/64s). Max notes depends on model, typically up to 16.
        count = len(notes)
        payload = [self.SONG, song_number, count]
        for note, dur in notes:
            payload.append(max(31, min(127, int(note))))  # MIDI note number range (approx.)
            payload.append(max(1, min(255, int(dur))))
        self._send(*payload)

    def play(self, song_number: int):
        self._send(self.PLAY, song_number)

    def drive(self, velocity_mm_s: int, radius_mm: int):
        # velocity: -500..500, radius: -2000..2000, with special cases: straight=0x8000, in-place cw=-1, ccw=1
        v = max(-500, min(500, int(velocity_mm_s)))
        r = int(radius_mm)
        vh = (v >> 8) & 0xFF
        vl = v & 0xFF
        rh = (r >> 8) & 0xFF
        rl = r & 0xFF
        self._send(self.DRIVE, vh, vl, rh, rl)

    def drive_special(self, velocity_mm_s: int, radius: str):
        radius = radius.lower()
        if radius in ("straight", "s"):
            r = 0x8000  # straight
        elif radius in ("left", "l", "ccw"):
            r = 1  # in place CCW
        elif radius in ("right", "r", "cw"):
            r = -1  # in place CW
        else:
            raise ValueError("Unknown radius keyword. Use straight|left|right or provide --radius integer.")
        self.drive(velocity_mm_s, r)


def parse_args(argv):
    p = argparse.ArgumentParser(description="Simple Roomba controller (serial OI)")
    p.add_argument("--port", required=True, help="Serial port, e.g., /dev/ttyUSB0 or COM3")
    p.add_argument("--baud", type=int, default=115200, help="Baud rate (e.g., 115200 or 57600)")
    p.add_argument("--no-start", action="store_true", help="Skip sending START+SAFE (if already in correct mode)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("clean", help="Start cleaning")
    sub.add_parser("spot", help="Spot clean")
    sub.add_parser("dock", help="Seek dock")
    sub.add_parser("safe", help="Enter SAFE mode")
    sub.add_parser("full", help="Enter FULL mode")
    sub.add_parser("poweroff", help="Power off the robot")

    sp_beep = sub.add_parser("beep", help="Play a short beep via SONG/PLAY")
    sp_beep.add_argument("--note", type=int, default=69, help="MIDI note (A4=69)")
    sp_beep.add_argument("--dur", type=int, default=16, help="Duration in 1/64s (16 ~= 0.25s)")

    sp_drive = sub.add_parser("drive", help="Drive with velocity and radius")
    sp_drive.add_argument("--velocity", type=int, required=True, help="Velocity mm/s (-500..500)")
    group = sp_drive.add_mutually_exclusive_group(required=False)
    group.add_argument("--radius", type=int, help="Turn radius mm (-2000..2000); special: -1=CW, 1=CCW, 32768=straight")
    group.add_argument("--turn", choices=["left", "right", "straight"], help="Use special radius keywords")
    sp_drive.add_argument("--duration", type=float, default=0.0, help="Optional duration in seconds, then stop")

    sp_vac_on = sub.add_parser("vacuum-on", help="Turn vacuum on (motors)")
    sp_vac_on.add_argument("--main", action="store_true", help="Also enable main brush")
    sp_vac_on.add_argument("--side", action="store_true", help="Also enable side brush")

    sub.add_parser("vacuum-off", help="Turn all cleaning motors off")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    rb = Roomba(args.port, baud=args.baud)

    with rb.connect():
        if not args.no_start:
            rb.start()
            # Some models prefer a small delay after START before SAFE
            time.sleep(0.05)
            rb.safe()

        if args.cmd == "clean":
            rb.clean()
        elif args.cmd == "spot":
            rb.spot()
        elif args.cmd == "dock":
            rb.dock()
        elif args.cmd == "safe":
            rb.safe()
        elif args.cmd == "full":
            rb.full()
        elif args.cmd == "poweroff":
            rb.power_off()
        elif args.cmd == "beep":
            # Load a simple 1-note song and play it
            rb.song(0, [(args.note, args.dur)])
            time.sleep(0.02)
            rb.play(0)
        elif args.cmd == "drive":
            if args.turn:
                rb.drive_special(args.velocity, args.turn)
            else:
                radius = args.radius if args.radius is not None else 0x8000
                rb.drive(args.velocity, radius)
            if args.duration and args.duration > 0:
                time.sleep(args.duration)
                # Stop motion by setting velocity=0
                rb.drive(0, 0x8000)
        elif args.cmd == "vacuum-on":
            rb.motors(main=args.main, vacuum=True, side=args.side)
        elif args.cmd == "vacuum-off":
            rb.motors(main=False, vacuum=False, side=False)
        else:
            raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

