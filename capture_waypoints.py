#!/usr/bin/env python3
"""
capture_waypoints.py — drive the rover around the path you want and this
automatically lays down waypoints as you go, writing lines in the format
gps_waypoint_follow.py expects:

    lat,lon

By default it runs in AUTO mode: every ~1s it checks the current GPS fix
and appends a new waypoint once the rover has moved at least
--min-distance-m (default 3m) from the last one. Stand still and nothing
new gets written — no button presses needed while driving. Stop with
Ctrl-C at the end of the path.

--manual switches to the old press-Enter-to-mark-a-point workflow, for
when you want to hand-place specific waypoints instead of recording a
continuous trail.

Reuses the same GpsReader (GPGGA parsing) that gps_waypoint_follow.py
uses, so position quality/behavior matches exactly between capture and
playback.

Usage
─────
    python capture_waypoints.py waypoints.txt                       # auto, drive around
    python capture_waypoints.py waypoints.txt --min-distance-m 5
    python capture_waypoints.py waypoints.txt --manual               # press Enter to mark points
    # Ctrl-C to stop and save at any point.
"""

import argparse
import logging
import threading
import time
from pathlib import Path

from gps_waypoint_follow import (
    GpsReader,
    haversine_distance_m,
    _DEFAULT_GPS_BAUD,
    _DEFAULT_GPS_PORT,
)

log = logging.getLogger("rover.capture_waypoints")

_DISPLAY_PERIOD_S  = 1.0
_AUTO_POLL_S       = 0.5
_FIX_STALE_S       = 3.0
_DEFAULT_MIN_DISTANCE_M = 3.0


def _status_line(fix) -> str:
    if fix.fix_quality == 0:
        return "waiting for GPS fix...                                   "
    age_s = time.monotonic() - fix.updated_at
    return (f"fix={fix.fix_quality} lat={fix.lat:.6f} lon={fix.lon:.6f} "
            f"age={age_s:4.1f}s                 ")


def _usable_fix(fix) -> bool:
    if fix.fix_quality == 0 or fix.updated_at == 0.0:
        return False
    return (time.monotonic() - fix.updated_at) <= _FIX_STALE_S


def _run_auto(gps: GpsReader, f, min_distance_m: float) -> list[tuple[float, float]]:
    captured: list[tuple[float, float]] = []
    last: tuple[float, float] | None = None

    print(f"Auto-capturing every >= {min_distance_m}m moved. Drive the path now. Ctrl-C to stop.\n")

    try:
        while True:
            fix = gps.get()
            print(f"\r  {_status_line(fix)} | {len(captured)} captured", end="", flush=True)

            if _usable_fix(fix):
                if last is None or haversine_distance_m(last[0], last[1], fix.lat, fix.lon) >= min_distance_m:
                    f.write(f"{fix.lat:.7f},{fix.lon:.7f}\n")
                    f.flush()
                    last = (fix.lat, fix.lon)
                    captured.append(last)
                    print(f"\n  captured waypoint {len(captured)}: {fix.lat:.7f}, {fix.lon:.7f}")

            time.sleep(_AUTO_POLL_S)
    except (KeyboardInterrupt, EOFError):
        pass

    return captured


def _run_manual(gps: GpsReader, f) -> list[tuple[float, float]]:
    captured: list[tuple[float, float]] = []
    offsets: list[int] = []  # file offset before each write, for undo
    stop_event = threading.Event()

    def _display_loop() -> None:
        while not stop_event.is_set():
            print(f"\r  {_status_line(gps.get())}", end="", flush=True)
            time.sleep(_DISPLAY_PERIOD_S)

    display_thread = threading.Thread(target=_display_loop, daemon=True, name="waypoint-display")
    display_thread.start()

    print("Enter = capture waypoint, u = undo last, q = quit\n")

    try:
        while True:
            cmd = input().strip().lower()

            if cmd == "q":
                break

            elif cmd == "u":
                if captured:
                    removed = captured.pop()
                    f.truncate(offsets.pop())
                    f.seek(0, 2)  # SEEK_END
                    print(f"\n  undone: {removed[0]:.6f}, {removed[1]:.6f} "
                          f"({len(captured)} waypoint(s) remaining)")
                else:
                    print("\n  nothing to undo")

            else:
                fix = gps.get()
                if not _usable_fix(fix):
                    print("\n  no usable GPS fix — not captured")
                    continue

                offsets.append(f.tell())
                f.write(f"{fix.lat:.7f},{fix.lon:.7f}\n")
                f.flush()
                captured.append((fix.lat, fix.lon))
                print(f"\n  captured waypoint {len(captured)}: {fix.lat:.7f}, {fix.lon:.7f}")

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        stop_event.set()

    return captured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_file", type=Path)
    parser.add_argument("--gps-port", default=_DEFAULT_GPS_PORT)
    parser.add_argument("--gps-baud", type=int, default=_DEFAULT_GPS_BAUD)
    parser.add_argument("--append", action="store_true",
                        help="Append to out_file instead of overwriting it")
    parser.add_argument("--manual", action="store_true",
                        help="Press Enter to mark each point instead of auto-recording a trail")
    parser.add_argument("--min-distance-m", type=float, default=_DEFAULT_MIN_DISTANCE_M,
                        help="Auto mode only: minimum distance between captured waypoints "
                             "(default: %(default)sm)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    file_mode = "a" if args.append and args.out_file.exists() else "w"

    gps = GpsReader(args.gps_port, args.gps_baud)
    gps.start()

    print(f"Writing waypoints to {args.out_file} ({'append' if file_mode == 'a' else 'overwrite'})")

    with open(args.out_file, file_mode) as f:
        if args.manual:
            captured = _run_manual(gps, f)
        else:
            captured = _run_auto(gps, f, args.min_distance_m)

    gps.stop()
    print(f"\n\nSaved {len(captured)} waypoint(s) to {args.out_file.resolve()}")


if __name__ == "__main__":
    main()
