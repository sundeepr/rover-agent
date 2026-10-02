#!/usr/bin/env python3
"""
capture_waypoints.py — walk or drive the rover to each point you want in
the path, hit Enter, and this appends the current GPS fix as a waypoint
line to a file in the format gps_waypoint_follow.py expects:

    lat,lon

Reuses the same GpsReader (GPGGA parsing) that gps_waypoint_follow.py
uses, so position quality/behavior matches exactly between capture and
playback.

Usage
─────
    python capture_waypoints.py waypoints.txt
    python capture_waypoints.py waypoints.txt --gps-port /dev/ttyTHS1

While running:
    <Enter>   capture the current GPS fix as a waypoint
    u<Enter>  undo (remove) the last captured waypoint
    q<Enter>  quit

Live fix quality/lat/lon/satellite count is printed continuously in the
background so you can see whether you have a usable fix before you
capture a point near it.
"""

import argparse
import logging
import threading
import time
from pathlib import Path

from gps_waypoint_follow import GpsReader, _DEFAULT_GPS_BAUD, _DEFAULT_GPS_PORT

log = logging.getLogger("rover.capture_waypoints")

_DISPLAY_PERIOD_S = 1.0


def _display_loop(gps: GpsReader, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        fix = gps.get()
        if fix.fix_quality == 0:
            print("\r  waiting for GPS fix...                                   ", end="", flush=True)
        else:
            age_s = time.monotonic() - fix.updated_at
            print(f"\r  fix={fix.fix_quality} lat={fix.lat:.6f} lon={fix.lon:.6f} "
                  f"age={age_s:4.1f}s                 ", end="", flush=True)
        time.sleep(_DISPLAY_PERIOD_S)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_file", type=Path)
    parser.add_argument("--gps-port", default=_DEFAULT_GPS_PORT)
    parser.add_argument("--gps-baud", type=int, default=_DEFAULT_GPS_BAUD)
    parser.add_argument("--append", action="store_true",
                        help="Append to out_file instead of overwriting it")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    mode = "a" if args.append and args.out_file.exists() else "w"
    captured: list[tuple[float, float]] = []
    offsets: list[int] = []  # file offset before each write, for undo

    gps = GpsReader(args.gps_port, args.gps_baud)
    gps.start()

    stop_event = threading.Event()
    display_thread = threading.Thread(target=_display_loop, args=(gps, stop_event),
                                      daemon=True, name="waypoint-display")
    display_thread.start()

    print(f"Writing waypoints to {args.out_file} ({'append' if mode == 'a' else 'overwrite'})")
    print("Enter = capture waypoint, u = undo last, q = quit\n")

    with open(args.out_file, mode) as f:
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
                    if fix.fix_quality == 0:
                        print("\n  no GPS fix yet — not captured")
                        continue
                    age_s = time.monotonic() - fix.updated_at
                    if age_s > 3.0:
                        print(f"\n  fix is {age_s:.1f}s stale — not captured")
                        continue

                    offsets.append(f.tell())
                    f.write(f"{fix.lat:.7f},{fix.lon:.7f}\n")
                    f.flush()
                    captured.append((fix.lat, fix.lon))
                    print(f"\n  captured waypoint {len(captured)}: {fix.lat:.7f}, {fix.lon:.7f}")

        except (KeyboardInterrupt, EOFError):
            pass

    stop_event.set()
    gps.stop()
    print(f"\n\nSaved {len(captured)} waypoint(s) to {args.out_file.resolve()}")


if __name__ == "__main__":
    main()
