#!/usr/bin/env python3
"""
test_atlas_imu_drive.py — drive the Atlas rover slowly while listening for
IMU telemetry on the same serial connection, and hold a fixed AUX output.

Format of the IMU data is NOT YET KNOWN — nothing else in this repo ever
reads from the Atlas serial port (atlas_controller.py's _send_cmd() only
ever calls _ser.write(), never _ser.read()). So this script's "IMU
listener" is deliberately generic: it just prints every line the STM32
sends back, verbatim, with a timestamp. Once you've seen a real line come
through, tell me its shape and I'll write a real parser instead of guessing
at one.

What it actually does
──────────────────────
  1. Connects to the Atlas board (or --dry-run to just log everything).
  2. Sets AUX to --aux-pct (default 30) — persists across every command
     after that, per AtlasController.set_aux()'s docstring.
  3. Starts a background thread that continuously reads lines off the
     serial port and logs them as "[IMU?] <raw line>" — best-effort
     UTF-8 decode, never crashes on garbage bytes.
  4. Drives straight forward at --power-pct (default 8 — the deadband
     floor per atlas_controller._MOTOR_DEADBAND_PCT, so this is close to
     the slowest speed that reliably overcomes static friction) for
     --duration-s, then stops.

Usage
─────
    python test_atlas_imu_drive.py --atlas-port /dev/ttyACM0

    # Dry run first — no serial port opened, just logs what would be sent:
    python test_atlas_imu_drive.py --atlas-port /dev/ttyACM0 --dry-run
"""

import argparse
import logging
import threading
import time

from atlas_controller import AtlasController

log = logging.getLogger("test_atlas_imu_drive")


def _imu_listener(ctrl: AtlasController, running: threading.Event) -> None:
    """
    Reads raw lines off the Atlas serial connection and logs them.

    Reaches into AtlasController's private _ser directly — there is no
    public read API on this controller today (it was only ever built to
    write commands). timeout=0.2 is already set on the Serial object by
    AtlasController.connect(), so readline() won't block indefinitely and
    this loop can check `running` regularly.
    """
    if ctrl.dry_run or ctrl._ser is None:
        log.info("[IMU listener] dry-run — no real serial port, nothing to read")
        return

    log.info("[IMU listener] started — printing every raw line the STM32 sends")
    while running.is_set():
        try:
            raw = ctrl._ser.readline()
        except Exception as e:
            log.warning("[IMU listener] read error: %s", e)
            time.sleep(0.2)
            continue
        if not raw:
            continue   # timeout, no data — normal, just loop
        text = raw.decode("utf-8", errors="replace").strip()
        if text:
            log.info("[IMU?] %s", text)
    log.info("[IMU listener] stopped")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--atlas-port", type=str, required=True,
                        help="Serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--dry-run", action="store_true",
                        help="Log commands but don't open the serial port / move anything")
    parser.add_argument("--power-pct", type=float, default=8.0, metavar="PCT",
                        help="%% power, both wheels, straight forward (default 8.0 — "
                             "atlas_controller.py's deadband floor)")
    parser.add_argument("--aux-pct", type=float, default=30.0, metavar="PCT",
                        help="AUX output %% to hold for the whole run (default 30.0)")
    parser.add_argument("--duration-s", type=float, default=5.0, metavar="SECS",
                        help="How long to drive before stopping (default 5.0)")
    args = parser.parse_args()

    ctrl = AtlasController(port=args.atlas_port, baud=args.baud, dry_run=args.dry_run)

    with ctrl.connect():
        log.info("Connected (dry_run=%s) on %s", args.dry_run, args.atlas_port)

        ctrl.set_aux(args.aux_pct)
        # set_aux() only updates the tracked value — send one command now so
        # AUX actually goes out immediately, before the drive even starts.
        ctrl._send_cmd(0, 0)

        running = threading.Event()
        running.set()
        listener = threading.Thread(target=_imu_listener, args=(ctrl, running),
                                    daemon=True, name="imu-listener")
        listener.start()

        p = int(args.power_pct)
        log.info("Driving straight forward at %d%%  AUX=%d%%  for %.1fs",
                 p, int(args.aux_pct), args.duration_s)
        # Raw L/R directly (same reasoning as test_atlas_motor_check.py) —
        # bypasses drive_raw()'s velocity/radius/deadband indirection so
        # the %power sent is exactly what was asked for, no guessing.
        ctrl._send_cmd(p, p)
        time.sleep(args.duration_s)
        ctrl.stop()

        log.info("Drive complete. Listening 2 more seconds for trailing IMU output…")
        time.sleep(2.0)

        running.clear()
        listener.join(timeout=1.0)

    log.info("Done.")


if __name__ == "__main__":
    main()
