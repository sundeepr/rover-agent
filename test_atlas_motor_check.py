#!/usr/bin/env python3
"""
test_atlas_motor_check.py — minimal, interactive Atlas motor sanity check.

Isolates "does the hardware move at all" from every other moving part
(Moondream, detection, the alignment state machine). Talks to
AtlasController directly — nothing else.

Atlas has no read-back / encoders (see atlas_controller.py's _send_cmd —
it's a fire-and-forget serial write, no ack from the STM32), so this
script can only confirm the port opens and commands are sent without
raising; whether the rover actually moves has to be confirmed by eye.
That's why each step pauses for Enter before running — so you can watch
one motion at a time instead of a canned sequence firing all at once.

Usage
─────
    # Step through: stop, spin right, spin left, forward, reverse —
    # confirm each one visually before continuing:
    python test_atlas_motor_check.py --atlas-port /dev/ttyACM0 --power-pct 20

    # Dry run first — just confirms the port opens, logs commands, no motion:
    python test_atlas_motor_check.py --atlas-port /dev/ttyACM0 --dry-run

    # One specific raw command instead of the guided sequence — e.g. to
    # find the real deadband threshold or check one wheel in isolation:
    python test_atlas_motor_check.py --atlas-port /dev/ttyACM0 \\
        --raw 15 -15 --duration-s 2
"""

import argparse
import logging
import sys
import time

from atlas_controller import AtlasController

log = logging.getLogger("test_atlas_motor_check")


def _pause(prompt: str) -> bool:
    """Wait for Enter; return False if the user typed 'q' to quit early."""
    try:
        ans = input(f"{prompt} [Enter=go, q=quit] ").strip().lower()
    except EOFError:
        return True
    return ans != "q"


def _run_step(ctrl: AtlasController, label: str, L: int, R: int, duration_s: float) -> None:
    log.info("[%s] sending L=%d%%  R=%d%%  for %.1fs", label, L, R, duration_s)
    # Deliberately reaches into AtlasController's private _send_cmd() rather
    # than going through drive_raw()'s velocity/radius/deadband conversion —
    # the whole point of this script is to test the raw L/R %power the
    # STM32 actually receives, with nothing else in between.
    ctrl._send_cmd(L, R)
    time.sleep(duration_s)
    ctrl.stop()
    log.info("[%s] done — did it move as expected?", label)


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
    parser.add_argument("--power-pct", type=float, default=20.0, metavar="PCT",
                        help="%% power to use for the guided spin/forward/reverse steps "
                             "(default 20.0 — motor deadband is ~8%%, see "
                             "atlas_controller._MOTOR_DEADBAND_PCT)")
    parser.add_argument("--duration-s", type=float, default=1.5, metavar="SECS",
                        help="How long each motion runs (default 1.5)")
    parser.add_argument("--raw", type=int, nargs=2, metavar=("L_PCT", "R_PCT"),
                        help="Skip the guided sequence — send this exact L/R %% pair once "
                             "and exit (e.g. --raw 15 -15)")
    parser.add_argument("--yes", action="store_true",
                        help="Don't pause for Enter between steps (guided sequence only)")
    args = parser.parse_args()

    ctrl = AtlasController(port=args.atlas_port, baud=args.baud, dry_run=args.dry_run)

    try:
        with ctrl.connect():
            log.info("Connected (dry_run=%s) on %s", args.dry_run, args.atlas_port)

            if args.raw is not None:
                L, R = args.raw
                log.info("Raw one-off command:")
                _run_step(ctrl, "RAW", L, R, args.duration_s)
                return

            p = args.power_pct
            steps = [
                ("STOP (baseline — confirms the port opens and a 0,0 write doesn't error)",
                 0, 0, 0.5),
                (f"SPIN RIGHT at {p:.0f}%  (left wheel fwd, right wheel back)",
                 int(p), -int(p), args.duration_s),
                (f"SPIN LEFT at {p:.0f}%  (left wheel back, right wheel fwd)",
                 -int(p), int(p), args.duration_s),
                (f"FORWARD at {p:.0f}%  (both wheels forward, same sign)",
                 int(p), int(p), args.duration_s),
                (f"REVERSE at {p:.0f}%  (both wheels back, same sign)",
                 -int(p), -int(p), args.duration_s),
            ]

            for label, L, R, dur in steps:
                if not args.yes and not _pause(f"Next: {label}"):
                    log.info("Stopped by user.")
                    return
                _run_step(ctrl, label, L, R, dur)

            log.info("All steps sent. If nothing moved: check the serial cable/port, "
                    "battery/power to the drive motors, and that --power-pct is above "
                    "the deadband (~8%%). If it moved the wrong direction: L/R or motor "
                    "wiring may be swapped relative to what atlas_controller.py assumes.")

    except Exception as e:
        log.error("Failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
