#!/usr/bin/env python3
"""
gps_waypoint_follow.py — drives the Atlas rover through a sequence of GPS
waypoints read from a file, one after another, using the onboard NEO GPS
module for position feedback.

Waypoints file format
──────────────────────
Plain text, one "lat,lon" pair per line (decimal degrees). Blank lines and
lines starting with # are ignored:

    # test loop around the field
    12.971600, 77.594600
    12.971700, 77.594700
    12.971600, 77.594800

How steering works (read this before trusting it in the field)
────────────────────────────────────────────────────────────────
This rover has no compass/IMU — see test_atlas_imu_drive.py. The only way
to know which way the rover is actually pointed is GPS *course over
ground* (COG), and COG is only meaningful while the rover is translating;
it's noise while stationary or spinning in place. So this script never
spins blindly — it always drives forward along a curved arc and steers by
varying the arc's radius, continuously re-aiming using fresh GPS fixes:

    heading_error = bearing_to_waypoint - last_known_moving_heading
    radius = tighter curve for larger heading_error, dead-banded to
             straight once heading_error is small

Before the first moving GPS fix exists (right after a stop, or at
startup), there's no heading yet, so it drives straight at a reduced
speed until one shows up.

Each waypoint is "reached" once the rover is within --tolerance-m of it;
the script then advances to the next line in the file. If a waypoint
can't be reached within --waypoint-timeout-s (GPS fix lost, waypoint
off the field, etc.) it's skipped with a warning rather than hanging the
whole run forever. GPS loss mid-drive always stops the rover immediately
— this script prioritizes not running off blind over keeping moving.

Usage
─────
    python gps_waypoint_follow.py waypoints.txt
    python gps_waypoint_follow.py waypoints.txt --atlas-port /dev/ttyACM0
    python gps_waypoint_follow.py waypoints.txt --gps-port /dev/ttyTHS1
    python gps_waypoint_follow.py waypoints.txt --dry-run   # logs commands, no serial port
    python gps_waypoint_follow.py waypoints.txt --cruise-vel 60 --tolerance-m 3
    # Ctrl-C stops the rover and exits cleanly at any point.
"""

import argparse
import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import serial

from atlas_controller import AtlasController

log = logging.getLogger("rover.gps_waypoint_follow")

_DEFAULT_ATLAS_PORT = "/dev/ttyACM0"
_DEFAULT_ATLAS_BAUD = 115200
_DEFAULT_GPS_PORT   = "/dev/ttyTHS1"
_DEFAULT_GPS_BAUD   = 9600

_EARTH_RADIUS_M            = 6371000.0
_KNOTS_TO_MPS              = 0.514444
_MIN_SPEED_FOR_COURSE_MPS  = 0.3    # below this, GPRMC's true_course is noise
_GPS_STALE_S               = 3.0    # no fresh fix in this long -> stop, don't dead-reckon
_CONTROL_PERIOD_S          = 0.5
_RECONNECT_S               = 3.0

_DEFAULT_TOLERANCE_M       = 2.0
_DEFAULT_CRUISE_VEL_MM_S   = 100
_DEFAULT_BOOTSTRAP_VEL_MM_S = 60
_DEFAULT_WAYPOINT_TIMEOUT_S = 180.0

_MIN_RADIUS_MM             = 300    # sharpest curve while still moving forward
_MAX_RADIUS_MM             = 3000   # gentle correction
_HEADING_ERROR_FOR_MIN_RADIUS_DEG = 45.0
_STRAIGHT_HEADING_ERROR_DEG       = 5.0   # dead-band: go straight below this


# ── Geo helpers ───────────────────────────────────────────────────────────────

def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dlmb   = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb   = math.radians(lon2 - lon1)
    y = math.sin(dlmb) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    return math.degrees(math.atan2(y, x)) % 360.0


def normalize_angle_deg(angle: float) -> float:
    """Wrap to (-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0


def steering_radius_mm(heading_error_deg: float) -> int:
    """
    Map a signed heading error to a drive_raw() radius.  Positive radius
    turns left (Roomba OI convention: CCW positive, matching
    atlas_controller._velocity_radius_to_lr), so a target to the RIGHT
    (positive heading_error) needs a negative radius.
    """
    err = min(abs(heading_error_deg), _HEADING_ERROR_FOR_MIN_RADIUS_DEG)
    frac = err / _HEADING_ERROR_FOR_MIN_RADIUS_DEG
    magnitude = _MAX_RADIUS_MM - frac * (_MAX_RADIUS_MM - _MIN_RADIUS_MM)
    return int(-magnitude if heading_error_deg > 0 else magnitude)


def load_waypoints(path: Path) -> list[tuple[float, float]]:
    waypoints = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 2:
            log.warning("waypoints file line %d: can't parse %r — skipping", lineno, raw)
            continue
        try:
            lat, lon = float(parts[0]), float(parts[1])
        except ValueError:
            log.warning("waypoints file line %d: can't parse %r — skipping", lineno, raw)
            continue
        waypoints.append((lat, lon))
    return waypoints


# ── GPS reader thread ────────────────────────────────────────────────────────

@dataclass
class GpsFix:
    lat: float = 0.0
    lon: float = 0.0
    fix_quality: int = 0
    heading_deg: Optional[float] = None   # only set from a valid *moving* GPRMC fix
    updated_at: float = 0.0               # time.monotonic() of last GGA position update


class GpsReader:
    """Background thread parsing $GPGGA (position/fix) and $GPRMC (course) off
    the onboard NEO module, exposing the latest fix via get()."""

    def __init__(self, port: str, baud: int):
        self._port = port
        self._baud = baud
        self._fix = GpsFix()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gps-reader")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    def get(self) -> GpsFix:
        with self._lock:
            return GpsFix(**vars(self._fix))

    def _loop(self) -> None:
        import pynmea2

        ser: Optional[serial.Serial] = None
        while not self._stop_event.is_set():
            if ser is None:
                try:
                    ser = serial.Serial(self._port, baudrate=self._baud, timeout=1.0)
                    log.info("gps: connected to %s @ %d", self._port, self._baud)
                except Exception as exc:
                    log.warning("gps: failed to open %s (%s) — retrying in %.0fs",
                                self._port, exc, _RECONNECT_S)
                    time.sleep(_RECONNECT_S)
                    continue

            try:
                raw = ser.readline()
            except Exception as exc:
                log.warning("gps: read error (%s) — reconnecting", exc)
                ser.close()
                ser = None
                time.sleep(_RECONNECT_S)
                continue

            if not raw:
                continue
            line = raw.decode("ascii", errors="replace").strip()

            if "$GPGGA" in line:
                try:
                    msg = pynmea2.parse(line)
                except Exception:
                    continue
                if msg.gps_qual is None:
                    continue
                with self._lock:
                    self._fix.fix_quality = int(msg.gps_qual)
                    if msg.gps_qual > 0 and msg.latitude and msg.longitude:
                        self._fix.lat = float(msg.latitude)
                        self._fix.lon = float(msg.longitude)
                        self._fix.updated_at = time.monotonic()

            elif "$GPRMC" in line:
                try:
                    msg = pynmea2.parse(line)
                except Exception:
                    continue
                if msg.status != "A" or msg.spd_over_grnd is None or msg.true_course is None:
                    continue
                speed_mps = float(msg.spd_over_grnd) * _KNOTS_TO_MPS
                if speed_mps >= _MIN_SPEED_FOR_COURSE_MPS:
                    with self._lock:
                        self._fix.heading_deg = float(msg.true_course)

        if ser is not None:
            ser.close()
        log.info("gps: reader stopped")


# ── Main control loop ────────────────────────────────────────────────────────

def _drive_toward(ctrl: AtlasController, fix: GpsFix, target: tuple[float, float],
                   cruise_vel: int, bootstrap_vel: int) -> float:
    """One control step toward `target`. Returns remaining distance in meters."""
    lat, lon = fix.lat, fix.lon
    distance_m = haversine_distance_m(lat, lon, *target)
    bearing_deg = initial_bearing_deg(lat, lon, *target)

    if fix.heading_deg is None:
        ctrl.drive_raw(bootstrap_vel, 0x8000)
        log.info("  bootstrapping heading — driving straight at %d (dist=%.1fm)",
                 bootstrap_vel, distance_m)
        return distance_m

    heading_error = normalize_angle_deg(bearing_deg - fix.heading_deg)
    if abs(heading_error) <= _STRAIGHT_HEADING_ERROR_DEG:
        ctrl.drive_raw(cruise_vel, 0x8000)
        log.info("  dist=%.1fm bearing=%.0f° heading=%.0f° err=%.1f° -> straight",
                 distance_m, bearing_deg, fix.heading_deg, heading_error)
    else:
        radius = steering_radius_mm(heading_error)
        ctrl.drive_raw(cruise_vel, radius)
        log.info("  dist=%.1fm bearing=%.0f° heading=%.0f° err=%.1f° -> radius=%dmm",
                 distance_m, bearing_deg, fix.heading_deg, heading_error, radius)

    return distance_m


def run_mission(ctrl: AtlasController, gps: GpsReader, waypoints: list[tuple[float, float]],
                 tolerance_m: float, cruise_vel: int, bootstrap_vel: int,
                 waypoint_timeout_s: float) -> None:
    for i, target in enumerate(waypoints, start=1):
        log.info("Waypoint %d/%d: %.6f, %.6f", i, len(waypoints), *target)
        start_t = time.monotonic()

        while True:
            fix = gps.get()
            now = time.monotonic()

            if waypoint_timeout_s > 0 and now - start_t > waypoint_timeout_s:
                log.warning("Waypoint %d timed out after %.0fs — skipping", i, waypoint_timeout_s)
                ctrl.stop()
                break

            if fix.fix_quality == 0 or fix.updated_at == 0.0:
                log.warning("No GPS fix yet — stopped, waiting")
                ctrl.stop()
                time.sleep(_CONTROL_PERIOD_S)
                continue

            if now - fix.updated_at > _GPS_STALE_S:
                log.warning("GPS fix stale (%.1fs old) — stopping until it recovers",
                            now - fix.updated_at)
                ctrl.stop()
                time.sleep(_CONTROL_PERIOD_S)
                continue

            distance_m = _drive_toward(ctrl, fix, target, cruise_vel, bootstrap_vel)
            if distance_m <= tolerance_m:
                log.info("Waypoint %d reached (dist=%.1fm)", i, distance_m)
                ctrl.stop()
                break

            time.sleep(_CONTROL_PERIOD_S)

    ctrl.stop()
    log.info("Mission complete — all %d waypoint(s) processed", len(waypoints))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("waypoints_file", type=Path)
    parser.add_argument("--atlas-port", default=_DEFAULT_ATLAS_PORT)
    parser.add_argument("--atlas-baud", type=int, default=_DEFAULT_ATLAS_BAUD)
    parser.add_argument("--gps-port", default=_DEFAULT_GPS_PORT)
    parser.add_argument("--gps-baud", type=int, default=_DEFAULT_GPS_BAUD)
    parser.add_argument("--tolerance-m", type=float, default=_DEFAULT_TOLERANCE_M,
                        help="Distance to a waypoint counted as 'reached' (default: %(default)sm)")
    parser.add_argument("--cruise-vel", type=int, default=_DEFAULT_CRUISE_VEL_MM_S,
                        help="Forward velocity while steering toward a waypoint (mm/s, default: %(default)s)")
    parser.add_argument("--bootstrap-vel", type=int, default=_DEFAULT_BOOTSTRAP_VEL_MM_S,
                        help="Forward velocity while no heading is known yet (mm/s, default: %(default)s)")
    parser.add_argument("--waypoint-timeout-s", type=float, default=_DEFAULT_WAYPOINT_TIMEOUT_S,
                        help="Skip a waypoint if unreached after this long, 0 to disable (default: %(default)ss)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log drive commands instead of opening the Atlas serial port")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    if not args.waypoints_file.exists():
        raise SystemExit(f"Waypoints file not found: {args.waypoints_file}")
    waypoints = load_waypoints(args.waypoints_file)
    if not waypoints:
        raise SystemExit(f"No valid waypoints found in {args.waypoints_file}")
    log.info("Loaded %d waypoint(s) from %s", len(waypoints), args.waypoints_file)

    gps = GpsReader(args.gps_port, args.gps_baud)
    gps.start()

    ctrl = AtlasController(args.atlas_port, baud=args.atlas_baud, dry_run=args.dry_run)
    try:
        with ctrl.connect():
            run_mission(ctrl, gps, waypoints, args.tolerance_m, args.cruise_vel,
                       args.bootstrap_vel, args.waypoint_timeout_s)
    except KeyboardInterrupt:
        log.info("Interrupted — stopping")
        ctrl.stop()
    finally:
        gps.stop()


if __name__ == "__main__":
    main()
