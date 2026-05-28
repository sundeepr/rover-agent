#!/usr/bin/env python3
"""
Rover calibration tool — reads the Atlas STM32's built-in IMU.

The Atlas rover streams IMU data on the same serial port used for motor
commands (/dev/ttyACM0).  This script opens that port once, reads IMU
frames in a background thread, and sends motor commands from the main
thread.

IMU frame format (sent at ~60 Hz by the Atlas STM32):
  $IMU,<ts_ms>,<qw>,<qx>,<qy>,<qz>,<gx>,<gy>,<gz>,<ax>,<ay>,<az>,<temp>,<voltage>#

  ts_ms   — timestamp in milliseconds (rover clock)
  qw/x/y/z — unit quaternion (orientation)
  gx/gy/gz — gyro in rad/s (or deg/s — treat as relative only)
  ax/ay/az — accelerometer in g
  temp     — temperature °C
  voltage  — battery voltage V

Heading (yaw) is derived from the quaternion:
  yaw_rad = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy² + qz²))

Other frames (informational, not used by calibration):
  $DBG,...#   — debug text from firmware
  $DATA,...#  — motor feedback, mode info
  $MODE,...#  — mode change notification

Usage examples
──────────────
  # Live IMU monitor (no movement, 60 s):
  python calibration/calibrate.py

  # 90° right turn:
  python calibration/calibrate.py --test turn --angle 90 --speed 40

  # Straight-line drift test (2 s):
  python calibration/calibrate.py --test straight --duration 2.0 --speed 40

  # Full calibration sequence:
  python calibration/calibrate.py --test full --speed 40

  # Dry-run (print commands only, no serial port):
  python calibration/calibrate.py --test full --dry-run
"""

import argparse
import math
import re
import sys
import threading
import time
import logging
from dataclasses import dataclass, field

log = logging.getLogger("rover.calibrate")

# ── Wire format ───────────────────────────────────────────────────────────────

_IMU_RE = re.compile(
    r"\$IMU,"
    r"(\d+),"                       # ts_ms
    r"([+-]?\d+\.?\d*),"            # qw
    r"([+-]?\d+\.?\d*),"            # qx
    r"([+-]?\d+\.?\d*),"            # qy
    r"([+-]?\d+\.?\d*),"            # qz
    r"([+-]?\d+\.?\d*),"            # gx
    r"([+-]?\d+\.?\d*),"            # gy
    r"([+-]?\d+\.?\d*),"            # gz
    r"([+-]?\d+\.?\d*),"            # ax
    r"([+-]?\d+\.?\d*),"            # ay
    r"([+-]?\d+\.?\d*),"            # az
    r"([+-]?\d+\.?\d*),"            # temp
    r"([+-]?\d+\.?\d*)#"            # voltage
)


def _atlas_cmd(L: int, R: int, AUX: int = 0) -> bytes:
    L   = max(-100, min(100, int(L)))
    R   = max(-100, min(100, int(R)))
    AUX = max(   0, min(100, int(AUX)))
    return f"$CMD,L={L},R={R},AUX={AUX}#\n".encode("ascii")


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class IMUReading:
    """One parsed IMU frame from the Atlas STM32."""
    ts_ms:    int   = 0
    qw:       float = 1.0
    qx:       float = 0.0
    qy:       float = 0.0
    qz:       float = 0.0
    gx:       float = 0.0
    gy:       float = 0.0
    gz:       float = 0.0
    ax:       float = 0.0
    ay:       float = 0.0
    az:       float = 0.0
    temp_c:   float = 0.0
    voltage:  float = 0.0
    wall_t:   float = 0.0    # time.time() when received
    valid:    bool  = False

    @property
    def heading_deg(self) -> float:
        """Yaw angle derived from quaternion, degrees, −180 to +180."""
        # Standard quaternion → yaw formula
        yaw = math.atan2(
            2.0 * (self.qw * self.qz + self.qx * self.qy),
            1.0 - 2.0 * (self.qy ** 2 + self.qz ** 2),
        )
        return math.degrees(yaw)

    @property
    def pitch_deg(self) -> float:
        sinp = 2.0 * (self.qw * self.qy - self.qz * self.qx)
        sinp = max(-1.0, min(1.0, sinp))
        return math.degrees(math.asin(sinp))

    @property
    def roll_deg(self) -> float:
        sinr = 2.0 * (self.qw * self.qx + self.qy * self.qz)
        cosr = 1.0 - 2.0 * (self.qx ** 2 + self.qy ** 2)
        return math.degrees(math.atan2(sinr, cosr))


def _parse_imu(line: str) -> IMUReading | None:
    """Try to parse a $IMU,...# line; return None on failure."""
    m = _IMU_RE.search(line)
    if not m:
        return None
    ts, qw, qx, qy, qz, gx, gy, gz, ax, ay, az, temp, volt = m.groups()
    return IMUReading(
        ts_ms   = int(ts),
        qw      = float(qw),
        qx      = float(qx),
        qy      = float(qy),
        qz      = float(qz),
        gx      = float(gx),
        gy      = float(gy),
        gz      = float(gz),
        ax      = float(ax),
        ay      = float(ay),
        az      = float(az),
        temp_c  = float(temp),
        voltage = float(volt),
        wall_t  = time.time(),
        valid   = True,
    )


# ── AtlasPort — one port, two directions ─────────────────────────────────────

class AtlasPort:
    """
    Manages the Atlas serial port for both command TX and IMU RX.

    A background thread continuously reads lines from the port, parses
    $IMU frames, and stores them for the main thread to query.

    Motor commands are written from the main thread.  The STM32 handles
    the interleaved traffic transparently.
    """

    def __init__(self, port: str, baud: int = 115200, dry_run: bool = False):
        self._port    = port
        self._baud    = baud
        self._dry_run = dry_run
        self._ser     = None

        self._latest: IMUReading = IMUReading()
        self._history: list[IMUReading] = []
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None

        self._armed   = False   # set when $DATA,SWD=ON (ARMED)# seen

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        if self._dry_run:
            log.info("[dry-run] AtlasPort: no serial port opened")
            return
        import serial
        self._ser = serial.Serial(
            self._port, self._baud, timeout=0.5, write_timeout=0.2)
        time.sleep(0.3)
        self._ser.reset_input_buffer()
        log.info("Opened %s @ %d baud", self._port, self._baud)
        self._running = True
        self._thread  = threading.Thread(target=self._reader_loop, daemon=True,
                                         name="atlas-imu-rx")
        self._thread.start()

    def close(self) -> None:
        self.send(0, 0)   # stop motors
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.5)
        if self._ser:
            self._ser.close()
            self._ser = None
            log.info("AtlasPort closed")

    # ── Command TX ────────────────────────────────────────────────────────────

    def send(self, L: int, R: int, AUX: int = 0) -> None:
        frame = _atlas_cmd(L, R, AUX)
        if self._dry_run:
            print(f"  [dry-run] {frame.decode().strip()}")
            return
        if self._ser:
            self._ser.write(frame)
            self._ser.flush()

    def stop(self) -> None:
        self.send(0, 0)

    def drive_straight(self, pct: int) -> None:
        self.send(pct, pct)

    def spin_right(self, pct: int) -> None:
        """Spin in place clockwise (positive yaw change)."""
        self.send(pct, -pct)

    def spin_left(self, pct: int) -> None:
        """Spin in place counter-clockwise (negative yaw change)."""
        self.send(-pct, pct)

    # ── IMU RX ────────────────────────────────────────────────────────────────

    def latest_imu(self) -> IMUReading:
        with self._lock:
            return self._latest

    def wait_for_imu(self, timeout: float = 5.0) -> bool:
        """Block until the first valid IMU reading arrives or timeout."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.latest_imu().valid:
                return True
            time.sleep(0.05)
        return False

    def snapshot_history(self) -> list[IMUReading]:
        """Return and clear the IMU history list."""
        with self._lock:
            h = list(self._history)
            self._history.clear()
        return h

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()

    @property
    def armed(self) -> bool:
        return self._armed

    # ── Background reader ─────────────────────────────────────────────────────

    def _reader_loop(self) -> None:
        while self._running:
            try:
                raw = self._ser.readline()
            except Exception as e:
                if self._running:
                    log.warning("Serial read error: %s", e)
                break
            if not raw:
                continue

            try:
                line = raw.decode("ascii", errors="replace").strip()
            except Exception:
                continue

            # Parse IMU frame
            reading = _parse_imu(line)
            if reading:
                with self._lock:
                    self._latest = reading
                    self._history.append(reading)
                continue

            # Track arming state
            if "ARMED" in line:
                if "SWD=ON" in line or "ARMED" in line:
                    self._armed = True
                    log.info("Atlas ARMED")
            elif "UNARMED" in line:
                self._armed = False
                log.debug("Atlas unarmed")

            # Log other informational frames at DEBUG level
            if line.startswith("$DBG,") or line.startswith("$MODE,"):
                log.debug("Atlas: %s", line)
            elif line.startswith("$DATA,") and "L=" not in line:
                # Motor feedback ($DATA,L=0,R=0,...) is too spammy; skip
                log.debug("Atlas: %s", line)


# ── Heading helpers ───────────────────────────────────────────────────────────

def _heading_diff(start: float, end: float) -> float:
    """Signed shortest-arc difference (degrees), result in (−180, +180]."""
    d = (end - start) % 360
    if d > 180:
        d -= 360
    return d


# ── Wheel-base geometry estimate ──────────────────────────────────────────────
#
# The Atlas M2 has a wheelbase of roughly 650 mm and MAX speed ~200 mm/s at
# 100% PWM.  Used only to estimate how long a turn should take; the actual
# heading is always measured from the IMU.
#
_WHEEL_BASE_MM    = 650
_MAX_VEL_MM_S     = 200    # at 100% PWM, approximate


def _estimated_turn_duration(target_deg: float, speed_pct: int) -> float:
    """Estimate spin duration in seconds (open-loop fallback)."""
    v_mm_s     = _MAX_VEL_MM_S * abs(speed_pct) / 100.0
    deg_per_s  = math.degrees(2.0 * v_mm_s / _WHEEL_BASE_MM)
    return abs(target_deg) / max(deg_per_s, 1.0)


# ── Test routines ─────────────────────────────────────────────────────────────

def test_imu_live(port: AtlasPort, duration: float = 60.0) -> None:
    """Print live IMU readings for `duration` seconds."""
    print(f"\n{'─'*72}")
    print(f"  Live IMU monitor  ({duration:.0f} s)   Ctrl-C to stop early")
    print(f"{'─'*72}")
    print(f"  {'Time':>6}  {'Heading':>8}  {'Pitch':>7}  {'Roll':>7}  "
          f"{'GyroZ':>8}  {'Volt':>5}  {'Temp':>5}")
    print(f"{'─'*72}")
    t0 = time.time()
    try:
        while time.time() - t0 < duration:
            r = port.latest_imu()
            if r.valid:
                print(f"\r  {time.time()-t0:6.1f}s  "
                      f"{r.heading_deg:7.2f}°  "
                      f"{r.pitch_deg:+6.2f}°  "
                      f"{r.roll_deg:+6.2f}°  "
                      f"{r.gz:+7.4f}r/s  "
                      f"{r.voltage:4.1f}V  "
                      f"{r.temp_c:4.1f}°C",
                      end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    print()


def test_turn(port: AtlasPort,
              target_deg: float = 90.0,
              speed_pct: int   = 40,
              settle_s: float  = 0.5,
              imu_settle_s: float = 0.3) -> dict:
    """
    Spin the rover and measure the actual heading change from the IMU.

    Returns a dict with commanded / measured / error values.
    """
    direction = "right" if target_deg > 0 else "left"
    est_dur   = _estimated_turn_duration(target_deg, speed_pct)

    print(f"\n{'─'*60}")
    print(f"  Turn test: {target_deg:+.0f}° ({direction})  "
          f"speed={speed_pct}%  est. {est_dur:.2f} s")
    print(f"{'─'*60}")

    # Settle, then record start heading
    time.sleep(settle_s)
    h_before = port.latest_imu().heading_deg
    print(f"  Heading before : {h_before:.3f}°")

    port.clear_history()

    # ── Execute turn ──
    if target_deg > 0:
        port.spin_right(speed_pct)
    else:
        port.spin_left(speed_pct)

    time.sleep(est_dur)
    port.stop()

    # Let the rover settle before measuring final heading
    time.sleep(imu_settle_s)

    h_after  = port.latest_imu().heading_deg
    measured = _heading_diff(h_before, h_after)
    error    = measured - target_deg

    print(f"  Heading after  : {h_after:.3f}°")
    print(f"  Commanded      : {target_deg:+.2f}°")
    print(f"  Measured       : {measured:+.2f}°")
    print(f"  Error          : {error:+.2f}°  ({abs(error)/max(abs(target_deg),1)*100:.1f}%)")

    if abs(error) < 5:
        print("  ✓  PASS (< ±5°)")
    elif abs(error) < 15:
        print("  ⚠  MARGINAL (5–15°) — adjust deg_per_s in atlas_controller.py")
    else:
        print("  ✗  FAIL (> 15°) — check battery, wheel grip, surface")

    return dict(test="turn",
                target_deg=target_deg,
                measured_deg=measured,
                error_deg=error,
                speed_pct=speed_pct,
                est_duration_s=est_dur)


def test_straight(port: AtlasPort,
                  duration: float = 2.0,
                  speed_pct: int  = 40,
                  settle_s: float = 0.5) -> dict:
    """
    Drive straight and measure heading drift.

    Good rover: < 3° over 2 s.
    """
    print(f"\n{'─'*60}")
    print(f"  Straight-line test: {duration:.1f} s  speed={speed_pct}%")
    print(f"{'─'*60}")

    time.sleep(settle_s)
    h_before = port.latest_imu().heading_deg
    print(f"  Heading before : {h_before:.3f}°")

    port.clear_history()
    port.drive_straight(speed_pct)
    time.sleep(duration)
    port.stop()
    time.sleep(settle_s)

    h_after = port.latest_imu().heading_deg
    drift   = _heading_diff(h_before, h_after)
    history = port.snapshot_history()

    max_drift = 0.0
    if history:
        diffs     = [_heading_diff(h_before, r.heading_deg) for r in history]
        max_drift = max(abs(d) for d in diffs)

    print(f"  Heading after  : {h_after:.3f}°")
    print(f"  Final drift    : {drift:+.2f}°")
    print(f"  Peak drift     : {max_drift:.2f}°")
    print(f"  Samples        : {len(history)}")

    if abs(drift) < 3.0:
        print("  ✓  PASS — drift < 3°")
    elif abs(drift) < 8.0:
        print("  ⚠  MARGINAL — drift 3–8°; trim motor L/R balance")
    else:
        print("  ✗  FAIL — drift > 8°; check alignment / motor mismatch")

    return dict(test="straight",
                duration_s=duration,
                drift_deg=drift,
                max_drift_deg=max_drift,
                speed_pct=speed_pct,
                samples=len(history))


def test_full(port: AtlasPort, speed_pct: int = 40) -> None:
    """Run the full calibration sequence and print a summary table."""
    results = []

    print("\n" + "═"*60)
    print("  FULL CALIBRATION SEQUENCE")
    print("═"*60)
    input("  Press Enter to start (rover will move)…")

    results.append(test_straight(port, duration=2.0,  speed_pct=speed_pct))
    time.sleep(1.0)

    results.append(test_turn(port, target_deg=+90,  speed_pct=speed_pct))
    time.sleep(1.0)

    results.append(test_turn(port, target_deg=-90,  speed_pct=speed_pct))
    time.sleep(1.0)

    results.append(test_turn(port, target_deg=+180, speed_pct=speed_pct))
    time.sleep(1.0)

    results.append(test_turn(port, target_deg=-180, speed_pct=speed_pct))

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  SUMMARY")
    print("═"*60)
    for r in results:
        if r["test"] == "turn":
            status = ("✓" if abs(r["error_deg"]) < 5 else
                      "⚠" if abs(r["error_deg"]) < 15 else "✗")
            print(f"  {status} Turn {r['target_deg']:+5.0f}°  →  "
                  f"measured {r['measured_deg']:+6.1f}°  "
                  f"error {r['error_deg']:+5.1f}°")
        elif r["test"] == "straight":
            status = ("✓" if abs(r["drift_deg"]) < 3 else
                      "⚠" if abs(r["drift_deg"]) < 8 else "✗")
            print(f"  {status} Straight {r['duration_s']:.1f}s  →  "
                  f"drift {r['drift_deg']:+5.1f}°  "
                  f"peak {r['max_drift_deg']:4.1f}°")
    print("═"*60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Rover calibration using Atlas STM32 built-in IMU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--port",  default="/dev/ttyACM0",
                        help="Atlas serial port (default: /dev/ttyACM0)")
    parser.add_argument("--baud",  type=int, default=115200)
    parser.add_argument("--speed", type=int, default=40,
                        help="Motor power %% for tests (default: 40)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands only, do not open serial port")

    parser.add_argument("--test",
                        choices=["imu", "turn", "straight", "full"],
                        default="imu",
                        help="Which test to run (default: imu — live monitor)")
    parser.add_argument("--angle",    type=float, default=90.0,
                        help="Target angle for turn test (degrees, default 90)")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Duration for straight test (seconds, default 2.0)")
    parser.add_argument("--monitor-time", type=float, default=60.0,
                        help="Seconds to run live IMU monitor (default 60)")

    args = parser.parse_args()

    # ── Open port ─────────────────────────────────────────────────────────────
    port = AtlasPort(port=args.port, baud=args.baud, dry_run=args.dry_run)

    print(f"\nOpening Atlas port: {args.port}  (dry-run={args.dry_run})")
    try:
        port.open()
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # Wait for first IMU reading (unless dry-run)
    if not args.dry_run:
        print("  Waiting for IMU data…", end="", flush=True)
        ok = port.wait_for_imu(timeout=5.0)
        if ok:
            r = port.latest_imu()
            print(f"\n  IMU OK — heading={r.heading_deg:.1f}°  "
                  f"pitch={r.pitch_deg:.1f}°  roll={r.roll_deg:.1f}°  "
                  f"volt={r.voltage:.1f}V  temp={r.temp_c:.1f}°C")
        else:
            print("\n  WARNING: no IMU data in 5 s — check serial port / cable")

    # ── Run test ──────────────────────────────────────────────────────────────
    try:
        if args.test == "imu":
            test_imu_live(port, duration=args.monitor_time)

        elif args.test == "turn":
            result = test_turn(port, target_deg=args.angle, speed_pct=args.speed)
            print(f"\nResult dict: {result}")

        elif args.test == "straight":
            result = test_straight(port, duration=args.duration, speed_pct=args.speed)
            print(f"\nResult dict: {result}")

        elif args.test == "full":
            test_full(port, speed_pct=args.speed)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        port.close()
        print("Done.")


if __name__ == "__main__":
    main()
