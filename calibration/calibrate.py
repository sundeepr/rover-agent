#!/usr/bin/env python3
"""
Rover calibration tool.

Sends motion commands to the Atlas rover and reads heading / gyro data
from an IMU to verify the rover is behaving as expected.

Supported IMUs (auto-detected if not specified):
  BNO055  — most common, I2C address 0x28 or 0x29, provides fused Euler angles
  MPU6050 — raw gyro/accel only, I2C address 0x68 or 0x69
  ICM-20948 — high-end, I2C 0x68 or 0x69

Usage examples
──────────────
  # Monitor IMU live (no movement):
  python calibration/calibrate.py --imu bno055 --atlas-port /dev/ttyACM0

  # Send a 90° right turn and measure actual heading change:
  python calibration/calibrate.py --imu bno055 --atlas-port /dev/ttyACM0 \\
      --test turn --angle 90

  # Drive straight for 2 s and record heading drift:
  python calibration/calibrate.py --imu bno055 --atlas-port /dev/ttyACM0 \\
      --test straight --duration 2.0

  # Full calibration sequence (turn left/right/straight):
  python calibration/calibrate.py --imu bno055 --atlas-port /dev/ttyACM0 \\
      --test full

  # Dry-run (no serial hardware, print commands only):
  python calibration/calibrate.py --imu bno055 --dry-run
"""

import argparse
import math
import sys
import time
import threading
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("rover.calibrate")

# ── Atlas serial command ──────────────────────────────────────────────────────

def _atlas_frame(L: int, R: int, AUX: int = 0) -> bytes:
    L   = max(-100, min(100, int(L)))
    R   = max(-100, min(100, int(R)))
    AUX = max(   0, min(100, int(AUX)))
    return f"$CMD,L={L},R={R},AUX={AUX}#\n".encode("ascii")


# ── IMU reading ───────────────────────────────────────────────────────────────

@dataclass
class IMUReading:
    """One snapshot from the IMU."""
    timestamp:  float   = 0.0
    heading:    float   = 0.0   # degrees, 0 = north / forward at calibration start
    pitch:      float   = 0.0   # degrees, nose up = positive
    roll:       float   = 0.0   # degrees, right lean = positive
    gyro_z:     float   = 0.0   # deg/s yaw rate (raw, before fusion)
    valid:      bool    = False


class BNO055Reader:
    """
    Reads fused Euler angles from a Bosch BNO055 over I2C.

    The BNO055 does all sensor fusion on-chip and outputs Euler angles
    (heading/pitch/roll) in degrees directly — no integration needed.

    I2C address: 0x28 (ADDR pin low) or 0x29 (ADDR pin high).
    """

    # BNO055 register addresses
    _CHIP_ID_REG      = 0x00
    _BNO055_CHIP_ID   = 0xA0
    _OPR_MODE_REG     = 0x3D
    _OPR_MODE_NDOF    = 0x0C   # 9-DOF fusion mode (best accuracy)
    _EULER_H_LSB      = 0x1A   # Euler heading low byte (6 bytes: H, R, P)
    _GYRO_DATA_Z_LSB  = 0x18   # raw gyro Z low byte
    _UNIT_SEL_REG     = 0x3B

    def __init__(self, i2c_bus: int = 1, address: int = 0x28):
        self._bus_num = i2c_bus
        self._addr    = address
        self._bus     = None

    def open(self) -> None:
        try:
            import smbus2
            self._bus = smbus2.SMBus(self._bus_num)
        except ImportError:
            raise RuntimeError(
                "smbus2 not installed.  Run: pip install smbus2")

        # Verify chip ID
        chip_id = self._bus.read_byte_data(self._addr, self._CHIP_ID_REG)
        if chip_id != self._BNO055_CHIP_ID:
            raise RuntimeError(
                f"BNO055 not found at 0x{self._addr:02X} on I2C bus {self._bus_num}. "
                f"Got chip ID 0x{chip_id:02X}, expected 0x{self._BNO055_CHIP_ID:02X}. "
                f"Check wiring and --i2c-address.")

        # Set NDOF fusion mode (takes ~650 ms to stabilise)
        self._bus.write_byte_data(self._addr, self._OPR_MODE_REG, self._OPR_MODE_NDOF)
        log.info("BNO055 found at 0x%02X on bus %d — waiting for fusion mode…",
                 self._addr, self._bus_num)
        time.sleep(0.7)
        log.info("BNO055 ready")

    def read(self) -> IMUReading:
        if self._bus is None:
            return IMUReading()
        try:
            # Read 6 bytes: Euler H (2), R (2), P (2) — little-endian, 1/16 deg
            data = self._bus.read_i2c_block_data(self._addr, self._EULER_H_LSB, 6)
            h = (data[1] << 8 | data[0])
            r = (data[3] << 8 | data[2])
            p = (data[5] << 8 | data[4])

            # Signed conversion
            if r > 32767: r -= 65536
            if p > 32767: p -= 65536

            # Read raw gyro Z (2 bytes) — 1/16 deg/s
            gz_data = self._bus.read_i2c_block_data(self._addr, self._GYRO_DATA_Z_LSB, 2)
            gz = (gz_data[1] << 8 | gz_data[0])
            if gz > 32767: gz -= 65536

            return IMUReading(
                timestamp = time.time(),
                heading   = h / 16.0,
                pitch     = p / 16.0,
                roll      = r / 16.0,
                gyro_z    = gz / 16.0,
                valid     = True,
            )
        except Exception as e:
            log.debug("BNO055 read error: %s", e)
            return IMUReading()

    def close(self) -> None:
        if self._bus:
            self._bus.close()
            self._bus = None


class MPU6050Reader:
    """
    Reads raw gyro data from an MPU-6050 over I2C and integrates to heading.

    The MPU-6050 has no on-chip fusion — only raw gyro.  Heading is obtained
    by integrating gyro Z.  Drift accumulates over time; use the BNO055 for
    accurate absolute heading.

    I2C address: 0x68 (AD0 low) or 0x69 (AD0 high).
    """

    _PWR_MGMT_1  = 0x6B
    _GYRO_CONFIG = 0x1B   # FS_SEL bits 3:4 (00=250°/s, 01=500°/s)
    _GYRO_ZOUT_H = 0x47   # raw gyro Z high byte
    _WHO_AM_I    = 0x75

    def __init__(self, i2c_bus: int = 1, address: int = 0x68):
        self._bus_num  = i2c_bus
        self._addr     = address
        self._bus      = None
        self._heading  = 0.0
        self._last_t   = None
        self._scale    = 250.0 / 32768.0   # deg/s per LSB (FS_SEL=0)

    def open(self) -> None:
        try:
            import smbus2
            self._bus = smbus2.SMBus(self._bus_num)
        except ImportError:
            raise RuntimeError("smbus2 not installed.  Run: pip install smbus2")

        who = self._bus.read_byte_data(self._addr, self._WHO_AM_I)
        if who not in (0x68, 0x69, 0x70, 0x12):
            log.warning("MPU6050 WHO_AM_I = 0x%02X (expected 0x68)", who)

        # Wake up (clear sleep bit)
        self._bus.write_byte_data(self._addr, self._PWR_MGMT_1, 0x00)
        time.sleep(0.1)
        log.info("MPU6050 at 0x%02X on bus %d ready", self._addr, self._bus_num)

    def read(self) -> IMUReading:
        if self._bus is None:
            return IMUReading()
        try:
            hi = self._bus.read_byte_data(self._addr, self._GYRO_ZOUT_H)
            lo = self._bus.read_byte_data(self._addr, self._GYRO_ZOUT_H + 1)
            raw = (hi << 8) | lo
            if raw > 32767: raw -= 65536
            gz = raw * self._scale   # deg/s

            now = time.time()
            if self._last_t is not None:
                dt = now - self._last_t
                self._heading += gz * dt
            self._last_t = now

            return IMUReading(
                timestamp = now,
                heading   = self._heading % 360,
                pitch     = 0.0,
                roll      = 0.0,
                gyro_z    = gz,
                valid     = True,
            )
        except Exception as e:
            log.debug("MPU6050 read error: %s", e)
            return IMUReading()

    def close(self) -> None:
        if self._bus:
            self._bus.close()
            self._bus = None


# ── IMU background thread ─────────────────────────────────────────────────────

class IMUMonitor:
    """Reads the IMU at ~50 Hz in a background thread."""

    def __init__(self, reader):
        self._reader  = reader
        self._latest  = IMUReading()
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None
        self._history: list[IMUReading] = []

    def start(self) -> None:
        self._reader.open()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self._reader.close()

    def latest(self) -> IMUReading:
        with self._lock:
            return self._latest

    def snapshot_history(self) -> list[IMUReading]:
        """Return and clear the reading history."""
        with self._lock:
            h = list(self._history)
            self._history.clear()
        return h

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()

    def _loop(self) -> None:
        while self._running:
            r = self._reader.read()
            if r.valid:
                with self._lock:
                    self._latest = r
                    self._history.append(r)
            time.sleep(0.02)   # 50 Hz


# ── Atlas driver (thin, calibration-specific) ─────────────────────────────────

class AtlasDriver:
    """Minimal Atlas serial driver for calibration (no rover_agent dependency)."""

    WHEEL_BASE_MM      = 650
    MAX_VEL_REF_MM_S   = 200
    DRIVE_SPEED_PCT    = 60

    def __init__(self, port: str, baud: int = 115200, dry_run: bool = False):
        self._port    = port
        self._baud    = baud
        self._dry_run = dry_run
        self._ser     = None

    def connect(self) -> None:
        if self._dry_run:
            log.info("[dry-run] Atlas driver ready (no serial port opened)")
            return
        import serial
        self._ser = serial.Serial(
            self._port, self._baud, timeout=0.2, write_timeout=0.2)
        time.sleep(0.2)
        log.info("Atlas connected on %s @ %d", self._port, self._baud)

    def disconnect(self) -> None:
        self.stop()
        if self._ser:
            self._ser.close()
            self._ser = None
            log.info("Atlas disconnected")

    def send(self, L: int, R: int) -> None:
        frame = _atlas_frame(L, R)
        if self._dry_run:
            print(f"  [dry-run] {frame.decode().strip()}")
        else:
            if self._ser:
                self._ser.write(frame)
                self._ser.flush()

    def stop(self) -> None:
        self.send(0, 0)

    def drive_straight(self, pct: int = 60) -> None:
        self.send(pct, pct)

    def spin_right(self, pct: int = 60) -> None:
        """Spin in place clockwise."""
        self.send(pct, -pct)

    def spin_left(self, pct: int = 60) -> None:
        """Spin in place counter-clockwise."""
        self.send(-pct, pct)


# ── Heading helpers ───────────────────────────────────────────────────────────

def _heading_diff(start: float, end: float) -> float:
    """Signed shortest-arc difference (degrees), result in [-180, 180]."""
    d = (end - start) % 360
    if d > 180:
        d -= 360
    return d


# ── Test routines ─────────────────────────────────────────────────────────────

def test_imu_live(imu: IMUMonitor, duration: float = 10.0) -> None:
    """Print live IMU readings for `duration` seconds."""
    print(f"\n{'─'*60}")
    print(f"  Live IMU  ({duration:.0f} s)   Ctrl-C to stop early")
    print(f"{'─'*60}")
    print(f"  {'Time':>6}  {'Heading':>8}  {'Pitch':>7}  {'Roll':>7}  {'GyroZ':>8}")
    print(f"{'─'*60}")
    t0 = time.time()
    try:
        while time.time() - t0 < duration:
            r = imu.latest()
            if r.valid:
                print(f"\r  {time.time()-t0:6.1f}s  "
                      f"{r.heading:7.2f}°  "
                      f"{r.pitch:+6.2f}°  "
                      f"{r.roll:+6.2f}°  "
                      f"{r.gyro_z:+7.2f}°/s",
                      end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    print()


def test_turn(atlas: AtlasDriver, imu: IMUMonitor,
              target_deg: float = 90.0,
              speed_pct: int = 40,
              settle_s: float = 0.5) -> dict:
    """
    Command a turn and measure the actual heading change via IMU.

    Returns a result dict with commanded vs measured angle.
    """
    direction = "right" if target_deg > 0 else "left"
    print(f"\n{'─'*60}")
    print(f"  Turn test: {target_deg:+.0f}° ({direction})  speed={speed_pct}%")
    print(f"{'─'*60}")

    # Settle before measuring
    time.sleep(settle_s)
    h_before = imu.latest().heading
    print(f"  Heading before : {h_before:.2f}°")

    # Estimate duration from wheel geometry
    # angular_rate (deg/s) = 2 * v_tangential / wheelbase
    v_tangential_mm_s  = AtlasDriver.MAX_VEL_REF_MM_S * speed_pct / 100
    deg_per_s          = math.degrees(2 * v_tangential_mm_s / AtlasDriver.WHEEL_BASE_MM)
    duration           = abs(target_deg) / deg_per_s
    print(f"  Estimated rate : {deg_per_s:.1f} °/s  →  {duration:.2f} s")

    imu.clear_history()

    # Execute
    if target_deg > 0:
        atlas.spin_right(speed_pct)
    else:
        atlas.spin_left(speed_pct)
    time.sleep(duration)
    atlas.stop()
    time.sleep(settle_s)

    h_after  = imu.latest().heading
    measured = _heading_diff(h_before, h_after)
    error    = measured - target_deg

    print(f"  Heading after  : {h_after:.2f}°")
    print(f"  Commanded      : {target_deg:+.2f}°")
    print(f"  Measured       : {measured:+.2f}°")
    print(f"  Error          : {error:+.2f}°  ({abs(error)/abs(target_deg)*100:.1f}%)")

    result = {
        "test":      "turn",
        "target_deg": target_deg,
        "measured_deg": measured,
        "error_deg":  error,
        "speed_pct":  speed_pct,
        "duration_s": duration,
    }
    return result


def test_straight(atlas: AtlasDriver, imu: IMUMonitor,
                  duration: float = 2.0,
                  speed_pct: int = 40,
                  settle_s: float = 0.5) -> dict:
    """
    Drive straight and measure heading drift.

    A well-aligned rover should drift < 5° over 2 seconds.
    """
    print(f"\n{'─'*60}")
    print(f"  Straight-line test: {duration:.1f} s  speed={speed_pct}%")
    print(f"{'─'*60}")

    time.sleep(settle_s)
    h_before = imu.latest().heading
    print(f"  Heading before : {h_before:.2f}°")

    imu.clear_history()
    atlas.drive_straight(speed_pct)
    time.sleep(duration)
    atlas.stop()
    time.sleep(settle_s)

    h_after = imu.latest().heading
    drift   = _heading_diff(h_before, h_after)
    history = imu.snapshot_history()
    max_drift = 0.0
    if history:
        headings = [_heading_diff(h_before, r.heading) for r in history]
        max_drift = max(abs(h) for h in headings)

    print(f"  Heading after  : {h_after:.2f}°")
    print(f"  Final drift    : {drift:+.2f}°")
    print(f"  Max drift      : {max_drift:.2f}°")
    if abs(drift) < 3.0:
        print("  ✓  PASS — drift < 3°")
    elif abs(drift) < 8.0:
        print("  ⚠  MARGINAL — drift 3–8°, consider motor trim")
    else:
        print("  ✗  FAIL — drift > 8°, check wheel alignment or motor balance")

    return {
        "test":        "straight",
        "duration_s":  duration,
        "drift_deg":   drift,
        "max_drift_deg": max_drift,
        "speed_pct":   speed_pct,
    }


def test_full(atlas: AtlasDriver, imu: IMUMonitor, speed_pct: int = 40) -> None:
    """Run the full calibration sequence."""
    results = []

    print("\n" + "═"*60)
    print("  FULL CALIBRATION SEQUENCE")
    print("═"*60)
    input("  Press Enter to start (rover will move)…")

    # 1. Straight
    results.append(test_straight(atlas, imu, duration=2.0, speed_pct=speed_pct))
    time.sleep(1.0)

    # 2. Turn right 90°
    results.append(test_turn(atlas, imu, target_deg=+90, speed_pct=speed_pct))
    time.sleep(1.0)

    # 3. Turn left 90° (back to start)
    results.append(test_turn(atlas, imu, target_deg=-90, speed_pct=speed_pct))
    time.sleep(1.0)

    # 4. Turn right 180°
    results.append(test_turn(atlas, imu, target_deg=+180, speed_pct=speed_pct))
    time.sleep(1.0)

    # 5. Turn left 180° (back to start)
    results.append(test_turn(atlas, imu, target_deg=-180, speed_pct=speed_pct))

    # Summary
    print("\n" + "═"*60)
    print("  SUMMARY")
    print("═"*60)
    for r in results:
        if r["test"] == "turn":
            print(f"  Turn {r['target_deg']:+.0f}°  →  "
                  f"measured {r['measured_deg']:+.1f}°  "
                  f"error {r['error_deg']:+.1f}°")
        elif r["test"] == "straight":
            print(f"  Straight {r['duration_s']:.1f}s  →  "
                  f"drift {r['drift_deg']:+.1f}°  "
                  f"max {r['max_drift_deg']:.1f}°")
    print("═"*60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Rover calibration — send motion commands and read IMU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # IMU
    parser.add_argument("--imu", choices=["bno055", "mpu6050"], default="bno055",
                        help="IMU model (default: bno055)")
    parser.add_argument("--i2c-bus",     type=int, default=1,
                        help="I2C bus number (default: 1)")
    parser.add_argument("--i2c-address", type=lambda x: int(x, 0), default=None,
                        help="I2C address override, e.g. 0x29 (default: auto)")

    # Atlas
    parser.add_argument("--atlas-port", default="/dev/ttyACM0",
                        help="Atlas serial port (default: /dev/ttyACM0)")
    parser.add_argument("--atlas-baud", type=int, default=115200)
    parser.add_argument("--speed",      type=int, default=40,
                        help="Motor power %% for tests (default: 40)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print commands, do not open serial port")

    # Test selection
    parser.add_argument("--test",
                        choices=["imu", "turn", "straight", "full"],
                        default="imu",
                        help="Which test to run (default: imu — live monitor)")
    parser.add_argument("--angle",    type=float, default=90.0,
                        help="Target angle for turn test in degrees (default: 90)")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Duration for straight test in seconds (default: 2.0)")

    args = parser.parse_args()

    # ── Build IMU reader ──────────────────────────────────────────────────────
    if args.imu == "bno055":
        addr = args.i2c_address if args.i2c_address is not None else 0x28
        reader = BNO055Reader(i2c_bus=args.i2c_bus, address=addr)
    else:
        addr = args.i2c_address if args.i2c_address is not None else 0x68
        reader = MPU6050Reader(i2c_bus=args.i2c_bus, address=addr)

    imu = IMUMonitor(reader)

    # ── Build Atlas driver ────────────────────────────────────────────────────
    atlas = AtlasDriver(
        port    = args.atlas_port,
        baud    = args.atlas_baud,
        dry_run = args.dry_run,
    )

    # ── Connect ───────────────────────────────────────────────────────────────
    print(f"\nConnecting IMU ({args.imu} @ I2C bus {args.i2c_bus} addr 0x{addr:02X})…")
    try:
        imu.start()
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    print("Connecting Atlas…")
    try:
        atlas.connect()
    except Exception as e:
        print(f"  ERROR: {e}")
        imu.stop()
        sys.exit(1)

    # Brief settle
    time.sleep(0.5)
    r = imu.latest()
    if r.valid:
        print(f"  IMU OK — heading={r.heading:.1f}°  pitch={r.pitch:.1f}°  roll={r.roll:.1f}°")
    else:
        print("  IMU: no valid reading yet — check wiring")

    # ── Run selected test ─────────────────────────────────────────────────────
    try:
        if args.test == "imu":
            test_imu_live(imu, duration=60.0)

        elif args.test == "turn":
            result = test_turn(atlas, imu,
                               target_deg=args.angle,
                               speed_pct=args.speed)
            print(f"\nResult: {result}")

        elif args.test == "straight":
            result = test_straight(atlas, imu,
                                   duration=args.duration,
                                   speed_pct=args.speed)
            print(f"\nResult: {result}")

        elif args.test == "full":
            test_full(atlas, imu, speed_pct=args.speed)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        atlas.stop()
        atlas.disconnect()
        imu.stop()
        print("Done.")


if __name__ == "__main__":
    main()
