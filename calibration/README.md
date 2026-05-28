# Rover Calibration

Scripts for verifying and tuning the Atlas-1 rover's motion using the
**built-in STM32 IMU** on `/dev/ttyACM0`.  No external IMU hardware needed.

## How it works

The Atlas STM32 continuously streams IMU frames on the same serial port used
for motor commands:

```
$IMU,<ts_ms>,<qw>,<qx>,<qy>,<qz>,<gx>,<gy>,<gz>,<ax>,<ay>,<az>,<temp>,<voltage>#
```

`calibrate.py` opens that port once, reads IMU frames in a background thread,
and sends motor commands from the main thread.  Heading is derived from the
quaternion on-the-fly.

## Install dependencies

```bash
pip install pyserial
```

## Usage

### 0. Sniff raw serial data (optional — to see what the rover is sending)

```bash
python calibration/read_serial.py --port /dev/ttyACM0
```

### 1. Live IMU monitor (no movement)

```bash
python calibration/calibrate.py
# or explicitly:
python calibration/calibrate.py --test imu --monitor-time 60
```

Prints heading, pitch, roll, gyro-Z, voltage, temperature at 10 Hz.
Use this to verify the IMU is alive before running any motion tests.

### 2. Turn test

```bash
# 90° right
python calibration/calibrate.py --test turn --angle 90 --speed 40

# 90° left
python calibration/calibrate.py --test turn --angle -90 --speed 40

# 180° right
python calibration/calibrate.py --test turn --angle 180 --speed 40
```

Commands the rover to spin and measures the actual heading change.
**Good result:** measured ≈ commanded ± 5°.

### 3. Straight-line test

```bash
python calibration/calibrate.py --test straight --duration 2.0 --speed 40
```

Drives forward and measures heading drift.
**Good result:** drift < 3° over 2 s.
If drift is consistently left or right, trim `L` vs `R` power offset in
`atlas_controller.py`.

### 4. Full calibration sequence

```bash
python calibration/calibrate.py --test full --speed 40
```

Runs straight → turn +90° → turn −90° → turn +180° → turn −180°.
Prints a ✓/⚠/✗ summary table at the end.

### Dry-run (no hardware)

```bash
python calibration/calibrate.py --test full --dry-run
```

Prints all commands without opening any serial port.

## Interpreting results

| Result | Meaning | Action |
|---|---|---|
| Turn error < ±5° | ✓ Good | No change needed |
| Turn error 5–15° | ⚠ Marginal | Adjust `deg_per_s` estimate in `atlas_controller.py` `_turn()` |
| Turn error > 15° | ✗ Poor | Check wheel slip, battery voltage, surface grip |
| Straight drift < 3° | ✓ Good | No change needed |
| Straight drift 3–8° | ⚠ Marginal | Trim one motor slightly (add/subtract 2–5% to L or R) |
| Straight drift > 8° | ✗ Poor | Wheel misalignment or large motor speed mismatch |

## Troubleshooting

**No IMU data in 5 s** — confirm port with `ls /dev/ttyACM*`, check USB cable.
Make sure nothing else (rover_agent, read_serial.py) has the port open.

**Permission denied on /dev/ttyACM0** — add yourself to the `dialout` group:
```bash
sudo usermod -a -G dialout $USER
# then log out and back in
```

**Heading stays at 0°** — the STM32 may not be in AUTO mode yet.  Wait a few
seconds after connecting; the `$DATA,SWD=ON (ARMED)#` message indicates the
IMU is live.
