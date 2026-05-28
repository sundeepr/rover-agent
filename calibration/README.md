# Rover Calibration

Scripts for verifying and tuning the Atlas-1 rover's motion against IMU measurements.

## Hardware needed

| Component | Details |
|---|---|
| IMU | **BNO055** (recommended) or MPU-6050 |
| Connection | I2C to Jetson — SDA→pin 3, SCL→pin 5, 3.3 V, GND |
| Atlas rover | Connected via `/dev/ttyACM0` as usual |

### Why BNO055?
The BNO055 does full sensor fusion on-chip (accelerometer + gyro + magnetometer) and outputs ready-to-use Euler angles. The MPU-6050 only outputs raw gyro data, which must be integrated — it drifts noticeably after a few seconds.

### Wiring (Jetson Orin / Xavier)
```
BNO055 VIN  →  3.3 V  (pin 1)
BNO055 GND  →  GND    (pin 6)
BNO055 SDA  →  SDA1   (pin 3)
BNO055 SCL  →  SCL1   (pin 5)
BNO055 ADDR →  GND    (→ I2C address 0x28)
```

Verify the IMU is visible:
```bash
sudo i2cdetect -y 1
# Should show 0x28 (BNO055) or 0x68 (MPU-6050)
```

## Install dependencies

```bash
pip install smbus2 pyserial
```

## Usage

### 1. Live IMU monitor (no movement)
```bash
python calibration/calibrate.py --imu bno055
```
Prints heading, pitch, roll, gyro-Z at 10 Hz for 60 s.  
Use this to verify the IMU is wired correctly before running any motion tests.

### 2. Turn test
```bash
# 90° right
python calibration/calibrate.py --imu bno055 --atlas-port /dev/ttyACM0 \
    --test turn --angle 90 --speed 40

# 90° left
python calibration/calibrate.py --imu bno055 --atlas-port /dev/ttyACM0 \
    --test turn --angle -90 --speed 40
```
Commands the rover to spin and measures the actual heading change.  
**Good result:** measured ≈ commanded ± 5°.

### 3. Straight-line test
```bash
python calibration/calibrate.py --imu bno055 --atlas-port /dev/ttyACM0 \
    --test straight --duration 2.0 --speed 40
```
Drives forward and measures heading drift.  
**Good result:** drift < 3° over 2 s.  
If drift is consistently left or right, the wheels need trimming (adjust `L` vs `R` power offset in `atlas_controller.py`).

### 4. Full calibration sequence
```bash
python calibration/calibrate.py --imu bno055 --atlas-port /dev/ttyACM0 \
    --test full --speed 40
```
Runs straight → turn 90° right → turn 90° left → turn 180° right → turn 180° left.  
Prints a summary table at the end.

### Dry-run (no hardware)
```bash
python calibration/calibrate.py --dry-run --test full
```
Prints all commands without opening any serial ports.

## Interpreting results

| Result | Meaning | Action |
|---|---|---|
| Turn error < ±5° | Good | No change needed |
| Turn error 5–15° | Marginal | Adjust `deg_per_s` estimate in `atlas_controller.py` `_turn()` |
| Turn error > 15° | Poor | Check wheel slip, battery voltage, surface grip |
| Straight drift < 3° | Good | No change needed |
| Straight drift 3–8° | Marginal | Trim one motor slightly (add/subtract 2–5% from L or R) |
| Straight drift > 8° | Poor | Wheel misalignment or large motor speed mismatch |

## Troubleshooting

**`BNO055 not found`** — run `sudo i2cdetect -y 1` to confirm address.  
Use `--i2c-address 0x29` if the ADDR pin is pulled high.

**`smbus2 not installed`** — run `pip install smbus2`.

**Atlas not responding** — confirm port with `ls /dev/ttyACM*`, check USB cable.

**Heading jumps around** — BNO055 needs to warm up for ~5 s after power-on. Wait before testing.
