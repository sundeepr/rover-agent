#!/usr/bin/env python3
"""
Continuously rotate the RoArm-M2-S base joint while reading camera frames.
If green is detected in a frame, the live joint angles are printed and the
frame is saved to disk.

Flow:
  1. Home all joints  (b=0 s=0 e=90 h=180 / EoAT closed)
  2. Start continuous base rotation (CMD_CONSTANT_CTRL T:123, direction=INCREASE)
  3. Read camera + poll joint angles in a tight loop
     - On each frame run HSV green-detection
     - If green pixel ratio exceeds threshold → print angles, save frame
  4. When base reaches +180° (or Ctrl-C), stop rotation and home

Usage:
    python arm_panorama_capture.py
    python arm_panorama_capture.py --port /dev/ttyUSB0 --cam 0 --spd 5 --threshold 0.01
    python arm_panorama_capture.py --out detections/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import serial

# ── Defaults ──────────────────────────────────────────────────────────────────
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE   = 115200
CAM_DEVICE  = 0
OUTPUT_DIR  = "panorama_output/detections"
DEFAULT_CONFIG = Path(__file__).parent / "arm_scan_config.json"

# HSV green range — tune if needed
GREEN_HSV_LO = np.array([ 35,  60,  60], dtype=np.uint8)
GREEN_HSV_HI = np.array([85, 255, 255], dtype=np.uint8)

# Minimum fraction of frame pixels that must be green to trigger a save
DEFAULT_GREEN_THRESHOLD = 0.01   # 1 %

# Continuous rotation speed coefficient (0-20 per docs)
DEFAULT_CONT_SPD = 5


# ── Serial helpers ─────────────────────────────────────────────────────────────

def send(ser: serial.Serial, cmd: dict) -> None:
    line = json.dumps(cmd) + "\n"
    ser.write(line.encode())


def load_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"ERROR: Config file not found: {path}")
    with open(path) as f:
        cfg = json.load(f)
    home = cfg.get("home", {})
    required = ["base_deg", "shoulder_deg", "elbow_deg", "eoat_deg"]
    missing = [k for k in required if k not in home]
    if missing:
        sys.exit(f"ERROR: Config missing fields in 'home': {missing}")
    print(f"Config loaded from {path}")
    print(f"  home: base={home['base_deg']}°  shoulder={home['shoulder_deg']}°  "
          f"elbow={home['elbow_deg']}°  eoat={home['eoat_deg']}°")
    return cfg


def home(ser: serial.Serial, cfg: dict) -> None:
    h = cfg["home"]
    print(f"\n-- Moving to scan home: base={h['base_deg']}°  shoulder={h['shoulder_deg']}°  "
          f"elbow={h['elbow_deg']}°  eoat={h['eoat_deg']}° --")
    send(ser, {"T": 121, "joint": 2, "angle": h["shoulder_deg"], "spd": 30, "acc": 10})
    time.sleep(2)
    send(ser, {"T": 121, "joint": 3, "angle": h["elbow_deg"],    "spd": 30, "acc": 10})
    time.sleep(2)
    send(ser, {"T": 121, "joint": 4, "angle": h["eoat_deg"],     "spd": 30, "acc": 10})
    time.sleep(1)
    send(ser, {"T": 121, "joint": 1, "angle": h["base_deg"],     "spd": 30, "acc": 10})
    time.sleep(3)


def start_base_rotation(ser: serial.Serial, spd: int) -> None:
    """Start continuous base rotation in the increasing direction (left)."""
    send(ser, {
        "T":    123,   # CMD_CONSTANT_CTRL
        "m":    0,     # angle control mode
        "axis": 1,     # base joint
        "cmd":  1,     # INCREASE (turns left)
        "spd":  spd,
    })


def stop_base_rotation(ser: serial.Serial) -> None:
    send(ser, {
        "T":    123,
        "m":    0,
        "axis": 1,
        "cmd":  0,     # STOP
        "spd":  0,
    })


def request_feedback(ser: serial.Serial) -> dict | None:
    """Send T:105 and read back the JSON feedback line. Returns None on timeout."""
    ser.reset_input_buffer()   # discard stale bytes from rotation responses
    send(ser, {"T": 105})
    deadline = time.time() + 0.3
    buf = b""
    while time.time() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf += chunk
            # Scan all complete lines in the buffer for the feedback response (T:1051)
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                try:
                    data = json.loads(line)
                    if data.get("T") == 1051:
                        return data
                except json.JSONDecodeError:
                    pass
    return None


# ── Camera helpers ─────────────────────────────────────────────────────────────

def open_camera(device: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        sys.exit(f"ERROR: Cannot open camera device {device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {device} opened at {w}x{h}")
    return cap


def read_frame(cap: cv2.VideoCapture) -> cv2.typing.MatLike | None:
    ret, frame = cap.read()
    return frame if ret else None


# ── Green detection ────────────────────────────────────────────────────────────

def green_ratio(frame: cv2.typing.MatLike) -> float:
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_HSV_LO, GREEN_HSV_HI)
    return float(np.count_nonzero(mask)) / mask.size


def annotate_frame(frame: cv2.typing.MatLike, feedback: dict, ratio: float) -> cv2.typing.MatLike:
    out  = frame.copy()
    b_deg = np.degrees(feedback.get("b", 0))
    s_deg = np.degrees(feedback.get("s", 0))
    e_deg = np.degrees(feedback.get("e", 0))
    t_deg = np.degrees(feedback.get("t", 0))
    lines = [
        f"GREEN {ratio*100:.1f}%",
        f"base={b_deg:.1f}  shoulder={s_deg:.1f}",
        f"elbow={e_deg:.1f}  eoat={t_deg:.1f}",
    ]
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.8
    thickness = 2
    y         = 30
    for line in lines:
        cv2.putText(out, line, (10, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
        y += 30
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sweep arm base continuously; save frames where green is detected"
    )
    parser.add_argument("--port",      default=SERIAL_PORT,
                        help="Serial port (default: /dev/ttyUSB0)")
    parser.add_argument("--cam",       type=int,   default=CAM_DEVICE,
                        help="Camera device index (default: 0)")
    parser.add_argument("--spd",       type=int,   default=DEFAULT_CONT_SPD,
                        help="Continuous rotation speed coefficient 0-20 (default: 5)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_GREEN_THRESHOLD,
                        help="Green pixel fraction to trigger save (default: 0.01 = 1%%)")
    parser.add_argument("--out",       default=OUTPUT_DIR,
                        help=f"Output directory for detections (default: {OUTPUT_DIR})")
    parser.add_argument("--config",    default=str(DEFAULT_CONFIG),
                        help=f"Path to arm config JSON (default: {DEFAULT_CONFIG})")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening serial port {args.port} at {BAUD_RATE} baud")
    ser = serial.Serial(args.port, BAUD_RATE, timeout=0.5)
    time.sleep(2)   # allow ESP32 to boot

    cap = open_camera(args.cam)

    detection_count = 0
    try:
        home(ser, cfg)

        # Move base to -180° and poll until it actually arrives
        print("-- Moving base to -180° start position --")
        send(ser, {"T": 121, "joint": 1, "angle": -180, "spd": 30, "acc": 10})
        while True:
            fb = request_feedback(ser)
            if fb is not None:
                base_now = np.degrees(fb.get("b", 0))
                print(f"  base={base_now:.1f}°", end="\r")
                if base_now <= -178.0:
                    print(f"\n  Reached {base_now:.1f}°, starting sweep")
                    break
            time.sleep(0.1)

        print(f"\n-- Starting continuous base rotation -180° → +180°  spd={args.spd} --")
        print(f"   Green threshold: {args.threshold*100:.2f}%  |  Ctrl-C to stop\n")
        start_base_rotation(ser, args.spd)

        while True:
            frame = read_frame(cap)
            if frame is None:
                print("WARNING: dropped frame")
                continue

            ratio = green_ratio(frame)

            # Poll arm angles on every frame (non-blocking best-effort)
            feedback = request_feedback(ser) or {}

            b_rad = feedback.get("b", float("nan"))
            s_rad = feedback.get("s", float("nan"))
            e_rad = feedback.get("e", float("nan"))
            t_rad = feedback.get("t", float("nan"))

            if ratio >= args.threshold:
                b_deg = np.degrees(b_rad)
                s_deg = np.degrees(s_rad)
                e_deg = np.degrees(e_rad)
                t_deg = np.degrees(t_rad)

                print(
                    f"GREEN DETECTED  {ratio*100:.1f}%  |  "
                    f"base={b_deg:.1f}°  shoulder={s_deg:.1f}°  "
                    f"elbow={e_deg:.1f}°  eoat={t_deg:.1f}°"
                )

                annotated = annotate_frame(frame, feedback, ratio)
                fname = out_dir / f"green_{detection_count:04d}_b{b_deg:.0f}.jpg"
                cv2.imwrite(str(fname), annotated)
                print(f"  Saved → {fname}")
                detection_count += 1

            # Stop when base has completed a full sweep to +180°
            if not np.isnan(b_rad) and np.degrees(b_rad) >= 179.0:
                print("\n-- Base reached +180°, stopping --")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        print("Stopping rotation...")
        stop_base_rotation(ser)
        time.sleep(0.5)
        print("-- Returning base to 0° --")
        send(ser, {"T": 121, "joint": 1, "angle": 0, "spd": 30, "acc": 10})
        time.sleep(4)
        home(ser, cfg)
        cap.release()
        ser.close()
        print(f"Done. {detection_count} green detection(s) saved to {out_dir}")


if __name__ == "__main__":
    main()
