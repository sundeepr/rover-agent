#!/usr/bin/env python3
"""
arm_panorama_capture.py — RoArm-M2-S green-plant scanner
==========================================================

Continuously rotates the arm base joint from -120° to +120° while streaming
the arm-tip camera feed. Any green plants detected in the frame are highlighted
with bounding boxes. When the largest bounding box centre falls inside a ±15%
centre zone (both horizontally and vertically), the arm LED turns on.

Hardware
--------
- RoArm-M2-S robotic arm (Waveshare), connected via USB serial (ESP32)
- Camera mounted at the tip of the arm (EoAT), default device index 0

Scan flow
---------
1. Load joint angles from arm_scan_config.json and move all joints to that pose
   (shoulder, elbow, EoAT first — then base, so the arm settles safely)
2. Move base to -120° using CMD_SINGLE_JOINT_ANGLE (T:121); wait for travel
3. Start continuous base rotation via CMD_CONSTANT_CTRL (T:123, cmd=1/INCREASE)
4. Per-frame loop:
   a. Read camera frame
   b. HSV green detection → contours → bounding boxes
   c. Poll live joint angles via CMD_SERVO_RAD_FEEDBACK (T:105 → response T:1051)
   d. If largest box centre is inside centre zone → LED on (T:114, led:255)
      else → LED off (T:114, led:0)
   e. Draw bounding boxes, HUD (angles + green %), and centre-zone crosshair
      on the live "Arm Scan" window
   f. Print to console when green detected above threshold
5. Stop when base reaches +120° (from feedback) or Q is pressed
6. Stop rotation, return base to 0°, home all joints, LED off

Key arm serial commands used
----------------------------
  T:102  CMD_JOINTS_RAD_CTRL      — move all joints (radians) — blocking
  T:105  CMD_SERVO_RAD_FEEDBACK   — request joint angle + coordinate feedback
  T:114  LED control              — led:255 = on, led:0 = off
  T:121  CMD_SINGLE_JOINT_ANGLE   — move one joint (degrees) — blocking on ESP32;
                                    do NOT poll T:105 during this move
  T:123  CMD_CONSTANT_CTRL        — continuous rotation; cmd:1=increase, cmd:0=stop

Config file: arm_scan_config.json
----------------------------------
Edit this file to change the home/scan pose without touching code:
  {
      "home": {
          "base_deg":     0,
          "shoulder_deg": -60,
          "elbow_deg":    150,
          "eoat_deg":     180
      }
  }
Shoulder -60° + elbow 150° keeps the camera roughly horizontal and pointed
forward. Adjust elbow by the same magnitude as shoulder to maintain level gaze
(e.g. shoulder -30° → elbow 120°).

Green detection
---------------
Uses an HSV mask (hue 35–85, sat 60–255, val 60–255) followed by morphological
closing to fill gaps inside plant blobs. Contours smaller than 500 px² are
discarded as noise. The "green %" in the HUD is the fraction of all frame pixels
that are green — 1% is the default trigger threshold (--threshold 0.01).

Sweep speed
-----------
--spd controls the continuous rotation speed coefficient (0–20, Waveshare units).
Default is 5 (slow). Use 8–12 for a faster sweep. At high speeds the arm may
overshoot +120° by a frame or two before the feedback loop catches it.

Usage
-----
    python arm_panorama_capture.py
    python arm_panorama_capture.py --spd 10
    python arm_panorama_capture.py --port /dev/ttyUSB1 --cam 2
    python arm_panorama_capture.py --threshold 0.02 --config my_pose.json

Arguments
---------
  --port      Serial port for the arm (default: /dev/ttyUSB0)
  --cam       Camera device index (default: 0)
  --spd       Continuous rotation speed 0–20 (default: 5)
  --threshold Green pixel fraction to log a detection (default: 0.01 = 1%)
  --out       Directory for any saved output (default: panorama_output/detections)
  --config    Path to JSON config file (default: arm_scan_config.json)

Press Q in the live window or Ctrl-C in the terminal to stop early.
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
    """Send T:105 and read back the T:1051 feedback line. Returns None on timeout."""
    send(ser, {"T": 105})
    deadline = time.time() + 0.5
    buf = b""
    while time.time() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and data.get("T") == 1051:
                        return data
                except (json.JSONDecodeError, ValueError):
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

# Minimum contour area (px²) to count as a plant — filters out noise
MIN_CONTOUR_AREA = 500


def detect_green(frame: cv2.typing.MatLike) -> tuple[float, list]:
    """
    Returns (green_ratio, contours) where contours are the bounding boxes
    of green regions sorted largest-first.
    """
    hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask    = cv2.inRange(hsv, GREEN_HSV_LO, GREEN_HSV_HI)
    # Morphological close to fill small holes inside plant blobs
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    ratio   = float(np.count_nonzero(mask)) / mask.size
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return ratio, contours


def draw_detections(frame: cv2.typing.MatLike, contours: list,
                    feedback: dict, ratio: float) -> cv2.typing.MatLike:
    out = frame.copy()

    # Draw bounding box + area label for each green contour
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        area = cv2.contourArea(c)
        cv2.putText(out, f"plant {i+1}  {area:.0f}px²",
                    (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    # HUD overlay — joint angles + green %
    b_deg = np.degrees(feedback.get("b", float("nan")))
    s_deg = np.degrees(feedback.get("s", float("nan")))
    e_deg = np.degrees(feedback.get("e", float("nan")))
    t_deg = np.degrees(feedback.get("t", float("nan")))
    hud = [
        f"green: {ratio*100:.1f}%   plants: {len(contours)}",
        f"base={b_deg:.1f}  shoulder={s_deg:.1f}",
        f"elbow={e_deg:.1f}  eoat={t_deg:.1f}",
    ]
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    y = 28
    for line in hud:
        # Dark shadow for readability over any background
        cv2.putText(out, line, (11, y + 1), font, scale, (0, 0, 0),     thickness + 1, cv2.LINE_AA)
        cv2.putText(out, line, (10, y),     font, scale, (0, 255, 0),   thickness,     cv2.LINE_AA)
        y += 28
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
    parser.add_argument("--no-ui",     action="store_true",
                        help="Disable the OpenCV display window (headless mode)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening serial port {args.port} at {BAUD_RATE} baud")
    # dsrdtr=False / rtscts=False prevents toggling DTR/RTS on open,
    # which would reset the ESP32 (Arduino bootloader behaviour).
    ser = serial.Serial(args.port, BAUD_RATE, timeout=0.5,
                        dsrdtr=False, rtscts=False)
    ser.dtr = False
    ser.rts = False
    time.sleep(4)   # allow ESP32 to be fully ready

    # Sweep parameters: config file takes precedence over CLI args
    sweep      = cfg.get("sweep", {})
    start_deg  = sweep.get("start_deg",        -120)
    end_deg    = sweep.get("end_deg",            120)
    sweep_spd  = sweep.get("spd",          args.spd)
    threshold  = sweep.get("green_threshold", args.threshold)

    cap = open_camera(args.cam)

    detection_count = 0
    try:
        home(ser, cfg)

        travel_time = abs(start_deg) / 30 + 3
        print(f"-- Moving base to {start_deg}° start position --")
        send(ser, {"T": 121, "joint": 1, "angle": start_deg, "spd": 30, "acc": 10})
        print(f"  Waiting {travel_time:.0f}s for base to reach {start_deg}°...")
        time.sleep(travel_time)
        print(f"  Base at {start_deg}°, starting sweep")

        print(f"\n-- Continuous base rotation {start_deg}° → {end_deg}°  spd={sweep_spd} --")
        print(f"   Green threshold: {threshold*100:.2f}%  |  Ctrl-C to stop\n")
        start_base_rotation(ser, sweep_spd)
        sweep_start_time = time.time()
        # Conservative max time: 240° at ~10°/s per speed unit + 5s headroom
        max_sweep_s = abs(end_deg - start_deg) / max(sweep_spd * 2, 1) + 5

        if not args.no_ui:
            cv2.namedWindow("Arm Scan", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Arm Scan", 1280, 720)

        led_on = False

        while True:
            frame = read_frame(cap)
            if frame is None:
                print("WARNING: dropped frame")
                continue

            ratio, contours = detect_green(frame)

            # Poll arm angles on every frame (non-blocking best-effort)
            feedback = request_feedback(ser) or {}
            b_rad = feedback.get("b", float("nan"))

            # Check if the largest green contour is centred in the frame (both axes)
            frame_cx = frame.shape[1] / 2
            frame_cy = frame.shape[0] / 2
            margin_x = frame.shape[1] * 0.15   # ±15% of frame width
            margin_y = frame.shape[0] * 0.15   # ±15% of frame height
            plant_centred = False
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                box_cx = x + w / 2
                box_cy = y + h / 2
                if abs(box_cx - frame_cx) <= margin_x and abs(box_cy - frame_cy) <= margin_y:
                    plant_centred = True

            # Turn LED on when a plant is centred, off otherwise
            if plant_centred and not led_on:
                send(ser, {"T": 114, "led": 255})
                led_on = True
                print("LED ON  — plant centred in frame")
            elif not plant_centred and led_on:
                send(ser, {"T": 114, "led": 0})
                led_on = False
                print("LED OFF — plant left frame centre")

            # Draw detections and show live window
            if not args.no_ui:
                display = draw_detections(frame, contours, feedback, ratio)
                # Draw centre-zone box (vertical + horizontal guide lines)
                cx, cy = int(frame_cx), int(frame_cy)
                mx, my = int(margin_x), int(margin_y)
                colour = (0, 255, 255) if plant_centred else (255, 255, 0)
                cv2.line(display, (cx - mx, 0),         (cx - mx, frame.shape[0]), colour, 1)
                cv2.line(display, (cx + mx, 0),         (cx + mx, frame.shape[0]), colour, 1)
                cv2.line(display, (0, cy - my),         (frame.shape[1], cy - my), colour, 1)
                cv2.line(display, (0, cy + my),         (frame.shape[1], cy + my), colour, 1)
                if plant_centred:
                    cv2.putText(display, "CENTRED", (cx - 50, frame.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Arm Scan", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nQ pressed — stopping.")
                    break

            if ratio >= threshold and contours:
                b_deg = np.degrees(b_rad)
                s_deg = np.degrees(feedback.get("s", float("nan")))
                e_deg = np.degrees(feedback.get("e", float("nan")))
                t_deg = np.degrees(feedback.get("t", float("nan")))
                print(
                    f"GREEN DETECTED  {ratio*100:.1f}%  plants={len(contours)}"
                    f"{'  [CENTRED]' if plant_centred else ''}  |  "
                    f"base={b_deg:.1f}°  shoulder={s_deg:.1f}°  "
                    f"elbow={e_deg:.1f}°  eoat={t_deg:.1f}°"
                )

            # Stop when base reaches end_deg (stop 5° early for deceleration)
            # or when max sweep time elapsed (feedback-unreliable fallback)
            elapsed = time.time() - sweep_start_time
            if elapsed >= max_sweep_s:
                print(f"\n-- Sweep time limit ({max_sweep_s:.0f}s) reached — stopping --")
                break
            if not np.isnan(b_rad) and np.degrees(b_rad) >= end_deg - 5.0:
                print("\n-- Base reached +120°, stopping --")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        print("Stopping rotation...")
        stop_base_rotation(ser)
        send(ser, {"T": 114, "led": 0})   # ensure LED off on exit
        time.sleep(0.5)
        print("-- Returning base to 0° --")
        send(ser, {"T": 121, "joint": 1, "angle": 0, "spd": 30, "acc": 10})
        time.sleep(4)
        home(ser, cfg)
        cap.release()
        ser.close()
        if not args.no_ui:
            cv2.destroyAllWindows()
        print(f"Done. {detection_count} green detection(s) saved to {out_dir}")


if __name__ == "__main__":
    main()
