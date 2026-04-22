#!/usr/bin/env python3
"""
Robotic arm logic validator — interactive UI for testing and debugging.

Usage:
    python arm_validator.py 0 2        # overhead cam at 0, arm cam at 2
    python arm_validator.py 0          # single camera, no visual servo
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import queue
import threading
import time
import tkinter as tk
import cv2
from PIL import Image, ImageTk
from camera_resolutions import STANDARD_RESOLUTIONS
from arm_controller import ArmController, pixel_to_arm_xyz, ARM_Z_MIN, ARM_Z_MAX

_RED_LOW  = (168, 200, 140)
_RED_HIGH = (178, 255, 180)
_MIN_AREA = 200

PREVIEW_W         = 960
PREVIEW_H         = 540

MOVE_SETTLE_S     = 2.0
DWELL_S           = 1.0
SERVO_SETTLE_S    = 1.0
Z_STEP_MM         = 10
MAX_XY_STEP_MM    = 20
MAX_BLIND_RETRIES = 5
DOT_SIZE_FRACTION = 1 / 8
ARM_CAM_FOCAL_PX  = 320   # tune for actual arm camera FOV


def _max_resolution(device: int) -> tuple[int, int] | None:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        return None
    best = None
    for w, h in STANDARD_RESOLUTIONS:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if aw == w and ah == h:
            best = (w, h)
    cap.release()
    return best


def _detect_dots(frame):
    """Return list of (cx, cy, w, h) for each red contour above _MIN_AREA."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _RED_LOW, _RED_HIGH)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    for cnt in contours:
        if cv2.contourArea(cnt) < _MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        dots.append((x + w // 2, y + h // 2, w, h))
    return dots


def _process_frame(frame, detect: bool):
    """Returns (PIL image, list of (cx, cy) centres)."""
    dots_xy = []
    if detect:
        for cx, cy, w, h in _detect_dots(frame):
            x, y = cx - w // 2, cy - h // 2
            dots_xy.append((cx, cy))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"({cx},{cy})", (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb), dots_xy


# ── Arm camera (camera 2, mounted on end-effector) ───────────────────────────

class ArmCamera:
    """Synchronous capture from the arm-mounted camera for visual servoing."""

    def __init__(self, device: int):
        res = _max_resolution(device)
        self._cap = cv2.VideoCapture(device)
        if res:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  res[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        self._img_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._img_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def img_w(self):
        return self._img_w

    @property
    def img_h(self):
        return self._img_h

    def get_dot(self):
        """Read one frame. Returns (cx, cy, dot_w, img_w, img_h) for largest dot, or None."""
        ret, frame = self._cap.read()
        if not ret:
            return None
        dots = _detect_dots(frame)
        if not dots:
            return None
        # pick largest by bounding box area
        cx, cy, dw, dh = max(dots, key=lambda d: d[2] * d[3])
        return cx, cy, dw, self._img_w, self._img_h

    def get_frame_pil(self):
        """Return latest frame as PIL image with dot overlay, or None."""
        ret, frame = self._cap.read()
        if not ret:
            return None
        for cx, cy, w, h in _detect_dots(frame):
            x, y = cx - w // 2, cy - h // 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
            # draw threshold line
            thresh_w = self._img_w // 8
            cv2.line(frame, (self._img_w // 2 - thresh_w // 2, cy),
                     (self._img_w // 2 + thresh_w // 2, cy), (0, 255, 0), 1)
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def close(self):
        self._cap.release()


# ── Per-camera panel ──────────────────────────────────────────────────────────

class CameraPanel:
    def __init__(self, parent: tk.Widget, device: int, detect: bool = False,
                 arm: ArmController | None = None,
                 arm_cam: ArmCamera | None = None,
                 arm_cam_feed: ArmCamera | None = None):
        self.device        = device
        self._detect       = detect
        self._arm          = arm
        self._arm_cam      = arm_cam
        self._arm_cam_feed = arm_cam_feed  # if set, use this for display instead of own cap
        self._alive   = False
        self._queue: queue.Queue = queue.Queue(maxsize=1)

        self._targets: list[tuple[float, float, float]] = []
        self._target_pixels: list[tuple[int, int]] = []
        self._targets_locked = False
        self._current_target_idx = -1

        self._frame = tk.LabelFrame(parent, text=f"Camera {device}", padx=6, pady=6)
        self._video_label = tk.Label(self._frame, bg="black")
        self._video_label.pack()
        self._status_var = tk.StringVar(value="Opening…")
        tk.Label(self._frame, textvariable=self._status_var,
                 fg="gray", anchor=tk.W).pack(fill=tk.X)

        self._start()

    def grid(self, **kwargs):
        self._frame.grid(**kwargs)

    def _start(self):
        self._alive = True
        threading.Thread(target=self._capture_loop, daemon=True,
                         name=f"cam-{self.device}").start()
        if self._arm:
            threading.Thread(target=self._arm_loop, daemon=True,
                             name=f"arm-{self.device}").start()
        self._poll_ui()

    def _capture_loop(self):
        # If this panel is for the arm camera, display frames from ArmCamera directly
        if self._arm_cam_feed is not None:
            self._status_var.set(
                f"{self._arm_cam_feed.img_w}×{self._arm_cam_feed.img_h} (arm cam)")
            while self._alive:
                pil = self._arm_cam_feed.get_frame_pil()
                if pil is None:
                    continue
                pil = pil.resize((PREVIEW_W, PREVIEW_H), Image.BILINEAR)
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                self._queue.put(pil)
            return

        res = _max_resolution(self.device)
        cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            self._status_var.set("Could not open device")
            return
        if res:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  res[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])

        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._status_var.set(f"{img_w}×{img_h}")

        while self._alive:
            ret, frame = cap.read()
            if not ret:
                continue

            pil, dots = _process_frame(frame, self._detect and not self._targets_locked)

            if dots and not self._targets_locked:
                self._target_pixels = list(dots)
                self._targets = [
                    pixel_to_arm_xyz(cx, cy, img_w, img_h) for cx, cy in dots
                ]
                self._targets_locked = True
                print(f"Image size: {img_w}×{img_h}")
                for idx, ((cx, cy), (ax, ay, az)) in enumerate(
                        zip(self._target_pixels, self._targets)):
                    print(f"  dot {idx+1}: pixel=({cx},{cy}) → arm=({ax:.1f},{ay:.1f},{az:.1f})")

            if self._targets_locked:
                arr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
                for i, (px, py) in enumerate(self._target_pixels):
                    color = (0, 255, 255) if i == self._current_target_idx else (0, 255, 0)
                    cv2.circle(arr, (px, py), 10, color, 2)
                    cv2.putText(arr, str(i + 1), (px + 12, py),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                pil = Image.fromarray(arr)

            pil = pil.resize((PREVIEW_W, PREVIEW_H), Image.BILINEAR)

            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(pil)

        cap.release()

    def _arm_loop(self):
        print("Scanning for targets…")
        self._status_var.set("Scanning…")

        while self._alive and not self._targets_locked:
            time.sleep(0.1)
        if not self._alive:
            return

        targets = self._targets
        print(f"Found {len(targets)} targets")

        for i, (ax, ay, az) in enumerate(targets):
            if not self._alive:
                break
            self._current_target_idx = i
            n = f"{i + 1}/{len(targets)}"

            print(f"Coarse move to target {n}")
            self._status_var.set(f"Moving to {n}")
            self._arm.move_to(ax, ay, az)
            time.sleep(MOVE_SETTLE_S)

            if self._arm_cam:
                ax, ay, az = self._servo_to_dot(ax, ay, az, n)

            self._status_var.set(f"Dwelling at {n}")
            time.sleep(DWELL_S)

        self._current_target_idx = -1
        print("All targets visited.")
        self._status_var.set("Done")

    def _servo_to_dot(self, x: float, y: float, z: float, label: str):
        """Visual servo using arm camera. Returns final (x, y, z)."""
        blind_retries = 0

        while self._alive:
            result = self._arm_cam.get_dot()

            if result is None:
                blind_retries += 1
                if blind_retries >= MAX_BLIND_RETRIES:
                    print(f"Servo {label}: lost dot after {MAX_BLIND_RETRIES} retries, skipping")
                    break
                z = min(z + Z_STEP_MM, ARM_Z_MAX)
                self._arm.move_to(x, y, z)
                time.sleep(SERVO_SETTLE_S)
                continue

            blind_retries = 0
            cx, cy, dot_w, img_w, img_h = result

            print(f"Servoing {label}: z={z:.0f} dot={dot_w}px/{img_w}")
            self._status_var.set(f"Servoing {label}: z={z:.0f} dot={dot_w}px/{img_w}")

            if dot_w >= img_w * DOT_SIZE_FRACTION:
                print(f"Servo {label}: reached target size")
                break

            # Pixel error from image centre
            err_x = cx - img_w / 2
            err_y = cy - img_h / 2

            scale = z / ARM_CAM_FOCAL_PX
            dx = float(_clamp(err_y * scale, -MAX_XY_STEP_MM, MAX_XY_STEP_MM))
            dy = float(_clamp(-err_x * scale, -MAX_XY_STEP_MM, MAX_XY_STEP_MM))

            x += dx
            y += dy
            z = max(z - Z_STEP_MM, ARM_Z_MIN)

            self._arm.move_to(x, y, z)
            time.sleep(SERVO_SETTLE_S)

        return x, y, z

    def _poll_ui(self):
        if not self._alive:
            return
        try:
            pil = self._queue.get_nowait()
            img = ImageTk.PhotoImage(pil)
            self._video_label.configure(image=img)
            self._video_label.image = img
        except queue.Empty:
            pass
        self._frame.after(33, self._poll_ui)

    def destroy(self):
        self._alive = False
        self._frame.destroy()


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ── Main app ──────────────────────────────────────────────────────────────────

class ArmValidatorApp:
    def __init__(self, root: tk.Tk, devices: list[int]):
        self.root = root
        self.root.title("Arm Validator")
        self.root.resizable(True, True)

        self._panels: list[CameraPanel] = []
        self._arm_cam: ArmCamera | None = None

        self._arm: ArmController | None = None
        try:
            self._arm = ArmController()
            print("Arm connected on /dev/ttyUSB0")
        except Exception as e:
            print(f"WARNING: could not connect to arm ({e}) — running without arm control")

        frame = tk.Frame(root, padx=8, pady=8)
        frame.pack(fill=tk.BOTH, expand=True)

        # Device index 2 is expected to be the arm camera — open it for servo
        arm_cam_device = 2
        if self._arm and arm_cam_device in devices:
            try:
                self._arm_cam = ArmCamera(arm_cam_device)
                print(f"Arm camera opened on device {arm_cam_device} "
                      f"({self._arm_cam.img_w}×{self._arm_cam.img_h})")
            except Exception as e:
                print(f"WARNING: could not open arm camera ({e})")

        for col, device in enumerate(devices):
            arm     = self._arm     if col == 0 else None
            arm_cam = self._arm_cam if col == 0 else None
            # ArmCamera already owns the cap for arm_cam_device — pass it so
            # CameraPanel reuses it for display instead of opening a second cap
            arm_cam_feed = self._arm_cam if device == arm_cam_device else None
            panel = CameraPanel(frame, device, detect=(col == 0),
                                arm=arm, arm_cam=arm_cam,
                                arm_cam_feed=arm_cam_feed)
            panel.grid(row=0, column=col, padx=8, pady=4, sticky="n")
            self._panels.append(panel)

    def on_close(self):
        for p in self._panels:
            p.destroy()
        if self._arm:
            self._arm.close()
        if self._arm_cam:
            self._arm_cam.close()
        self.root.destroy()


def main():
    devices = [int(d) for d in sys.argv[1:]] if len(sys.argv) > 1 else [0]
    root = tk.Tk()
    app = ArmValidatorApp(root, devices)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
