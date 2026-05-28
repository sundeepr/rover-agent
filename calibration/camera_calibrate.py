#!/usr/bin/env python3
"""
Camera calibration for OmniVLA waypoint perspective projection.

Place a scale with red dots exactly 100 mm apart on the ground in front
of the camera.  Two dot arrangements are supported:

  --axis forward  (recommended)
      Dots in a line going AWAY from the camera along the ground.
      Nearest dot is at --distance mm; each subsequent dot is
      --dot-spacing mm further.  Calibrates the vertical (fy) focal
      length and the horizon line from the perspective convergence.
      cx is taken as the mean x-pixel of all dots (should be ~image centre).
      fx = fy (square-pixel assumption).

  --axis horizontal
      Dots in a line across the camera (perpendicular to forward axis).
      Ruler is at --distance mm; needs --camera-height to derive the
      horizon line.  Calibrates fx only; fy = fx.

The JSON is loaded by the rover agent to project OmniVLA BEV waypoints
(x_m lateral, y_m forward) to image pixel coordinates:

    u = cx + (x_m / y_m) * fx
    v = vy_horizon + (h_cam_m / y_m) * fy

Usage
─────
Forward dots (your ruler going away from camera):
    python calibration/camera_calibrate.py \\
        --image frame.jpg \\
        --axis forward \\
        --distance 320 \\
        --camera-height 820

Horizontal dots (ruler across the frame):
    python calibration/camera_calibrate.py \\
        --image frame.jpg \\
        --axis horizontal \\
        --distance 500 \\
        --camera-height 220

Live capture:
    python calibration/camera_calibrate.py \\
        --device 0 --axis forward --distance 320 --camera-height 820
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np


# ── Red dot detection ─────────────────────────────────────────────────────────

def _detect_red_blobs(frame: np.ndarray,
                      min_area: int = 30,
                      max_area: int = 8000) -> list[tuple[float, float]]:
    """Find red blob centroids in frame using HSV masking."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lo1 = np.array([  0, 100, 60])
    hi1 = np.array([ 10, 255, 255])
    lo2 = np.array([165, 100, 60])
    hi2 = np.array([180, 255, 255])

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lo1, hi1),
        cv2.inRange(hsv, lo2, hi2),
    )

    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        a = cv2.contourArea(c)
        if min_area <= a <= max_area:
            M = cv2.moments(c)
            if M["m00"] > 0:
                blobs.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))

    return blobs


# ── Calibration: forward axis ─────────────────────────────────────────────────

def _calibrate_forward(blobs: list[tuple[float, float]],
                       start_dist_mm: float,
                       dot_spacing_mm: float,
                       h_cam_mm: float,
                       frame_shape: tuple) -> dict:
    """
    Calibrate from dots arranged going away from the camera.

    Blobs are sorted bottom-to-top (nearest first).
    Each blob i is at forward distance  y_i = start_dist + i * dot_spacing.

    Model:  v = vy_horizon + (h_cam / y) * fy

    Fit fy and vy_horizon using least-squares over all blobs.
    cx = mean x-pixel of all blobs.
    fx = fy (square pixels).
    """
    h, w = frame_shape[:2]

    # Sort bottom-to-top (large v = near, small v = far)
    blobs_sorted = sorted(blobs, key=lambda b: b[1], reverse=True)
    n = len(blobs_sorted)

    # Build arrays:  v_i = vy_horizon + (h_cam / y_i) * fy
    # Rewrite as:    v_i = A * (1/y_i) + B
    # where A = h_cam * fy,  B = vy_horizon
    y_mm = np.array([start_dist_mm + i * dot_spacing_mm for i in range(n)])
    v_px = np.array([b[1] for b in blobs_sorted])
    x_px = np.array([b[0] for b in blobs_sorted])

    # Least-squares:  [1/y_i, 1] * [A, B]^T = v_i
    X = np.column_stack([1.0 / y_mm, np.ones(n)])
    coeffs, _, _, _ = np.linalg.lstsq(X, v_px, rcond=None)
    A, B = coeffs   # A = h_cam * fy,  B = vy_horizon

    fy = A / h_cam_mm
    vy_horizon = B
    fx = fy          # square pixels
    cx = float(np.mean(x_px))

    # Per-dot residuals for diagnostics
    v_pred = A / y_mm + B
    residuals = v_px - v_pred

    return dict(
        fx             = round(fx, 2),
        fy             = round(fy, 2),
        cx             = round(cx, 2),
        cy             = round(h / 2.0, 2),
        vy_horizon     = round(vy_horizon, 2),
        camera_height_m= round(h_cam_mm / 1000.0, 4),
        image_w        = w,
        image_h        = h,
        dot_spacing_mm = dot_spacing_mm,
        method         = "forward_axis",
        n_dots         = n,
        rmse_px        = round(float(np.sqrt(np.mean(residuals**2))), 2),
        _dots_y_mm     = y_mm.tolist(),
        _dots_v_px     = v_px.tolist(),
        _dots_v_pred   = v_pred.tolist(),
    )


# ── Calibration: horizontal axis ──────────────────────────────────────────────

def _calibrate_horizontal(blobs: list[tuple[float, float]],
                          distance_mm: float,
                          dot_spacing_mm: float,
                          h_cam_mm: float,
                          frame_shape: tuple) -> dict:
    """
    Calibrate from dots arranged horizontally across the frame.

    fx = pixel_spacing * distance / dot_spacing
    vy_horizon derived from ruler's pixel row and camera height.
    fy = fx (square pixels).
    """
    h, w = frame_shape[:2]
    blobs_sorted = sorted(blobs, key=lambda b: b[0])   # left to right

    spacings = [blobs_sorted[i+1][0] - blobs_sorted[i][0]
                for i in range(len(blobs_sorted)-1)]
    px_spacing = float(np.mean(spacings))

    fx = px_spacing * distance_mm / dot_spacing_mm
    fy = fx
    cx = w / 2.0
    blob_v = float(np.mean([b[1] for b in blobs_sorted]))
    vy_horizon = blob_v - (h_cam_mm / distance_mm) * fy

    return dict(
        fx             = round(fx, 2),
        fy             = round(fy, 2),
        cx             = round(cx, 2),
        cy             = round(h / 2.0, 2),
        vy_horizon     = round(vy_horizon, 2),
        camera_height_m= round(h_cam_mm / 1000.0, 4),
        image_w        = w,
        image_h        = h,
        dot_spacing_mm = dot_spacing_mm,
        method         = "horizontal_axis",
        n_dots         = len(blobs_sorted),
        rmse_px        = 0.0,
    )


# ── Projection helper ─────────────────────────────────────────────────────────

def bev_to_pixel(x_m, y_m, calib, frame_w, frame_h):
    if y_m < 0.05:
        return None
    u = calib["cx"] + (x_m / y_m) * calib["fx"]
    v = calib["vy_horizon"] + (calib["camera_height_m"] / y_m) * calib["fy"]
    u = int(round(max(0, min(frame_w - 1, u))))
    v = int(round(max(0, min(frame_h - 1, v))))
    return u, v


# ── Visualisation ─────────────────────────────────────────────────────────────

def _annotate(frame, blobs, calib, axis, start_dist_mm, dot_spacing_mm):
    out   = frame.copy()
    h, w  = out.shape[:2]

    # Sort blobs same way as calibration
    if axis == "forward":
        blobs_s = sorted(blobs, key=lambda b: b[1], reverse=True)
    else:
        blobs_s = sorted(blobs, key=lambda b: b[0])

    # Draw detected blobs with distance label
    for i, (bx, by) in enumerate(blobs_s):
        dist = start_dist_mm + i * dot_spacing_mm
        cv2.circle(out, (int(bx), int(by)), 8, (0, 255, 0), 2)
        cv2.putText(out, f"{dist:.0f}", (int(bx)+10, int(by)+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Horizon line
    vy = int(calib["vy_horizon"])
    if 0 <= vy < h:
        cv2.line(out, (0, vy), (w, vy), (255, 100, 0), 1)
        cv2.putText(out, "horizon", (5, vy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1)
    else:
        cv2.putText(out, f"horizon @ y={vy}px (outside frame)",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,100,0), 1)

    # Project test waypoints at 0.3, 0.5, 0.8, 1.0, 1.5, 2.0 m along centre
    for y_m in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        pt = bev_to_pixel(0.0, y_m, calib, w, h)
        if pt:
            cv2.circle(out, pt, 5, (0, 100, 255), -1)
            cv2.putText(out, f"{y_m:.1f}m", (pt[0]+6, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

    # RMSE badge
    rmse = calib.get("rmse_px", 0)
    label = (f"fx={calib['fx']:.0f}px  fy={calib['fy']:.0f}px  "
             f"h={calib['camera_height_m']*1000:.0f}mm  "
             f"horizon={calib['vy_horizon']:.0f}px  "
             f"RMSE={rmse:.1f}px  n={calib['n_dots']}")
    cv2.putText(out, label, (8, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 230, 255), 1)
    return out


# ── Capture ───────────────────────────────────────────────────────────────────

def _capture(device, warmup=20):
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(warmup):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Capture failed")
    return frame


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate camera for OmniVLA waypoint projection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--device",        type=int,   default=0)
    parser.add_argument("--image",         type=str,   default=None)
    parser.add_argument("--axis",          choices=["forward", "horizontal"],
                        default="forward",
                        help="Dot arrangement: forward (going away) or "
                             "horizontal (across frame). Default: forward")
    parser.add_argument("--distance",      type=float, required=True,
                        help="Distance to nearest dot (forward) or to ruler "
                             "(horizontal), in mm")
    parser.add_argument("--camera-height", type=float, required=True,
                        help="Camera height above ground in mm")
    parser.add_argument("--dot-spacing",   type=float, default=100.0,
                        help="Real-world spacing between dots in mm (default 100)")
    parser.add_argument("--output",        type=str,
                        default="camera_calibration.json")
    parser.add_argument("--no-show",       action="store_true")
    args = parser.parse_args()

    # ── Load frame ────────────────────────────────────────────────────────────
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"ERROR: cannot read {args.image}"); sys.exit(1)
        print(f"Loaded: {args.image}  {frame.shape[1]}×{frame.shape[0]}")
    else:
        print(f"Capturing from device {args.device}…")
        frame = _capture(args.device)
        fname = f"calibration_frame_{int(time.time())}.jpg"
        cv2.imwrite(fname, frame)
        print(f"Saved frame → {fname}")

    # ── Detect blobs ──────────────────────────────────────────────────────────
    print("Detecting red blobs…")
    blobs = _detect_red_blobs(frame)
    print(f"  Found {len(blobs)} blob(s)")
    if blobs:
        if args.axis == "forward":
            blobs_s = sorted(blobs, key=lambda b: b[1], reverse=True)
        else:
            blobs_s = sorted(blobs, key=lambda b: b[0])
        for i, (bx, by) in enumerate(blobs_s):
            dist = args.distance + i * args.dot_spacing
            print(f"  dot {i:2d}: pixel ({bx:6.1f}, {by:6.1f})  "
                  f"→ {dist:.0f} mm from camera")

    if len(blobs) < 2:
        print("\nERROR: need at least 2 red blobs. Check lighting / dot colour.")
        sys.exit(1)

    # ── Compute calibration ───────────────────────────────────────────────────
    if args.axis == "forward":
        calib = _calibrate_forward(blobs, args.distance, args.dot_spacing,
                                   args.camera_height, frame.shape)
    else:
        calib = _calibrate_horizontal(blobs, args.distance, args.dot_spacing,
                                      args.camera_height, frame.shape)

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print("  CALIBRATION RESULT")
    print(f"{'═'*55}")
    print(f"  Axis             : {args.axis}")
    print(f"  Dots used        : {calib['n_dots']}")
    print(f"  fx / fy          : {calib['fx']:.1f} / {calib['fy']:.1f} px")
    print(f"  Principal pt cx  : {calib['cx']:.1f} px")
    print(f"  Horizon line vy  : {calib['vy_horizon']:.1f} px")
    print(f"  Camera height    : {calib['camera_height_m']*1000:.1f} mm")
    print(f"  Fit RMSE         : {calib['rmse_px']:.2f} px")
    print(f"{'═'*55}")

    if calib['rmse_px'] > 5:
        print("  ⚠  RMSE > 5 px — possible fisheye distortion or measurement error")
        print("     Calibration still usable but projection accuracy is limited.")

    # ── Save ──────────────────────────────────────────────────────────────────
    # Strip internal debug keys before saving
    out_calib = {k: v for k, v in calib.items() if not k.startswith("_")}
    with open(args.output, "w") as f:
        json.dump(out_calib, f, indent=2)
    print(f"\n  Saved → {args.output}")

    # ── Visualise ─────────────────────────────────────────────────────────────
    has_display = (bool(os.environ.get("DISPLAY"))
                   or sys.platform == "darwin"
                   or os.name == "nt")
    vis = _annotate(frame, blobs, calib, args.axis,
                    args.distance, args.dot_spacing)
    vis_path = args.output.replace(".json", "_annotated.jpg")
    cv2.imwrite(vis_path, vis)
    print(f"  Saved annotated → {vis_path}")

    if not args.no_show and has_display:
        cv2.imshow("Calibration — press any key", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\nDone. Use with:")
    print(f"  python rover_agent.py --camera-calibration {args.output} ...")


if __name__ == "__main__":
    main()
