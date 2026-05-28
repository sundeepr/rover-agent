#!/usr/bin/env python3
"""
Camera calibration for OmniVLA waypoint perspective projection.

Place a scale (ruler) with red dots exactly 100 mm apart on the ground
in front of the camera.  This script:

  1. Captures a frame (or loads a saved image).
  2. Auto-detects red dots using HSV masking.
  3. Computes the camera focal length (fx) from the pixel spacing and
     the known real-world spacing at a known distance.
  4. Derives the horizon line (vy_horizon) and camera height.
  5. Saves the calibration to  camera_calibration.json.

The JSON is loaded by the rover agent at startup to project OmniVLA
BEV waypoints (x_m lateral, y_m forward) into camera pixel coordinates:

    u = cx + (x_m / y_m) * fx
    v = vy_horizon + (h_cam_m / y_m) * fy

Usage
─────
Single ruler at a known distance:
    python calibration/camera_calibrate.py \\
        --device 0 --distance 600 --camera-height 220

Two rulers at different distances (auto-derives h_cam & horizon):
    python calibration/camera_calibrate.py \\
        --device 0 --distance 400 --distance2 800

From a saved image:
    python calibration/camera_calibrate.py \\
        --image frame.jpg --distance 600 --camera-height 220

Arguments
─────────
--device          Camera index (default 0)
--image           Path to a saved image instead of live camera
--distance        Distance from camera to ruler in mm (required)
--distance2       Distance to a second ruler (enables auto h_cam + horizon)
--camera-height   Camera height above ground in mm (single-ruler mode)
--dot-spacing     Real-world dot spacing in mm (default 100)
--output          Output JSON path (default camera_calibration.json)
--show            Show annotated frame (default: show if display available)
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
                      min_area: int = 50,
                      max_area: int = 8000) -> list[tuple[float, float]]:
    """
    Find red blob centroids in frame using HSV masking.

    Returns list of (x, y) pixel centroids sorted left-to-right.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red wraps around hue 0/180 in OpenCV HSV
    lo1 = np.array([  0, 100, 80])
    hi1 = np.array([ 10, 255, 255])
    lo2 = np.array([165, 100, 80])
    hi2 = np.array([180, 255, 255])

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lo1, hi1),
        cv2.inRange(hsv, lo2, hi2),
    )

    # Morphological clean-up
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        a = cv2.contourArea(c)
        if min_area <= a <= max_area:
            M  = cv2.moments(c)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                blobs.append((cx, cy))

    # Sort left-to-right
    blobs.sort(key=lambda p: p[0])
    return blobs


def _pixel_spacing(blobs: list[tuple[float, float]]) -> float:
    """Average horizontal pixel distance between consecutive blobs."""
    if len(blobs) < 2:
        raise ValueError(f"Need ≥ 2 red blobs, found {len(blobs)}")
    spacings = [blobs[i+1][0] - blobs[i][0] for i in range(len(blobs) - 1)]
    return float(np.mean(spacings))


def _mean_y(blobs: list[tuple[float, float]]) -> float:
    """Average pixel row of blobs (vertical position of ruler in frame)."""
    return float(np.mean([b[1] for b in blobs]))


# ── Calibration math ──────────────────────────────────────────────────────────

def _compute_single(px_spacing: float, blob_y: float,
                    distance_mm: float, dot_spacing_mm: float,
                    h_cam_mm: float, frame_shape: tuple) -> dict:
    """
    Compute calibration from a single ruler placement.

      px_spacing    – pixel distance between adjacent dots
      blob_y        – pixel row of the ruler
      distance_mm   – real-world distance from camera to ruler
      dot_spacing_mm– real-world dot spacing (default 100 mm)
      h_cam_mm      – camera height above ground in mm
    """
    h, w = frame_shape[:2]
    fx = px_spacing * distance_mm / dot_spacing_mm
    fy = fx   # assume square pixels

    cx = w / 2.0
    cy = h / 2.0

    # ruler appears at pixel row blob_y and at known distance
    # blob_y = vy_horizon + (h_cam_mm / distance_mm) * fy
    vy_horizon = blob_y - (h_cam_mm / distance_mm) * fy

    return dict(
        fx            = round(fx, 2),
        fy            = round(fy, 2),
        cx            = round(cx, 2),
        cy            = round(cy, 2),
        vy_horizon    = round(vy_horizon, 2),
        camera_height_m = round(h_cam_mm / 1000.0, 4),
        image_w       = w,
        image_h       = h,
        dot_spacing_mm = dot_spacing_mm,
        method        = "single_ruler",
    )


def _compute_dual(px_spacing1: float, blob_y1: float, d1_mm: float,
                  px_spacing2: float, blob_y2: float, d2_mm: float,
                  dot_spacing_mm: float, frame_shape: tuple) -> dict:
    """
    Compute calibration from two ruler placements at different distances.
    Automatically solves for h_cam and vy_horizon without user measurement.

    From the pinhole model:
        blob_y = vy_horizon + (h_cam / d) * fy

    Two equations, two unknowns (vy_horizon, h_cam * fy):
        blob_y1 = vy_horizon + K / d1
        blob_y2 = vy_horizon + K / d2
        where K = h_cam_mm * fy

    Solving:
        K          = (blob_y1 - blob_y2) / (1/d1 - 1/d2)
        vy_horizon = blob_y1 - K / d1
        h_cam_mm   = K / fy
    """
    h, w = frame_shape[:2]
    cx   = w / 2.0
    cy   = h / 2.0

    # Use average of both ruler's fx estimates
    fx1 = px_spacing1 * d1_mm / dot_spacing_mm
    fx2 = px_spacing2 * d2_mm / dot_spacing_mm
    fx  = (fx1 + fx2) / 2.0
    fy  = fx

    # Solve for K = h_cam_mm * fy and vy_horizon
    inv_d1 = 1.0 / d1_mm
    inv_d2 = 1.0 / d2_mm
    K          = (blob_y1 - blob_y2) / (inv_d1 - inv_d2)
    vy_horizon = blob_y1 - K * inv_d1
    h_cam_mm   = K / fy

    return dict(
        fx            = round(fx, 2),
        fy            = round(fy, 2),
        cx            = round(cx, 2),
        cy            = round(cy, 2),
        vy_horizon    = round(vy_horizon, 2),
        camera_height_m = round(h_cam_mm / 1000.0, 4),
        image_w       = w,
        image_h       = h,
        dot_spacing_mm = dot_spacing_mm,
        method        = "dual_ruler",
    )


# ── Projection helpers (also written to JSON as a usage note) ─────────────────

def bev_to_pixel(x_m: float, y_m: float, calib: dict) -> tuple[int, int] | None:
    """
    Project a ground-plane BEV point to image pixel coordinates.

      x_m  – lateral offset in metres  (positive = right)
      y_m  – forward distance in metres (positive = ahead)

    Returns (u, v) pixel tuple, or None if the point is behind the camera.
    """
    if y_m < 0.05:
        return None
    fx = calib["fx"]
    fy = calib["fy"]
    cx = calib["cx"]
    vy = calib["vy_horizon"]
    h  = calib["camera_height_m"]

    u = cx + (x_m / y_m) * fx
    v = vy + (h  / y_m) * fy
    return int(round(u)), int(round(v))


# ── Visualisation ─────────────────────────────────────────────────────────────

def _annotate_frame(frame: np.ndarray,
                    blobs: list[tuple[float, float]],
                    calib: dict | None,
                    label: str) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Draw detected blobs
    for i, (bx, by) in enumerate(blobs):
        cv2.circle(out, (int(bx), int(by)), 10, (0, 255, 0), 2)
        cv2.putText(out, str(i), (int(bx) + 12, int(by) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw inter-blob spacing lines
    for i in range(len(blobs) - 1):
        x0, y0 = int(blobs[i][0]),   int(blobs[i][1])
        x1, y1 = int(blobs[i+1][0]), int(blobs[i+1][1])
        cv2.line(out, (x0, y0), (x1, y1), (0, 200, 255), 2)
        mid_x = (x0 + x1) // 2
        mid_y = (y0 + y1) // 2
        dx    = blobs[i+1][0] - blobs[i][0]
        cv2.putText(out, f"{dx:.0f}px", (mid_x, mid_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    # Draw horizon line if calibrated
    if calib is not None:
        vy = int(calib["vy_horizon"])
        cv2.line(out, (0, vy), (w, vy), (255, 100, 0), 1)
        cv2.putText(out, "horizon", (5, vy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1)

        # Project test waypoints: 0.5 m, 1 m, 2 m ahead, centred
        for y_m in [0.5, 1.0, 2.0]:
            pt = bev_to_pixel(0.0, y_m, calib)
            if pt and 0 <= pt[1] < h:
                cv2.circle(out, pt, 6, (0, 120, 255), -1)
                cv2.putText(out, f"{y_m:.1f}m", (pt[0] + 8, pt[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 255), 1)

    cv2.putText(out, label, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 230, 255), 2)
    return out


# ── Capture helper ────────────────────────────────────────────────────────────

def _capture_frame(device: int, warmup_frames: int = 20) -> np.ndarray:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera device {device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(warmup_frames):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("Failed to capture frame from camera")
    return frame


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate camera for OmniVLA waypoint projection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--device",         type=int,   default=0)
    parser.add_argument("--image",          type=str,   default=None,
                        help="Path to saved image (skips live capture)")
    parser.add_argument("--distance",       type=float, required=True,
                        help="Distance from camera to ruler #1 in mm")
    parser.add_argument("--distance2",      type=float, default=None,
                        help="Distance to ruler #2 (enables auto h_cam/horizon)")
    parser.add_argument("--camera-height",  type=float, default=None,
                        help="Camera height above ground in mm (single-ruler mode)")
    parser.add_argument("--dot-spacing",    type=float, default=100.0,
                        help="Real-world dot spacing in mm (default 100)")
    parser.add_argument("--output",         type=str,
                        default="camera_calibration.json")
    parser.add_argument("--no-show",        action="store_true",
                        help="Do not open a display window")
    args = parser.parse_args()

    dual_mode = args.distance2 is not None

    if not dual_mode and args.camera_height is None:
        parser.error(
            "Single-ruler mode requires --camera-height.\n"
            "Or use --distance2 to enable dual-ruler mode (auto h_cam).")

    # ── Load / capture frame ──────────────────────────────────────────────────
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"ERROR: cannot read image {args.image}")
            sys.exit(1)
        print(f"Loaded image: {args.image}  ({frame.shape[1]}×{frame.shape[0]})")
    else:
        print(f"Capturing from camera device {args.device}…")
        try:
            frame = _capture_frame(args.device)
        except RuntimeError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        # Save captured frame for reference
        fname = f"calibration_frame_{int(time.time())}.jpg"
        cv2.imwrite(fname, frame)
        print(f"Saved captured frame → {fname}")

    # ── Detect blobs in ruler #1 ──────────────────────────────────────────────
    print(f"\nDetecting red blobs (ruler at {args.distance:.0f} mm)…")
    blobs1 = _detect_red_blobs(frame)
    print(f"  Found {len(blobs1)} blob(s): {[(f'{b[0]:.0f}', f'{b[1]:.0f}') for b in blobs1]}")

    if len(blobs1) < 2:
        print("\nERROR: fewer than 2 red blobs detected.")
        print("Tips:")
        print("  • Make sure the red dots are clearly visible and well lit")
        print("  • Avoid other red objects in the frame")
        print("  • Save the frame with --image and inspect it manually")
        if not args.no_show:
            mask_dbg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cv2.imshow("frame", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        sys.exit(1)

    spacing1 = _pixel_spacing(blobs1)
    y1       = _mean_y(blobs1)
    print(f"  Avg pixel spacing: {spacing1:.1f} px")
    print(f"  Avg pixel row    : {y1:.1f}")

    # ── Dual-ruler mode: capture / detect second ruler ────────────────────────
    blobs2   = None
    spacing2 = None
    y2       = None

    if dual_mode:
        print(f"\nDual-ruler mode: now place ruler at {args.distance2:.0f} mm.")
        if args.image:
            img2_path = input("  Path to second image: ").strip()
            frame2    = cv2.imread(img2_path)
            if frame2 is None:
                print(f"ERROR: cannot read {img2_path}")
                sys.exit(1)
        else:
            input("  Press Enter when ruler is in position…")
            frame2 = _capture_frame(args.device)

        print(f"Detecting red blobs (ruler at {args.distance2:.0f} mm)…")
        blobs2   = _detect_red_blobs(frame2)
        print(f"  Found {len(blobs2)} blob(s)")

        if len(blobs2) < 2:
            print("ERROR: fewer than 2 blobs found in second frame.")
            sys.exit(1)

        spacing2 = _pixel_spacing(blobs2)
        y2       = _mean_y(blobs2)
        print(f"  Avg pixel spacing: {spacing2:.1f} px")
        print(f"  Avg pixel row    : {y2:.1f}")

    # ── Compute calibration ───────────────────────────────────────────────────
    if dual_mode:
        calib = _compute_dual(
            spacing1, y1, args.distance,
            spacing2, y2, args.distance2,
            args.dot_spacing, frame.shape,
        )
    else:
        calib = _compute_single(
            spacing1, y1,
            args.distance, args.dot_spacing,
            args.camera_height, frame.shape,
        )

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'═'*50}")
    print("  CALIBRATION RESULT")
    print(f"{'═'*50}")
    print(f"  fx / fy          : {calib['fx']:.1f} / {calib['fy']:.1f} px")
    print(f"  Principal point  : ({calib['cx']:.1f}, {calib['cy']:.1f})")
    print(f"  Horizon line     : y = {calib['vy_horizon']:.1f} px  "
          f"({'above' if calib['vy_horizon'] < 0 else 'below'} top edge)")
    print(f"  Camera height    : {calib['camera_height_m']*1000:.1f} mm  "
          f"({calib['camera_height_m']:.4f} m)")
    print(f"  Method           : {calib['method']}")
    print(f"{'═'*50}")

    # Sanity checks
    warnings = []
    if calib['fx'] < 200 or calib['fx'] > 2000:
        warnings.append(f"fx={calib['fx']:.0f} is unusual — check distance and dot spacing")
    if calib['camera_height_m'] < 0.05 or calib['camera_height_m'] > 1.5:
        warnings.append(f"camera_height={calib['camera_height_m']*1000:.0f} mm seems wrong")
    for w in warnings:
        print(f"  ⚠  {w}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\n  Saved → {args.output}")

    # ── Visualise ─────────────────────────────────────────────────────────────
    has_display = bool(os.environ.get("DISPLAY") or sys.platform == "darwin"
                       or os.name == "nt")
    if not args.no_show and has_display:
        label  = (f"fx={calib['fx']:.0f}px  h={calib['camera_height_m']*1000:.0f}mm"
                  f"  horizon={calib['vy_horizon']:.0f}px")
        vis    = _annotate_frame(frame, blobs1, calib, label)
        cv2.imshow("Camera Calibration — press any key to close", vis)
        # Also save annotated frame
        vis_path = args.output.replace(".json", "_annotated.jpg")
        cv2.imwrite(vis_path, vis)
        print(f"  Saved annotated frame → {vis_path}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # Headless: save annotated frame anyway
        label    = f"fx={calib['fx']:.0f}px h={calib['camera_height_m']*1000:.0f}mm"
        vis      = _annotate_frame(frame, blobs1, calib, label)
        vis_path = args.output.replace(".json", "_annotated.jpg")
        cv2.imwrite(vis_path, vis)
        print(f"  Saved annotated frame → {vis_path}")

    print("\nDone.  Load camera_calibration.json in rover_agent with:")
    print("  python rover_agent.py --camera-calibration camera_calibration.json")


if __name__ == "__main__":
    main()
