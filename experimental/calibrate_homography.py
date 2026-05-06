#!/usr/bin/env python3
"""
calibrate_homography.py — Camera-to-ground homography calibration for Atlas rover.

Computes a homography matrix H that maps ground positions (in metres, rover frame)
to pixel coordinates in the forward camera image. Once computed, H can replace the
linear scale in _annotate() so waypoint dots land on the correct image pixels.

Rover coordinate frame
──────────────────────
  dx = forward (metres away from rover)
  dy = lateral (metres left/right; negative = left, positive = right)
  origin = rover camera position on the ground

Step-by-step usage
──────────────────
1. Park the rover on a flat surface. Do NOT move it during calibration.

2. Place tape crosses (or printed markers) on the ground at these positions
   relative to the front of the rover:

       Label   dx (forward)   dy (lateral)
       ─────   ────────────   ────────────
         A        0.50 m         0.00 m    (50 cm ahead, centre)
         B        1.00 m         0.00 m    (1 m ahead, centre)
         C        0.50 m        -0.20 m    (50 cm ahead, 20 cm left)
         D        0.50 m        +0.20 m    (50 cm ahead, 20 cm right)
         E        1.00 m        -0.20 m    (1 m ahead, 20 cm left)
         F        1.00 m        +0.20 m    (1 m ahead, 20 cm right)

   Tip: use a tape measure and chalk / coloured tape so the markers are
   clearly visible in the camera image.

3. Capture a single still frame from the rover camera:
       python experimental/calibrate_homography.py --capture --device 0

   This saves calibration_frame.jpg in the current directory.

4. Run the interactive click tool:
       python experimental/calibrate_homography.py --click calibration_frame.jpg

   A window opens. Click each marker IN ORDER A→B→C→D→E→F.
   Press 'u' to undo the last click, 'q' or Enter when all 6 are done.

5. The script saves homography.npy next to calibration_frame.jpg and prints
   a reprojection error. Aim for < 5 pixels.

6. To use H in _annotate() (omnivla_strategy.py), replace the linear scale block:

       # OLD (linear, no perspective):
       scale = min(h, w) * 0.3
       px = int(cx - dy * scale)
       py = int(cy - dx * scale)

       # NEW (homography):
       H = np.load("homography.npy")
       pt = np.array([[[dy, dx]]], dtype=np.float32)   # (lateral, forward)
       px_pt = cv2.perspectiveTransform(pt, H)
       px = int(px_pt[0][0][0])
       py = int(px_pt[0][0][1])

Re-calibrate whenever the camera angle or mount position changes.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Ground truth marker positions (dy, dx) in metres ─────────────────────────
# Order must match the click order: A B C D E F
GROUND_PTS = np.array([
    [ 0.00,  0.50],   # A — 50 cm ahead, centre
    [ 0.00,  1.00],   # B — 1 m ahead,   centre
    [-0.20,  0.50],   # C — 50 cm ahead, 20 cm left
    [ 0.20,  0.50],   # D — 50 cm ahead, 20 cm right
    [-0.20,  1.00],   # E — 1 m ahead,   20 cm left
    [ 0.20,  1.00],   # F — 1 m ahead,   20 cm right
], dtype=np.float32)

LABELS = ["A", "B", "C", "D", "E", "F"]


# ── Capture ───────────────────────────────────────────────────────────────────

def capture(device: int, out_path: Path) -> None:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
    print(f"Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("Press SPACE to capture, q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture — SPACE to save", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            cv2.imwrite(str(out_path), frame)
            print(f"Saved: {out_path}")
            break
        if key in (ord('q'), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


# ── Interactive click tool ────────────────────────────────────────────────────

def click_markers(img_path: Path) -> np.ndarray:
    img_orig = cv2.imread(str(img_path))
    if img_orig is None:
        sys.exit(f"Cannot read: {img_path}")

    n_required = len(GROUND_PTS)
    clicks: list[tuple[int, int]] = []
    img = img_orig.copy()

    def redraw():
        nonlocal img
        img = img_orig.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
            cv2.putText(img, LABELS[i], (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        remaining = n_required - len(clicks)
        if remaining > 0:
            next_label = LABELS[len(clicks)]
            cv2.putText(img, f"Click marker {next_label}  ({remaining} remaining)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        else:
            cv2.putText(img, "All markers clicked — press Enter or q to finish",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2)
        cv2.imshow("Click markers", img)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < n_required:
            clicks.append((x, y))
            print(f"  {LABELS[len(clicks)-1]}: pixel ({x}, {y})")
            redraw()

    cv2.imshow("Click markers", img)
    cv2.setMouseCallback("Click markers", on_click)
    redraw()

    print(f"\nClick the {n_required} markers in order: {', '.join(LABELS)}")
    print("Keys: u = undo last click   Enter / q = done\n")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('u') and clicks:
            removed = clicks.pop()
            print(f"  Undo: removed {LABELS[len(clicks)]} at {removed}")
            redraw()
        elif key in (13, ord('q'), 27):   # Enter or q or Esc
            break

    cv2.destroyAllWindows()

    if len(clicks) < n_required:
        sys.exit(f"Need {n_required} clicks, got {len(clicks)}. Aborting.")

    return np.array(clicks, dtype=np.float32)


# ── Homography computation ────────────────────────────────────────────────────

def compute_and_save(img_path: Path, pixel_pts: np.ndarray) -> None:
    H, mask = cv2.findHomography(GROUND_PTS, pixel_pts, cv2.RANSAC, 3.0)
    if H is None:
        sys.exit("Homography computation failed — check your clicks.")

    # Reprojection error
    proj = cv2.perspectiveTransform(GROUND_PTS.reshape(-1, 1, 2), H)
    errors = np.linalg.norm(pixel_pts.reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
    print(f"\nReprojection errors (pixels):")
    for label, err in zip(LABELS, errors):
        print(f"  {label}: {err:.2f} px")
    print(f"  Mean: {errors.mean():.2f} px  Max: {errors.max():.2f} px")
    if errors.mean() > 10:
        print("WARNING: mean error > 10 px — consider re-clicking the markers.")

    out_path = img_path.parent / "homography.npy"
    np.save(str(out_path), H)
    print(f"\nHomography saved to: {out_path}")
    print(f"Matrix H:\n{H}")

    # Overlay verification image
    img = cv2.imread(str(img_path))
    proj_pts = proj.reshape(-1, 2).astype(int)
    pixel_pts_i = pixel_pts.astype(int)
    for i, (pp, gp) in enumerate(zip(pixel_pts_i, proj_pts)):
        cv2.circle(img, tuple(pp),  10, (0, 255, 0), 2)   # clicked (green)
        cv2.circle(img, tuple(gp),  6,  (0, 0, 255), -1)  # reprojected (red)
        cv2.putText(img, LABELS[i], (pp[0]+14, pp[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    verify_path = img_path.parent / "homography_verify.jpg"
    cv2.imwrite(str(verify_path), img)
    print(f"Verification image saved to: {verify_path}")
    print("Green circles = your clicks, Red dots = reprojected positions.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Camera-to-ground homography calibration for Atlas rover")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true",
                      help="Capture a calibration frame from the camera")
    mode.add_argument("--click",   metavar="IMAGE",
                      help="Load an existing image and click the markers")
    parser.add_argument("--device", type=int, default=0,
                        help="Camera device index for --capture (default: 0)")
    parser.add_argument("--out",    default="calibration_frame.jpg",
                        help="Output path for captured frame (default: calibration_frame.jpg)")
    args = parser.parse_args()

    if args.capture:
        capture(args.device, Path(args.out))
        print(f"\nNow run:\n  python experimental/calibrate_homography.py --click {args.out}")
    else:
        img_path = Path(args.click)
        pixel_pts = click_markers(img_path)
        compute_and_save(img_path, pixel_pts)


if __name__ == "__main__":
    main()
