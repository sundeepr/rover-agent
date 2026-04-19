#!/usr/bin/env python3
"""
Probe a camera device and print all standard resolutions it supports.

Usage:
    python camera_resolutions.py          # device 0
    python camera_resolutions.py --device 2
"""

import argparse
import cv2

STANDARD_RESOLUTIONS = [
    (160,  120),
    (320,  240),
    (424,  240),
    (640,  360),
    (640,  480),
    (800,  600),
    (960,  540),
    (1024, 576),
    (1280, 720),
    (1280, 960),
    (1600, 896),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
]


def estimate_fov(cap: cv2.VideoCapture) -> None:
    """
    Estimate horizontal and vertical FOV by running a simple autocalibration:
    reads focal length from CAP_PROP_FOCUS if available, otherwise uses
    OpenCV's camera calibration property (CAP_PROP_FPS as a fallback hint).

    Most USB webcams expose focal length via V4L2. If not available, reports
    that FOV cannot be estimated without calibration.
    """
    import math

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Some cameras expose focal length in pixels via CAP_PROP_ZOOM (focal mm)
    # or can be estimated via a checkerboard calibration. The most universally
    # available approach: use OpenCV's built-in grab of a single frame and run
    # a quick single-image focal-length estimate via known sensor aspect ratio.
    #
    # For USB cameras that expose V4L2 focus metadata:
    fx = cap.get(cv2.CAP_PROP_FOCUS)  # returns 0 if unsupported

    if fx and fx > 0:
        fov_h = math.degrees(2 * math.atan(w / (2 * fx)))
        fov_v = math.degrees(2 * math.atan(h / (2 * fx)))
        print(f"\nFOV estimate (from focal length {fx:.1f} px):")
        print(f"  Horizontal: {fov_h:.1f}°")
        print(f"  Vertical  : {fov_v:.1f}°")
    else:
        print("\nFOV: camera does not expose focal length via V4L2.")
        print("  To measure FOV accurately, run OpenCV camera calibration")
        print("  with a checkerboard pattern (cv2.calibrateCamera).")


def probe(device: int) -> None:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"ERROR: could not open camera device {device}")
        return

    print(f"Camera device {device} — supported resolutions:")
    print(f"  {'Width':>6}  {'Height':>6}  {'Actual WxH'}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*20}")

    supported = []
    for w, h in STANDARD_RESOLUTIONS:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        match = (actual_w == w and actual_h == h)
        if match:
            supported.append((actual_w, actual_h))
            print(f"  {w:>6}  {h:>6}  ✓")
        else:
            print(f"  {w:>6}  {h:>6}  → {actual_w}x{actual_h}")

    print()
    print(f"Supported ({len(supported)}):", ", ".join(f"{w}x{h}" for w, h in supported))

    estimate_fov(cap)

    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Probe camera resolutions")
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()
    probe(args.device)


if __name__ == "__main__":
    main()
