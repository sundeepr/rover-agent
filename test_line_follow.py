#!/usr/bin/env python3
"""
test_line_follow.py — replay raw.avi through the line_follow detection pipeline.

Usage
─────
    python test_line_follow.py /path/to/session/raw.avi
    python test_line_follow.py /path/to/session/raw.avi --color black
    python test_line_follow.py /path/to/session/raw.avi --out output.avi --speed 2.0
    python test_line_follow.py /path/to/session/raw.avi --no-display --out output.avi

Keys (during playback)
──────────────────────
    SPACE  — pause / resume
    q      — quit
    ,  .   — step back / forward one frame (when paused)
"""

import argparse
import sys

import cv2
import numpy as np

from line_follow_strategy import _detect, _annotate, _COLOUR_BOUNDS


def main():
    parser = argparse.ArgumentParser(description="Test line_follow on raw.avi")
    parser.add_argument("video", help="Path to raw.avi")
    parser.add_argument("--color",      default="black",
                        choices=list(_COLOUR_BOUNDS),
                        help="Target line colour (default: black)")
    parser.add_argument("--out",        default=None,
                        help="Save annotated video here (.avi)")
    parser.add_argument("--speed",      type=float, default=1.0,
                        help="Playback speed multiplier (default 1.0)")
    parser.add_argument("--no-display", action="store_true",
                        help="Don't open a window (useful with --out)")
    parser.add_argument("--start",      type=int, default=0,
                        help="Start at this frame index")
    args = parser.parse_args()

    hsv_bounds = _COLOUR_BOUNDS[args.color]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video  : {total} frames  {src_w}x{src_h}  {fps:.1f} fps")
    print(f"Colour : {args.color}")
    print("Keys   : SPACE=pause  q=quit  ,/.=step frame")

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        # Output size determined after first frame processed — set up lazily
        writer = args.out   # placeholder

    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    paused    = False
    frame_idx = args.start
    raw       = None
    delay     = max(1, int(1000 / fps / args.speed))
    vwriter   = None

    while True:
        if not paused:
            ret, raw = cap.read()
            if not ret:
                break
            frame_idx += 1

        line_col, error_norm, area, proc, mask, best_stats, strip_y, cx = \
            _detect(raw, hsv_bounds)

        vel    = 40 if line_col is not None else 0
        radius = 0x8000
        result = "following" if line_col is not None else "line_lost"

        out = _annotate(proc, mask, best_stats, line_col, cx, strip_y,
                        vel, radius, error_norm, result, area, args.color)


        # Lazy writer init once we know the output frame size
        if writer and vwriter is None:
            oh, ow = out.shape[:2]
            vwriter = cv2.VideoWriter(writer, cv2.VideoWriter_fourcc(*"MJPG"),
                                      fps, (ow, oh))
            print(f"Saving : {writer}  ({ow}x{oh})")
        if vwriter:
            vwriter.write(out)

        if not args.no_display:
            cv2.imshow("line_follow test", out)
            key = cv2.waitKey(1 if paused else delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('.') and paused:
                ret, raw = cap.read()
                if ret:
                    frame_idx += 1
            elif key == ord(',') and paused and frame_idx > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 2)
                ret, raw = cap.read()
                if ret:
                    frame_idx -= 1
        else:
            if frame_idx % 30 == 0:
                sym = "OK" if line_col is not None else "--"
                print(f"  [{frame_idx/total*100:5.1f}%] frame {frame_idx:4d}  "
                      f"{sym}  col={str(line_col):>5}  "
                      f"error={error_norm:+.3f}  area={area}")

    cap.release()
    if vwriter:
        vwriter.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
