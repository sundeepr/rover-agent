#!/usr/bin/env python3
"""
CP Plus RTSP stream viewer.

Connects to a CP Plus IP camera over RTSP and displays the live feed.
Tries the two most common CP Plus URL formats and falls back to a generic one.

Usage:
    python experimental/cpplus_stream.py
    python experimental/cpplus_stream.py --ip 192.168.1.100 --channel 2
    python experimental/cpplus_stream.py --sub          # sub-stream (lower res)
    python experimental/cpplus_stream.py --save out.avi # also save to file

Controls (display window):
    q / Esc  — quit
    s        — save a snapshot to cpplus_snapshot_<timestamp>.jpg
    f        — toggle fullscreen
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

# ── CP Plus RTSP URL templates ────────────────────────────────────────────────
# subtype=0 → main stream (high res)
# subtype=1 → sub stream  (low res / fluent)

def _build_urls(ip: str, user: str, password: str,
                channel: int, sub: bool) -> list[str]:
    subtype = 1 if sub else 0
    return [
        # Most common CP Plus / Dahua format
        f"rtsp://{user}:{password}@{ip}:554/cam/realmonitor"
        f"?channel={channel}&subtype={subtype}",
        # Alternate Dahua/CP Plus format
        f"rtsp://{user}:{password}@{ip}:554/h264/ch{channel}/{'sub' if sub else 'main'}/av_stream",
        # Generic fallback
        f"rtsp://{user}:{password}@{ip}/stream{channel}",
    ]


def open_stream(urls: list[str]) -> tuple[cv2.VideoCapture, str]:
    """Try each URL in order; return the first that opens successfully."""
    for url in urls:
        # Log URL with password masked
        safe = url.replace(url.split("@")[0].split("//")[1], "****:****")
        print(f"Trying: {safe}")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"Connected: {safe}")
                return cap, url
            cap.release()
        else:
            cap.release()
    raise RuntimeError("Could not connect to camera on any URL. "
                       "Check IP, credentials, and that the camera is reachable.")


def main() -> None:
    parser = argparse.ArgumentParser(description="CP Plus RTSP stream viewer")
    parser.add_argument("--ip",       default="192.168.1.100",
                        help="Camera IP address (default: 192.168.1.100)")
    parser.add_argument("--user",     default="camera1",
                        help="RTSP username (default: camera1)")
    parser.add_argument("--password", default="Camera_1234",
                        help="RTSP password (default: Camera_1234)")
    parser.add_argument("--channel",  type=int, default=1,
                        help="Camera channel number (default: 1)")
    parser.add_argument("--sub",      action="store_true",
                        help="Use sub-stream (lower resolution, less bandwidth)")
    parser.add_argument("--save",     type=str, default=None, metavar="FILE",
                        help="Also save the stream to a video file (e.g. out.avi)")
    parser.add_argument("--url",      type=str, default=None,
                        help="Override with a specific RTSP URL")
    args = parser.parse_args()

    urls = [args.url] if args.url else _build_urls(
        args.ip, args.user, args.password, args.channel, args.sub
    )

    print("Connecting to CP Plus camera…")
    cap, active_url = open_stream(urls)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stream = "sub" if args.sub else "main"
    print(f"Stream: {width}x{height} @ {fps:.1f} fps  ({stream}-stream  ch{args.channel})")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
        print(f"Saving to: {args.save}")

    win = "CP Plus — press Q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(width, 1280), min(height, 720))

    fullscreen = False
    frame_count = 0
    t0 = time.time()
    display_fps = 0.0

    print("Streaming… (q/Esc=quit  s=snapshot  f=fullscreen)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed — reconnecting…")
            cap.release()
            time.sleep(2.0)
            try:
                cap, _ = open_stream(urls)
            except RuntimeError as e:
                print(f"Reconnect failed: {e}")
                break
            continue

        frame_count += 1
        elapsed = time.time() - t0
        if elapsed >= 2.0:
            display_fps = frame_count / elapsed
            frame_count = 0
            t0 = time.time()

        # Overlay: timestamp + fps
        ts  = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts, (10, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{display_fps:.1f} fps", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        if writer:
            writer.write(frame)

        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):   # q or Esc
            break
        elif key == ord("s"):
            fname = f"cpplus_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fname, frame)
            print(f"Snapshot saved: {fname}")
        elif key == ord("f"):
            fullscreen = not fullscreen
            flag = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, flag)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
