#!/usr/bin/env python3
"""
CP Plus RTSP stream viewer — cv2 only, no external ffmpeg required.

Sets OPENCV_FFMPEG_CAPTURE_OPTIONS to force TCP transport and longer
timeouts before opening the stream, which resolves the 401 auth issue
seen with OpenCV's default UDP RTSP handling on CP Plus cameras.

Usage:
    python experimental/cpplus_stream.py
    python experimental/cpplus_stream.py --ip 192.168.1.100 --channel 2
    python experimental/cpplus_stream.py --sub          # sub-stream (lower res)
    python experimental/cpplus_stream.py --save out.avi # also save to file

Controls:
    q / Esc  — quit
    s        — save snapshot to cpplus_snapshot_<timestamp>.jpg
    f        — toggle fullscreen
"""

import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np

# Force TCP transport BEFORE any VideoCapture is created.
# This resolves the 401 Unauthorized seen with UDP on many CP Plus cameras.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|timeout;10000000"
)


def _build_urls(ip: str, user: str, password: str,
                channel: int, sub: bool) -> list[str]:
    subtype = 1 if sub else 0
    sub_str = "sub" if sub else "main"
    return [
        f"rtsp://{user}:{password}@{ip}:554/cam/realmonitor"
        f"?channel={channel}&subtype={subtype}",
        f"rtsp://{user}:{password}@{ip}:554/h264/ch{channel}/{sub_str}/av_stream",
        f"rtsp://{user}:{password}@{ip}/stream{channel}",
        f"rtsp://{user}:{password}@{ip}:554/Streaming/Channels/{channel}01",
    ]


def _mask(url: str) -> str:
    try:
        proto, rest = url.split("://", 1)
        _, host = rest.split("@", 1)
        return f"{proto}://****:****@{host}"
    except ValueError:
        return url


def try_connect(urls: list[str]) -> tuple[cv2.VideoCapture, str, int, int, float]:
    for url in urls:
        safe = _mask(url)
        print(f"Trying: {safe}")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"  Could not open")
            cap.release()
            continue
        # Read a few frames — first frame sometimes fails even when open
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                print(f"  Connected: {w}x{h} @ {fps:.1f} fps")
                return cap, url, w, h, fps
        print(f"  Opened but no frames received")
        cap.release()

    raise RuntimeError(
        "Could not connect to camera on any URL.\n"
        "  • Verify the camera is reachable: ping 192.168.1.100\n"
        "  • Log in to the camera web UI and confirm the credentials\n"
        "  • Try --url with the exact RTSP URL from the camera's RTSP settings"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CP Plus RTSP stream viewer")
    parser.add_argument("--ip",       default="192.168.1.100")
    parser.add_argument("--user",     default="admin")
    parser.add_argument("--password", default="Cam3ra_1234")
    parser.add_argument("--channel",  type=int, default=1)
    parser.add_argument("--sub",      action="store_true",
                        help="Use sub-stream (lower resolution)")
    parser.add_argument("--save",     type=str, default=None, metavar="FILE",
                        help="Also record to a video file (e.g. out.avi)")
    parser.add_argument("--url",      type=str, default=None,
                        help="Override with a specific RTSP URL")
    args = parser.parse_args()

    urls = ([args.url] if args.url
            else _build_urls(args.ip, args.user, args.password,
                             args.channel, args.sub))

    print("Connecting to CP Plus camera…")
    cap, active_url, width, height, fps = try_connect(urls)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
        print(f"Saving to: {args.save}")

    win = "CP Plus — q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(width, 1280), min(height, 720))

    fullscreen  = False
    frame_count = 0
    t0          = time.time()
    display_fps = 0.0
    fail_count  = 0

    print("Streaming…  q/Esc=quit  s=snapshot  f=fullscreen")

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            fail_count += 1
            if fail_count > 10:
                print("Too many read failures — reconnecting…")
                cap.release()
                time.sleep(2.0)
                cap, active_url, width, height, fps = try_connect(urls)
                fail_count = 0
            continue
        fail_count = 0

        frame_count += 1
        elapsed = time.time() - t0
        if elapsed >= 2.0:
            display_fps = frame_count / elapsed
            frame_count = 0
            t0 = time.time()

        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts,
                    (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{display_fps:.1f} fps",
                    (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (160, 160, 160), 1, cv2.LINE_AA)

        if writer:
            writer.write(frame)

        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            fname = f"cpplus_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fname, frame)
            print(f"Snapshot saved: {fname}")
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                win, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
