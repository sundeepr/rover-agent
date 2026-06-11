#!/usr/bin/env python3
"""
CP Plus RTSP stream viewer.

Uses ffmpeg subprocess piping instead of cv2.VideoCapture so RTSP digest
authentication works reliably (OpenCV's FFmpeg backend fails at the
OPTIONS stage on many CP Plus cameras).

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
import subprocess
import sys
import time
from datetime import datetime

import cv2
import numpy as np

# ── CP Plus RTSP URL templates ────────────────────────────────────────────────

def _build_urls(ip: str, user: str, password: str,
                channel: int, sub: bool) -> list[str]:
    subtype = 1 if sub else 0
    sub_str = "sub" if sub else "main"
    return [
        # Most common CP Plus / Dahua format
        f"rtsp://{user}:{password}@{ip}:554/cam/realmonitor"
        f"?channel={channel}&subtype={subtype}",
        # Alternate Dahua/CP Plus format
        f"rtsp://{user}:{password}@{ip}:554/h264/ch{channel}/{sub_str}/av_stream",
        # Generic fallback
        f"rtsp://{user}:{password}@{ip}/stream{channel}",
    ]


def _mask_url(url: str) -> str:
    """Replace user:password in URL with ****:**** for logging."""
    try:
        proto, rest = url.split("://", 1)
        creds, host = rest.split("@", 1)
        return f"{proto}://****:****@{host}"
    except ValueError:
        return url


def _probe_url(url: str) -> tuple[int, int, float]:
    """
    Use ffprobe to get width, height, fps from the stream.
    Returns (width, height, fps) or raises RuntimeError on failure.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-rtsp_transport", "tcp",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "csv=p=0",
        url,
    ]
    try:
        out = subprocess.check_output(cmd, timeout=10, stderr=subprocess.DEVNULL)
        parts = out.decode().strip().split(",")
        w, h = int(parts[0]), int(parts[1])
        num, den = parts[2].split("/")
        fps = float(num) / float(den)
        return w, h, fps
    except Exception as exc:
        raise RuntimeError(f"ffprobe failed: {exc}")


def open_ffmpeg(url: str, width: int, height: int) -> subprocess.Popen:
    """Launch ffmpeg to decode the stream and pipe raw BGR frames."""
    cmd = [
        "ffmpeg",
        "-loglevel",    "warning",
        "-rtsp_transport", "tcp",      # TCP is more reliable than UDP through NAT
        "-i",           url,
        "-vf",          f"scale={width}:{height}",
        "-f",           "rawvideo",
        "-pix_fmt",     "bgr24",
        "-",                            # pipe to stdout
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def try_connect(urls: list[str]) -> tuple[str, int, int, float]:
    """Try each URL; return (url, width, height, fps) for the first that works."""
    for url in urls:
        safe = _mask_url(url)
        print(f"Trying: {safe}")
        try:
            w, h, fps = _probe_url(url)
            print(f"Connected: {safe}  →  {w}x{h} @ {fps:.1f} fps")
            return url, w, h, fps
        except RuntimeError as e:
            print(f"  Failed: {e}")
    raise RuntimeError(
        "Could not connect on any URL.\n"
        "  • Check the camera is reachable: ping 192.168.1.100\n"
        "  • Verify credentials in the camera web UI\n"
        "  • Try --url with the exact RTSP URL from the camera's settings"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CP Plus RTSP stream viewer")
    parser.add_argument("--ip",       default="192.168.1.100")
    parser.add_argument("--user",     default="admin")
    parser.add_argument("--password", default="Cam3ra_1234")
    parser.add_argument("--channel",  type=int, default=1)
    parser.add_argument("--sub",      action="store_true",
                        help="Sub-stream (lower res / less bandwidth)")
    parser.add_argument("--width",    type=int, default=0,
                        help="Override display width  (0 = auto from stream)")
    parser.add_argument("--height",   type=int, default=0,
                        help="Override display height (0 = auto from stream)")
    parser.add_argument("--save",     type=str, default=None, metavar="FILE",
                        help="Also save stream to a video file")
    parser.add_argument("--url",      type=str, default=None,
                        help="Override with a specific RTSP URL")
    args = parser.parse_args()

    urls = ([args.url] if args.url else
            _build_urls(args.ip, args.user, args.password,
                        args.channel, args.sub))

    print("Connecting to CP Plus camera…")
    active_url, stream_w, stream_h, fps = try_connect(urls)

    width  = args.width  or stream_w
    height = args.height or stream_h
    fps    = fps or 25.0

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
        print(f"Saving to: {args.save}")

    win = "CP Plus — q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(width, 1280), min(height, 720))

    fullscreen   = False
    frame_bytes  = width * height * 3
    frame_count  = 0
    t0           = time.time()
    display_fps  = 0.0

    print("Streaming…  q/Esc=quit  s=snapshot  f=fullscreen")

    proc = open_ffmpeg(active_url, width, height)

    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                print("Stream ended or ffmpeg exited — reconnecting…")
                proc.terminate()
                time.sleep(2.0)
                _, stream_w, stream_h, fps = try_connect(urls)
                width      = args.width  or stream_w
                height     = args.height or stream_h
                frame_bytes = width * height * 3
                proc = open_ffmpeg(active_url, width, height)
                continue

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))

            frame_count += 1
            elapsed = time.time() - t0
            if elapsed >= 2.0:
                display_fps = frame_count / elapsed
                frame_count = 0
                t0 = time.time()

            # Overlay: timestamp + fps
            ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
            cv2.putText(frame, ts, (10, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, f"{display_fps:.1f} fps", (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1,
                        cv2.LINE_AA)

            if writer:
                writer.write(frame)

            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                fname = f"cpplus_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(fname, frame)
                print(f"Snapshot: {fname}")
            elif key == ord("f"):
                fullscreen = not fullscreen
                flag = (cv2.WINDOW_FULLSCREEN if fullscreen
                        else cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, flag)

    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
