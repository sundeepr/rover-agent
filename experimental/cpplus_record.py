#!/usr/bin/env python3
"""
CP Plus RTSP stream recorder — no display, no frame buffering problems.

Uses a two-thread pipeline:
  Reader thread  — reads frames as fast as the camera delivers them,
                   pushes every frame into a queue (no drops)
  Writer thread  — pulls from the queue and writes to disk

Separating these means the writer never slows down the reader and the
reader never blocks waiting for the disk write to finish.

Usage:
    python experimental/cpplus_record.py                    # saves stream.avi
    python experimental/cpplus_record.py --out clip.avi
    python experimental/cpplus_record.py --duration 60      # record 60 seconds
    python experimental/cpplus_record.py --preview          # show live window too
"""

import argparse
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

# Low-latency capture flags — reduce buffering so frames arrive fresh
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp"
    "|timeout;5000000"
    "|fflags;nobuffer"
    "|flags;low_delay"
    "|probesize;32768"
    "|analyzeduration;0"
)

IP       = "192.168.1.100"
USER     = "admin"
PASSWORD = "Cam3ra_1234"
RTSP_URL = f"rtsp://{USER}:{PASSWORD}@{IP}:554/"


def open_capture(url: str) -> tuple[cv2.VideoCapture, int, int, float]:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {url}")
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            return cap, w, h, fps
    cap.release()
    raise RuntimeError("Stream opened but no frames received")


class StreamRecorder:
    """
    Two-thread RTSP recorder.

    Reader thread: reads every frame from cv2.VideoCapture and puts it
                   on a queue.  Never drops frames.
    Writer thread: pulls frames from the queue and writes them to disk.
    """

    _SENTINEL = None   # signals writer to flush and stop

    def __init__(self, url: str, out_path: str, duration_s: float | None,
                 preview: bool) -> None:
        self._url        = url
        self._out_path   = out_path
        self._duration_s = duration_s
        self._preview    = preview
        self._q: queue.Queue = queue.Queue(maxsize=256)
        self._stop       = threading.Event()
        self._frames_read    = 0
        self._frames_written = 0

    def run(self) -> None:
        cap, width, height, fps = open_capture(self._url)
        print(f"Stream: {width}x{height} @ {fps:.1f} fps")

        out_path = self._out_path or f"stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
        fourcc  = cv2.VideoWriter_fourcc(*"XVID")
        writer  = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open output file: {out_path}")
        print(f"Recording → {out_path}  (Ctrl-C to stop)")

        if self._preview:
            cv2.namedWindow("Recording — q to stop", cv2.WINDOW_NORMAL)

        reader_t = threading.Thread(target=self._reader, args=(cap,), daemon=True)
        writer_t = threading.Thread(target=self._writer, args=(writer,), daemon=True)

        t_start = time.time()
        reader_t.start()
        writer_t.start()

        try:
            while not self._stop.is_set():
                elapsed = time.time() - t_start
                if self._duration_s and elapsed >= self._duration_s:
                    print(f"\nDuration reached ({self._duration_s:.0f}s)")
                    break

                # Status line every 2 seconds
                qsize = self._q.qsize()
                print(f"\r  {elapsed:6.1f}s | read={self._frames_read}  "
                      f"written={self._frames_written}  queue={qsize}   ",
                      end="", flush=True)
                time.sleep(0.5)

                if self._preview:
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break

        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self._stop.set()
            self._q.put(self._SENTINEL)   # wake writer so it can exit
            writer_t.join(timeout=5.0)
            cap.release()
            writer.release()
            if self._preview:
                cv2.destroyAllWindows()
            elapsed = time.time() - t_start
            print(f"\nDone — {self._frames_written} frames in {elapsed:.1f}s  → {out_path}")

    def _reader(self, cap: cv2.VideoCapture) -> None:
        """Reads every frame and enqueues it.  Drops nothing."""
        fail_streak = 0
        while not self._stop.is_set():
            ret, frame = cap.read()
            if ret and frame is not None:
                fail_streak = 0
                self._frames_read += 1
                try:
                    self._q.put(frame, timeout=2.0)
                except queue.Full:
                    # Writer can't keep up — drop oldest to make room
                    try:
                        self._q.get_nowait()
                    except queue.Empty:
                        pass
                    self._q.put_nowait(frame)
            else:
                fail_streak += 1
                if fail_streak > 30:
                    print("\nReader: too many failures — stopping")
                    self._stop.set()
                    self._q.put(self._SENTINEL)
                    break

    def _writer(self, writer: cv2.VideoWriter) -> None:
        """Pulls frames from queue and writes to disk."""
        while True:
            try:
                frame = self._q.get(timeout=2.0)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue

            if frame is self._SENTINEL:
                break

            writer.write(frame)
            self._frames_written += 1

            if self._preview:
                cv2.imshow("Recording — q to stop", frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="CP Plus RTSP recorder")
    parser.add_argument("--url",      default=RTSP_URL,
                        help=f"RTSP URL (default: {RTSP_URL.replace(PASSWORD, '****')})")
    parser.add_argument("--out",      default=None, metavar="FILE",
                        help="Output file (default: stream_<timestamp>.avi)")
    parser.add_argument("--duration", type=float, default=None, metavar="SECS",
                        help="Stop after N seconds (default: run until Ctrl-C)")
    parser.add_argument("--preview",  action="store_true",
                        help="Show live preview window while recording")
    args = parser.parse_args()

    recorder = StreamRecorder(args.url, args.out, args.duration, args.preview)
    recorder.run()


if __name__ == "__main__":
    main()
