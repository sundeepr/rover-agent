#!/usr/bin/env python3
"""
extract_mcap_video.py — pulls JPEG frames out of foxglove.CompressedImage
topics in an .mcap recording (as written by record_mcap.py) and re-encodes
each camera topic back into a plain .avi file playable in any video player.

Each topic gets its own output fps, estimated from the actual message
timestamps in the file (first-to-last message span / frame count) rather
than assumed — camera threads don't guarantee a fixed capture rate.

Usage
─────
    python extract_mcap_video.py sessions/20260930_120000/recording.mcap
    python extract_mcap_video.py recording.mcap --out-dir sessions/20260930_120000
    python extract_mcap_video.py recording.mcap --topics /cam/cam_10_0_1_101
    python extract_mcap_video.py recording.mcap --fps 15   # skip timestamp-based estimate
"""

import argparse
import base64
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader

log = logging.getLogger("rover.extract_mcap_video")

_IMAGE_SCHEMA_NAME = "foxglove.CompressedImage"
_FALLBACK_FPS = 15.0


def _discover_image_topics(mcap_path: Path) -> list[str]:
    with open(mcap_path, "rb") as f:
        summary = make_reader(f).get_summary()
        if summary is None:
            return []
        topics = []
        for channel in summary.channels.values():
            schema = summary.schemas.get(channel.schema_id)
            if schema is not None and schema.name == _IMAGE_SCHEMA_NAME:
                topics.append(channel.topic)
        return topics


def _estimate_fps(mcap_path: Path, topic: str) -> float:
    first_ts = last_ts = None
    count = 0
    with open(mcap_path, "rb") as f:
        for _, _, message in make_reader(f).iter_messages(topics=[topic]):
            if first_ts is None:
                first_ts = message.log_time
            last_ts = message.log_time
            count += 1

    if count < 2 or last_ts == first_ts:
        return _FALLBACK_FPS

    duration_s = (last_ts - first_ts) / 1e9
    fps = (count - 1) / duration_s
    return fps if fps > 0.1 else _FALLBACK_FPS


def _safe_filename(topic: str) -> str:
    return topic.strip("/").replace("/", "_") + ".avi"


def _extract_topic(mcap_path: Path, topic: str, out_dir: Path, fps: float) -> None:
    out_path = out_dir / _safe_filename(topic)
    writer = None
    n_frames = 0

    with open(mcap_path, "rb") as f:
        for _, _, message in make_reader(f).iter_messages(topics=[topic]):
            payload = json.loads(message.data)
            jpeg = base64.b64decode(payload["data"])
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue  # corrupt/partial frame — skip rather than abort the whole export

            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
                log.info("[%s] %dx%d @ %.2ffps -> %s", topic, w, h, fps, out_path)

            writer.write(frame)
            n_frames += 1

    if writer is not None:
        writer.release()
        log.info("[%s] wrote %d frames -> %s", topic, n_frames, out_path)
    else:
        log.warning("[%s] no frames found — nothing written", topic)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mcap_path", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Directory for extracted .avi files (default: same dir as the .mcap)")
    parser.add_argument("--topics", nargs="+", default=None,
                        help="Specific image topics to extract (default: all "
                             "foxglove.CompressedImage topics found in the file)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Force output fps instead of estimating from message timestamps")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    if not args.mcap_path.exists():
        raise SystemExit(f"File not found: {args.mcap_path}")

    out_dir = args.out_dir or args.mcap_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    topics = args.topics or _discover_image_topics(args.mcap_path)
    if not topics:
        raise SystemExit("No image topics found (and none given via --topics)")

    log.info("Extracting %d topic(s) from %s", len(topics), args.mcap_path)
    for topic in topics:
        fps = args.fps if args.fps is not None else _estimate_fps(args.mcap_path, topic)
        _extract_topic(args.mcap_path, topic, out_dir, fps)


if __name__ == "__main__":
    main()
