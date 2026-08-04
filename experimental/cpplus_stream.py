#!/usr/bin/env python3
"""
CP Plus IP camera RTSP viewer — ONVIF discovery + cv2 display.

Uses ONVIF to ask the camera for its own stream URI (the only reliable
way to get the correct URL for a direct IP camera).  Falls back to a
list of common CP Plus / generic ONVIF RTSP URL patterns if the library
is not installed.

Install the ONVIF library once:
    pip install onvif-zeep

Usage:
    python experimental/cpplus_stream.py
    python experimental/cpplus_stream.py --ip 192.168.1.100 --onvif-port 80
    python experimental/cpplus_stream.py --profile 1   # use second profile
    python experimental/cpplus_stream.py --sub         # request sub-stream
    python experimental/cpplus_stream.py --save out.avi

Controls:
    q / Esc  — quit
    s        — snapshot to cpplus_snapshot_<timestamp>.jpg
    f        — toggle fullscreen
    p        — print all discovered ONVIF profiles to console
"""

import argparse
import os
import threading
import time
from datetime import datetime

import cv2

# Low-latency FFmpeg options (must be set before any VideoCapture is created):
#   nobuffer      — disable stream buffering; deliver frames as soon as decoded
#   low_delay     — disable H.264 B-frame reorder buffer (biggest single fix)
#   probesize=32K — stop probing early (default 5 MB causes multi-second startup)
#   analyzeduration=0 — skip stream analysis phase
#   framedrop     — drop late frames instead of queuing them
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp"
    "|timeout;5000000"
    "|fflags;nobuffer"
    "|flags;low_delay"
    "|probesize;32768"
    "|analyzeduration;0"
    "|framedrop;1"
)

IP       = "192.168.1.101"
USER     = "admin"
PASSWORD = "Cam3ra_1234"


# ── ONVIF discovery ───────────────────────────────────────────────────────────

def onvif_get_stream_uri(ip: str, user: str, password: str,
                         onvif_port: int, profile_index: int,
                         sub: bool) -> tuple[str, list]:
    """
    Connect to the camera via ONVIF and retrieve the RTSP stream URI.

    Returns (stream_uri, profiles_list).
    Raises ImportError if onvif-zeep is not installed.
    Raises RuntimeError on connection / auth failure.
    """
    try:
        from onvif import ONVIFCamera
    except ImportError:
        raise ImportError(
            "onvif-zeep not installed.\n"
            "Run:  pip install onvif-zeep\n"
            "Then re-run this script."
        )

    # Disable SSL verification globally — the camera uses a self-signed cert
    # and the onvif-zeep library fetches WSDL over HTTPS internally, ignoring
    # any per-session verify=False we pass in.
    import ssl
    import requests
    import urllib3

    ssl._create_default_https_context = ssl._create_unverified_context
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Monkey-patch requests.Session.send so every request skips verification
    _orig_send = requests.Session.send
    def _send_no_verify(self, request, **kwargs):
        kwargs["verify"] = False
        return _orig_send(self, request, **kwargs)
    requests.Session.send = _send_no_verify

    print(f"ONVIF: connecting to {ip}:{onvif_port} as {user}…")
    try:
        cam      = ONVIFCamera(ip, onvif_port, user, password, encrypt=False)
        media    = cam.create_media_service()
        profiles = media.GetProfiles()
        if not profiles:
            raise RuntimeError("ONVIF: camera returned no media profiles")
    except Exception as exc:
        raise RuntimeError(f"ONVIF failed: {exc}")

    print(f"ONVIF: found {len(profiles)} profile(s):")
    for i, p in enumerate(profiles):
        enc = getattr(getattr(p, "VideoEncoderConfiguration", None),
                      "Encoding", "?")
        res = getattr(getattr(p, "VideoEncoderConfiguration", None),
                      "Resolution", None)
        res_str = f"{res.Width}x{res.Height}" if res else "?"
        print(f"  [{i}] {p.Name}  token={p.token}  enc={enc}  res={res_str}")

    idx   = min(profile_index, len(profiles) - 1)
    token = profiles[idx].token
    print(f"ONVIF: requesting stream URI for profile [{idx}] token={token}")

    req = media.create_type("GetStreamUri")
    req.ProfileToken = token
    req.StreamSetup  = {
        "Stream":    "RTP-Unicast",
        "Transport": {"Protocol": "RTSP"},
    }
    uri_resp = media.GetStreamUri(req)
    uri      = uri_resp.Uri

    # Inject credentials into the URI if the camera returns a bare URL
    if "@" not in uri:
        proto, rest = uri.split("://", 1)
        uri = f"{proto}://{user}:{password}@{rest}"
    else:
        # Replace whatever credentials the camera put in with ours
        proto, rest = uri.split("://", 1)
        _, host_path = rest.split("@", 1)
        uri = f"{proto}://{user}:{password}@{host_path}"

    return uri, profiles


# ── Fallback URL list (direct IP camera patterns, not NVR) ───────────────────

def _fallback_urls(ip: str, user: str, password: str, sub: bool) -> list[str]:
    subtype = 1 if sub else 0
    return [
        # CP Plus documented format (confirmed working)
        f"rtsp://{user}:{password}@{ip}:554/video/live?channel=1&subtype={subtype}",
        # Root path fallback
        f"rtsp://{user}:{password}@{ip}:554/",
        f"rtsp://{user}:{password}@{ip}:554/onvif{'2' if sub else '1'}",
        f"rtsp://{user}:{password}@{ip}:554/stream{'2' if sub else '1'}",
    ]


# ── Threaded frame reader ─────────────────────────────────────────────────────

class LatestFrameReader:
    """
    Reads frames from a VideoCapture in a background thread and always
    exposes the most recent one.  The display thread never waits for a
    decode — it always gets the freshest frame available, eliminating
    queue build-up that causes visible latency.
    """

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap     = cap
        self._frame   = None
        self._lock    = threading.Lock()
        self._running = True
        self._t       = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame

    def read(self) -> tuple[bool, cv2.typing.MatLike | None]:
        with self._lock:
            f = self._frame
        return (f is not None, f)

    def stop(self) -> None:
        self._running = False


# ── Stream helpers ────────────────────────────────────────────────────────────

def _mask(url: str) -> str:
    try:
        proto, rest = url.split("://", 1)
        _, host = rest.split("@", 1)
        return f"{proto}://****:****@{host}"
    except ValueError:
        return url


def open_capture(url: str) -> tuple[cv2.VideoCapture, int, int, float]:
    """Open RTSP URL with cv2; return (cap, width, height, fps)."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("VideoCapture could not open URL")
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
            return cap, w, h, fps
    cap.release()
    raise RuntimeError("Stream opened but no frames received")


def try_urls(urls: list[str]) -> tuple[cv2.VideoCapture, str, int, int, float]:
    for url in urls:
        safe = _mask(url)
        print(f"Trying: {safe}")
        try:
            cap, w, h, fps = open_capture(url)
            print(f"  Connected: {w}x{h} @ {fps:.1f} fps")
            return cap, url, w, h, fps
        except RuntimeError as e:
            print(f"  Failed: {e}")
    raise RuntimeError(
        "Could not connect on any URL.\n"
        "  • Confirm camera reachable: ping 192.168.1.100\n"
        "  • Log into the camera web UI and check RTSP / ONVIF settings\n"
        "  • Run with --url to specify the exact RTSP URL\n"
        "  • Install onvif-zeep for automatic URL discovery: pip install onvif-zeep"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CP Plus IP camera RTSP viewer")
    parser.add_argument("--ip",         default=IP)
    parser.add_argument("--user",       default=USER)
    parser.add_argument("--password",   default=PASSWORD)
    parser.add_argument("--onvif-port", type=int, default=80,
                        help="ONVIF HTTP port (try 80, 8080, or 8899, default 80)")
    parser.add_argument("--profile",    type=int, default=0,
                        help="ONVIF profile index to stream (default 0 = main)")
    parser.add_argument("--sub",        action="store_true",
                        help="Request sub-stream / second profile")
    parser.add_argument("--save",       type=str, default=None, metavar="FILE")
    parser.add_argument("--url",        type=str, default=None,
                        help="Skip discovery and use this RTSP URL directly")
    args = parser.parse_args()

    profile_idx = args.profile if not args.sub else max(args.profile, 1)

    # ── 1. Try ONVIF discovery ────────────────────────────────────────────────
    stream_url = args.url
    if not stream_url:
        try:
            stream_url, _ = onvif_get_stream_uri(
                args.ip, args.user, args.password,
                args.onvif_port, profile_idx, args.sub,
            )
            print(f"ONVIF stream URI: {_mask(stream_url)}")
        except ImportError as e:
            print(f"\n{e}\n")
            print("Falling back to common CP Plus direct-camera URL patterns…")
        except Exception as e:
            print(f"ONVIF failed ({e}) — trying fallback URLs…")

    # ── 2. Open capture ───────────────────────────────────────────────────────
    if stream_url:
        print(f"Opening: {_mask(stream_url)}")
        try:
            cap, width, height, fps = open_capture(stream_url)
            active_url = stream_url
        except RuntimeError as e:
            print(f"Direct URL failed ({e}) — trying fallback patterns…")
            stream_url = None

    if not stream_url:
        cap, active_url, width, height, fps = try_urls(
            _fallback_urls(args.ip, args.user, args.password, args.sub)
        )

    print(f"Streaming {width}x{height} @ {fps:.1f} fps")

    # ── 3. Display loop ───────────────────────────────────────────────────────
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
        print(f"Saving to: {args.save}")

    win = "CP Plus — q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(width, 1280), min(height, 720))

    # Background reader — always holds the freshest decoded frame so the
    # display thread never blocks waiting for a slow decode.
    reader = LatestFrameReader(cap)

    fullscreen  = False
    frame_count = 0
    t0          = time.time()
    display_fps = 0.0
    no_frame_streak = 0

    print("q/Esc=quit  s=snapshot  f=fullscreen")

    while True:
        ret, frame = reader.read()

        if not ret or frame is None:
            no_frame_streak += 1
            if no_frame_streak > 100:   # ~1 s with 10 ms sleep
                print("No frames — reconnecting…")
                reader.stop()
                cap.release()
                time.sleep(2.0)
                try:
                    cap, active_url, width, height, fps = try_urls([active_url])
                    reader = LatestFrameReader(cap)
                    no_frame_streak = 0
                except RuntimeError:
                    break
            time.sleep(0.01)
            continue
        no_frame_streak = 0

        frame_count += 1
        elapsed = time.time() - t0
        if elapsed >= 2.0:
            display_fps = frame_count / elapsed
            frame_count = 0
            t0 = time.time()

        h, w = frame.shape[:2]
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts,
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{display_fps:.1f} fps",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
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
            print(f"Snapshot: {fname}")
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                win, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )

    reader.stop()
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
