#!/usr/bin/env python3
"""
Rover navigation agent — web UI.

Opens the camera, queries Gemini for navigation waypoints, and serves
a live webpage at http://localhost:5000 showing:
  - Real-time MJPEG camera feed (raw, 30 fps)
  - Frozen frame that was sent to Gemini with waypoint overlay
  - Mission status, reasoning, confidence and history panel

Usage:
    python rover_agent.py
    python rover_agent.py --device 1 --interval 5 --port 5000
    python rover_agent.py --roomba-port /dev/ttyUSB0
"""

import argparse
import collections
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, send_file

import gemini_client
import prompts
import roomba_controller

# ── Logging ────────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"rover_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("rover")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                                       datefmt="%Y-%m-%d %H:%M:%S"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Log file: %s", log_file.resolve())
    return logger


log = setup_logging()

# ── Shared state (written by agent thread, read by Flask) ──────────────────────

_raw_lock = threading.Lock()
_llm_lock = threading.Lock()
_result_lock = threading.Lock()

_raw_frame = None          # numpy BGR array, updated at ~30 fps
_llm_frame = None          # numpy BGR array, last frame sent to Gemini + overlay
_latest_result: dict = {}
_trajectory: list[dict] = []          # structured per-step record: step, phase, x, y, description
_step = 0
_phase = 1

# Rolling buffer of JPEG bytes for the last 3 query frames (oldest first)
_FRAME_BUFFER_SIZE = 3
_frame_buffer: collections.deque = collections.deque(maxlen=_FRAME_BUFFER_SIZE)

# LLM query timing (seconds since epoch)
_llm_query_start: float = 0.0    # set when query is sent; 0 = idle
_llm_response_s: float = 0.0     # elapsed time of last completed query



def set_raw_frame(frame):
    global _raw_frame
    with _raw_lock:
        _raw_frame = frame.copy()

def set_llm_frame(frame):
    global _llm_frame
    with _llm_lock:
        _llm_frame = frame.copy()

def set_result(result: dict):
    global _latest_result
    with _result_lock:
        _latest_result = result

def get_result() -> dict:
    with _result_lock:
        return dict(_latest_result)


# ── Overlay drawing ────────────────────────────────────────────────────────────

# Rank 1 = green (best), rank 2 = yellow, rank 3 = orange
_WP_COLORS = {1: (0, 255, 0), 2: (0, 220, 255), 3: (0, 140, 255)}


def draw_overlay(frame, result: dict, step: int):
    if not result:
        return frame

    out = frame.copy()

    for wp in result.get("waypoints", []):
        rank = wp.get("rank", 1)
        color = _WP_COLORS.get(rank, (255, 255, 255))
        x, y = int(wp["x"]), int(wp["y"])

        cv2.drawMarker(out, (x, y), color, cv2.MARKER_CROSS, markerSize=26, thickness=2)
        cv2.circle(out, (x, y), 14, color, 2)

        prob = wp.get("probability", 0)
        label = f"#{rank} {prob:.0%} {wp.get('description','')[:30]}"
        cv2.putText(out, label, (x + 16, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    lines = [
        f"Step {step}  Phase {result.get('phase','?')}  {result.get('goal_status','')}",
        f"Confidence: {result.get('confidence', 0):.0%}",
    ]
    y_pos = 24
    for line in lines:
        cv2.putText(out, line, (11, y_pos + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, line, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y_pos += 22

    return out


# ── Agent loop (runs in background thread) ─────────────────────────────────────

def _gemini_query(step: int, phase: int, query_frame, captures_dir: Path,
                   roomba_ctrl):
    """Runs in its own thread — sends frame to Gemini and updates shared state."""
    global _phase, _llm_query_start, _llm_response_s, _frame_buffer

    _, jpeg_buf = cv2.imencode(".jpg", query_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    image_bytes = jpeg_buf.tobytes()

    # Build image list: up to last 3 prior frames + current frame
    image_frames = list(_frame_buffer) + [image_bytes]
    # Add current frame to buffer for future queries
    _frame_buffer.append(image_bytes)

    user_prompt = prompts.build_user_prompt(phase, step, list(_trajectory))

    log.info("--- Step %d | Phase %d | %d image(s) sent ---", step, phase, len(image_frames))

    # Save the raw frame sent to Gemini
    cv2.imwrite(str(captures_dir / f"step_{step:04d}_raw.jpg"), query_frame)

    t0 = time.time()
    _llm_query_start = t0
    try:
        result = gemini_client.get_waypoint(
            image_frames, prompts.SYSTEM_PROMPT, user_prompt
        )
        elapsed = time.time() - t0
        _llm_response_s = elapsed
        _llm_query_start = 0.0
        set_result(result)

        annotated = draw_overlay(query_frame, result, step)
        set_llm_frame(annotated)
        cv2.imwrite(str(captures_dir / f"step_{step:04d}_annotated.jpg"), annotated)

        status     = result.get("goal_status", "unknown")
        reasoning  = result.get("reasoning", "")
        confidence = result.get("confidence", 0)

        log.info("Status     : %s", status)
        log.info("Confidence : %.0f%%", confidence * 100)
        log.info("Reasoning  : %s", reasoning)
        log.info("Response time: %.2fs", elapsed)
        for wp in result.get("waypoints", []):
            log.info("Waypoint #%d: (%d, %d) p=%.0f%% — %s",
                     wp.get("rank"), wp["x"], wp["y"],
                     wp.get("probability", 0) * 100,
                     wp.get("description", ""))

        top = next((w for w in result.get("waypoints", []) if w.get("rank") == 1), None)
        if top:
            _trajectory.append({
                "step": step, "phase": phase,
                "x": top["x"], "y": top["y"],
                "description": top.get("description", ""),
            })

        if roomba_ctrl and status == "in_progress" and top:
            try:
                roomba_ctrl.navigate_to_waypoint(top)
            except Exception as e:
                log.error("Roomba drive error: %s", e, exc_info=True)

        if status == "phase1_complete":
            _phase = 2
            log.info(">> Phase 1 complete — executing U-turn")
            if roomba_ctrl:
                try:
                    roomba_ctrl.uturn()
                except Exception as e:
                    log.error("U-turn error: %s", e, exc_info=True)
        elif status == "mission_complete":
            log.info(">> Mission complete after %d steps!", step)

    except Exception as e:
        _llm_query_start = 0.0
        log.error("Gemini error after %.2fs: %s", time.time() - t0, e, exc_info=True)


def agent_loop(device: int, interval: float, roomba_port: str | None = None, dry_run: bool = False):
    global _step

    # Optionally connect to the Roomba
    roomba_ctrl: roomba_controller.RoombaController | None = None
    roomba_ctx = None
    if roomba_port:
        roomba_ctrl = roomba_controller.RoombaController(port=roomba_port, dry_run=dry_run)
        roomba_ctx = roomba_ctrl.connect()
        roomba_ctx.__enter__()
        log.info("Roomba controller active on %s%s", roomba_port, " (dry-run)" if dry_run else "")

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        log.error("Could not open camera at device %d", device)
        if roomba_ctx:
            roomba_ctx.__exit__(None, None, None)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, prompts.IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, prompts.IMAGE_HEIGHT)
    log.info("Camera opened: %dx%d",
             int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    captures_dir = Path("captures")
    captures_dir.mkdir(exist_ok=True)
    log.info("Saving LLM frames to: %s", captures_dir.resolve())

    last_query_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            log.error("Failed to grab frame")
            break

        # ── Always push raw frame to realtime stream (never blocked by Gemini) ──
        set_raw_frame(frame)

        now = time.time()

        # ── Fire Gemini query in a separate thread so camera loop never blocks ──
        if now - last_query_time >= interval:
            last_query_time = now
            _step += 1
            threading.Thread(
                target=_gemini_query,
                args=(_step, _phase, frame.copy(), captures_dir, roomba_ctrl),
                daemon=True,
            ).start()

        time.sleep(0.033)   # ~30 fps

    cap.release()
    log.info("Camera released")

    if roomba_ctx:
        roomba_ctx.__exit__(None, None, None)


# ── Flask app ──────────────────────────────────────────────────────────────────

app = Flask(__name__)

_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Rover Agent</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0f0f0f; color: #e0e0e0; font-family: monospace;
           display: flex; flex-direction: column; height: 100vh; }
    header { padding: 10px 16px; background: #1a1a1a; border-bottom: 1px solid #333;
             font-size: 1.1em; letter-spacing: 0.05em; color: #7ecfff; flex-shrink: 0; }
    .main { display: flex; flex: 1; overflow: hidden; }

    /* Video column — two stacked feeds */
    .video-column { flex: 1; display: flex; flex-direction: column; background: #000;
                    gap: 2px; overflow: hidden; }
    .video-box { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-height: 0; }
    .video-box .label { background: #111; color: #555; font-size: 0.68em;
                        text-transform: uppercase; letter-spacing: 0.1em;
                        padding: 4px 10px; flex-shrink: 0; }
    .video-box img { flex: 1; width: 100%; object-fit: contain; display: block; min-height: 0; }

    /* Status panel */
    .status-panel { width: 300px; background: #141414; border-left: 1px solid #2a2a2a;
                    display: flex; flex-direction: column; overflow: hidden; flex-shrink: 0; }
    .status-panel h2 { padding: 10px 12px; font-size: 0.8em; text-transform: uppercase;
                       letter-spacing: 0.1em; color: #555; border-bottom: 1px solid #2a2a2a; }
    .kv { padding: 8px 12px; border-bottom: 1px solid #1e1e1e; }
    .kv .label { font-size: 0.7em; color: #555; text-transform: uppercase;
                 letter-spacing: 0.08em; margin-bottom: 3px; }
    .kv .value { font-size: 0.9em; word-break: break-word; }
    .status-ok   { color: #4caf50; }
    .status-done { color: #2196f3; }
    .status-err  { color: #f44336; }

    .history { flex: 1; overflow-y: auto; padding: 8px 12px; }
    .history-item { font-size: 0.78em; color: #666; padding: 3px 0;
                    border-bottom: 1px solid #1a1a1a; }
    .history-item span { color: #999; }
    .log-link { display: block; font-size: 0.75em; padding: 4px 0;
                border-bottom: 1px solid #1a1a1a; color: #7ecfff;
                text-decoration: none; }
    .log-link:hover { color: #fff; }
  </style>
</head>
<body>
  <header>&#x25B6; Rover Navigation Agent</header>
  <div class="main">

    <div class="video-column">
      <div class="video-box">
        <div class="label">&#x1F534; Live camera</div>
        <img src="/video/realtime" alt="live feed">
      </div>
      <div class="video-box">
        <div class="label">&#x1F9E0; Last Gemini query — with waypoints</div>
        <img src="/video/llm" alt="LLM frame">
      </div>
    </div>

    <div class="status-panel">
      <h2>Mission Status</h2>
      <div class="kv"><div class="label">Phase</div>
        <div class="value" id="phase">—</div></div>
      <div class="kv"><div class="label">Step</div>
        <div class="value" id="step">—</div></div>
      <div class="kv"><div class="label">Status</div>
        <div class="value" id="status">—</div></div>
      <div class="kv"><div class="label">Confidence</div>
        <div class="value" id="confidence">—</div></div>
      <div class="kv"><div class="label">Waypoints</div>
        <div class="value" id="waypoints">—</div></div>
      <div class="kv"><div class="label">Reasoning</div>
        <div class="value" id="reasoning">—</div></div>
      <div class="kv"><div class="label">LLM Timer</div>
        <div class="value" id="llm-timer">—</div></div>

      <h2 style="margin-top:4px">History</h2>
      <div class="history" id="history"></div>

      <h2 style="margin-top:4px">Logs</h2>
      <div id="log-list" style="padding:8px 12px; overflow-y:auto; max-height:120px;"></div>
    </div>

  </div>

  <script>
    const statusColors = {
      in_progress:      'status-ok',
      phase1_complete:  'status-done',
      mission_complete: 'status-done',
      no_path:          'status-err',
    };

    let _queryStart = 0;      // unix seconds; 0 = not querying
    let _lastResponseS = 0;
    let _timerInterval = null;

    function updateTimer() {
      const el = document.getElementById('llm-timer');
      if (_queryStart > 0) {
        const elapsed = (Date.now() / 1000 - _queryStart).toFixed(1);
        el.textContent = '⏱ querying... ' + elapsed + 's';
        el.style.color = '#ffeb3b';
      } else if (_lastResponseS > 0) {
        el.textContent = '✓ responded in ' + _lastResponseS.toFixed(2) + 's';
        el.style.color = '#4caf50';
      }
    }

    async function loadLogs() {
      try {
        const r = await fetch('/logs');
        const files = await r.json();
        document.getElementById('log-list').innerHTML = files.length
          ? files.map(f =>
              `<a class="log-link" href="/logs/${encodeURIComponent(f)}" download="${f}">&#x2B07; ${f}</a>`
            ).join('')
          : '<span style="font-size:0.75em;color:#555">No logs yet</span>';
      } catch(_) {}
    }
    loadLogs();
    setInterval(loadLogs, 10000);   // refresh list every 10s

    async function poll() {
      try {
        const r = await fetch('/status');
        const d = await r.json();

        document.getElementById('phase').textContent      = d.phase ?? '—';
        document.getElementById('step').textContent       = d.step  ?? '—';
        document.getElementById('confidence').textContent =
          d.confidence != null ? (d.confidence * 100).toFixed(0) + '%' : '—';
        document.getElementById('reasoning').textContent  = d.reasoning ?? '—';

        const statusEl = document.getElementById('status');
        statusEl.textContent  = d.goal_status ?? '—';
        statusEl.className    = 'value ' + (statusColors[d.goal_status] ?? '');

        const wps = d.waypoints ?? [];
        document.getElementById('waypoints').innerHTML = wps.length
          ? wps.map(w =>
              `<div style="margin-bottom:4px">
                <span style="color:${['#4caf50','#ffeb3b','#ff9800'][w.rank-1] ?? '#fff'}">
                  #${w.rank} ${(w.probability*100).toFixed(0)}%
                </span>
                (${w.x}, ${w.y}) ${w.description ?? ''}
              </div>`).join('')
          : 'none';

        const hist = document.getElementById('history');
        hist.innerHTML = (d.history ?? []).slice().reverse()
          .map(h => `<div class="history-item"><span>${h}</span></div>`)
          .join('');

        // Update timer state from server
        _queryStart   = d.llm_query_start ?? 0;
        _lastResponseS = d.llm_response_s ?? 0;
        if (_queryStart > 0 && !_timerInterval) {
          _timerInterval = setInterval(updateTimer, 100);
        } else if (_queryStart === 0 && _timerInterval) {
          clearInterval(_timerInterval);
          _timerInterval = null;
          updateTimer();
        }
      } catch(_) {}
      setTimeout(poll, 1000);
    }
    poll();
  </script>
</body>
</html>"""


_BLANK_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def _stream(frame_lock, get_frame_fn, placeholder_text: str):
    """MJPEG generator — matches momanip_navigation.py pattern exactly."""
    blank = _BLANK_FRAME.copy()
    cv2.putText(blank, placeholder_text, (30, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)

    while True:
        with frame_lock:
            frame = get_frame_fn()
            if frame is None:
                ret, buf = cv2.imencode(".jpg", blank)
            else:
                ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            frame_bytes = buf.tobytes()

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.05)


@app.route("/")
def index():
    return render_template_string(_HTML)


@app.route("/video/realtime")
def video_realtime():
    return Response(_stream(_raw_lock, lambda: _raw_frame, "Waiting for camera..."),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video/llm")
def video_llm():
    return Response(_stream(_llm_lock, lambda: _llm_frame, "Waiting for first Gemini query..."),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/logs")
def list_logs():
    files = sorted(Path("logs").glob("rover_*.log"), reverse=True)
    return jsonify([f.name for f in files])


@app.route("/logs/<path:filename>")
def download_log(filename):
    log_path = Path("logs") / filename
    if not log_path.exists() or log_path.parent != Path("logs"):
        return "Not found", 404
    return send_file(log_path.resolve(), as_attachment=True, download_name=filename)


@app.route("/status")
def status():
    result = get_result()
    result["step"] = _step
    result["history"] = [
        f"Step {t['step']}: ({t['x']},{t['y']}) {t['description']}"
        for t in _trajectory
    ]
    result["llm_query_start"] = _llm_query_start
    result["llm_response_s"] = _llm_response_s
    return jsonify(result)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rover navigation agent")
    parser.add_argument("--device",      type=int,   default=0,    help="Camera device index")
    parser.add_argument("--interval",    type=float, default=3.0,  help="Seconds between Gemini queries")
    parser.add_argument("--port",        type=int,   default=5000, help="Web server port")
    parser.add_argument("--roomba-port", type=str,   default=None, help="Roomba serial port (e.g. /dev/ttyUSB0); omit to disable")
    parser.add_argument("--dry-run",     action="store_true",      help="Log Roomba commands but do not send them")
    args = parser.parse_args()

    log.info("=== Rover agent starting ===")
    log.info("Camera device : %d", args.device)
    log.info("Query interval: %.1fs", args.interval)
    log.info("Model         : %s", gemini_client.MODEL)
    log.info("Web UI        : http://localhost:%d", args.port)
    if args.roomba_port:
        log.info("Roomba port   : %s%s", args.roomba_port, " (dry-run)" if args.dry_run else "")
    else:
        log.info("Roomba        : disabled (pass --roomba-port to enable)")

    t = threading.Thread(
        target=agent_loop,
        args=(args.device, args.interval, args.roomba_port, args.dry_run),
        daemon=True,
    )
    t.start()

    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
