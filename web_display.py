"""
WebDisplay — Flask web server for the rover navigation agent.

Streams the live camera feed and the last LLM-annotated frame as MJPEG,
and serves a mission status panel via a JSON endpoint.

Usage:
    display = WebDisplay(state, log_dir=Path("logs"))
    display.run(host="0.0.0.0", port=5000)   # blocks; call from main thread
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, send_file

from navigation_strategy import AgentState

log = logging.getLogger("rover.web")

# ── HTML template ──────────────────────────────────────────────────────────────

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
      <div class="kv"><div class="label">Nav Mode</div>
        <div class="value" id="nav-mode">—</div></div>
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
    const navModeColors = {
      aligning:  '#ffeb3b',
      following: '#4caf50',
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

        const navEl = document.getElementById('nav-mode');
        navEl.textContent = d.navigation_mode ?? '—';
        navEl.style.color = navModeColors[d.navigation_mode] ?? '#e0e0e0';

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


# ── WebDisplay ─────────────────────────────────────────────────────────────────

class WebDisplay:
    """
    Owns the Flask application. Receives an AgentState at construction —
    that is its only coupling to the rest of the system.
    """

    def __init__(self, state: AgentState, log_dir: Path = Path("logs")):
        self._state = state
        self._log_dir = log_dir
        self._blank = np.zeros((480, 640, 3), dtype=np.uint8)
        self._app = Flask(__name__)
        self._register_routes()

    @property
    def app(self) -> Flask:
        """Expose the Flask app (useful for testing or WSGI integration)."""
        return self._app

    def run(self, host: str = "0.0.0.0", port: int = 5000) -> None:
        """Start Flask on the calling thread (blocks). Call from main thread."""
        # Route werkzeug access logs to the rover log file only (not stdout)
        werkzeug_log = logging.getLogger("werkzeug")
        werkzeug_log.handlers = []
        werkzeug_log.propagate = False
        for handler in logging.getLogger("rover").handlers:
            if isinstance(handler, logging.FileHandler):
                werkzeug_log.addHandler(handler)
        werkzeug_log.setLevel(logging.DEBUG)

        self._app.run(host=host, port=port, debug=False,
                      use_reloader=False, threaded=True)

    # ── Route registration ─────────────────────────────────────────────────────

    def _register_routes(self) -> None:
        app = self._app
        app.add_url_rule("/",                        "index",        self._index)
        app.add_url_rule("/video/realtime",          "video_realtime", self._video_realtime)
        app.add_url_rule("/video/llm",               "video_llm",    self._video_llm)
        app.add_url_rule("/status",                  "status",       self._status)
        app.add_url_rule("/logs",                    "list_logs",    self._list_logs)
        app.add_url_rule("/logs/<path:filename>",    "download_log", self._download_log)

    # ── Route handlers ─────────────────────────────────────────────────────────

    def _index(self):
        return render_template_string(_HTML)

    def _video_realtime(self):
        return Response(
            self._stream(self._state.raw_lock,
                         lambda: self._state.raw_frame,
                         "Waiting for camera..."),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def _video_llm(self):
        return Response(
            self._stream(self._state.llm_lock,
                         lambda: self._state.llm_frame,
                         "Waiting for first Gemini query..."),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def _status(self):
        with self._state.result_lock:
            result      = dict(self._state.latest_result)
            step        = self._state.step
            trajectory  = list(self._state.trajectory)
            query_start = self._state.llm_query_start
            response_s  = self._state.llm_response_s
        result["step"] = step
        result["history"] = [
            f"Step {t['step']}: ({t['x']},{t['y']}) {t['description']}"
            for t in trajectory
        ]
        result["llm_query_start"] = query_start
        result["llm_response_s"]  = response_s
        return jsonify(result)

    def _list_logs(self):
        files = sorted(self._log_dir.glob("rover_*.log"), reverse=True)
        return jsonify([f.name for f in files])

    def _download_log(self, filename: str):
        log_path = self._log_dir / filename
        if not log_path.exists() or log_path.parent.resolve() != self._log_dir.resolve():
            return "Not found", 404
        return send_file(log_path.resolve(), as_attachment=True, download_name=filename)

    # ── MJPEG streaming ────────────────────────────────────────────────────────

    def _stream(self, lock, get_frame_fn, placeholder_text: str):
        """MJPEG generator — encodes inside lock to match momanip pattern."""
        blank = self._blank.copy()
        cv2.putText(blank, placeholder_text, (30, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)

        while True:
            with lock:
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
