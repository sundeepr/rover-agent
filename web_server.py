#!/usr/bin/env python3
"""
Standalone web server for the rover navigation agent.

Runs independently of the agent — survives agent restarts and crashes.
The agent process connects to this server and publishes frames + status
via HTTP POST. The browser always talks to this server only.

Usage:
    python web_server.py                      # default: 0.0.0.0:5001
    python web_server.py --port 5001

Then start the agent:
    python rover_agent.py --web-server http://localhost:5001 ...
"""

import argparse
import logging
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request, send_file

log = logging.getLogger("rover.web_server")

# How long without a push before the agent is considered disconnected.
AGENT_TIMEOUT_S = 10.0

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
             font-size: 1.1em; letter-spacing: 0.05em; color: #7ecfff; flex-shrink: 0;
             display: flex; align-items: center; gap: 16px; }
    #pause-btn { padding: 5px 18px; border: none; border-radius: 4px; cursor: pointer;
                 font-family: monospace; font-size: 0.9em; font-weight: bold;
                 background: #c0392b; color: #fff; transition: background 0.15s; }
    #pause-btn:hover { filter: brightness(1.2); }
    #pause-btn.paused { background: #27ae60; }
    #agent-indicator { font-size: 0.75em; margin-left: auto; display: flex;
                       align-items: center; gap: 6px; }
    #agent-dot { font-size: 1.1em; }
    #agent-dot.connected    { color: #4caf50; }
    #agent-dot.disconnected { color: #f44336; }
    .main { display: flex; flex: 1; overflow: hidden; }

    /* Video column */
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
  <header>
    <span>&#x25B6; Rover Navigation Agent</span>
    <button id="pause-btn" onclick="togglePause()">&#x23F8; Pause</button>
    <div id="agent-indicator">
      <span id="agent-dot" class="disconnected">&#x25CF;</span>
      <span id="agent-label">Agent disconnected</span>
    </div>
  </header>
  <div class="main">

    <div class="video-column">
      <div class="video-box">
        <div class="label">&#x1F534; Live camera</div>
        <img src="/video/realtime" alt="live feed">
      </div>
      <div class="video-box">
        <div class="label">&#x1F9E0; Last query — with waypoints</div>
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
      path_lost:        'status-err',
      initializing:     '',
    };
    const navModeColors = {
      aligning:  '#ffeb3b',
      following: '#4caf50',
    };

    let _queryStart = 0;
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
    setInterval(loadLogs, 10000);

    async function togglePause() {
      const r = await fetch('/pause', { method: 'POST' });
      const d = await r.json();
      updatePauseButton(d.paused);
    }

    function updatePauseButton(paused) {
      const btn = document.getElementById('pause-btn');
      if (paused) {
        btn.textContent = '▶ Resume';
        btn.classList.add('paused');
      } else {
        btn.textContent = '⏸ Pause';
        btn.classList.remove('paused');
      }
    }

    function updateAgentIndicator(connected) {
      const dot   = document.getElementById('agent-dot');
      const label = document.getElementById('agent-label');
      if (connected) {
        dot.className = 'connected';
        label.textContent = 'Agent connected';
      } else {
        dot.className = 'disconnected';
        label.textContent = 'Agent disconnected';
      }
    }

    async function poll() {
      try {
        const r = await fetch('/status');
        const d = await r.json();

        updatePauseButton(d.paused ?? false);
        updateAgentIndicator(d.agent_connected ?? false);

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

        _queryStart    = d.llm_query_start ?? 0;
        _lastResponseS = d.llm_response_s  ?? 0;
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


# ── Server state ───────────────────────────────────────────────────────────────

class _ServerState:
    """Thread-safe buffer for frames and status received from the agent."""

    def __init__(self):
        self._lock       = threading.Lock()
        self.raw_jpeg    = None          # bytes | None
        self.llm_jpeg    = None          # bytes | None
        self.status      = {}            # latest JSON from agent
        self.paused      = False
        self.last_push   = 0.0           # epoch seconds

    @property
    def agent_connected(self) -> bool:
        return (time.time() - self.last_push) < AGENT_TIMEOUT_S

    def touch(self):
        self.last_push = time.time()

    @property
    def lock(self):
        return self._lock


# ── Web server ─────────────────────────────────────────────────────────────────

class WebServer:
    """
    Standalone Flask server.

    Receives frames and status from the agent via POST /agent/frame and
    POST /agent/status. Serves MJPEG streams and a status JSON endpoint
    to the browser. Survives agent restarts and crashes.
    """

    def __init__(self, log_dir: Path = Path("logs")):
        self._state   = _ServerState()
        self._log_dir = log_dir
        self._blank   = np.zeros((480, 640, 3), dtype=np.uint8)
        self._app     = Flask(__name__)
        self._register_routes()

    def run(self, host: str = "0.0.0.0", port: int = 5001) -> None:
        werkzeug_log = logging.getLogger("werkzeug")
        werkzeug_log.handlers   = []
        werkzeug_log.propagate  = False
        werkzeug_log.setLevel(logging.WARNING)

        log.info("Web server listening on http://%s:%d", host, port)
        self._app.run(host=host, port=port, debug=False,
                      use_reloader=False, threaded=True)

    # ── Route registration ────────────────────────────────────────────────────

    def _register_routes(self) -> None:
        app = self._app
        app.add_url_rule("/",                       "index",        self._index)
        app.add_url_rule("/video/realtime",         "v_realtime",   self._video_realtime)
        app.add_url_rule("/video/llm",              "v_llm",        self._video_llm)
        app.add_url_rule("/status",                 "status",       self._status)
        app.add_url_rule("/pause",                  "pause",        self._pause,        methods=["POST"])
        app.add_url_rule("/agent/frame",            "agent_frame",  self._agent_frame,  methods=["POST"])
        app.add_url_rule("/agent/status",           "agent_status", self._agent_status, methods=["POST"])
        app.add_url_rule("/logs",                   "list_logs",    self._list_logs)
        app.add_url_rule("/logs/<path:filename>",   "dl_log",       self._download_log)

    # ── Agent push endpoints ──────────────────────────────────────────────────

    def _agent_frame(self):
        """POST /agent/frame?stream=realtime|llm  body: raw JPEG bytes."""
        stream = request.args.get("stream", "realtime")
        jpeg   = request.get_data()
        with self._state.lock:
            self._state.touch()
            if stream == "llm":
                self._state.llm_jpeg = jpeg
            else:
                self._state.raw_jpeg = jpeg
            paused = self._state.paused
        return jsonify({"ok": True, "paused": paused})

    def _agent_status(self):
        """POST /agent/status  body: JSON status dict."""
        data = request.get_json(force=True) or {}
        with self._state.lock:
            self._state.touch()
            self._state.status = data
            paused = self._state.paused
        return jsonify({"ok": True, "paused": paused})

    # ── Browser endpoints ─────────────────────────────────────────────────────

    def _index(self):
        return render_template_string(_HTML)

    def _video_realtime(self):
        return Response(
            self._stream(lambda: self._state.raw_jpeg, "Waiting for agent..."),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def _video_llm(self):
        return Response(
            self._stream(lambda: self._state.llm_jpeg, "Waiting for first query..."),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def _pause(self):
        """Toggle pause. The agent reads the new state on its next push."""
        with self._state.lock:
            self._state.paused = not self._state.paused
            paused = self._state.paused
        log.info("Pause toggled → %s", paused)
        return jsonify({"paused": paused})

    def _status(self):
        with self._state.lock:
            result              = dict(self._state.status)
            result["paused"]    = self._state.paused
            result["agent_connected"] = self._state.agent_connected
        return jsonify(result)

    def _list_logs(self):
        files = sorted(self._log_dir.glob("rover_*.log"), reverse=True)
        return jsonify([f.name for f in files])

    def _download_log(self, filename: str):
        log_path = self._log_dir / filename
        if not log_path.exists() or log_path.parent.resolve() != self._log_dir.resolve():
            return "Not found", 404
        return send_file(log_path.resolve(), as_attachment=True, download_name=filename)

    # ── MJPEG stream ──────────────────────────────────────────────────────────

    def _stream(self, get_jpeg_fn, placeholder_text: str):
        """
        MJPEG generator at ~20 fps.

        get_jpeg_fn is called under the state lock and returns bytes or None.
        When None (no frame yet or agent never connected), sends a grey
        placeholder image.
        """
        blank = self._blank.copy()
        cv2.putText(blank, placeholder_text, (30, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
        _, buf      = cv2.imencode(".jpg", blank)
        blank_bytes = buf.tobytes()

        while True:
            with self._state.lock:
                frame_bytes = get_jpeg_fn()
            data = frame_bytes if frame_bytes is not None else blank_bytes
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            time.sleep(0.05)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Rover web server (standalone)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", default=5001, type=int,
                        help="HTTP port (default: 5001)")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    server = WebServer(log_dir=Path("logs"))
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
