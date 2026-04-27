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
    header { padding: 8px 16px; background: #1a1a1a; border-bottom: 1px solid #333;
             font-size: 1.1em; letter-spacing: 0.05em; color: #7ecfff; flex-shrink: 0;
             display: flex; align-items: center; gap: 16px; }
    #agent-indicator { font-size: 0.75em; margin-left: auto; display: flex;
                       align-items: center; gap: 6px; }
    #agent-dot { font-size: 1.1em; }
    #agent-dot.connected    { color: #4caf50; }
    #agent-dot.disconnected { color: #f44336; }
    .main { display: flex; flex: 1; overflow: hidden; }

    /* Content area */
    .content-area { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

    /* Videos side by side */
    .videos-row { flex: 1; display: flex; flex-direction: row; background: #000;
                  gap: 2px; overflow: hidden; min-height: 0; }
    .video-box { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-height: 0; }
    .video-box .label { background: #111; color: #555; font-size: 0.68em;
                        text-transform: uppercase; letter-spacing: 0.1em;
                        padding: 4px 10px; flex-shrink: 0; }
    .video-box img { flex: 1; width: 100%; object-fit: contain; display: block; min-height: 0; }

    /* Bottom bar */
    .bottom-bar { height: 190px; display: flex; flex-direction: row; flex-shrink: 0;
                  background: #111; border-top: 1px solid #2a2a2a; }

    /* Chat */
    .chat-section { flex: 1; display: flex; flex-direction: column;
                    padding: 8px 12px; gap: 6px; overflow: hidden; min-width: 0; }
    .chat-history { flex: 1; overflow-y: auto; display: flex; flex-direction: column;
                    gap: 4px; justify-content: flex-end; }
    .chat-msg { font-size: 0.78em; padding: 4px 10px; border-radius: 4px;
                max-width: 85%; word-break: break-word; line-height: 1.4; }
    .chat-msg.user  { align-self: flex-end; background: #1a3a5c; color: #7ecfff; }
    .chat-msg.agent { align-self: flex-start; background: #1e2a1e; color: #4caf50; }
    .chat-msg.err   { align-self: flex-start; background: #2a1a1a; color: #f44336; }
    .chat-input-row { display: flex; gap: 6px; flex-shrink: 0; }
    .chat-input-row input { flex: 1; background: #0a0a0a; border: 1px solid #333;
                             color: #e0e0e0; padding: 7px 10px; font-family: monospace;
                             font-size: 0.85em; border-radius: 4px; outline: none; }
    .chat-input-row input:focus { border-color: #7ecfff; }
    .chat-input-row input::placeholder { color: #444; }
    .chat-input-row button { padding: 7px 16px; background: #1a3a5c; border: none;
                              color: #7ecfff; font-family: monospace; font-size: 0.85em;
                              border-radius: 4px; cursor: pointer; white-space: nowrap; }
    .chat-input-row button:hover { background: #234d7a; }

    /* Controls: joystick + pause */
    .controls-section { width: 190px; display: flex; flex-direction: column;
                        align-items: center; justify-content: center; gap: 10px;
                        border-left: 1px solid #2a2a2a; flex-shrink: 0; padding: 10px 8px; }
    #pause-btn { padding: 6px 0; width: 110px; border: none; border-radius: 4px;
                 cursor: pointer; font-family: monospace; font-size: 0.82em;
                 font-weight: bold; background: #c0392b; color: #fff;
                 transition: background 0.15s; }
    #pause-btn:hover { filter: brightness(1.2); }
    #pause-btn.paused { background: #27ae60; }

    /* Joystick */
    #joystick-wrap { display: flex; flex-direction: column; align-items: center;
                     gap: 4px; user-select: none; }
    #joystick-base { width: 100px; height: 100px; border-radius: 50%;
                     background: rgba(255,255,255,0.07); border: 2px solid rgba(255,255,255,0.18);
                     position: relative; cursor: default; }
    #joystick-knob { width: 36px; height: 36px; border-radius: 50%;
                     background: rgba(200,200,200,0.25); border: 2px solid rgba(255,255,255,0.5);
                     position: absolute; top: 50%; left: 50%;
                     transform: translate(-50%, -50%);
                     transition: background 0.1s; cursor: grab; }
    #joystick-knob.active { cursor: grabbing; }
    #joystick-knob.moving { background: rgba(76,175,80,0.55); }
    #joystick-readout { font-size: 0.6em; color: rgba(255,255,255,0.4);
                        text-align: center; letter-spacing: 0.04em; }
    #joy-status { font-size: 0.68em; color: #555; padding: 2px 0;
                  white-space: nowrap; flex-shrink: 0; letter-spacing: 0.03em; }

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
                border-bottom: 1px solid #1a1a1a; color: #7ecfff; text-decoration: none; }
    .log-link:hover { color: #fff; }
  </style>
</head>
<body>
  <header>
    <span>&#x25B6; Rover Navigation Agent</span>
    <div id="agent-indicator">
      <span id="agent-dot" class="disconnected">&#x25CF;</span>
      <span id="agent-label">Agent disconnected</span>
    </div>
  </header>

  <div class="main">
    <div class="content-area">

      <div class="videos-row">
        <div class="video-box">
          <div class="label">&#x1F534; Live camera</div>
          <img src="/video/realtime" alt="live feed">
        </div>
        <div class="video-box">
          <div class="label">&#x1F9E0; Last query — with waypoints</div>
          <img src="/video/llm" alt="LLM frame">
        </div>
        <div class="video-box" id="down-cam-box">
          <div class="label">&#x1F4F7; Down camera — row centering</div>
          <img src="/video/down" alt="down camera">
        </div>
      </div>

      <div class="bottom-bar">
        <div class="chat-section">
          <div class="chat-history" id="chat-history"></div>
          <div id="joy-status">—</div>
          <div class="chat-input-row">
            <input type="text" id="chat-input"
                   placeholder="Set goal, e.g. Follow the dirt path…"
                   onkeydown="if(event.key==='Enter') sendChat()">
            <button onclick="sendChat()">&#x27A4; Send</button>
          </div>
        </div>
        <div class="controls-section">
          <div id="joystick-wrap">
            <div id="joystick-base"><div id="joystick-knob"></div></div>
            <div id="joystick-readout">joystick</div>
          </div>
          <button id="pause-btn" onclick="togglePause()">&#x23F8; Pause</button>
        </div>
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
      <div class="kv" id="centering-kv" style="display:none">
        <div class="label">Row Centering</div>
        <div class="value" id="centering">—</div></div>
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
    const navModeColors = { aligning: '#ffeb3b', following: '#4caf50' };

    let _queryStart = 0, _lastResponseS = 0, _timerInterval = null;
    let _paused = false;
    let _chatIdx = 0;   // how many agent chat_history entries we've already shown

    function updateTimer() {
      const el = document.getElementById('llm-timer');
      if (_queryStart > 0) {
        el.textContent = '⏱ querying... ' + (Date.now()/1000 - _queryStart).toFixed(1) + 's';
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
          ? files.map(f => `<a class="log-link" href="/logs/${encodeURIComponent(f)}" download="${f}">&#x2B07; ${f}</a>`).join('')
          : '<span style="font-size:0.75em;color:#555">No logs yet</span>';
      } catch(_) {}
    }
    loadLogs();
    setInterval(loadLogs, 10000);

    // ── WebSocket control channel (port 5002) ────────────────────────────
    // Direct path to drive_raw() — bypasses the 50 ms agent_publisher poll.
    // Falls back to HTTP /chat if WS is unavailable (agent not running yet,
    // port blocked, etc.).
    let _ctrlWs = null;
    (function _connectCtrl() {
      try {
        _ctrlWs = new WebSocket(`ws://${location.hostname}:5002`);
        _ctrlWs.onopen  = () => log.debug && console.debug('ctrl-ws open');
        _ctrlWs.onclose = () => { _ctrlWs = null; setTimeout(_connectCtrl, 2000); };
        _ctrlWs.onerror = () => {};   // onclose fires after onerror; suppress console noise
      } catch(_) {}
    })();

    function sendMovement(fwd, turn) {
      if (_ctrlWs && _ctrlWs.readyState === WebSocket.OPEN) {
        _ctrlWs.send(JSON.stringify({fwd, turn}));
      } else {
        // HTTP fallback (agent_publisher path, ~50ms latency)
        chat({ type: 'movement', fwd, turn }).catch(() => {});
      }
    }

    // ── Chat ──────────────────────────────────────────────────────────────
    function addChatMsg(text, type) {
      const el = document.createElement('div');
      el.className = 'chat-msg ' + type;
      el.textContent = text;
      const hist = document.getElementById('chat-history');
      hist.appendChild(el);
      hist.scrollTop = hist.scrollHeight;
    }

    async function chat(payload) {
      const r = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      return r.json();
    }

    async function sendChat() {
      const input = document.getElementById('chat-input');
      const text  = input.value.trim();
      if (!text) return;
      input.value = '';
      addChatMsg(text, 'user');
      try {
        const d = await chat({ type: 'goal', text });
        addChatMsg(d.message ?? 'Goal set', 'agent');
      } catch(e) {
        addChatMsg('Error: could not reach server', 'err');
      }
    }

    // ── Pause / Resume ────────────────────────────────────────────────────
    async function togglePause() {
      const action = _paused ? 'resume' : 'pause';
      addChatMsg(action, 'user');
      try {
        const d = await chat({ type: 'control', action });
        _paused = d.paused ?? _paused;
        updatePauseButton(_paused);
        addChatMsg(d.message ?? action, 'agent');
      } catch(_) {
        addChatMsg('Error: could not reach server', 'err');
      }
    }

    function updatePauseButton(paused) {
      _paused = paused;
      const btn = document.getElementById('pause-btn');
      btn.textContent = paused ? '▶ Resume' : '⏸ Pause';
      btn.classList.toggle('paused', paused);
    }

    function updateAgentIndicator(connected) {
      const dot = document.getElementById('agent-dot');
      dot.className = connected ? 'connected' : 'disconnected';
      document.getElementById('agent-label').textContent =
        connected ? 'Agent connected' : 'Agent disconnected';
    }

    // ── Status poll ───────────────────────────────────────────────────────
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
        statusEl.textContent = d.goal_status ?? '—';
        statusEl.className   = 'value ' + (statusColors[d.goal_status] ?? '');

        const navEl = document.getElementById('nav-mode');
        navEl.textContent = d.navigation_mode ?? '—';
        navEl.style.color = navModeColors[d.navigation_mode] ?? '#e0e0e0';

        const wps = d.waypoints ?? [];
        document.getElementById('waypoints').innerHTML = wps.length
          ? wps.map(w =>
              `<div style="margin-bottom:4px">
                <span style="color:${['#4caf50','#ffeb3b','#ff9800'][w.rank-1]??'#fff'}">
                  #${w.rank} ${(w.probability*100).toFixed(0)}%
                </span> (${w.x},${w.y}) ${w.description??''}
              </div>`).join('')
          : 'none';

        document.getElementById('history').innerHTML =
          (d.history ?? []).slice().reverse()
            .map(h => `<div class="history-item"><span>${h}</span></div>`).join('');

        _queryStart = d.llm_query_start ?? 0;
        _lastResponseS = d.llm_response_s ?? 0;
        if (_queryStart > 0 && !_timerInterval)
          _timerInterval = setInterval(updateTimer, 100);
        else if (_queryStart === 0 && _timerInterval) {
          clearInterval(_timerInterval); _timerInterval = null; updateTimer();
        }

        // Row centering stats (only shown for row_centering_omnivla strategy)
        const hasCentering = d.has_down_cam || d.row_lateral_error_px != null;
        document.getElementById('centering-kv').style.display = hasCentering ? '' : 'none';
        // down-cam-box is always visible — hiding it drops the MJPEG connection
        if (hasCentering) {
          const err = d.row_lateral_error_px;
          const on  = d.centering_applied;
          const cEl = document.getElementById('centering');
          cEl.textContent = `err=${err > 0 ? '+' : ''}${err.toFixed(1)}px  ${on ? 'correcting' : 'no rows'}`;
          cEl.style.color = on ? '#4caf50' : '#ffeb3b';
        }

        // Merge new agent-pushed chat messages (e.g. "Ready")
        const serverChat = d.chat_history ?? [];
        for (let i = _chatIdx; i < serverChat.length; i++) {
          const m = serverChat[i];
          addChatMsg(m.text, m.role === 'user' ? 'user' : 'agent');
        }
        _chatIdx = serverChat.length;
      } catch(_) {}
      setTimeout(poll, 1000);
    }
    poll();

    // ── Joystick ──────────────────────────────────────────────────────────
    (function () {
      const BASE_R = 48, KNOB_R = 18, DEAD = 7;
      const AXIS_DEAD = 12;   // per-axis dead zone (%) — suppresses cross-axis noise
      const base   = document.getElementById('joystick-base');
      const knob   = document.getElementById('joystick-knob');
      const readout = document.getElementById('joystick-readout');

      const joyStatus = document.getElementById('joy-status');
      let dragging = false, rect;
      let _joyFwd = 0, _joyTurn = 0, _joyTimer = null;

      function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

      function applyPos(cx, cy) {
        const dist  = Math.hypot(cx, cy);
        const limit = BASE_R - KNOB_R;
        const scale = dist > limit ? limit / dist : 1;
        knob.style.transform =
          `translate(calc(-50% + ${cx*scale}px), calc(-50% + ${cy*scale}px))`;
        const moving = dist > DEAD;
        knob.classList.toggle('moving', moving);
        if (!moving) {
          _joyFwd = 0; _joyTurn = 0;
          readout.textContent = 'joystick';
          joyStatus.textContent = '—';
          joyStatus.style.color = '#555';
          return;
        }
        let rawFwd  = clamp(Math.round(-cy / (BASE_R - KNOB_R) * 100), -100, 100);
        let rawTurn = clamp(Math.round( cx / (BASE_R - KNOB_R) * 100), -100, 100);
        // Per-axis dead zone: suppress small cross-axis noise.
        // e.g. pushing forward with a slight right offset won't trigger a turn.
        _joyFwd  = Math.abs(rawFwd)  < AXIS_DEAD ? 0 : rawFwd;
        _joyTurn = Math.abs(rawTurn) < AXIS_DEAD ? 0 : rawTurn;
        const fwdS  = _joyFwd  > 0 ? `FWD ${_joyFwd}%`  : _joyFwd  < 0 ? `REV ${-_joyFwd}%`  : '';
        const turnS = _joyTurn > 0 ? `R ${_joyTurn}%`   : _joyTurn < 0 ? `L ${-_joyTurn}%`   : '';
        readout.textContent = [fwdS, turnS].filter(Boolean).join('  ') || 'holding';
        joyStatus.textContent = `● manual  ${[fwdS, turnS].filter(Boolean).join('  ')}`;
        joyStatus.style.color = '#4caf50';
      }

      function centre() {
        knob.style.transform = 'translate(-50%, -50%)';
        knob.classList.remove('moving', 'active');
        readout.textContent = 'joystick';
        joyStatus.textContent = '—';
        joyStatus.style.color = '#555';
        _joyFwd = 0; _joyTurn = 0;
      }

      function startDrag(e) {
        e.preventDefault();
        dragging = true;
        rect = base.getBoundingClientRect();
        knob.classList.add('active');
        // Send at 20 Hz while held to keep the publisher's 350 ms safety window open
        _joyTimer = setInterval(() => {
          if (_joyFwd !== 0 || _joyTurn !== 0)
            sendMovement(_joyFwd, _joyTurn);
        }, 50);
      }

      function moveDrag(e) {
        if (!dragging) return;
        e.preventDefault();
        const c = e.touches ? e.touches[0] : e;
        applyPos(c.clientX - rect.left - BASE_R, c.clientY - rect.top - BASE_R);
      }

      function endDrag() {
        if (!dragging) return;
        dragging = false;
        clearInterval(_joyTimer); _joyTimer = null;
        centre();
        sendMovement(0, 0);
      }

      knob.addEventListener('mousedown',  startDrag, { passive: false });
      knob.addEventListener('touchstart', startDrag, { passive: false });
      window.addEventListener('mousemove',  moveDrag, { passive: false });
      window.addEventListener('touchmove',  moveDrag, { passive: false });
      window.addEventListener('mouseup',    endDrag);
      window.addEventListener('touchend',   endDrag);
    })();
  </script>
</body>
</html>"""


# ── Server state ───────────────────────────────────────────────────────────────

class _ServerState:
    """Thread-safe buffer for frames and status received from the agent."""

    def __init__(self):
        self._lock        = threading.Lock()
        self.raw_jpeg     = None          # bytes | None
        self.llm_jpeg     = None          # bytes | None
        self.down_jpeg    = None          # bytes | None  (downward-facing camera)
        self.status       = {}            # latest JSON from agent
        self.paused       = False
        self.last_push    = 0.0           # epoch seconds
        self.goal         = ""            # latest goal set via /chat
        self.movement     = {"fwd": 0, "turn": 0}  # latest joystick from /chat
        self.chat_history: list = []      # [{"role","text","ts"}] — agent-pushed messages

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
        app.add_url_rule("/video/down",             "v_down",       self._video_down)
        app.add_url_rule("/status",                 "status",       self._status)
        app.add_url_rule("/pause",                  "pause",        self._pause,        methods=["POST"])
        app.add_url_rule("/agent/frame",            "agent_frame",  self._agent_frame,  methods=["POST"])
        app.add_url_rule("/agent/status",           "agent_status", self._agent_status, methods=["POST"])
        app.add_url_rule("/chat",                   "chat",         self._chat,         methods=["POST"])
        app.add_url_rule("/agent/chat",             "agent_chat",   self._agent_chat,   methods=["POST"])
        app.add_url_rule("/logs",                   "list_logs",    self._list_logs)
        app.add_url_rule("/logs/<path:filename>",   "dl_log",       self._download_log)

    # ── Agent push endpoints ──────────────────────────────────────────────────

    def _agent_frame(self):
        """POST /agent/frame?stream=realtime|llm|down  body: raw JPEG bytes."""
        stream = request.args.get("stream", "realtime")
        jpeg   = request.get_data()
        with self._state.lock:
            self._state.touch()
            if stream == "llm":
                self._state.llm_jpeg  = jpeg
            elif stream == "down":
                self._state.down_jpeg = jpeg
            else:
                self._state.raw_jpeg  = jpeg
            paused = self._state.paused
        return jsonify({"ok": True, "paused": paused})

    def _agent_status(self):
        """POST /agent/status  body: JSON status dict."""
        data = request.get_json(force=True) or {}
        with self._state.lock:
            self._state.touch()
            self._state.status = data
            paused = self._state.paused
            goal   = self._state.goal
            mv     = dict(self._state.movement)
            # Movement is NOT single-consumed here — the publisher reads it every
            # cycle and drives continuously. The browser sends fwd:0,turn:0 on
            # release; the publisher's 350 ms safety expiry stops the rover if
            # the browser goes silent before sending the release.
        return jsonify({"ok": True, "paused": paused, "goal": goal, "movement": mv})

    def _agent_chat(self):
        """POST /agent/chat  body: {"role": "agent", "text": "..."}"""
        data = request.get_json(force=True) or {}
        msg  = {"role": data.get("role", "agent"),
                "text": data.get("text", ""),
                "ts":   time.time()}
        with self._state.lock:
            self._state.chat_history.append(msg)
        return jsonify({"ok": True})

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

    def _video_down(self):
        return Response(
            self._stream(lambda: self._state.down_jpeg, "No down camera"),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def _chat(self):
        """POST /chat — unified command endpoint from the browser UI."""
        data     = request.get_json(force=True) or {}
        msg_type = data.get("type")

        if msg_type == "goal":
            text = data.get("text", "").strip()
            with self._state.lock:
                self._state.goal = text
            log.info("Goal set via chat: %s", text)
            return jsonify({"ok": True, "message": f"Goal set: {text}"})

        elif msg_type == "movement":
            fwd  = int(data.get("fwd",  0))
            turn = int(data.get("turn", 0))
            with self._state.lock:
                self._state.movement = {"fwd": fwd, "turn": turn}
            return jsonify({"ok": True})

        elif msg_type == "control":
            action = data.get("action")
            if action in ("pause", "resume"):
                with self._state.lock:
                    self._state.paused = (action == "pause")
                    paused = self._state.paused
                log.info("Control action: %s", action)
                return jsonify({"ok": True, "paused": paused,
                                "message": "Paused" if paused else "Resumed"})
            return jsonify({"ok": False, "message": f"Unknown action: {action}"}), 400

        return jsonify({"ok": False, "message": f"Unknown type: {msg_type}"}), 400

    def _pause(self):
        """Toggle pause (legacy endpoint — kept for agent_publisher compatibility)."""
        with self._state.lock:
            self._state.paused = not self._state.paused
            paused = self._state.paused
        log.info("Pause toggled → %s", paused)
        return jsonify({"paused": paused})

    def _status(self):
        with self._state.lock:
            result                    = dict(self._state.status)
            result["paused"]          = self._state.paused
            result["agent_connected"] = self._state.agent_connected
            result["goal"]            = self._state.goal
            result["chat_history"]    = list(self._state.chat_history[-50:])
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
