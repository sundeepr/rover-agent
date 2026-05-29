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
import asyncio
import logging
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request, send_file

log = logging.getLogger("rover.web_server")

# ── WebRTC imports (optional — falls back to MJPEG if not installed) ──────────
try:
    import fractions
    from av import VideoFrame
    from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
    from aiortc.contrib.media import MediaBlackhole
    _WEBRTC_AVAILABLE = True
except ImportError:
    _WEBRTC_AVAILABLE = False
    log.warning("aiortc not installed — WebRTC disabled, MJPEG fallback active. "
                "Install with: pip install aiortc")

# How long without a push before the agent is considered disconnected.
AGENT_TIMEOUT_S = 10.0

# ── WebRTC video track ────────────────────────────────────────────────────────

if _WEBRTC_AVAILABLE:
    class JpegVideoTrack(MediaStreamTrack):
        """
        Pulls the latest JPEG from _ServerState and yields VideoFrames at ~20 fps.
        Always sends the most recent frame — never queues stale frames.
        """
        kind = "video"

        def __init__(self, get_jpeg_fn, blank_jpeg: bytes):
            super().__init__()
            self._get_jpeg = get_jpeg_fn
            self._blank    = blank_jpeg
            self._pts      = 0
            self._clock    = fractions.Fraction(1, 20)   # 20 fps time base

        async def recv(self):
            jpeg = self._get_jpeg()
            data = jpeg if jpeg is not None else self._blank

            buf = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            frame = VideoFrame.from_ndarray(img_rgb, format="rgb24")
            frame.pts      = self._pts
            frame.time_base = self._clock
            self._pts      += 1

            # Pace at 20 fps
            await asyncio.sleep(0.05)
            return frame

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

    /* 2×2 video grid */
    .videos-grid { display: grid;
                   grid-template-columns: 1fr 1fr;
                   grid-template-rows: auto auto;
                   gap: 2px; background: #000; flex-shrink: 0; }
    .video-box { display: flex; flex-direction: column; overflow: hidden; }
    .video-box .label { background: #111; color: #555; font-size: 0.68em;
                        text-transform: uppercase; letter-spacing: 0.1em;
                        padding: 4px 10px; flex-shrink: 0; }
    /* img is exactly 4:3 — no letterbox bars, no distortion */
    .video-box img { width: 100%; aspect-ratio: 4/3; object-fit: fill; display: block; }

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

      <div class="videos-grid">
        <!-- Row 1: navigation cameras -->
        <div class="video-box" style="position:relative;">
          <div class="label">&#x1F534; Live camera — click to add waypoints</div>
          <img id="live-img" src="/video/realtime" style="width:100%;display:block;background:#000;">
          <canvas id="waypoint-canvas" style="position:absolute;top:0;left:0;width:100%;height:100%;cursor:crosshair;"></canvas>
        </div>
        <div class="video-box" style="position:relative;">
          <div class="label">&#x1F9E0; Last query — with waypoints</div>
          <img id="llm-img" src="/video/llm" style="width:100%;display:block;background:#000;">
          <canvas id="llm-canvas" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;"></canvas>
        </div>
        <!-- Row 2: wheel cameras -->
        <div class="video-box" style="position:relative;">
          <div class="label">&#x1F6DE; Left wheel camera</div>
          <img id="left-wheel-img" src="/video/left_wheel" style="width:100%;display:block;background:#000;">
        </div>
        <div class="video-box" style="position:relative;">
          <div class="label">&#x1F6DE; Right wheel camera</div>
          <img id="right-wheel-img" src="/video/right_wheel" style="width:100%;display:block;background:#000;">
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
      <div class="kv"><div class="label">Response Times</div>
        <div class="value" id="response-times">—</div></div>

      <h2 style="margin-top:4px">History</h2>
      <div class="history" id="history"></div>

      <h2 style="margin-top:4px">Logs</h2>
      <div id="log-list" style="padding:8px 12px; overflow-y:auto; max-height:120px;"></div>

      <h2 style="margin-top:4px">Teleop Recording</h2>
      <div style="padding:8px 12px; display:flex; flex-direction:column; gap:6px;">
        <div style="font-size:0.72em; color:#555;">Instruction</div>
        <input id="tp-instruction" type="text" placeholder="e.g. drive between the crop rows"
          style="background:#1e1e1e;border:1px solid #333;color:#ddd;padding:4px 6px;font-family:monospace;font-size:0.8em;border-radius:3px;">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;">
          <input id="tp-crop"   type="text" placeholder="Crop (e.g. chili)"
            style="background:#1e1e1e;border:1px solid #333;color:#ddd;padding:4px 6px;font-family:monospace;font-size:0.75em;border-radius:3px;">
          <input id="tp-loc"    type="text" placeholder="Location ID"
            style="background:#1e1e1e;border:1px solid #333;color:#ddd;padding:4px 6px;font-family:monospace;font-size:0.75em;border-radius:3px;">
          <input id="tp-stage"  type="text" placeholder="Growth stage"
            style="background:#1e1e1e;border:1px solid #333;color:#ddd;padding:4px 6px;font-family:monospace;font-size:0.75em;border-radius:3px;">
          <input id="tp-robot"  type="text" placeholder="Robot ID"
            style="background:#1e1e1e;border:1px solid #333;color:#ddd;padding:4px 6px;font-family:monospace;font-size:0.75em;border-radius:3px;">
        </div>
        <button onclick="teleopExecute()"
          style="width:100%;padding:8px;background:#e65100;border:none;color:#fff;border-radius:4px;cursor:pointer;font-family:monospace;font-size:0.9em;font-weight:bold;letter-spacing:0.05em;">
          &#x25B6; EXECUTE WAYPOINTS
        </button>
        <div style="display:flex;gap:6px;margin-top:2px;">
          <button id="tp-start-btn" onclick="teleopCmd('start')"
            style="flex:1;padding:6px;background:#1b5e20;border:none;color:#a5d6a7;border-radius:4px;cursor:pointer;font-family:monospace;font-size:0.8em;">
            &#x25CF; Start Episode
          </button>
          <button id="tp-stop-btn" onclick="teleopCmd('stop')"
            style="flex:1;padding:6px;background:#1a237e;border:none;color:#90caf9;border-radius:4px;cursor:pointer;font-family:monospace;font-size:0.8em;">
            &#x25A0; Stop
          </button>
          <button onclick="teleopCmd('discard')"
            style="padding:6px 10px;background:#311;border:none;color:#ef9a9a;border-radius:4px;cursor:pointer;font-family:monospace;font-size:0.8em;">
            &#x2715;
          </button>
        </div>
        <div id="tp-status" style="font-size:0.75em;color:#666;">idle — 0 frames</div>
        <div style="font-size:0.72em;color:#555;margin-top:4px;">Waypoints (right-click canvas to clear)</div>
        <div id="tp-waypoints" style="font-size:0.72em;color:#888;max-height:80px;overflow-y:auto;"></div>
      </div>
    </div>
  </div>

  <script>
    // ── WebRTC video setup ────────────────────────────────────────────────────
    async function _startWebRTC(elementId, stream) {
      const pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
      });

      pc.ontrack = e => {
        const el = document.getElementById(elementId);
        if (el && e.streams[0]) el.srcObject = e.streams[0];
      };

      // Dummy transceiver to receive video
      pc.addTransceiver('video', { direction: 'recvonly' });

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      // Wait for ICE gathering to complete
      await new Promise(resolve => {
        if (pc.iceGatheringState === 'complete') { resolve(); return; }
        pc.onicegatheringstatechange = () => {
          if (pc.iceGatheringState === 'complete') resolve();
        };
        setTimeout(resolve, 3000);  // fallback timeout
      });

      try {
        const r = await fetch('/webrtc/offer/' + stream, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sdp:  pc.localDescription.sdp,
            type: pc.localDescription.type,
          }),
        });
        if (!r.ok) throw new Error('offer rejected: ' + r.status);
        const answer = await r.json();
        await pc.setRemoteDescription(new RTCSessionDescription(answer));
      } catch (e) {
        console.warn('WebRTC failed for', stream, '— falling back to MJPEG:', e);
        // Fallback: replace <video> with <img> MJPEG
        const el = document.getElementById(elementId);
        if (el) {
          const img = document.createElement('img');
          img.id    = elementId;
          img.src   = '/video/' + stream;
          img.style = el.getAttribute('style') || '';
          el.parentNode.replaceChild(img, el);
        }
      }
    }

    // ── Teleop waypoint canvas ────────────────────────────────────────────────
    const _waypoints = [];  // [{nx, ny}, ...]

    /**
     * Return the pixel rect {x, y, w, h} of the actual image content in
     * CANVAS coordinates.
     *
     * Uses getBoundingClientRect() on both elements — reliable in all
     * browsers and layout modes (flex, absolute, etc.) unlike offsetTop
     * which is ambiguous inside flex containers.
     *
     * The img uses aspect-ratio:4/3 + object-fit:fill so the video content
     * fills the entire img element — no letterbox bars. The content rect
     * is therefore the same as the img bounding rect.
     */
    function _imgContentRect(img, canvas) {
      const cr = canvas.getBoundingClientRect();
      const ir = img.getBoundingClientRect();
      return {
        x: ir.left - cr.left,
        y: ir.top  - cr.top,
        w: ir.width,
        h: ir.height,
      };
    }

    function _redrawCanvas() {
      const img    = document.getElementById('live-img');
      const canvas = document.getElementById('waypoint-canvas');
      // Size canvas to its own display dimensions (full video-box, not img)
      canvas.width  = canvas.clientWidth;
      canvas.height = canvas.clientHeight;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const cr = _imgContentRect(img, canvas);   // actual image content bounds

      // ── Calibration lines: vertical dashed yellow at 20 / 50 / 80 % ──────
      ctx.save();
      ctx.setLineDash([8, 6]);
      ctx.lineWidth    = 1.5;
      ctx.strokeStyle  = 'rgba(255, 220, 0, 0.85)';
      ctx.font         = 'bold 11px monospace';
      ctx.fillStyle    = 'rgba(255, 220, 0, 0.9)';
      ctx.textBaseline = 'bottom';
      [0.20, 0.50, 0.80].forEach(frac => {
        const x = Math.round(cr.x + frac * cr.w);
        ctx.beginPath();
        ctx.moveTo(x, cr.y + cr.h);
        ctx.lineTo(x, cr.y);
        ctx.stroke();
        ctx.textAlign = frac < 0.5 ? 'left' : frac > 0.5 ? 'right' : 'center';
        ctx.fillText(Math.round(frac * 100) + '%',
                     x + (frac < 0.5 ? 3 : frac > 0.5 ? -3 : 0),
                     cr.y + cr.h - 4);
      });
      ctx.restore();

      // Fixed anchor at bottom-centre of image content (rover position)
      const ax = cr.x + cr.w / 2;
      const ay = cr.y + cr.h;

      // Build chain: anchor → wp[0] → wp[1] → ...
      const chain = [{px: ax, py: ay, anchor: true}];
      _waypoints.forEach(wp => chain.push({
        px: cr.x + wp.nx * cr.w,
        py: cr.y + wp.ny * cr.h,
      }));

      // Draw connecting lines
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = 'rgba(0, 200, 120, 0.75)';
      ctx.lineWidth   = 2;
      for (let i = 0; i < chain.length - 1; i++) {
        ctx.beginPath();
        ctx.moveTo(chain[i].px, chain[i].py);
        ctx.lineTo(chain[i+1].px, chain[i+1].py);
        ctx.stroke();
      }
      ctx.setLineDash([]);

      // Draw anchor
      ctx.beginPath();
      ctx.arc(ax, ay - 2, 8, 0, 2*Math.PI);
      ctx.fillStyle   = '#fff';
      ctx.strokeStyle = '#0af';
      ctx.lineWidth   = 2;
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle   = '#0af';
      ctx.font        = 'bold 9px monospace';
      ctx.textAlign   = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('⊕', ax, ay - 2);

      // Draw waypoints
      _waypoints.forEach((wp, i) => {
        const px = cr.x + wp.nx * cr.w;
        const py = cr.y + wp.ny * cr.h;
        ctx.beginPath();
        ctx.arc(px, py, 10, 0, 2*Math.PI);
        ctx.fillStyle = i === 0 ? '#00ff64' : '#00cc50';
        ctx.fill();
        ctx.fillStyle    = '#000';
        ctx.font         = 'bold 11px monospace';
        ctx.textAlign    = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(i + 1, px, py);
      });

      // Sidebar list
      const el = document.getElementById('tp-waypoints');
      el.innerHTML = _waypoints.map((wp, i) =>
        `<div>#${i+1} x=${wp.nx.toFixed(3)} y=${wp.ny.toFixed(3)}</div>`
      ).join('') || '<div style="color:#444">none — left-click on camera to add</div>';
    }

    function _redrawLlmCanvas() {
      const img    = document.getElementById('llm-img');
      const canvas = document.getElementById('llm-canvas');
      canvas.width  = img.offsetWidth;
      canvas.height = img.offsetHeight;
      if (!canvas.width || !canvas.height) return;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const x = Math.round(0.5 * canvas.width);
      ctx.save();
      ctx.setLineDash([8, 6]);
      ctx.lineWidth   = 1.5;
      ctx.strokeStyle = 'rgba(255, 220, 0, 0.85)';
      ctx.beginPath();
      ctx.moveTo(x, canvas.height);
      ctx.lineTo(x, 0);
      ctx.stroke();
      ctx.font         = 'bold 11px monospace';
      ctx.fillStyle    = 'rgba(255, 220, 0, 0.9)';
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText('50%', x, canvas.height - 4);
      ctx.restore();
    }
    setInterval(_redrawLlmCanvas, 500);

    // down canvas removed — wheel cams now have their own grid cells

    function _sendWaypoints() {
      fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({type:'waypoints', waypoints: _waypoints.map(w => [w.nx, w.ny])})
      });
    }

    document.addEventListener('DOMContentLoaded', () => {
      const canvas = document.getElementById('waypoint-canvas');
      canvas.addEventListener('click', e => {
        const r   = canvas.getBoundingClientRect();
        const img = document.getElementById('live-img');
        const cr  = _imgContentRect(img, canvas);
        // Map click to 0-1 coords within the actual image content
        const nx = Math.max(0, Math.min(1, (e.clientX - r.left - cr.x) / cr.w));
        const ny = Math.max(0, Math.min(1, (e.clientY - r.top  - cr.y) / cr.h));
        _waypoints.push({nx, ny});
        _redrawCanvas();
        _sendWaypoints();
      });
      canvas.addEventListener('contextmenu', e => {
        e.preventDefault();
        _waypoints.length = 0;
        _redrawCanvas();
        _sendWaypoints();
      });
      window.addEventListener('resize', _redrawCanvas);
      setInterval(_redrawCanvas, 500);  // keep in sync with img resize
    });

    function teleopExecute() {
      if (_waypoints.length === 0) {
        alert('No waypoints set. Click on the camera feed to add waypoints first.');
        return;
      }
      // Send execute command with current waypoints, then clear canvas
      const meta = {
        instruction:  document.getElementById('tp-instruction').value.trim(),
        crop:         document.getElementById('tp-crop').value.trim(),
        location_id:  document.getElementById('tp-loc').value.trim(),
        growth_stage: document.getElementById('tp-stage').value.trim(),
        robot_id:     document.getElementById('tp-robot').value.trim(),
        date:         new Date().toISOString().slice(0,10),
        collection_mode: 'human_teleop',
        task:         'row_following',
      };
      fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          type: 'episode_cmd', cmd: 'execute', meta,
          waypoints: _waypoints.map(w => [w.nx, w.ny]),
        })
      });
      // Clear canvas so user can set next waypoints immediately
      _waypoints.length = 0;
      _redrawCanvas();
    }

    function teleopCmd(cmd) {
      const meta = {
        instruction:  document.getElementById('tp-instruction').value.trim(),
        crop:         document.getElementById('tp-crop').value.trim(),
        location_id:  document.getElementById('tp-loc').value.trim(),
        growth_stage: document.getElementById('tp-stage').value.trim(),
        robot_id:     document.getElementById('tp-robot').value.trim(),
        date:         new Date().toISOString().slice(0,10),
        collection_mode: 'human_teleop',
        task:         'row_following',
      };
      fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({type:'episode_cmd', cmd, meta})
      });
      if (cmd === 'start') {
        document.getElementById('tp-status').style.color = '#f44336';
        document.getElementById('tp-status').textContent = '● recording — 0 frames';
      } else if (cmd === 'stop' || cmd === 'discard') {
        document.getElementById('tp-status').style.color = '#666';
        document.getElementById('tp-status').textContent = `${cmd} — 0 frames`;
      }
    }
    // ── End teleop ────────────────────────────────────────────────────────────

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

        const rtEl = document.getElementById('response-times');
        const rts = d.response_times ?? [];
        rtEl.innerHTML = rts.length
          ? rts.map((t, i) => `<span style="color:${i===rts.length-1?'#4caf50':'#aaa'}">${t}s</span>`).join(' · ')
          : '—';

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

        // Teleop frame counter
        if (d.recording) {
          const f = d.frames ?? 0;
          const el = document.getElementById('tp-status');
          el.style.color = '#f44336';
          el.textContent = `● recording — ${f} frames`;
        }
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
        self.raw_jpeg        = None          # bytes | None
        self.llm_jpeg        = None          # bytes | None
        self.down_jpeg       = None          # bytes | None  (downward-facing camera)
        self.left_wheel_jpeg = None          # bytes | None  (left wheel camera)
        self.right_wheel_jpeg= None          # bytes | None  (right wheel camera)
        self.status       = {}            # latest JSON from agent
        self.paused       = False
        self.last_push    = 0.0           # epoch seconds
        self.goal         = ""            # latest goal set via /chat
        self.movement     = {"fwd": 0, "turn": 0}  # latest joystick from /chat
        self.chat_history: list = []      # [{"role","text","ts"}] — agent-pushed messages
        # Teleop data collection
        self.teleop_waypoints: list = []  # [[nx,ny],...] normalised image coords
        self.teleop_episode_cmd: str = "" # "start"|"stop"|"discard" — single-consume
        self.teleop_episode_meta: dict = {}

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

        # Blank JPEG for WebRTC tracks before agent connects
        _, buf = cv2.imencode(".jpg", self._blank)
        self._blank_jpeg = buf.tobytes()

        # Asyncio loop for aiortc (runs in a daemon thread)
        self._loop = asyncio.new_event_loop()
        self._pcs: set = set()   # active RTCPeerConnections
        if _WEBRTC_AVAILABLE:
            threading.Thread(target=self._loop.run_forever, daemon=True,
                             name="webrtc-loop").start()

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
        app.add_url_rule("/",                           "index",        self._index)
        app.add_url_rule("/video/realtime",             "v_realtime",     self._video_realtime)
        app.add_url_rule("/video/llm",                  "v_llm",          self._video_llm)
        app.add_url_rule("/video/down",                 "v_down",         self._video_down)
        app.add_url_rule("/video/left_wheel",           "v_left_wheel",   self._video_left_wheel)
        app.add_url_rule("/video/right_wheel",          "v_right_wheel",  self._video_right_wheel)
        app.add_url_rule("/webrtc/offer/<stream>",      "webrtc_offer", self._webrtc_offer, methods=["POST"])
        app.add_url_rule("/status",                     "status",       self._status)
        app.add_url_rule("/pause",                      "pause",        self._pause,        methods=["POST"])
        app.add_url_rule("/agent/frame",                "agent_frame",  self._agent_frame,  methods=["POST"])
        app.add_url_rule("/agent/status",               "agent_status", self._agent_status, methods=["POST"])
        app.add_url_rule("/chat",                       "chat",         self._chat,         methods=["POST"])
        app.add_url_rule("/agent/chat",                 "agent_chat",   self._agent_chat,   methods=["POST"])
        app.add_url_rule("/logs",                       "list_logs",    self._list_logs)
        app.add_url_rule("/logs/<path:filename>",       "dl_log",       self._download_log)

    # ── Agent push endpoints ──────────────────────────────────────────────────

    def _agent_frame(self):
        """POST /agent/frame?stream=realtime|llm|down|left_wheel|right_wheel  body: raw JPEG bytes."""
        stream = request.args.get("stream", "realtime")
        jpeg   = request.get_data()
        with self._state.lock:
            self._state.touch()
            if stream == "llm":
                self._state.llm_jpeg         = jpeg
            elif stream == "down":
                self._state.down_jpeg        = jpeg
            elif stream == "left_wheel":
                self._state.left_wheel_jpeg  = jpeg
            elif stream == "right_wheel":
                self._state.right_wheel_jpeg = jpeg
            else:
                self._state.raw_jpeg         = jpeg
            paused = self._state.paused
        return jsonify({"ok": True, "paused": paused})

    def _agent_status(self):
        """POST /agent/status  body: JSON status dict."""
        data = request.get_json(force=True) or {}
        with self._state.lock:
            self._state.touch()
            self._state.status = data
            paused   = self._state.paused
            goal     = self._state.goal
            mv       = dict(self._state.movement)
            # Teleop fields — episode_cmd is single-consume (cleared after read)
            t_wpts   = list(self._state.teleop_waypoints)
            t_cmd    = self._state.teleop_episode_cmd
            t_meta   = dict(self._state.teleop_episode_meta)
            self._state.teleop_episode_cmd = ""
        return jsonify({
            "ok": True, "paused": paused, "goal": goal, "movement": mv,
            "teleop_waypoints": t_wpts,
            "teleop_episode_cmd": t_cmd,
            "teleop_episode_meta": t_meta,
        })

    def _agent_chat(self):
        """POST /agent/chat  body: {"role": "agent", "text": "..."}"""
        data = request.get_json(force=True) or {}
        msg  = {"role": data.get("role", "agent"),
                "text": data.get("text", ""),
                "ts":   time.time()}
        with self._state.lock:
            self._state.chat_history.append(msg)
        return jsonify({"ok": True})

    # ── WebRTC offer/answer ───────────────────────────────────────────────────

    def _webrtc_offer(self, stream: str):
        """POST /webrtc/offer/<stream> — SDP offer from browser, returns SDP answer."""
        if not _WEBRTC_AVAILABLE:
            return jsonify({"error": "aiortc not installed"}), 503

        data = request.get_json(force=True) or {}
        sdp  = data.get("sdp", "")
        kind = data.get("type", "offer")

        # Run async negotiation on the dedicated event loop and block for result
        future = asyncio.run_coroutine_threadsafe(
            self._do_webrtc_offer(stream, sdp, kind), self._loop
        )
        try:
            answer_sdp, answer_type = future.result(timeout=10.0)
        except Exception as e:
            log.error("WebRTC offer error for %s: %s", stream, e)
            return jsonify({"error": str(e)}), 500

        return jsonify({"sdp": answer_sdp, "type": answer_type})

    async def _do_webrtc_offer(self, stream: str, sdp: str, kind: str):
        pc = RTCPeerConnection()
        self._pcs.add(pc)

        # Build getter for the right JPEG stream
        state = self._state
        if stream == "llm":
            def get_jpeg():
                with state.lock: return state.llm_jpeg
        elif stream == "down":
            def get_jpeg():
                with state.lock: return state.down_jpeg
        else:
            def get_jpeg():
                with state.lock: return state.raw_jpeg

        track = JpegVideoTrack(get_jpeg, self._blank_jpeg)
        pc.addTrack(track)

        @pc.on("connectionstatechange")
        async def on_state():
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await pc.close()
                self._pcs.discard(pc)
                log.debug("WebRTC %s connection %s", stream, pc.connectionState)

        offer = RTCSessionDescription(sdp=sdp, type=kind)
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return pc.localDescription.sdp, pc.localDescription.type

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

    def _video_left_wheel(self):
        return Response(
            self._stream(lambda: self._state.left_wheel_jpeg, "Left cam initialising..."),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def _video_right_wheel(self):
        return Response(
            self._stream(lambda: self._state.right_wheel_jpeg, "Right cam initialising..."),
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

        elif msg_type == "waypoints":
            wpts = data.get("waypoints", [])
            with self._state.lock:
                self._state.teleop_waypoints = wpts
            return jsonify({"ok": True, "count": len(wpts)})

        elif msg_type == "episode_cmd":
            cmd  = data.get("cmd", "")
            meta = data.get("meta", {})
            with self._state.lock:
                self._state.teleop_episode_cmd  = cmd
                self._state.teleop_episode_meta = meta
                if cmd == "execute":
                    # Embed waypoints in the command so the rover receives them
                    # in the same publisher cycle. Server state is cleared since
                    # the canvas already reset on the browser side.
                    wpts = data.get("waypoints", [])
                    self._state.teleop_episode_meta["_waypoints"] = wpts
                    self._state.teleop_waypoints = []
            log.info("Teleop episode cmd: %s", cmd)
            return jsonify({"ok": True})

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
