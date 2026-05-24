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
import itertools
import json
import logging
import os
import subprocess
import sys
import time
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request, send_file

log = logging.getLogger("rover.web_server")

# How long without a push before the agent is considered disconnected.
AGENT_TIMEOUT_S = 10.0

# ── Agent configuration ───────────────────────────────────────────────────────

_CONFIG_FILE = Path("rover_config.json")

_DEFAULT_CONFIG: dict = {
    "device":             0,
    "down_device":        "",
    "strategy":           "teleop",
    "rover":              "atlas",
    "rover_port":         "/dev/ttyACM0",
    "interval":           0.1,
    "dry_run":            False,
    "web_server":         "http://localhost:5001",
    "control_port":       5002,
    "line_vel":           40,
    "line_kp":            2000.0,
    "line_color":         "black",
    "dataset_dir":        "./dataset",
    "teleop_instruction": "",
    "teleop_fps":         10,
    "ollama_model":       "qwen2.5vl",
    "ollama_server":      "http://localhost:11434",
    "goal":               "",
    "cloud_server":       "ws://localhost:8765",
    "omnivla_velocity":   25,
    "crop_type":          "plant",
    "fwd_vel":            80,
    "steering_kp":        0.003,
}


def _load_config() -> dict:
    if _CONFIG_FILE.exists():
        try:
            return {**_DEFAULT_CONFIG, **json.loads(_CONFIG_FILE.read_text())}
        except Exception:
            pass
    return dict(_DEFAULT_CONFIG)


def _save_config(cfg: dict) -> None:
    _CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def _build_agent_cmd(config: dict) -> list[str]:
    """Turn a config dict into rover_agent.py argv."""
    here = Path(__file__).parent
    cmd  = [sys.executable, str(here / "rover_agent.py")]
    cmd += ["--device",       str(config.get("device", 0))]
    cmd += ["--strategy",     str(config.get("strategy", "teleop"))]
    cmd += ["--rover",        str(config.get("rover", "atlas"))]
    cmd += ["--interval",     str(config.get("interval", 0.1))]
    cmd += ["--web-server",   str(config.get("web_server", "http://localhost:5001"))]
    cmd += ["--control-port", str(config.get("control_port", 5002))]

    rover = config.get("rover", "atlas")
    port  = str(config.get("rover_port", "")).strip()
    if port:
        cmd += [f"--{rover}-port", port]

    if config.get("dry_run"):
        cmd += ["--dry-run"]

    dd = str(config.get("down_device", "")).strip()
    if dd:
        cmd += ["--down-device", dd]

    strategy = config.get("strategy", "teleop")
    if strategy == "line_follow":
        cmd += ["--line-vel",   str(config.get("line_vel",   40)),
                "--line-kp",    str(config.get("line_kp",    2000.0)),
                "--line-color", str(config.get("line_color", "black"))]
    elif strategy == "teleop":
        cmd += ["--dataset-dir",        str(config.get("dataset_dir",        "./dataset")),
                "--teleop-instruction", str(config.get("teleop_instruction", "")),
                "--teleop-fps",         str(config.get("teleop_fps",         10))]
    elif strategy == "ollama":
        cmd += ["--ollama-model",  str(config.get("ollama_model",  "qwen2.5vl")),
                "--ollama-server", str(config.get("ollama_server", "http://localhost:11434"))]
    elif strategy in ("cloud_omnivla", "omnivla_full"):
        cmd += ["--cloud-server",     str(config.get("cloud_server",     "ws://localhost:8765")),
                "--omnivla-velocity", str(config.get("omnivla_velocity", 25))]
    elif strategy in ("crop_row", "hough_crop_row"):
        cmd += ["--crop-type",   str(config.get("crop_type",   "plant")),
                "--fwd-vel",     str(config.get("fwd_vel",     80)),
                "--steering-kp", str(config.get("steering_kp", 0.003))]

    goal = str(config.get("goal", "")).strip()
    if goal:
        cmd += ["--goal", goal]
    return cmd


class _AgentRunner:
    """Manages the rover_agent.py subprocess and buffers its stdout/stderr."""

    def __init__(self):
        self._proc:       subprocess.Popen | None = None
        self._start_time: float = 0.0
        self._log_buf:    deque  = deque(maxlen=500)
        self._lock        = threading.Lock()

    def start(self, config: dict) -> None:
        self.stop()
        cmd = _build_agent_cmd(config)
        log.info("Starting agent: %s", " ".join(cmd))
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        with self._lock:
            self._proc       = proc
            self._start_time = time.time()
        threading.Thread(target=self._drain, args=(proc,), daemon=True).start()

    def stop(self) -> None:
        with self._lock:
            proc       = self._proc
            self._proc = None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def status(self) -> dict:
        with self._lock:
            proc    = self._proc
            running = proc is not None and proc.poll() is None
            pid     = proc.pid if running else None
            uptime  = round(time.time() - self._start_time, 1) if running else 0.0
        return {"running": running, "pid": pid, "uptime_s": uptime}

    def logs(self, n: int = 300) -> list[dict]:
        with self._lock:
            return list(itertools.islice(reversed(self._log_buf), n))[::-1]

    def _drain(self, proc: subprocess.Popen) -> None:
        try:
            for line in proc.stdout:
                with self._lock:
                    self._log_buf.append({"ts": time.time(), "text": line.rstrip()})
        except Exception:
            pass
        proc.wait()
        log.info("Agent process exited (rc=%s)", proc.returncode)


# Module-level runner singleton (used by WebServer route handlers)
_runner = _AgentRunner()

# ── HTML template ──────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Rover Agent</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0f0f0f; color: #e0e0e0; font-family: monospace;
           display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
    header { padding: 6px 16px; background: #1a1a1a; border-bottom: 1px solid #333;
             font-size: 1.05em; letter-spacing: 0.05em; color: #7ecfff; flex-shrink: 0;
             display: flex; align-items: center; gap: 12px; }
    #agent-indicator { font-size: 0.75em; margin-left: auto; display: flex;
                       align-items: center; gap: 6px; }
    #agent-dot { font-size: 1.1em; }
    #agent-dot.connected    { color: #4caf50; }
    #agent-dot.disconnected { color: #f44336; }

    /* ── Tab nav ── */
    .tab-nav { display: flex; background: #141414; border-bottom: 1px solid #2a2a2a;
               flex-shrink: 0; padding: 0 8px; }
    .tab-btn { padding: 7px 20px; background: transparent; border: none; color: #555;
               cursor: pointer; font-family: monospace; font-size: 0.85em;
               border-bottom: 2px solid transparent; transition: color 0.15s; letter-spacing: 0.04em; }
    .tab-btn:hover { color: #aaa; }
    .tab-btn.active { color: #7ecfff; border-bottom-color: #7ecfff; }
    .tab-pane { display: none; flex: 1; overflow: hidden; min-height: 0; }
    .tab-pane.active { display: flex; }

    /* ── Configure tab ── */
    #tab-configure { flex-direction: column; overflow-y: auto; }
    .cfg-scroll { flex: 1; overflow-y: auto; display: flex; flex-direction: column;
                  align-items: center; padding: 16px 20px 30px; }
    .cfg-form { width: 100%; max-width: 860px; display: flex; flex-direction: column; gap: 14px; }
    .cfg-agent-bar { display: flex; align-items: center; gap: 8px; padding: 12px 16px;
                     background: #141414; border: 1px solid #252525; border-radius: 6px; }
    #runner-badge { margin-left: auto; font-size: 0.82em; color: #555; }
    .cfg-section { border: 1px solid #252525; border-radius: 6px; overflow: hidden; background: #111; }
    .cfg-section-title { padding: 7px 14px; background: #1a1a1a; font-size: 0.7em;
                         text-transform: uppercase; letter-spacing: 0.1em; color: #555;
                         border-bottom: 1px solid #252525; }
    .cfg-row { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; padding: 10px 14px; }
    .cfg-label { font-size: 0.8em; color: #888; white-space: nowrap; min-width: 110px; }
    .cfg-input { background: #0a0a0a; border: 1px solid #333; color: #e0e0e0;
                 padding: 5px 9px; font-family: monospace; font-size: 0.85em;
                 border-radius: 4px; outline: none; }
    .cfg-input:focus { border-color: #7ecfff; }
    .cfg-input[type=number] { width: 80px; }
    select.cfg-input { cursor: pointer; }
    .cfg-check-row { display: flex; align-items: center; gap: 6px; font-size: 0.8em; color: #888; cursor: pointer; }
    .btn-start   { padding: 7px 18px; background: #1b5e20; border: none; color: #a5d6a7;
                   font-family: monospace; font-size: 0.88em; border-radius: 4px; cursor: pointer; font-weight: bold; }
    .btn-start:hover   { background: #2e7d32; }
    .btn-stop    { padding: 7px 18px; background: #7f0000; border: none; color: #ef9a9a;
                   font-family: monospace; font-size: 0.88em; border-radius: 4px; cursor: pointer; font-weight: bold; }
    .btn-stop:hover    { background: #b71c1c; }
    .btn-restart { padding: 7px 14px; background: #212121; border: 1px solid #444;
                   color: #aaa; font-family: monospace; font-size: 0.82em; border-radius: 4px; cursor: pointer; }
    .btn-restart:hover { background: #333; }

    /* ── Logs tab ── */
    #tab-logs { flex-direction: column; padding: 10px; gap: 8px; }
    .log-toolbar { display: flex; align-items: center; gap: 8px; flex-shrink: 0; font-size: 0.8em; color: #666; }
    .log-toolbar button { padding: 4px 12px; background: #1a1a1a; border: 1px solid #333; color: #aaa;
                          font-family: monospace; font-size: 0.85em; border-radius: 4px; cursor: pointer; }
    .log-toolbar button:hover { background: #2a2a2a; }
    #log-output { flex: 1; overflow-y: auto; overflow-x: auto; background: #060606; border: 1px solid #1e1e1e;
                  border-radius: 4px; padding: 10px 14px; font-family: monospace; font-size: 0.78em;
                  color: #7a7a7a; white-space: pre; min-height: 0; }
    #log-output .log-err  { color: #ef9a9a; }
    #log-output .log-warn { color: #ffcc80; }
    #log-output .log-info { color: #8a8a8a; }

    /* ── Live tab (existing layout) ── */
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

  <div class="tab-nav">
    <button class="tab-btn active" data-tab="configure" onclick="switchTab('configure')">&#x2699; Configure</button>
    <button class="tab-btn" data-tab="live"      onclick="switchTab('live')">&#x1F4F9; Live</button>
    <button class="tab-btn" data-tab="logs"      onclick="switchTab('logs')">&#x1F4DC; Logs</button>
  </div>

  <!-- ── Configure tab ──────────────────────────────────────────────────── -->
  <div id="tab-configure" class="tab-pane active">
    <div class="cfg-scroll">
      <div class="cfg-form">

        <!-- Agent control -->
        <div class="cfg-agent-bar">
          <button class="btn-start"   onclick="agentStart()">&#x25B6; Start</button>
          <button class="btn-stop"    onclick="agentStop()">&#x25A0; Stop</button>
          <button class="btn-restart" onclick="serverRestart()">&#x21BA; Restart Server</button>
          <span id="runner-badge">&#x25CB; stopped</span>
        </div>

        <!-- Cameras -->
        <div class="cfg-section">
          <div class="cfg-section-title">&#x1F4F7; Cameras</div>
          <div class="cfg-row">
            <label class="cfg-label">Main device</label>
            <input class="cfg-input" id="c-device" type="number" value="0" style="width:70px;" title="Camera index passed to cv2.VideoCapture">
            <label class="cfg-label" style="margin-left:16px;">Down device</label>
            <input class="cfg-input" id="c-down-device" type="text" placeholder="blank = disabled" style="width:160px;">
          </div>
        </div>

        <!-- Rover -->
        <div class="cfg-section">
          <div class="cfg-section-title">&#x1F916; Rover</div>
          <div class="cfg-row">
            <label class="cfg-label">Type</label>
            <select class="cfg-input" id="c-rover" style="width:110px;">
              <option value="atlas">atlas</option>
              <option value="roomba">roomba</option>
            </select>
            <label class="cfg-label" style="margin-left:16px;">Serial port</label>
            <input class="cfg-input" id="c-rover-port" type="text" placeholder="/dev/ttyACM0" style="width:180px;">
            <label class="cfg-check-row" style="margin-left:16px;">
              <input type="checkbox" id="c-dry-run"> Dry run
            </label>
          </div>
        </div>

        <!-- Strategy -->
        <div class="cfg-section">
          <div class="cfg-section-title">&#x1F9E0; Strategy</div>
          <div class="cfg-row">
            <label class="cfg-label">Strategy</label>
            <select class="cfg-input" id="c-strategy" style="width:180px;" onchange="onStrategyChange()">
              <option value="teleop">teleop</option>
              <option value="line_follow">line_follow</option>
              <option value="ollama">ollama</option>
              <option value="gemini">gemini</option>
              <option value="cloud_omnivla">cloud_omnivla</option>
              <option value="omnivla_full">omnivla_full</option>
              <option value="crop_row">crop_row</option>
              <option value="hough_crop_row">hough_crop_row</option>
              <option value="omnivla">omnivla</option>
            </select>
            <label class="cfg-label" style="margin-left:16px;">Interval</label>
            <input class="cfg-input" id="c-interval" type="number" step="0.05" value="0.1" style="width:75px;">
            <span style="font-size:0.8em;color:#555;">s between queries</span>
          </div>

          <!-- line_follow params -->
          <div class="cfg-strategy-params" id="sp-line_follow" style="display:none; border-top:1px solid #1e1e1e;">
            <div class="cfg-row">
              <label class="cfg-label">Velocity</label>
              <input class="cfg-input" id="c-line-vel" type="number" value="40" style="width:75px;">
              <span style="font-size:0.8em;color:#555;">mm/s</span>
              <label class="cfg-label" style="margin-left:16px;">Steering Kp</label>
              <input class="cfg-input" id="c-line-kp" type="number" value="2000" style="width:90px;">
              <label class="cfg-label" style="margin-left:16px;">Color</label>
              <select class="cfg-input" id="c-line-color" style="width:110px;">
                <option value="black">black</option>
                <option value="blue">blue</option>
                <option value="orange">orange</option>
                <option value="red">red</option>
                <option value="grey">grey</option>
              </select>
            </div>
          </div>

          <!-- teleop params -->
          <div class="cfg-strategy-params" id="sp-teleop" style="display:none; border-top:1px solid #1e1e1e;">
            <div class="cfg-row">
              <label class="cfg-label">Dataset dir</label>
              <input class="cfg-input" id="c-dataset-dir" type="text" value="./dataset" style="width:180px;">
              <label class="cfg-label" style="margin-left:16px;">FPS</label>
              <input class="cfg-input" id="c-teleop-fps" type="number" value="10" style="width:65px;">
            </div>
            <div class="cfg-row" style="padding-top:0;">
              <label class="cfg-label">Instruction</label>
              <input class="cfg-input" id="c-teleop-instruction" type="text"
                     placeholder="e.g. drive between the crop rows" style="flex:1;min-width:200px;">
            </div>
          </div>

          <!-- ollama params -->
          <div class="cfg-strategy-params" id="sp-ollama" style="display:none; border-top:1px solid #1e1e1e;">
            <div class="cfg-row">
              <label class="cfg-label">Model</label>
              <input class="cfg-input" id="c-ollama-model" type="text" value="qwen2.5vl" style="width:160px;">
              <label class="cfg-label" style="margin-left:16px;">Server</label>
              <input class="cfg-input" id="c-ollama-server" type="text" value="http://localhost:11434" style="width:220px;">
            </div>
          </div>

          <!-- cloud_omnivla / omnivla_full params -->
          <div class="cfg-strategy-params" id="sp-omnivla" style="display:none; border-top:1px solid #1e1e1e;">
            <div class="cfg-row">
              <label class="cfg-label">Cloud server</label>
              <input class="cfg-input" id="c-cloud-server" type="text" value="ws://localhost:8765" style="width:230px;">
              <label class="cfg-label" style="margin-left:16px;">Velocity</label>
              <input class="cfg-input" id="c-omnivla-velocity" type="number" value="25" style="width:75px;">
              <span style="font-size:0.8em;color:#555;">mm/s</span>
            </div>
          </div>

          <!-- crop_row / hough_crop_row params -->
          <div class="cfg-strategy-params" id="sp-crop_row" style="display:none; border-top:1px solid #1e1e1e;">
            <div class="cfg-row">
              <label class="cfg-label">Crop type</label>
              <input class="cfg-input" id="c-crop-type" type="text" value="plant" style="width:110px;">
              <label class="cfg-label" style="margin-left:16px;">Fwd velocity</label>
              <input class="cfg-input" id="c-fwd-vel" type="number" value="80" style="width:75px;">
              <span style="font-size:0.8em;color:#555;">mm/s</span>
              <label class="cfg-label" style="margin-left:16px;">Steering Kp</label>
              <input class="cfg-input" id="c-steering-kp" type="number" step="0.001" value="0.003" style="width:90px;">
            </div>
          </div>
        </div>

        <!-- Goal -->
        <div class="cfg-section">
          <div class="cfg-section-title">&#x1F3AF; Goal (optional)</div>
          <div class="cfg-row">
            <label class="cfg-label">Goal text</label>
            <input class="cfg-input" id="c-goal" type="text"
                   placeholder="e.g. Follow the dirt path" style="flex:1;min-width:200px;">
          </div>
        </div>

      </div><!-- /cfg-form -->
    </div><!-- /cfg-scroll -->
  </div><!-- /tab-configure -->

  <!-- ── Logs tab ───────────────────────────────────────────────────────── -->
  <div id="tab-logs" class="tab-pane">
    <div class="log-toolbar">
      <span>Agent output</span>
      <button onclick="clearLogDisplay()">Clear display</button>
      <label class="cfg-check-row">
        <input type="checkbox" id="log-autoscroll" checked onchange="_autoScroll=this.checked"> Auto-scroll
      </label>
    </div>
    <pre id="log-output">(no output yet)</pre>
  </div>

  <!-- ── Live tab ───────────────────────────────────────────────────────── -->
  <div id="tab-live" class="tab-pane">
  <div class="main">
    <div class="content-area">

      <div class="videos-row">
        <div class="video-box" style="position:relative;">
          <div class="label">&#x1F534; Live camera — click to add waypoints</div>
          <img id="live-img" src="/video/realtime" alt="live feed" style="width:100%;display:block;">
          <canvas id="waypoint-canvas" style="position:absolute;top:0;left:0;width:100%;height:100%;cursor:crosshair;"></canvas>
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
  </div><!-- /main (live) -->
  </div><!-- /tab-live -->

  <script>
    // ── Tab switching ─────────────────────────────────────────────────────────
    let _logsActive = false;
    let _autoScroll = true;

    function switchTab(name) {
      document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById('tab-' + name).classList.add('active');
      document.querySelector('.tab-btn[data-tab="' + name + '"]').classList.add('active');
      _logsActive = (name === 'logs');
      if (_logsActive) pollLogs();
    }

    // ── Configure: strategy param visibility ─────────────────────────────────
    const _strategyParamMap = {
      line_follow:    ['sp-line_follow'],
      teleop:         ['sp-teleop'],
      ollama:         ['sp-ollama'],
      cloud_omnivla:  ['sp-omnivla'],
      omnivla_full:   ['sp-omnivla'],
      crop_row:       ['sp-crop_row'],
      hough_crop_row: ['sp-crop_row'],
    };

    function onStrategyChange() {
      const strategy = document.getElementById('c-strategy').value;
      document.querySelectorAll('.cfg-strategy-params').forEach(el => el.style.display = 'none');
      (_strategyParamMap[strategy] || []).forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = '';
      });
    }

    // ── Configure: collect / populate ────────────────────────────────────────
    function collectConfig() {
      return {
        device:             parseInt(document.getElementById('c-device').value) || 0,
        down_device:        document.getElementById('c-down-device').value.trim(),
        strategy:           document.getElementById('c-strategy').value,
        rover:              document.getElementById('c-rover').value,
        rover_port:         document.getElementById('c-rover-port').value.trim(),
        interval:           parseFloat(document.getElementById('c-interval').value) || 0.1,
        dry_run:            document.getElementById('c-dry-run').checked,
        web_server:         'http://' + location.hostname + ':5001',
        control_port:       5002,
        line_vel:           parseInt(document.getElementById('c-line-vel').value) || 40,
        line_kp:            parseFloat(document.getElementById('c-line-kp').value) || 2000,
        line_color:         document.getElementById('c-line-color').value,
        dataset_dir:        document.getElementById('c-dataset-dir').value || './dataset',
        teleop_instruction: document.getElementById('c-teleop-instruction').value,
        teleop_fps:         parseInt(document.getElementById('c-teleop-fps').value) || 10,
        ollama_model:       document.getElementById('c-ollama-model').value,
        ollama_server:      document.getElementById('c-ollama-server').value,
        goal:               document.getElementById('c-goal').value.trim(),
        cloud_server:       document.getElementById('c-cloud-server').value,
        omnivla_velocity:   parseInt(document.getElementById('c-omnivla-velocity').value) || 25,
        crop_type:          document.getElementById('c-crop-type').value,
        fwd_vel:            parseInt(document.getElementById('c-fwd-vel').value) || 80,
        steering_kp:        parseFloat(document.getElementById('c-steering-kp').value) || 0.003,
      };
    }

    function populateConfig(cfg) {
      const set = (id, v) => { const el = document.getElementById(id); if (el) el.value = v ?? el.value; };
      const chk = (id, v) => { const el = document.getElementById(id); if (el) el.checked = !!v; };
      set('c-device',             cfg.device ?? 0);
      set('c-down-device',        cfg.down_device ?? '');
      set('c-strategy',           cfg.strategy ?? 'teleop');
      set('c-rover',              cfg.rover ?? 'atlas');
      set('c-rover-port',         cfg.rover_port ?? '');
      set('c-interval',           cfg.interval ?? 0.1);
      chk('c-dry-run',            cfg.dry_run);
      set('c-line-vel',           cfg.line_vel ?? 40);
      set('c-line-kp',            cfg.line_kp ?? 2000);
      set('c-line-color',         cfg.line_color ?? 'black');
      set('c-dataset-dir',        cfg.dataset_dir ?? './dataset');
      set('c-teleop-instruction', cfg.teleop_instruction ?? '');
      set('c-teleop-fps',         cfg.teleop_fps ?? 10);
      set('c-ollama-model',       cfg.ollama_model ?? 'qwen2.5vl');
      set('c-ollama-server',      cfg.ollama_server ?? 'http://localhost:11434');
      set('c-goal',               cfg.goal ?? '');
      set('c-cloud-server',       cfg.cloud_server ?? 'ws://localhost:8765');
      set('c-omnivla-velocity',   cfg.omnivla_velocity ?? 25);
      set('c-crop-type',          cfg.crop_type ?? 'plant');
      set('c-fwd-vel',            cfg.fwd_vel ?? 80);
      set('c-steering-kp',        cfg.steering_kp ?? 0.003);
      onStrategyChange();
    }

    async function loadConfig() {
      try {
        const r = await fetch('/api/config');
        populateConfig(await r.json());
      } catch(_) {}
    }

    // ── Agent start / stop / restart ──────────────────────────────────────────
    async function agentStart() {
      const cfg = collectConfig();
      await fetch('/api/config', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(cfg),
      });
      const r = await fetch('/api/agent/start', {method: 'POST'});
      if (r.ok) switchTab('logs');
    }

    async function agentStop() {
      await fetch('/api/agent/stop', {method: 'POST'});
    }

    async function serverRestart() {
      if (!confirm('Restart the web server? The page will reconnect automatically.')) return;
      document.getElementById('runner-badge').textContent = '⟳ restarting…';
      fetch('/api/server/restart', {method: 'POST'}).catch(() => {});
      setTimeout(async function poll() {
        try { await fetch('/api/agent/status'); location.reload(); }
        catch(_) { setTimeout(poll, 600); }
      }, 1200);
    }

    // ── Runner status badge (2 s poll) ────────────────────────────────────────
    async function pollRunnerStatus() {
      try {
        const r = await fetch('/api/agent/status');
        const s = await r.json();
        const badge = document.getElementById('runner-badge');
        if (s.running) {
          badge.style.color = '#4caf50';
          badge.textContent = '● running — PID ' + s.pid + '  (' + s.uptime_s + 's)';
        } else {
          badge.style.color = '#666';
          badge.textContent = '○ stopped';
        }
      } catch(_) {}
    }
    setInterval(pollRunnerStatus, 2000);
    pollRunnerStatus();

    // ── Log tab ───────────────────────────────────────────────────────────────
    let _logDisplayCleared = false;

    async function pollLogs() {
      if (!_logsActive) return;
      try {
        const r = await fetch('/api/agent/logs?n=300');
        const d = await r.json();
        const el = document.getElementById('log-output');
        if (_logDisplayCleared) return;
        if (!d.lines || d.lines.length === 0) {
          el.textContent = '(no output yet — start the agent first)';
          return;
        }
        el.textContent = d.lines.map(l => {
          const ts = new Date(l.ts * 1000).toLocaleTimeString();
          return '[' + ts + '] ' + l.text;
        }).join('\n');
        if (_autoScroll) el.scrollTop = el.scrollHeight;
      } catch(_) {}
    }
    setInterval(pollLogs, 1000);

    function clearLogDisplay() {
      document.getElementById('log-output').textContent = '(display cleared — agent still running)';
      _logDisplayCleared = true;
      setTimeout(() => { _logDisplayCleared = false; }, 2000);
    }

    // ── Init ──────────────────────────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', () => {
      loadConfig();
    });

    // ── Teleop waypoint canvas ────────────────────────────────────────────────
    const _waypoints = [];  // [{nx, ny}, ...]

    function _redrawCanvas() {
      const img    = document.getElementById('live-img');
      const canvas = document.getElementById('waypoint-canvas');
      canvas.width  = img.offsetWidth;
      canvas.height = img.offsetHeight;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Fixed anchor at bottom-centre (rover position)
      const ax = canvas.width  / 2;
      const ay = canvas.height;

      // Build chain: anchor → wp[0] → wp[1] → ...
      const chain = [{px: ax, py: ay, anchor: true}];
      _waypoints.forEach(wp => chain.push({
        px: wp.nx * canvas.width,
        py: wp.ny * canvas.height,
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
        const px = wp.nx * canvas.width;
        const py = wp.ny * canvas.height;
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

    function _sendWaypoints() {
      fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({type:'waypoints', waypoints: _waypoints.map(w => [w.nx, w.ny])})
      });
    }

    document.addEventListener('DOMContentLoaded', () => {
      const canvas = document.getElementById('waypoint-canvas');
      canvas.addEventListener('click', e => {
        const r = canvas.getBoundingClientRect();
        const nx = (e.clientX - r.left) / r.width;
        const ny = (e.clientY - r.top)  / r.height;
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
        self.raw_jpeg     = None          # bytes | None
        self.llm_jpeg     = None          # bytes | None
        self.down_jpeg    = None          # bytes | None  (downward-facing camera)
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
        # ── Agent runner API ──
        app.add_url_rule("/api/config",             "api_cfg_get",  self._api_config_get)
        app.add_url_rule("/api/config",             "api_cfg_post", self._api_config_post,      methods=["POST"])
        app.add_url_rule("/api/agent/start",        "api_ag_start", self._api_agent_start,      methods=["POST"])
        app.add_url_rule("/api/agent/stop",         "api_ag_stop",  self._api_agent_stop,       methods=["POST"])
        app.add_url_rule("/api/agent/status",       "api_ag_stat",  self._api_agent_status_run)
        app.add_url_rule("/api/agent/logs",         "api_ag_logs",  self._api_agent_logs)
        app.add_url_rule("/api/server/restart",     "api_srv_rst",  self._api_server_restart,   methods=["POST"])

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

    # ── Agent runner API ──────────────────────────────────────────────────────

    def _api_config_get(self):
        return jsonify(_load_config())

    def _api_config_post(self):
        cfg = request.get_json(force=True) or {}
        merged = {**_DEFAULT_CONFIG, **cfg}
        _save_config(merged)
        return jsonify({"ok": True})

    def _api_agent_start(self):
        cfg = _load_config()
        _runner.start(cfg)
        return jsonify({"ok": True})

    def _api_agent_stop(self):
        _runner.stop()
        return jsonify({"ok": True})

    def _api_agent_status_run(self):
        return jsonify(_runner.status())

    def _api_agent_logs(self):
        n = int(request.args.get("n", 300))
        return jsonify({"lines": _runner.logs(n)})

    def _api_server_restart(self):
        _runner.stop()
        time.sleep(0.3)
        os.execv(sys.executable, [sys.executable] + sys.argv)

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
    try:
        server.run(host=args.host, port=args.port)
    finally:
        _runner.stop()


if __name__ == "__main__":
    main()
