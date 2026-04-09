"""
AgentPublisher — publishes camera frames and status to the standalone web server.

Runs as a background thread inside the agent process. Reads from AgentState
and POSTs to web_server.py over HTTP. The webserver returns the current pause
state and goal in every response so the agent stays in sync without a separate poll.

Usage (internal — called from rover_agent.py):
    pub = AgentPublisher("http://localhost:5001")
    threading.Thread(target=pub.run, args=(state, rover_ctrl, strategy), daemon=True).start()
"""

import logging
import threading
import time

import cv2
import numpy as np

log = logging.getLogger("rover.publisher")

# How often the publish loop runs (seconds). Raw frames are sent every cycle;
# status is sent every cycle; LLM frame only when it changes.
_INTERVAL = 0.05   # 20 fps


class AgentPublisher:
    """
    Reads from AgentState and pushes frames + status to the web server.

    The web server's response to every POST includes {"paused": bool, "goal": str,
    "movement": dict}. The publisher syncs pause state, detects new goals, and
    forwards joystick movement to the rover controller.
    """

    def __init__(self, server_url: str, http_timeout: float = 2.0):
        self._url                  = server_url.rstrip("/")
        self._timeout              = http_timeout
        self._last_goal            = ""    # tracks last goal forwarded to strategy
        self._operator_cmd_expires = 0.0   # safety expiry for continuous joystick drive

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, state, rover_ctrl=None, strategy=None) -> None:
        """
        Main publish loop — intended to run on a daemon thread.

        Continuously reads raw_frame, llm_frame, and latest_result from
        AgentState and pushes them to the web server. Syncs pause state,
        new goals, and operator movement commands from the server response.
        """
        sess = self._make_session()
        last_llm_id   = None
        posted_ready  = False

        while True:
            t0 = time.time()

            # ── Raw frame (every cycle) ───────────────────────────────────
            with state.raw_lock:
                raw = state.raw_frame
            if raw is not None:
                self._push_frame(sess, raw, "realtime")

            # ── LLM frame (only when it changes) ─────────────────────────
            with state.llm_lock:
                llm = state.llm_frame
            llm_id = id(llm)
            if llm is not None and llm_id != last_llm_id:
                self._push_frame(sess, llm, "llm")
                last_llm_id = llm_id

            # ── Status + sync (every cycle) ──────────────────────────────
            status = self._build_status(state)
            resp   = self._push_status(sess, status)
            self._sync_pause(state, rover_ctrl, resp.get("paused", False))
            self._sync_goal(state, strategy, resp.get("goal", ""))
            self._sync_movement(state, rover_ctrl, resp)

            # ── Post "Ready" once when strategy models are loaded ─────────
            if (not posted_ready and strategy is not None
                    and hasattr(strategy, "_loaded") and strategy._loaded.is_set()):
                self._post_agent_chat(sess, "Ready — send a goal to start navigation")
                posted_ready = True

            # ── Pace loop ────────────────────────────────────────────────
            elapsed = time.time() - t0
            remaining = _INTERVAL - elapsed
            if remaining > 0:
                time.sleep(remaining)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_session(self):
        import requests
        s = requests.Session()
        # Keep-alive so each POST reuses the TCP connection
        s.headers.update({"Connection": "keep-alive"})
        return s

    def _push_frame(self, sess, frame_bgr: np.ndarray, stream: str) -> None:
        """Encode frame as JPEG and POST to /agent/frame."""
        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        try:
            sess.post(
                f"{self._url}/agent/frame",
                params={"stream": stream},
                data=buf.tobytes(),
                headers={"Content-Type": "image/jpeg"},
                timeout=self._timeout,
            )
        except Exception as e:
            log.debug("push_frame(%s) failed: %s", stream, e)

    def _push_status(self, sess, status: dict) -> dict:
        """POST status JSON. Returns the full server response dict."""
        try:
            r = sess.post(
                f"{self._url}/agent/status",
                json=status,
                timeout=self._timeout,
            )
            return r.json()
        except Exception as e:
            log.debug("push_status failed: %s", e)
            return {}

    def _post_agent_chat(self, sess, text: str) -> None:
        """Push a message from the agent into the web server's chat history."""
        try:
            sess.post(
                f"{self._url}/agent/chat",
                json={"role": "agent", "text": text},
                timeout=self._timeout,
            )
        except Exception as e:
            log.debug("post_agent_chat failed: %s", e)

    @staticmethod
    def _build_status(state) -> dict:
        """Snapshot AgentState into a JSON-serialisable dict."""
        with state.result_lock:
            result             = dict(state.latest_result)
            result["step"]     = state.step
            result["paused"]   = state.paused.is_set()
            result["history"]  = [
                f"Step {t['step']}: ({t['x']},{t['y']}) {t['description']}"
                for t in state.trajectory
            ]
            result["llm_query_start"] = state.llm_query_start
            result["llm_response_s"]  = state.llm_response_s
        return result

    @staticmethod
    def _sync_pause(state, rover_ctrl, remote_paused: bool) -> None:
        """Reconcile agent pause state with the value from the web server."""
        was_paused = state.paused.is_set()
        if remote_paused and not was_paused:
            state.paused.set()
            log.info("Paused by web UI")
            if rover_ctrl:
                try:
                    rover_ctrl.stop()
                except Exception as e:
                    log.error("Stop error on remote pause: %s", e)
        elif not remote_paused and was_paused:
            state.paused.clear()
            log.info("Resumed by web UI")

    def _sync_goal(self, state, strategy, server_goal: str) -> None:
        """Forward a new goal from the web UI to the strategy and AgentState."""
        if not server_goal or server_goal == self._last_goal:
            return
        self._last_goal = server_goal
        log.info("Goal received from web UI: '%s'", server_goal)
        if strategy is not None:
            strategy.set_goal(server_goal)
        with state.result_lock:
            state.goal = server_goal
        state.goal_ready.set()

    def _sync_movement(self, state, rover_ctrl, resp: dict) -> None:
        """Apply operator joystick movement from the web server response.

        Movement is driven at full publisher rate (20 Hz) for smooth continuous
        motion. A 350 ms safety expiry stops the rover if the browser goes silent
        (tab closed, network loss, joystick released).
        """
        mv  = resp.get("movement")
        now = time.time()

        if mv is not None:
            fwd  = mv.get("fwd",  0)
            turn = mv.get("turn", 0)
            with state.result_lock:
                if fwd != 0 or turn != 0:
                    state.operator_control     = mv
                    state.operator_until       = now + state.query_interval + 0.5
                    self._operator_cmd_expires = now + 0.35   # 350 ms safety window
                else:
                    # Explicit zero — joystick released; clear immediately
                    state.operator_control     = None
                    state.operator_until       = 0.0
                    self._operator_cmd_expires = 0.0

        # Drive every publisher cycle while the safety window is open
        if self._operator_cmd_expires <= now:
            return
        with state.result_lock:
            oc = state.operator_control
        if not oc or state.paused.is_set() or rover_ctrl is None:
            return
        fwd    = oc.get("fwd",  0)
        turn   = oc.get("turn", 0)
        vel    = fwd * 50 // 100   # cap at 50 mm/s for manual control
        radius = 0x8000 if turn == 0 else int(-2000 / (turn / 100))
        try:
            rover_ctrl.drive_raw(vel, radius)
        except Exception as e:
            log.warning("Manual drive error: %s", e)
