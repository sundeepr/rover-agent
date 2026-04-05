"""
AgentPublisher — publishes camera frames and status to the standalone web server.

Runs as a background thread inside the agent process. Reads from AgentState
and POSTs to web_server.py over HTTP. The webserver returns the current pause
state in every response so the agent stays in sync without a separate poll.

Usage (internal — called from rover_agent.py):
    pub = AgentPublisher("http://localhost:5001")
    threading.Thread(target=pub.run, args=(state, rover_ctrl), daemon=True).start()
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

    The web server's response to every POST includes {"paused": bool}.
    When the pause state differs from the agent's current state the publisher
    updates state.paused and calls rover_ctrl.stop() on a pause transition.
    """

    def __init__(self, server_url: str, http_timeout: float = 2.0):
        self._url     = server_url.rstrip("/")
        self._timeout = http_timeout
        self._session = None   # created lazily to avoid import at module load

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, state, rover_ctrl=None) -> None:
        """
        Main publish loop — intended to run on a daemon thread.

        Continuously reads raw_frame, llm_frame, and latest_result from
        AgentState and pushes them to the web server. Syncs pause state
        from the server response.
        """
        sess = self._make_session()
        last_llm_id  = None
        last_status_push = 0.0

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

            # ── Status + pause sync (every cycle) ────────────────────────
            status = self._build_status(state)
            remote_paused = self._push_status(sess, status)
            self._sync_pause(state, rover_ctrl, remote_paused)

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

    def _push_status(self, sess, status: dict) -> bool:
        """POST status JSON. Returns the server's paused state (or False on error)."""
        try:
            r = sess.post(
                f"{self._url}/agent/status",
                json=status,
                timeout=self._timeout,
            )
            return bool(r.json().get("paused", False))
        except Exception as e:
            log.debug("push_status failed: %s", e)
            return False

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
