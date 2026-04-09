"""
ControlServer — low-latency WebSocket joystick control channel.

Runs a lightweight asyncio WebSocket server (port 5002 by default) that accepts
joystick commands from the browser or an Android app and calls rover_ctrl.drive_raw()
immediately — no polling hop, no intermediate HTTP round-trip.

Protocol (same for all clients):
    send while held:  {"fwd": <-100..100>, "turn": <-100..100>}
    send on release:  {"fwd": 0, "turn": 0}

Multiple clients can connect; the last writer wins.
A 300 ms watchdog stops the rover if no message arrives (protects against
browser tab closes, network loss, or app crashes).

Usage (from rover_agent.py):
    from control_server import ControlServer
    ctrl_srv = ControlServer(state, rover_ctrl, port=5002)
    ctrl_srv.start()   # starts asyncio loop in a daemon thread
"""

import asyncio
import json
import logging
import threading
import time

log = logging.getLogger("rover.control_server")

# Maximum manual velocity in mm/s.  Higher than the HTTP-path 50 mm/s cap
# because WS commands arrive in real time so the operator can feel speed changes.
_MAX_VEL_MM_S = 150

# Safety watchdog: stop the rover if no message arrives within this window.
_WATCHDOG_S = 0.3


def _joy_to_drive(fwd: int, turn: int, max_vel: int) -> tuple[int, int]:
    """Convert joystick axes to (velocity mm/s, radius mm) for drive_raw().

    fwd  ∈ [-100, 100]  — forward/back
    turn ∈ [-100, 100]  — right is positive, left is negative

    Roomba OI / Atlas drive_raw conventions:
      radius = 0x8000 → straight
      radius = -1     → spin CW  (right in place)
      radius =  1     → spin CCW (left  in place)
      radius negative → arc right;  positive → arc left
    """
    if turn == 0:
        # Straight forward/back
        return fwd * max_vel // 100, 0x8000

    if fwd == 0:
        # Pure rotation — spin in place, speed proportional to turn amount
        vel    = abs(turn) * max_vel // 100
        radius = -1 if turn > 0 else 1   # right=CW=-1, left=CCW=1
        return vel, radius

    # Arc: linear radius mapping keeps gentle turns gentle.
    # turn=100 → radius=-2000 (tight right)
    # turn=10  → radius= -200 (gentle right arc)
    # (old formula used division which gave -20000 for turn=10 — always tight)
    vel    = fwd * max_vel // 100
    radius = int(-2000 * turn / 100)
    return vel, radius


class ControlServer:
    """
    WebSocket control server for real-time joystick input.

    Parameters
    ----------
    state : AgentState
        Shared agent state — operator_control / operator_until are updated
        so the navigation strategy knows to yield to manual control.
    rover_ctrl : RoombaController | AtlasController | None
        Connected rover controller. If None, commands are logged but not sent.
    port : int
        TCP port to listen on (default 5002).
    """

    def __init__(self, state, rover_ctrl, port: int = 5002):
        self._state      = state
        self._rover_ctrl = rover_ctrl
        self._port       = port
        self._watchdog_expires = 0.0   # epoch timestamp; 0 = no active command
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        """Start the WebSocket server in a background daemon thread."""
        t = threading.Thread(target=self._run_loop, daemon=True, name="ctrl-ws")
        t.start()

    # ── asyncio entry point ───────────────────────────────────────────────────

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            log.error("ControlServer fatal error: %s", e)

    async def _serve(self) -> None:
        try:
            import websockets
        except ImportError:
            log.error("'websockets' not installed — WS control disabled. "
                      "Run: pip install 'websockets>=12.0'")
            return

        log.info("WS control server listening on ws://0.0.0.0:%d", self._port)
        async with websockets.serve(self._handle, "0.0.0.0", self._port,
                                    ping_interval=5, ping_timeout=10):
            await self._watchdog_task()

    # ── Per-connection handler ────────────────────────────────────────────────

    async def _handle(self, ws) -> None:
        client = ws.remote_address
        log.info("Control client connected: %s", client)
        try:
            async for raw in ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                self._apply(data)
        except Exception:
            pass
        finally:
            log.info("Control client disconnected: %s", client)

    # ── Command application ───────────────────────────────────────────────────

    def _apply(self, data: dict) -> None:
        fwd  = int(data.get("fwd",  0))
        turn = int(data.get("turn", 0))
        now  = time.time()

        if fwd != 0 or turn != 0:
            vel, radius = _joy_to_drive(fwd, turn, _MAX_VEL_MM_S)

            if self._rover_ctrl is not None:
                try:
                    self._rover_ctrl.drive_raw(vel, radius)
                except Exception as e:
                    log.warning("drive_raw error: %s", e)
            else:
                log.debug("WS drive (dry): vel=%d radius=%d", vel, radius)

            # Refresh watchdog and update shared operator state
            self._watchdog_expires = now + _WATCHDOG_S
            with self._state.result_lock:
                self._state.operator_control = {"fwd": fwd, "turn": turn}
                self._state.operator_until   = now + self._state.query_interval + 0.5
        else:
            # Explicit zero — joystick released, stop immediately
            self._watchdog_expires = 0.0
            self._stop()

    def _stop(self) -> None:
        if self._rover_ctrl is not None:
            try:
                self._rover_ctrl.stop()
            except Exception as e:
                log.warning("stop error: %s", e)
        with self._state.result_lock:
            self._state.operator_control = None
            self._state.operator_until   = 0.0

    # ── Watchdog ──────────────────────────────────────────────────────────────

    async def _watchdog_task(self) -> None:
        """Stop the rover if no command arrives within _WATCHDOG_S seconds."""
        while True:
            await asyncio.sleep(0.05)
            if self._watchdog_expires > 0 and time.time() > self._watchdog_expires:
                log.info("WS watchdog expired — stopping rover")
                self._watchdog_expires = 0.0
                self._stop()
