#!/usr/bin/env python3
"""
RoArm teleop socket server.

Supported client payload:
  {
    "type": "teleop_delta",
    "seq": 12,
    "time_ms": 123456789,
    "control_active": true,
    "recenter": false,
    "mode": "xyz",
    "delta": {"x": 0.01, "y": -0.02, "z": 0.00}
  }

The server owns:
  - persistent EE target state
  - workspace clamping
  - serial command emission
  - clamp logging
"""

import argparse
import asyncio
import collections
import json
import math
import ssl
import time
from dataclasses import dataclass

import serial
import websockets


SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 9876
VERBOSE_STREAM_LOGS = False

HOME_X_MM = 250.0
HOME_Y_MM = 0.0
HOME_Z_MM = 0.0
HOME_T_RAD = 3.14
MM_PER_METER = 1250.0

MIN_X_MM = 0.0
MAX_X_MM = 500.0
MIN_Y_MM = -500.0
MAX_Y_MM = 500.0
MIN_Z_MM = -500.0
MAX_Z_MM = 500.0

INPUT_MOVE_EPS_M = 1e-4
TARGET_EPS_MM = 0.5
RECLUTCH_HOLDOFF_S = 0.30
MOVING_AVERAGE_WINDOW = 20

INIT_COMMAND = {"T": 100}
FEEDBACK_COMMAND = {"T": 105}
FEEDBACK_TIMEOUT_S = 0.5
def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


@dataclass
class EeTarget:
    x: float
    y: float
    z: float
    t: float = HOME_T_RAD


class TeleopState:
    def __init__(self) -> None:
        self.target = EeTarget(HOME_X_MM, HOME_Y_MM, HOME_Z_MM, HOME_T_RAD)
        self.control_anchor_target = EeTarget(HOME_X_MM, HOME_Y_MM, HOME_Z_MM, HOME_T_RAD)
        self.control_active = False
        self.control_resume_time = 0.0
        self.mode = "xyz"
        self.last_delta = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.filtered_delta = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.last_seq = 0
        self.messages_received = 0
        self.commands_sent = 0
        self.clamp_events = 0
        self.history_x = collections.deque(maxlen=40)
        self.history_y = collections.deque(maxlen=40)
        self.history_z = collections.deque(maxlen=40)
        self.delta_x_window = collections.deque(maxlen=MOVING_AVERAGE_WINDOW)
        self.delta_y_window = collections.deque(maxlen=MOVING_AVERAGE_WINDOW)
        self.delta_z_window = collections.deque(maxlen=MOVING_AVERAGE_WINDOW)


def same_target(a: EeTarget, b: EeTarget) -> bool:
    return (
        abs(a.x - b.x) < TARGET_EPS_MM and
        abs(a.y - b.y) < TARGET_EPS_MM and
        abs(a.z - b.z) < TARGET_EPS_MM and
        abs(a.t - b.t) < 1e-3
    )


def ee_command(target: EeTarget) -> str:
    return json.dumps({
        "T": 1041,
        "x": round(target.x, 3),
        "y": round(target.y, 3),
        "z": round(target.z, 3),
        "t": round(target.t, 6),
    })


def send_json(ser: serial.Serial, payload: dict) -> None:
    ser.write((json.dumps(payload) + "\n").encode())


def request_feedback(ser: serial.Serial) -> dict | None:
    send_json(ser, FEEDBACK_COMMAND)
    deadline = time.time() + FEEDBACK_TIMEOUT_S
    buf = b""
    while time.time() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict) and data.get("T") == 1051:
                    return data
    return None


def extract_target_from_feedback(feedback: dict | None) -> EeTarget | None:
    if not isinstance(feedback, dict):
        return None
    try:
        x = float(feedback["x"])
        y = float(feedback["y"])
        z = float(feedback["z"])
    except (KeyError, TypeError, ValueError):
        return None
    try:
        t = float(feedback.get("t", HOME_T_RAD))
    except (TypeError, ValueError):
        t = HOME_T_RAD
    return EeTarget(
        x=clamp(x, MIN_X_MM, MAX_X_MM),
        y=clamp(y, MIN_Y_MM, MAX_Y_MM),
        z=clamp(z, MIN_Z_MM, MAX_Z_MM),
        t=t,
    )


def log_clamp(payload: dict, previous_target: EeTarget, new_target: EeTarget, mode: str) -> None:
    print(
        "[clamp]",
        json.dumps({
            "seq": payload.get("seq"),
            "mode": mode,
            "delta": payload.get("delta"),
            "previous_target": previous_target.__dict__,
            "new_target": new_target.__dict__,
        }),
    )


def apply_mode(anchor: EeTarget, delta: dict, mode: str) -> tuple[EeTarget, bool]:
    dx = float(delta.get("x", 0.0))
    dy = float(delta.get("y", 0.0))
    dz = float(delta.get("z", 0.0))

    requested_target = EeTarget(anchor.x, anchor.y, anchor.z, anchor.t)
    clamped = False

    if mode in ("xyz", "x-only"):
        requested_x = anchor.x - dz * MM_PER_METER
        target_x = clamp(requested_x, MIN_X_MM, MAX_X_MM)
        clamped |= abs(target_x - requested_x) > 1e-6
        requested_target.x = target_x
    if mode in ("xyz", "y-only"):
        requested_y = anchor.y - dx * MM_PER_METER
        target_y = clamp(requested_y, MIN_Y_MM, MAX_Y_MM)
        clamped |= abs(target_y - requested_y) > 1e-6
        requested_target.y = target_y
    if mode in ("xyz", "z-only"):
        requested_z = anchor.z + dy * MM_PER_METER
        target_z = clamp(requested_z, MIN_Z_MM, MAX_Z_MM)
        clamped |= abs(target_z - requested_z) > 1e-6
        requested_target.z = target_z

    return requested_target, clamped


def controller_moved(delta: dict) -> bool:
    return any(abs(float(delta.get(axis, 0.0))) >= INPUT_MOVE_EPS_M for axis in ("x", "y", "z"))


def axis_bar(value: float, minimum: float, maximum: float, width: int = 24) -> str:
    if maximum <= minimum:
        return "-" * width
    ratio = (value - minimum) / (maximum - minimum)
    ratio = clamp(ratio, 0.0, 1.0)
    filled = int(round(ratio * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "-" * (width - filled)


def sparkline(values: collections.deque[float], minimum: float, maximum: float) -> str:
    if not values:
        return ""
    glyphs = "▁▂▃▄▅▆▇█"
    span = maximum - minimum
    if span <= 0.0:
        return glyphs[0] * len(values)
    chars = []
    for value in values:
        ratio = clamp((value - minimum) / span, 0.0, 1.0)
        index = min(len(glyphs) - 1, int(round(ratio * (len(glyphs) - 1))))
        chars.append(glyphs[index])
    return "".join(chars)


def render_dashboard(state: TeleopState) -> None:
    lines = [
        "\x1b[2J\x1b[H",
        "RoArm Teleop Dashboard",
        f"seq={state.last_seq}  mode={state.mode}  control_active={state.control_active}",
        f"messages={state.messages_received}  sent={state.commands_sent}  clamps={state.clamp_events}",
        "",
        "Current target",
        f"  X {state.target.x:8.2f} |{axis_bar(state.target.x, MIN_X_MM, MAX_X_MM)}|",
        f"  Y {state.target.y:8.2f} |{axis_bar(state.target.y, MIN_Y_MM, MAX_Y_MM)}|",
        f"  Z {state.target.z:8.2f} |{axis_bar(state.target.z, MIN_Z_MM, MAX_Z_MM)}|",
        "",
        "Clutch anchor",
        f"  X {state.control_anchor_target.x:8.2f} |{axis_bar(state.control_anchor_target.x, MIN_X_MM, MAX_X_MM)}|",
        f"  Y {state.control_anchor_target.y:8.2f} |{axis_bar(state.control_anchor_target.y, MIN_Y_MM, MAX_Y_MM)}|",
        f"  Z {state.control_anchor_target.z:8.2f} |{axis_bar(state.control_anchor_target.z, MIN_Z_MM, MAX_Z_MM)}|",
        "",
        f"Raw delta:      x={state.last_delta['x']:+.5f}  y={state.last_delta['y']:+.5f}  z={state.last_delta['z']:+.5f}",
        f"Filtered delta: x={state.filtered_delta['x']:+.5f}  y={state.filtered_delta['y']:+.5f}  z={state.filtered_delta['z']:+.5f}",
        "",
        "History",
        f"  X {sparkline(state.history_x, MIN_X_MM, MAX_X_MM)}",
        f"  Y {sparkline(state.history_y, MIN_Y_MM, MAX_Y_MM)}",
        f"  Z {sparkline(state.history_z, MIN_Z_MM, MAX_Z_MM)}",
        "",
    ]
    print("\n".join(lines), end="", flush=True)


def append_history(state: TeleopState) -> None:
    state.history_x.append(state.target.x)
    state.history_y.append(state.target.y)
    state.history_z.append(state.target.z)


def reset_delta_filter(state: TeleopState) -> None:
    state.delta_x_window.clear()
    state.delta_y_window.clear()
    state.delta_z_window.clear()
    state.filtered_delta = {"x": 0.0, "y": 0.0, "z": 0.0}


def apply_moving_average_filter(state: TeleopState, delta: dict) -> dict:
    dx = float(delta.get("x", 0.0))
    dy = float(delta.get("y", 0.0))
    dz = float(delta.get("z", 0.0))
    state.delta_x_window.append(dx)
    state.delta_y_window.append(dy)
    state.delta_z_window.append(dz)
    filtered = {
        "x": sum(state.delta_x_window) / len(state.delta_x_window),
        "y": sum(state.delta_y_window) / len(state.delta_y_window),
        "z": sum(state.delta_z_window) / len(state.delta_z_window),
    }
    state.filtered_delta = filtered
    return filtered


def handle_teleop_message(payload: dict, state: TeleopState, ser: serial.Serial) -> None:
    state.messages_received += 1
    delta = payload.get("delta", {})
    moved = controller_moved(delta)
    state.last_seq = int(payload.get("seq", state.last_seq))
    state.mode = str(payload.get("mode", state.mode))
    state.last_delta = {
        "x": float(delta.get("x", 0.0)),
        "y": float(delta.get("y", 0.0)),
        "z": float(delta.get("z", 0.0)),
    }

    if payload.get("recenter"):
        state.target = EeTarget(HOME_X_MM, HOME_Y_MM, HOME_Z_MM, HOME_T_RAD)
        state.control_anchor_target = EeTarget(HOME_X_MM, HOME_Y_MM, HOME_Z_MM, HOME_T_RAD)
        reset_delta_filter(state)
        ser.write((ee_command(state.target) + "\n").encode())
        state.commands_sent += 1
        if VERBOSE_STREAM_LOGS:
            print(f"[arm] recenter sent seq={payload.get('seq')} target={state.target.__dict__}")
        append_history(state)
        render_dashboard(state)
        return

    requested_control_active = bool(payload.get("control_active", False))
    if requested_control_active and not state.control_active:
        state.control_anchor_target = EeTarget(state.target.x, state.target.y, state.target.z, state.target.t)
        state.control_resume_time = time.monotonic() + RECLUTCH_HOLDOFF_S
        reset_delta_filter(state)
        if VERBOSE_STREAM_LOGS:
            print(f"[teleop] control engaged seq={payload.get('seq')} anchor_target={state.control_anchor_target.__dict__}")
    elif not requested_control_active and state.control_active:
        if VERBOSE_STREAM_LOGS:
            print(f"[teleop] control released seq={payload.get('seq')} target={state.target.__dict__}")
        state.control_anchor_target = EeTarget(state.target.x, state.target.y, state.target.z, state.target.t)
        state.control_resume_time = 0.0
        reset_delta_filter(state)

    state.control_active = requested_control_active
    if not state.control_active or not moved:
        append_history(state)
        render_dashboard(state)
        return

    filtered_delta = apply_moving_average_filter(state, delta)

    if time.monotonic() < state.control_resume_time:
        append_history(state)
        render_dashboard(state)
        return

    previous_target = EeTarget(state.target.x, state.target.y, state.target.z, state.target.t)
    new_target, clamped = apply_mode(state.control_anchor_target, filtered_delta, str(payload.get("mode", "xyz")))
    if clamped:
        state.clamp_events += 1
        log_clamp(payload, previous_target, new_target, str(payload.get("mode", "xyz")))

    if same_target(previous_target, new_target):
        return

    command = ee_command(new_target)
    ser.write((command + "\n").encode())
    state.commands_sent += 1
    state.target = new_target
    if VERBOSE_STREAM_LOGS:
        print(f"[arm] seq={payload.get('seq')} target={new_target.__dict__} command={command}")
    append_history(state)
    render_dashboard(state)


def relay_command(raw: str, addr, ser: serial.Serial, state: TeleopState) -> None:
    if VERBOSE_STREAM_LOGS:
        print(f"[>] data received from {addr}: {raw!r}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[!] invalid JSON from {addr}: {e}")
        return

    if payload.get("type") != "teleop_delta":
        print(f"[!] unsupported payload type from {addr}: {payload.get('type')!r}")
        return

    handle_teleop_message(payload, state, ser)


async def handle_raw_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, ser: serial.Serial, state: TeleopState) -> None:
    addr = writer.get_extra_info("peername")
    print(f"[+] new connection from {addr}")
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            relay_command(line.decode().strip(), addr, ser, state)
    except (ConnectionResetError, asyncio.IncompleteReadError):
        pass
    finally:
        writer.close()
        print(f"[-] disconnected {addr}")


async def run_raw(host: str, port: int, ser: serial.Serial, state: TeleopState) -> None:
    print(f"Raw TCP server listening on {host}:{port}")
    server = await asyncio.start_server(
        lambda r, w: handle_raw_client(r, w, ser, state), host, port
    )
    async with server:
        await server.serve_forever()


async def handle_ws_client(ws, ser: serial.Serial, state: TeleopState) -> None:
    addr = ws.remote_address
    print(f"[+] new connection from {addr}")
    try:
        async for message in ws:
            relay_command(message, addr, ser, state)
    except websockets.ConnectionClosed:
        pass
    finally:
        print(f"[-] disconnected {addr}")


async def run_ws(host: str, port: int, ser: serial.Serial, state: TeleopState, ssl_ctx=None) -> None:
    scheme = "wss" if ssl_ctx else "ws"
    print(f"WebSocket server listening on {scheme}://{host}:{port}")
    async with websockets.serve(lambda ws: handle_ws_client(ws, ser, state), host, port, ssl=ssl_ctx):
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description="RoArm teleop socket server")
    parser.add_argument("--socket-type", choices=["raw", "ws", "wss"], default="ws")
    parser.add_argument("--port", type=int, default=LISTEN_PORT)
    parser.add_argument("--serial", type=str, default=SERIAL_PORT)
    parser.add_argument("--baud", type=int, default=BAUD_RATE)
    parser.add_argument("--cert", type=str, default=None)
    parser.add_argument("--key", type=str, default=None)
    args = parser.parse_args()

    if args.socket_type == "wss":
        if not (args.cert and args.key):
            parser.error("--cert and --key are required for --socket-type wss")
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(args.cert, args.key)
    else:
        if args.cert or args.key:
            parser.error("--cert / --key are only used with --socket-type wss")
        ssl_ctx = None

    print(f"Opening serial {args.serial} @ {args.baud}")
    ser = serial.Serial(args.serial, args.baud, timeout=0.05)
    time.sleep(2)
    state = TeleopState()
    send_json(ser, INIT_COMMAND)
    feedback = request_feedback(ser)
    feedback_target = extract_target_from_feedback(feedback)
    if feedback_target is not None:
        state.target = feedback_target
        state.control_anchor_target = EeTarget(
            feedback_target.x,
            feedback_target.y,
            feedback_target.z,
            feedback_target.t,
        )
        print(f"[init] arm feedback target={state.target.__dict__}")
    else:
        print(f"[init] arm feedback unavailable; using home target={state.target.__dict__}")
    append_history(state)
    render_dashboard(state)

    try:
        if args.socket_type == "raw":
            asyncio.run(run_raw(LISTEN_HOST, args.port, ser, state))
        else:
            asyncio.run(run_ws(LISTEN_HOST, args.port, ser, state, ssl_ctx))
    except KeyboardInterrupt:
        print("\nShutting down.")
        print(
            f"Summary: received={state.messages_received} sent={state.commands_sent} "
            f"clamps={state.clamp_events}"
        )
    finally:
        ser.close()


if __name__ == "__main__":
    main()
