#!/usr/bin/env python3
"""
Rover teleop WebSocket server — single endpoint for controlling the full rover
platform: two arms and rover drive.

Clients send one JSON message per command.  The "type" field routes it:

  Right arm (XYZ via T:1041):
    {"type": "right_arm", "x": 150, "y": 0, "z": 80, "t": 0, "spd": 200}

  Left arm (XYZ via T:1041):
    {"type": "left_arm",  "x": 150, "y": 0, "z": 80, "t": 0, "spd": 200}

  Rover (Atlas L/R motor %, -100..100):
    {"type": "rover", "L": 60, "R": 60}

  Rover (joystick style, converted to L/R internally):
    {"type": "rover", "fwd": 80, "turn": 20}

  Stop rover:
    {"type": "rover", "L": 0, "R": 0}

Socket types (--socket-type):
  raw   Plain TCP, newline-delimited JSON
  ws    WebSocket (default)
  wss   WebSocket over TLS — requires --cert and --key

Usage:
    python experimental/rover_teleop_server.py
    python experimental/rover_teleop_server.py \\
        --right-arm-port /dev/ttyUSB0 \\
        --left-arm-port  /dev/ttyUSB1 \\
        --rover-port     /dev/ttyACM0 \\
        --socket-type ws --port 9000

    # WSS
    python experimental/rover_teleop_server.py --socket-type wss \\
        --cert cert.pem --key key.pem

Generate a self-signed cert for WSS testing:
    openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

Test with wscat:
    wscat -c ws://localhost:9000
    > {"type":"rover","fwd":60,"turn":0}
    > {"type":"right_arm","x":150,"y":0,"z":80,"t":0}
"""

import argparse
import asyncio
import json
import math
import ssl
import time
import threading
import serial
import websockets


LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 9000
BAUD_RATE   = 115200

# Atlas rover motor constants (mirrors atlas_controller.py)
_MAX_VELOCITY_REF_MM_S = 200
_MOTOR_DEADBAND_PCT    = 8
_DRIVE_SPEED_PCT       = 60
_MAX_ANG_RAD_S         = 0.5


# ── Serial helpers ────────────────────────────────────────────────────────────

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


class SerialPort:
    """Thread-safe wrapper around a serial.Serial instance."""

    def __init__(self, port: str, baud: int = BAUD_RATE):
        self._ser  = serial.Serial(port, baud, timeout=1)
        self._lock = threading.Lock()
        time.sleep(2)   # let ESP32/STM32 boot after open

    def write(self, data: bytes) -> None:
        with self._lock:
            self._ser.write(data)

    def close(self) -> None:
        self._ser.close()


# ── Arm command (T:1041 XYZ) ─────────────────────────────────────────────────

ARM_REQUIRED = {"x", "y", "z"}

def relay_arm(cmd: dict, ser: SerialPort, label: str, addr) -> None:
    missing = ARM_REQUIRED - cmd.keys()
    if missing:
        print(f"[!] {label} from {addr}: ignored — missing keys {missing}")
        return

    payload = {
        "T":   1041,
        "x":   float(cmd["x"]),
        "y":   float(cmd["y"]),
        "z":   float(cmd["z"]),
        "t":   float(cmd.get("t", 0)),
        "spd": int(cmd.get("spd", 200)),
    }
    line = json.dumps(payload) + "\n"
    ser.write(line.encode())
    print(f"[{label}] {addr} → x={payload['x']} y={payload['y']} z={payload['z']}")


# ── Rover command (Atlas $CMD,L=,R=,AUX=#) ───────────────────────────────────

def _apply_deadband(pct: float) -> int:
    if pct == 0:
        return 0
    sign   = 1 if pct > 0 else -1
    scaled = _MOTOR_DEADBAND_PCT + (100 - _MOTOR_DEADBAND_PCT) * abs(pct) / 100.0
    return _clamp(int(sign * scaled), -100, 100)


def _fwd_turn_to_lr(fwd: int, turn: int) -> tuple[int, int]:
    """Convert joystick fwd/turn (-100..100) to Atlas L/R motor percentages."""
    vel_mm_s = fwd * _MAX_VELOCITY_REF_MM_S // 100
    if turn == 0:
        pct = vel_mm_s / _MAX_VELOCITY_REF_MM_S * 100 if vel_mm_s != 0 else 0
        p   = _apply_deadband(pct)
        return p, p

    ang_rad_s = (turn / 100.0) * _MAX_ANG_RAD_S
    if vel_mm_s == 0:
        spin = int(abs(ang_rad_s) / _MAX_ANG_RAD_S * _DRIVE_SPEED_PCT)
        return (spin, -spin) if turn > 0 else (-spin, spin)

    radius_mm = int(math.copysign(
        min(32767, abs(vel_mm_s / ang_rad_s)), -ang_rad_s))
    ratio = _clamp(_MAX_VELOCITY_REF_MM_S / (2 * radius_mm), -0.9, 0.9)
    v_r = vel_mm_s * (1 + ratio)
    v_l = vel_mm_s * (1 - ratio)
    max_v = max(abs(v_r), abs(v_l), _MAX_VELOCITY_REF_MM_S)
    return _apply_deadband(v_l / max_v * 100), _apply_deadband(v_r / max_v * 100)


def relay_rover(cmd: dict, ser: SerialPort, addr) -> None:
    if "L" in cmd or "R" in cmd:
        L = _clamp(int(cmd.get("L", 0)), -100, 100)
        R = _clamp(int(cmd.get("R", 0)), -100, 100)
    elif "fwd" in cmd or "turn" in cmd:
        L, R = _fwd_turn_to_lr(int(cmd.get("fwd", 0)), int(cmd.get("turn", 0)))
    else:
        print(f"[!] rover from {addr}: ignored — need L/R or fwd/turn")
        return

    AUX   = _clamp(int(cmd.get("AUX", 0)), 0, 100)
    frame = f"$CMD,L={L},R={R},AUX={AUX}#\n".encode("ascii")
    ser.write(frame)
    print(f"[rover] {addr} → L={L} R={R} AUX={AUX}")


# ── Message dispatcher ────────────────────────────────────────────────────────

def dispatch(raw: str, addr, ports: dict) -> None:
    try:
        cmd = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[!] invalid JSON from {addr}: {e}")
        return

    kind = cmd.get("type")
    if kind == "right_arm":
        if ports["right_arm"]:
            relay_arm(cmd, ports["right_arm"], "right_arm", addr)
        else:
            print("[!] right_arm command received but --right-arm-port not configured")
    elif kind == "left_arm":
        if ports["left_arm"]:
            relay_arm(cmd, ports["left_arm"], "left_arm", addr)
        else:
            print("[!] left_arm command received but --left-arm-port not configured")
    elif kind == "rover":
        if ports["rover"]:
            relay_rover(cmd, ports["rover"], addr)
        else:
            print("[!] rover command received but --rover-port not configured")
    else:
        print(f"[!] unknown type {kind!r} from {addr} — expected right_arm / left_arm / rover")


# ── Raw TCP mode ──────────────────────────────────────────────────────────────

async def handle_raw_client(reader, writer, ports: dict) -> None:
    addr = writer.get_extra_info("peername")
    print(f"[+] new connection from {addr}")
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            print(f"[>] {addr}: {line.decode().strip()!r}")
            dispatch(line.decode().strip(), addr, ports)
    except (ConnectionResetError, asyncio.IncompleteReadError):
        pass
    finally:
        writer.close()
        print(f"[-] disconnected {addr}")


async def run_raw(host: str, port: int, ports: dict) -> None:
    print(f"Raw TCP server listening on {host}:{port}")
    server = await asyncio.start_server(
        lambda r, w: handle_raw_client(r, w, ports), host, port
    )
    async with server:
        await server.serve_forever()


# ── WebSocket mode ────────────────────────────────────────────────────────────

async def handle_ws_client(ws, ports: dict) -> None:
    addr = ws.remote_address
    print(f"[+] new connection from {addr}")
    try:
        async for message in ws:
            print(f"[>] {addr}: {message!r}")
            dispatch(message, addr, ports)
    except websockets.ConnectionClosed:
        pass
    finally:
        print(f"[-] disconnected {addr}")


async def run_ws(host: str, port: int, ports: dict, ssl_ctx=None) -> None:
    scheme = "wss" if ssl_ctx else "ws"
    print(f"WebSocket server listening on {scheme}://{host}:{port}")
    async with websockets.serve(
        lambda ws: handle_ws_client(ws, ports), host, port, ssl=ssl_ctx
    ):
        await asyncio.Future()   # run forever


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rover teleop WebSocket server")
    parser.add_argument("--socket-type", choices=["raw", "ws", "wss"], default="ws")
    parser.add_argument("--port",           type=int, default=LISTEN_PORT)
    parser.add_argument("--cert",           default=None, help="TLS cert (wss only)")
    parser.add_argument("--key",            default=None, help="TLS key  (wss only)")
    parser.add_argument("--right-arm-port", default=None, metavar="DEV",
                        help="Serial port for right arm (e.g. /dev/ttyUSB0)")
    parser.add_argument("--left-arm-port",  default=None, metavar="DEV",
                        help="Serial port for left arm  (e.g. /dev/ttyUSB1)")
    parser.add_argument("--rover-port",     default=None, metavar="DEV",
                        help="Serial port for rover     (e.g. /dev/ttyACM0)")
    parser.add_argument("--baud",           type=int, default=BAUD_RATE)
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

    if not any([args.right_arm_port, args.left_arm_port, args.rover_port]):
        parser.error("Specify at least one of --right-arm-port, --left-arm-port, --rover-port")

    ports: dict = {"right_arm": None, "left_arm": None, "rover": None}

    try:
        if args.right_arm_port:
            print(f"Opening right arm serial: {args.right_arm_port}")
            ports["right_arm"] = SerialPort(args.right_arm_port, args.baud)

        if args.left_arm_port:
            print(f"Opening left arm serial:  {args.left_arm_port}")
            ports["left_arm"] = SerialPort(args.left_arm_port, args.baud)

        if args.rover_port:
            print(f"Opening rover serial:     {args.rover_port}")
            ports["rover"] = SerialPort(args.rover_port, args.baud)

        try:
            if args.socket_type == "raw":
                asyncio.run(run_raw(LISTEN_HOST, args.port, ports))
            else:
                asyncio.run(run_ws(LISTEN_HOST, args.port, ports, ssl_ctx))
        except KeyboardInterrupt:
            print("\nShutting down.")

    finally:
        for name, p in ports.items():
            if p:
                p.close()
                print(f"Closed {name} serial port.")


if __name__ == "__main__":
    main()
