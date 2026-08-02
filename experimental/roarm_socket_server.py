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
  - IK
  - serial command emission
  - no-motion logging
"""

import argparse
import asyncio
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

LINK1_MM = 320.0
LINK2_MM = 212.0
BASE_MIN_RAD = -3.14
BASE_MAX_RAD = 3.14
SHOULDER_MIN_RAD = -1.57
SHOULDER_MAX_RAD = 1.57
ELBOW_MIN_RAD = 0.0
ELBOW_MAX_RAD = 3.14

INPUT_MOVE_EPS_M = 1e-4
TARGET_EPS_MM = 0.5
JOINT_EPS_RAD = 1e-3
def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


@dataclass
class EeTarget:
    x: float
    y: float
    z: float
    t: float = HOME_T_RAD


@dataclass
class JointTarget:
    base: float
    shoulder: float
    elbow: float
    hand: float


class TeleopState:
    def __init__(self) -> None:
        self.target = EeTarget(HOME_X_MM, HOME_Y_MM, HOME_Z_MM, HOME_T_RAD)
        self.control_anchor_target = EeTarget(HOME_X_MM, HOME_Y_MM, HOME_Z_MM, HOME_T_RAD)
        self.last_joint_target: JointTarget | None = None
        self.control_active = False
        self.messages_received = 0
        self.commands_sent = 0
        self.ik_failures = 0
        self.clamp_events = 0


def same_target(a: EeTarget, b: EeTarget) -> bool:
    return (
        abs(a.x - b.x) < TARGET_EPS_MM and
        abs(a.y - b.y) < TARGET_EPS_MM and
        abs(a.z - b.z) < TARGET_EPS_MM and
        abs(a.t - b.t) < JOINT_EPS_RAD
    )


def same_joints(a: JointTarget | None, b: JointTarget | None) -> bool:
    if a is None or b is None:
        return False
    return (
        abs(a.base - b.base) < JOINT_EPS_RAD and
        abs(a.shoulder - b.shoulder) < JOINT_EPS_RAD and
        abs(a.elbow - b.elbow) < JOINT_EPS_RAD and
        abs(a.hand - b.hand) < JOINT_EPS_RAD
    )


def solve_ik(target: EeTarget) -> JointTarget | None:
    x, y, z = target.x, target.y, target.z
    base = math.atan2(y, x)
    radial = math.sqrt(x * x + y * y)
    reach = math.sqrt(radial * radial + z * z)
    min_reach = abs(LINK1_MM - LINK2_MM) + 1.0
    max_reach = (LINK1_MM + LINK2_MM) - 1.0
    safe_reach = clamp(reach, min_reach, max_reach)

    safe_radial = radial
    safe_z = z
    if reach > 1e-3 and abs(safe_reach - reach) > 1e-3:
        scale = safe_reach / reach
        safe_radial *= scale
        safe_z *= scale

    cos_elbow = clamp(
        (safe_radial * safe_radial + safe_z * safe_z - LINK1_MM * LINK1_MM - LINK2_MM * LINK2_MM) /
        (2.0 * LINK1_MM * LINK2_MM),
        -1.0,
        1.0,
    )
    elbow = math.acos(cos_elbow)
    shoulder = math.atan2(safe_z, safe_radial) - math.atan2(
        LINK2_MM * math.sin(elbow),
        LINK1_MM + LINK2_MM * math.cos(elbow),
    )

    return JointTarget(
        base=clamp(base, BASE_MIN_RAD, BASE_MAX_RAD),
        shoulder=clamp(shoulder, SHOULDER_MIN_RAD, SHOULDER_MAX_RAD),
        elbow=clamp(elbow, ELBOW_MIN_RAD, ELBOW_MAX_RAD),
        hand=target.t,
    )


def joint_command(joints: JointTarget) -> str:
    return json.dumps({
        "T": 102,
        "base": round(joints.base, 6),
        "shoulder": round(joints.shoulder, 6),
        "elbow": round(joints.elbow, 6),
        "hand": round(joints.hand, 6),
        "spd": 0,
        "acc": 10,
    })


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


def handle_teleop_message(payload: dict, state: TeleopState, ser: serial.Serial) -> None:
    state.messages_received += 1
    delta = payload.get("delta", {})
    moved = controller_moved(delta)

    if payload.get("recenter"):
        state.target = EeTarget(HOME_X_MM, HOME_Y_MM, HOME_Z_MM, HOME_T_RAD)
        state.control_anchor_target = EeTarget(HOME_X_MM, HOME_Y_MM, HOME_Z_MM, HOME_T_RAD)
        joints = solve_ik(state.target)
        if joints is None:
            state.ik_failures += 1
            print(f"[!] IK failed for recenter seq={payload.get('seq')}")
            return
        ser.write((joint_command(joints) + "\n").encode())
        state.commands_sent += 1
        state.last_joint_target = joints
        print(f"[arm] recenter sent seq={payload.get('seq')} target={state.target.__dict__}")
        return

    requested_control_active = bool(payload.get("control_active", False))
    if requested_control_active and not state.control_active:
        state.control_anchor_target = EeTarget(state.target.x, state.target.y, state.target.z, state.target.t)
        print(f"[teleop] control engaged seq={payload.get('seq')} anchor_target={state.control_anchor_target.__dict__}")
    elif not requested_control_active and state.control_active:
        print(f"[teleop] control released seq={payload.get('seq')} target={state.target.__dict__}")
        state.control_anchor_target = EeTarget(state.target.x, state.target.y, state.target.z, state.target.t)

    state.control_active = requested_control_active
    if not state.control_active or not moved:
        return

    previous_target = EeTarget(state.target.x, state.target.y, state.target.z, state.target.t)
    previous_joints = state.last_joint_target
    new_target, clamped = apply_mode(state.control_anchor_target, delta, str(payload.get("mode", "xyz")))
    if clamped:
        state.clamp_events += 1
        log_clamp(payload, previous_target, new_target, str(payload.get("mode", "xyz")))

    if same_target(previous_target, new_target):
        return

    joints = solve_ik(new_target)
    if joints is None:
        state.ik_failures += 1
        print(
            "[ik_failed]",
            json.dumps({
                "seq": payload.get("seq"),
                "mode": payload.get("mode"),
                "delta": payload.get("delta"),
                "previous_target": previous_target.__dict__,
                "new_target": new_target.__dict__,
            }),
        )
        return

    if same_joints(previous_joints, joints):
        state.target = new_target
        return

    command = joint_command(joints)
    ser.write((command + "\n").encode())
    state.commands_sent += 1
    state.target = new_target
    state.last_joint_target = joints
    print(f"[arm] seq={payload.get('seq')} target={new_target.__dict__} joints={joints.__dict__} command={command}")


def relay_command(raw: str, addr, ser: serial.Serial, state: TeleopState) -> None:
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

    try:
        if args.socket_type == "raw":
            asyncio.run(run_raw(LISTEN_HOST, args.port, ser, state))
        else:
            asyncio.run(run_ws(LISTEN_HOST, args.port, ser, state, ssl_ctx))
    except KeyboardInterrupt:
        print("\nShutting down.")
        print(
            f"Summary: received={state.messages_received} sent={state.commands_sent} "
            f"ik_failures={state.ik_failures} clamps={state.clamp_events}"
        )
    finally:
        ser.close()


if __name__ == "__main__":
    main()
