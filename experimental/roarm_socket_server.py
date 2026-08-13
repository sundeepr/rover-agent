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
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import serial
import websockets

ROVER_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(ROVER_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(ROVER_AGENT_ROOT))

from atlas_controller import AtlasController
from control_server import _joy_to_drive


SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 9876
VERBOSE_STREAM_LOGS = False

GRIPPER_OPEN_RAD = math.pi / 2.0
GRIPPER_CLOSED_RAD = math.pi
GRIPPER_SPD = 0
GRIPPER_ACC = 10

HOME_X_MM = 250.0
HOME_Y_MM = 0.0
HOME_Z_MM = 0.0
HOME_T_RAD = GRIPPER_OPEN_RAD
MM_PER_METER = 1500.0
MOTION_SCALE = 0.4

# RoArm-M2 geometry and angular offsets from Waveshare's RoArm-M2_config.h.
ARM_L1_LENGTH_MM = 126.06
ARM_L2_LENGTH_MM_A = 236.82
ARM_L2_LENGTH_MM_B = 30.00
ARM_L3_LENGTH_MM_A = 280.15
ARM_L3_LENGTH_MM_B = 1.73
ARM_L2_LENGTH_MM = math.hypot(ARM_L2_LENGTH_MM_A, ARM_L2_LENGTH_MM_B)
ARM_L3_LENGTH_MM = math.hypot(ARM_L3_LENGTH_MM_A, ARM_L3_LENGTH_MM_B)
T2_RAD = math.atan2(ARM_L2_LENGTH_MM_B, ARM_L2_LENGTH_MM_A)
T3_RAD = math.atan2(ARM_L3_LENGTH_MM_B, ARM_L3_LENGTH_MM_A)
BASE_MIN_RAD = -3.14
BASE_MAX_RAD = 3.14
SHOULDER_MIN_RAD = -1.57
SHOULDER_MAX_RAD = 1.57
ELBOW_MIN_RAD = -0.78539815
ELBOW_MAX_RAD = 3.14
JOINT_SPD = 0
JOINT_ACC = 10

MIN_X_MM = 0.0
MAX_X_MM = 490.0
MIN_Y_MM = -490.0
MAX_Y_MM = 490.0
MIN_Z_MM = -490.0
MAX_Z_MM = 490.0
MIN_RADIAL_MM = 80.0

INPUT_MOVE_EPS_M = 1e-4
TARGET_EPS_MM = 0.5
RECLUTCH_HOLDOFF_S = 0.30
EMA_ALPHA = 1.0

INIT_COMMAND = {"T": 100}
FEEDBACK_COMMAND = {"T": 105}
FEEDBACK_TIMEOUT_S = 0.5
ROVER_MAX_VEL_MM_S = 80
ROVER_WATCHDOG_S = 0.3
ROVER_GRIP_AUX_PCT = 50
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


class KinematicsError(ValueError):
    pass


class RoverDriveState:
    def __init__(self) -> None:
        self.last_command_time = 0.0
        self.active = False
        self.commands_received = 0
        self.fwd = 0
        self.turn = 0
        self.aux_pct = 0


class TeleopState:
    def __init__(self, arm_name: str) -> None:
        self.arm_name = arm_name
        self.target = EeTarget(HOME_X_MM, HOME_Y_MM, HOME_Z_MM, HOME_T_RAD)
        self.control_anchor_target = EeTarget(HOME_X_MM, HOME_Y_MM, HOME_Z_MM, HOME_T_RAD)
        self.control_active = False
        self.gripper_closed = None
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


def stop_rover(
        rover_ctrl: AtlasController | None,
        rover_state: RoverDriveState,
        source: str) -> None:
    try:
        if source != "release" and rover_state.aux_pct != 0 and rover_ctrl is not None:
            rover_ctrl.set_aux(0)
            rover_state.aux_pct = 0
        if (rover_state.active or source != "release") and rover_ctrl is not None:
            rover_ctrl.stop()
    except Exception as error:
        print(f"[!] rover stop failed source={source}: {error}")
    finally:
        rover_state.active = False
        rover_state.fwd = 0
        rover_state.turn = 0
    print(f"[rover] stop source={source}")
def same_target(a: EeTarget, b: EeTarget) -> bool:
    return (
        abs(a.x - b.x) < TARGET_EPS_MM and
        abs(a.y - b.y) < TARGET_EPS_MM and
        abs(a.z - b.z) < TARGET_EPS_MM and
        abs(a.t - b.t) < 1e-3
    )


def checked_acos(value: float, label: str) -> float:
    if value < -1.000001 or value > 1.000001:
        raise KinematicsError(f"unreachable target ({label}={value:.6f})")
    return math.acos(clamp(value, -1.0, 1.0))


def solve_planar_ik(radial_mm: float, z_mm: float) -> tuple[float, float]:
    """Port of Waveshare simpleLinkageIkRad for EEMode 0."""
    if radial_mm <= 1e-6:
        raise KinematicsError("target lies on the base axis")

    if abs(z_mm) < 1e-6:
        psi = checked_acos(
            (ARM_L2_LENGTH_MM**2 + radial_mm**2 - ARM_L3_LENGTH_MM**2) /
            (2.0 * ARM_L2_LENGTH_MM * radial_mm),
            "psi",
        ) + T2_RAD
        shoulder = math.pi / 2.0 - psi
        omega = checked_acos(
            (radial_mm**2 + ARM_L3_LENGTH_MM**2 - ARM_L2_LENGTH_MM**2) /
            (2.0 * radial_mm * ARM_L3_LENGTH_MM),
            "omega",
        )
    else:
        reach_squared = radial_mm**2 + z_mm**2
        reach = math.sqrt(reach_squared)
        elevation = math.atan2(z_mm, radial_mm)
        psi = checked_acos(
            (ARM_L2_LENGTH_MM**2 + reach_squared - ARM_L3_LENGTH_MM**2) /
            (2.0 * ARM_L2_LENGTH_MM * reach),
            "psi",
        ) + T2_RAD
        shoulder = math.pi / 2.0 - elevation - psi
        omega = checked_acos(
            (ARM_L3_LENGTH_MM**2 + reach_squared - ARM_L2_LENGTH_MM**2) /
            (2.0 * reach * ARM_L3_LENGTH_MM),
            "omega",
        )

    elbow = psi + omega - T3_RAD
    return shoulder, elbow


def solve_ik(target: EeTarget) -> JointTarget:
    """Cartesian -> polar -> Waveshare planar IK decomposition."""
    radial = math.hypot(target.x, target.y)
    base = math.atan2(target.y, target.x)
    shoulder, elbow = solve_planar_ik(radial, target.z)

    joint_values = (
        ("base", base, BASE_MIN_RAD, BASE_MAX_RAD),
        ("shoulder", shoulder, SHOULDER_MIN_RAD, SHOULDER_MAX_RAD),
        ("elbow", elbow, ELBOW_MIN_RAD, ELBOW_MAX_RAD),
    )
    for name, value, minimum, maximum in joint_values:
        if value < minimum or value > maximum:
            raise KinematicsError(
                f"{name}={value:.6f} outside [{minimum:.6f}, {maximum:.6f}]"
            )

    return JointTarget(base=base, shoulder=shoulder, elbow=elbow, hand=target.t)


def compute_fk(joints: JointTarget) -> EeTarget:
    """Port of Waveshare RoArmM2_computePosbyJointRad for EEMode 0."""
    link2_angle = math.pi / 2.0 - (joints.shoulder + T2_RAD)
    link3_angle = math.pi / 2.0 - (joints.elbow + joints.shoulder)
    radial = (
        ARM_L2_LENGTH_MM * math.cos(link2_angle) +
        ARM_L3_LENGTH_MM * math.cos(link3_angle)
    )
    z = (
        ARM_L2_LENGTH_MM * math.sin(link2_angle) +
        ARM_L3_LENGTH_MM * math.sin(link3_angle)
    )
    return EeTarget(
        x=radial * math.cos(joints.base),
        y=radial * math.sin(joints.base),
        z=z,
        t=joints.hand,
    )


def joint_command(target: EeTarget) -> str:
    joints = solve_ik(target)
    return json.dumps({
        "T": 102,
        "base": round(joints.base, 6),
        "shoulder": round(joints.shoulder, 6),
        "elbow": round(joints.elbow, 6),
        "hand": round(joints.hand, 6),
        "spd": JOINT_SPD,
        "acc": JOINT_ACC,
    })


def gripper_command(closed: bool) -> str:
    return json.dumps({
        "T": 106,
        "cmd": GRIPPER_CLOSED_RAD if closed else GRIPPER_OPEN_RAD,
        "spd": GRIPPER_SPD,
        "acc": GRIPPER_ACC,
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


def log_ik_rejection(payload: dict, target: EeTarget, error: KinematicsError) -> None:
    print(
        "[ik_rejected]",
        json.dumps({
            "seq": payload.get("seq"),
            "target": target.__dict__,
            "reason": str(error),
        }),
    )


def apply_mode(anchor: EeTarget, delta: dict, mode: str) -> tuple[EeTarget, bool]:
    dx = float(delta.get("x", 0.0))
    dy = float(delta.get("y", 0.0))
    dz = float(delta.get("z", 0.0))

    requested_target = EeTarget(anchor.x, anchor.y, anchor.z, anchor.t)
    clamped = False

    if mode in ("xyz", "xyz_rate", "x-only"):
        requested_x = anchor.x + dx * MM_PER_METER * MOTION_SCALE
        target_x = clamp(requested_x, MIN_X_MM, MAX_X_MM)
        clamped |= abs(target_x - requested_x) > 1e-6
        requested_target.x = target_x
    if mode in ("xyz", "xyz_rate", "y-only"):
        requested_y = anchor.y + dy * MM_PER_METER * MOTION_SCALE
        target_y = clamp(requested_y, MIN_Y_MM, MAX_Y_MM)
        clamped |= abs(target_y - requested_y) > 1e-6
        requested_target.y = target_y
    if mode in ("xyz", "xyz_rate", "z-only"):
        requested_z = anchor.z + dz * MM_PER_METER * MOTION_SCALE
        target_z = clamp(requested_z, MIN_Z_MM, MAX_Z_MM)
        clamped |= abs(target_z - requested_z) > 1e-6
        requested_target.z = target_z

    radial = math.hypot(requested_target.x, requested_target.y)
    if radial < MIN_RADIAL_MM:
        if radial < 1e-6:
            requested_target.x = MIN_RADIAL_MM
            requested_target.y = 0.0
        else:
            scale = MIN_RADIAL_MM / radial
            requested_target.x *= scale
            requested_target.y *= scale
        clamped = True

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
        f"RoArm Teleop Dashboard [{state.arm_name}]",
        f"seq={state.last_seq}  mode={state.mode}  control_active={state.control_active}",
        f"messages={state.messages_received}  sent={state.commands_sent}  clamps={state.clamp_events}",
        f"gripper={'unknown' if state.gripper_closed is None else ('closed' if state.gripper_closed else 'open')}",
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
    state.filtered_delta = {"x": 0.0, "y": 0.0, "z": 0.0}


def apply_ema_filter(state: TeleopState, delta: dict) -> dict:
    dx = float(delta.get("x", 0.0))
    dy = float(delta.get("y", 0.0))
    dz = float(delta.get("z", 0.0))
    filtered = {
        "x": state.filtered_delta["x"] + EMA_ALPHA * (dx - state.filtered_delta["x"]),
        "y": state.filtered_delta["y"] + EMA_ALPHA * (dy - state.filtered_delta["y"]),
        "z": state.filtered_delta["z"] + EMA_ALPHA * (dz - state.filtered_delta["z"]),
    }
    state.filtered_delta = filtered
    return filtered


def handle_gripper_message(
        payload: dict,
        state: TeleopState,
        ser: serial.Serial,
        rover_ctrl: AtlasController | None,
        rover_state: RoverDriveState) -> None:
    closed = payload.get("closed")
    if not isinstance(closed, bool):
        print(f"[!] invalid gripper state: {closed!r}")
        return
    if state.gripper_closed is closed:
        return

    command = gripper_command(closed)
    ser.write((command + "\n").encode())
    angle = GRIPPER_CLOSED_RAD if closed else GRIPPER_OPEN_RAD
    state.gripper_closed = closed
    state.target.t = angle
    state.control_anchor_target.t = angle
    state.commands_sent += 1
    print(f"[gripper] state={'closed' if closed else 'open'} command={command}")

    if state.arm_name == "right" and rover_ctrl is not None:
        aux_pct = ROVER_GRIP_AUX_PCT if closed else 0
        if aux_pct != rover_state.aux_pct:
            try:
                rover_ctrl.set_aux(aux_pct)
                rover_state.aux_pct = aux_pct
                velocity, radius = _joy_to_drive(
                    rover_state.fwd,
                    rover_state.turn,
                    ROVER_MAX_VEL_MM_S,
                )
                rover_ctrl.drive_raw(velocity, radius)
                print(f"[rover] aux={aux_pct}% source=right_grip")
            except Exception as error:
                print(f"[!] rover AUX command failed: {error}")
    render_dashboard(state)


def handle_rover_drive_message(
        payload: dict,
        rover_ctrl: AtlasController | None,
        rover_state: RoverDriveState) -> None:
    try:
        fwd = int(clamp(int(payload.get("fwd", 0)), -100, 100))
        turn = int(clamp(int(payload.get("turn", 0)), -100, 100))
    except (TypeError, ValueError):
        print(f"[!] invalid rover drive payload: {payload!r}")
        return

    rover_state.commands_received += 1
    rover_state.last_command_time = time.monotonic()
    rover_state.fwd = fwd
    rover_state.turn = turn

    try:
        if fwd == 0 and turn == 0:
            stop_rover(rover_ctrl, rover_state, "release")
            return

        velocity, radius = _joy_to_drive(fwd, turn, ROVER_MAX_VEL_MM_S)
        if rover_ctrl is not None:
            rover_ctrl.drive_raw(velocity, radius)
        rover_state.active = True
        if VERBOSE_STREAM_LOGS:
            print(
                f"[rover] fwd={fwd} turn={turn} velocity={velocity} radius={radius}"
            )
    except Exception as error:
        rover_state.active = False
        print(f"[!] rover command failed: {error}")


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
        current_hand_angle = state.target.t
        state.target = EeTarget(HOME_X_MM, HOME_Y_MM, HOME_Z_MM, current_hand_angle)
        state.control_anchor_target = EeTarget(HOME_X_MM, HOME_Y_MM, HOME_Z_MM, current_hand_angle)
        reset_delta_filter(state)
        try:
            command = joint_command(state.target)
        except KinematicsError as error:
            log_ik_rejection(payload, state.target, error)
            return
        ser.write((command + "\n").encode())
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

    filtered_delta = apply_ema_filter(state, delta)

    if time.monotonic() < state.control_resume_time:
        append_history(state)
        render_dashboard(state)
        return

    previous_target = EeTarget(state.target.x, state.target.y, state.target.z, state.target.t)
    mode = str(payload.get("mode", "xyz"))
    new_target, clamped = apply_mode(state.control_anchor_target, filtered_delta, mode)
    if clamped:
        state.clamp_events += 1
        log_clamp(payload, previous_target, new_target, str(payload.get("mode", "xyz")))

    if same_target(previous_target, new_target):
        return

    try:
        command = joint_command(new_target)
    except KinematicsError as error:
        state.clamp_events += 1
        log_ik_rejection(payload, new_target, error)
        return
    ser.write((command + "\n").encode())
    state.commands_sent += 1
    state.target = new_target
    if VERBOSE_STREAM_LOGS:
        print(f"[arm] seq={payload.get('seq')} target={new_target.__dict__} command={command}")
    append_history(state)
    render_dashboard(state)


def relay_command(
        raw: str,
        addr,
        serial_ports: dict[str, serial.Serial],
        states: dict[str, TeleopState],
        rover_ctrl: AtlasController | None,
        rover_state: RoverDriveState) -> None:
    if VERBOSE_STREAM_LOGS:
        print(f"[>] data received from {addr}: {raw!r}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[!] invalid JSON from {addr}: {e}")
        return

    payload_type = payload.get("type")
    if payload_type == "rover_drive":
        handle_rover_drive_message(payload, rover_ctrl, rover_state)
        return

    arm_name = str(payload.get("arm", "right")).lower()
    ser = serial_ports.get(arm_name)
    state = states.get(arm_name)
    if ser is None or state is None:
        print(f"[!] arm {arm_name!r} is not configured")
        return

    if payload_type == "teleop_delta":
        handle_teleop_message(payload, state, ser)
    elif payload_type == "gripper_state":
        handle_gripper_message(payload, state, ser, rover_ctrl, rover_state)
    else:
        print(f"[!] unsupported payload type from {addr}: {payload_type!r}")


async def handle_raw_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        serial_ports: dict[str, serial.Serial],
        states: dict[str, TeleopState],
        rover_ctrl: AtlasController | None,
        rover_state: RoverDriveState) -> None:
    addr = writer.get_extra_info("peername")
    print(f"[+] new connection from {addr}")
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            relay_command(
                line.decode().strip(),
                addr,
                serial_ports,
                states,
                rover_ctrl,
                rover_state,
            )
    except (ConnectionResetError, asyncio.IncompleteReadError):
        pass
    finally:
        if rover_state.active or rover_state.aux_pct != 0:
            stop_rover(rover_ctrl, rover_state, "disconnect")
        writer.close()
        print(f"[-] disconnected {addr}")


async def run_raw(
        host: str,
        port: int,
        serial_ports: dict[str, serial.Serial],
        states: dict[str, TeleopState],
        rover_ctrl: AtlasController | None,
        rover_state: RoverDriveState) -> None:
    print(f"Raw TCP server listening on {host}:{port}")
    server = await asyncio.start_server(
        lambda r, w: handle_raw_client(
            r, w, serial_ports, states, rover_ctrl, rover_state
        ), host, port
    )
    watchdog_task = asyncio.create_task(rover_watchdog(rover_ctrl, rover_state))
    try:
        async with server:
            await server.serve_forever()
    finally:
        watchdog_task.cancel()


async def handle_ws_client(
        ws,
        serial_ports: dict[str, serial.Serial],
        states: dict[str, TeleopState],
        rover_ctrl: AtlasController | None,
        rover_state: RoverDriveState) -> None:
    addr = ws.remote_address
    print(f"[+] new connection from {addr}")
    try:
        async for message in ws:
            relay_command(
                message, addr, serial_ports, states, rover_ctrl, rover_state
            )
    except websockets.ConnectionClosed:
        pass
    finally:
        if rover_state.active or rover_state.aux_pct != 0:
            stop_rover(rover_ctrl, rover_state, "disconnect")
        print(f"[-] disconnected {addr}")


async def run_ws(
        host: str,
        port: int,
        serial_ports: dict[str, serial.Serial],
        states: dict[str, TeleopState],
        rover_ctrl: AtlasController | None,
        rover_state: RoverDriveState,
        ssl_ctx=None) -> None:
    scheme = "wss" if ssl_ctx else "ws"
    print(f"WebSocket server listening on {scheme}://{host}:{port}")
    watchdog_task = asyncio.create_task(rover_watchdog(rover_ctrl, rover_state))
    try:
        async with websockets.serve(
                lambda ws: handle_ws_client(
                    ws, serial_ports, states, rover_ctrl, rover_state
                ),
                host,
                port,
                ssl=ssl_ctx):
            await asyncio.Future()
    finally:
        watchdog_task.cancel()


async def rover_watchdog(
        rover_ctrl: AtlasController | None,
        rover_state: RoverDriveState) -> None:
    while True:
        await asyncio.sleep(0.05)
        if (
            rover_state.active and
            time.monotonic() - rover_state.last_command_time > ROVER_WATCHDOG_S
        ):
            stop_rover(rover_ctrl, rover_state, "watchdog")


def initialize_arm(name: str, ser: serial.Serial) -> TeleopState:
    state = TeleopState(name)
    send_json(ser, INIT_COMMAND)
    feedback = request_feedback(ser)
    feedback_target = extract_target_from_feedback(feedback)
    if feedback_target is not None:
        print(f"[init] {name} arm startup feedback={feedback_target.__dict__}")
    else:
        print(f"[init] {name} arm startup feedback unavailable")

    home_command = joint_command(state.target)
    ser.write((home_command + "\n").encode())
    print(
        f"[init] {name} arm moving to configured home={state.target.__dict__} "
        f"command={home_command}"
    )
    append_history(state)
    render_dashboard(state)
    return state


def main():
    parser = argparse.ArgumentParser(description="RoArm teleop socket server")
    parser.add_argument("--socket-type", choices=["raw", "ws", "wss"], default="ws")
    parser.add_argument("--port", type=int, default=LISTEN_PORT)
    parser.add_argument(
        "--serial",
        dest="right_serial",
        type=str,
        default=SERIAL_PORT,
        help="right-arm serial device",
    )
    parser.add_argument(
        "--left-serial",
        type=str,
        default=None,
        help="left-arm serial device; omit for right-arm-only operation",
    )
    parser.add_argument("--baud", type=int, default=BAUD_RATE)
    parser.add_argument(
        "--rover-serial",
        type=str,
        default=None,
        help="Atlas rover serial device; omit to disable rover driving",
    )
    parser.add_argument("--rover-baud", type=int, default=115200)
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

    serial_paths = {"right": args.right_serial}
    if args.left_serial:
        serial_paths["left"] = args.left_serial

    serial_ports = {}
    for name, path in serial_paths.items():
        print(f"Opening {name} arm serial {path} @ {args.baud}")
        serial_ports[name] = serial.Serial(path, args.baud, timeout=0.05)
    time.sleep(2)
    states = {
        name: initialize_arm(name, ser)
        for name, ser in serial_ports.items()
    }
    rover_context = None
    rover_ctrl = None
    rover_state = RoverDriveState()
    if args.rover_serial:
        rover_context = AtlasController(
            port=args.rover_serial,
            baud=args.rover_baud,
        ).connect()
        rover_ctrl = rover_context.__enter__()

    try:
        if args.socket_type == "raw":
            asyncio.run(run_raw(
                LISTEN_HOST,
                args.port,
                serial_ports,
                states,
                rover_ctrl,
                rover_state,
            ))
        else:
            asyncio.run(run_ws(
                LISTEN_HOST,
                args.port,
                serial_ports,
                states,
                rover_ctrl,
                rover_state,
                ssl_ctx,
            ))
    except KeyboardInterrupt:
        print("\nShutting down.")
        for name, state in states.items():
            print(
                f"Summary [{name}]: received={state.messages_received} "
                f"sent={state.commands_sent} clamps={state.clamp_events}"
            )
    finally:
        if rover_context is not None:
            rover_context.__exit__(None, None, None)
        for ser in serial_ports.values():
            ser.close()


if __name__ == "__main__":
    main()
