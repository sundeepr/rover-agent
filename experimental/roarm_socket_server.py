#!/usr/bin/env python3
"""
RoArm M2 WebSocket server — accepts WS connections and relays JSON move
commands to the arm over serial.

Clients send:
    {"T":1041,"x":<x>,"y":<y>,"z":<z>,"t":<wrist_angle>}

The server forwards each command verbatim (plus a newline) to the arm's
serial port.  Multiple clients can be connected simultaneously.

Usage:
    python roarm_socket_server.py [--port 9000] [--serial /dev/ttyUSB0]

Test with wscat:
    wscat -c ws://localhost:9000
    > {"T":1041,"x":150,"y":0,"z":80,"t":3.14}
"""

import argparse
import asyncio
import json
import time
import serial
import websockets


SERIAL_PORT   = "/dev/ttyUSB0"
BAUD_RATE     = 115200
LISTEN_HOST   = "0.0.0.0"
LISTEN_PORT   = 9000
REQUIRED_KEYS = {"T", "x", "y", "z", "t"}


async def handle_client(ws, ser: serial.Serial) -> None:
    addr = ws.remote_address
    print(f"[+] new connection from {addr}")
    try:
        async for message in ws:
            print(f"[>] data received from {addr}: {message!r}")
            try:
                cmd = json.loads(message)
            except json.JSONDecodeError as e:
                print(f"[!] invalid JSON from {addr}: {e}")
                continue

            missing = REQUIRED_KEYS - cmd.keys()
            if missing:
                print(f"[!] ignored — missing keys: {missing}")
                continue

            payload = json.dumps(cmd) + "\n"
            ser.write(payload.encode())
            print(f"[arm] sending to serial → x={cmd['x']} y={cmd['y']} z={cmd['z']} t={cmd['t']}")

    except websockets.ConnectionClosed:
        pass
    finally:
        print(f"[-] disconnected {addr}")


async def run(host: str, port: int, ser: serial.Serial) -> None:
    print(f"WebSocket server listening on ws://{host}:{port}")
    async with websockets.serve(lambda ws: handle_client(ws, ser), host, port):
        await asyncio.Future()   # run forever


def main():
    parser = argparse.ArgumentParser(description="RoArm M2 WebSocket relay server")
    parser.add_argument("--port",   type=int, default=LISTEN_PORT, help="WebSocket listen port")
    parser.add_argument("--serial", type=str, default=SERIAL_PORT, help="Serial device for arm")
    parser.add_argument("--baud",   type=int, default=BAUD_RATE)
    args = parser.parse_args()

    print(f"Opening serial {args.serial} @ {args.baud}")
    ser = serial.Serial(args.serial, args.baud, timeout=1)
    time.sleep(2)   # let ESP32 boot after serial open

    try:
        asyncio.run(run(LISTEN_HOST, args.port, ser))
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
