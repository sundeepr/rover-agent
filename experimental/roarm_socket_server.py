#!/usr/bin/env python3
"""
RoArm M2 socket server — relays JSON move commands to the arm over serial.
Supports three socket modes selected with --socket-type:

  raw   Plain TCP.  Each newline-terminated JSON message is forwarded to serial.
  ws    WebSocket (default).
  wss   WebSocket over TLS — requires --cert and --key.

Clients send:
    {"T":1041,"x":<x>,"y":<y>,"z":<z>,"t":<wrist_angle>}

Usage:
    python roarm_socket_server.py --socket-type raw  [--port 9876] [--serial /dev/ttyUSB0]
    python roarm_socket_server.py --socket-type ws   [--port 9876]
    python roarm_socket_server.py --socket-type wss  --cert cert.pem --key key.pem [--port 9876]

Generate a self-signed cert for WSS testing:
    openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

Test with:
    nc localhost 9876                              # raw
    wscat -c ws://localhost:9876                  # ws
    wscat -c wss://localhost:9876 --no-check      # wss (self-signed)
    > {"T":1041,"x":150,"y":0,"z":80,"t":3.14}
"""

import argparse
import asyncio
import json
import ssl
import time
import serial
import websockets


SERIAL_PORT   = "/dev/ttyUSB0"
BAUD_RATE     = 115200
LISTEN_HOST   = "0.0.0.0"
LISTEN_PORT   = 9876

# ---------------------------------------------------------------------------
# Shared command handler
# ---------------------------------------------------------------------------

def relay_command(raw: str, addr, ser: serial.Serial) -> None:
    print(f"[>] data received from {addr}: {raw!r}")
    try:
        json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[!] invalid JSON from {addr}: {e}")
        return

    ser.write((raw.strip() + "\n").encode())
    print(f"[arm] sending to serial → {raw.strip()}")


# ---------------------------------------------------------------------------
# Raw TCP mode
# ---------------------------------------------------------------------------

async def handle_raw_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, ser: serial.Serial) -> None:
    addr = writer.get_extra_info("peername")
    print(f"[+] new connection from {addr}")
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            relay_command(line.decode().strip(), addr, ser)
    except (ConnectionResetError, asyncio.IncompleteReadError):
        pass
    finally:
        writer.close()
        print(f"[-] disconnected {addr}")


async def run_raw(host: str, port: int, ser: serial.Serial) -> None:
    print(f"Raw TCP server listening on {host}:{port}")
    server = await asyncio.start_server(
        lambda r, w: handle_raw_client(r, w, ser), host, port
    )
    async with server:
        await server.serve_forever()


# ---------------------------------------------------------------------------
# WebSocket mode (ws / wss)
# ---------------------------------------------------------------------------

async def handle_ws_client(ws, ser: serial.Serial) -> None:
    addr = ws.remote_address
    print(f"[+] new connection from {addr}")
    try:
        async for message in ws:
            relay_command(message, addr, ser)
    except websockets.ConnectionClosed:
        pass
    finally:
        print(f"[-] disconnected {addr}")


async def run_ws(host: str, port: int, ser: serial.Serial, ssl_ctx=None) -> None:
    scheme = "wss" if ssl_ctx else "ws"
    print(f"WebSocket server listening on {scheme}://{host}:{port}")
    async with websockets.serve(lambda ws: handle_ws_client(ws, ser), host, port, ssl=ssl_ctx):
        await asyncio.Future()   # run forever


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RoArm M2 socket relay server")
    parser.add_argument("--socket-type", choices=["raw", "ws", "wss"], default="ws",
                        help="Socket type: raw (TCP), ws (WebSocket), wss (WebSocket+TLS)")
    parser.add_argument("--port",   type=int, default=LISTEN_PORT, help="Listen port")
    parser.add_argument("--serial", type=str, default=SERIAL_PORT, help="Serial device for arm")
    parser.add_argument("--baud",   type=int, default=BAUD_RATE)
    parser.add_argument("--cert",   type=str, default=None, help="TLS certificate file (wss only)")
    parser.add_argument("--key",    type=str, default=None, help="TLS private key file (wss only)")
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
    ser = serial.Serial(args.serial, args.baud, timeout=1)
    time.sleep(2)   # let ESP32 boot after serial open

    try:
        if args.socket_type == "raw":
            asyncio.run(run_raw(LISTEN_HOST, args.port, ser))
        else:
            asyncio.run(run_ws(LISTEN_HOST, args.port, ser, ssl_ctx))
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
