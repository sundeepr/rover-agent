#!/usr/bin/env python3
"""
RoArm M2 socket server — listens for JSON move commands and relays them to the arm via serial.

Clients send:
    {"T":1041,"x":<x>,"y":<y>,"z":<z>,"t":<wrist_angle>}

The server forwards it verbatim (plus a newline) to the arm's serial port.

Usage:
    python roarm_socket_server.py [--port 9000] [--serial /dev/ttyUSB0]
"""

import argparse
import json
import socket
import threading
import time
import serial


SERIAL_PORT  = "/dev/ttyUSB0"
BAUD_RATE    = 115200
LISTEN_HOST  = "0.0.0.0"
LISTEN_PORT  = 9000
REQUIRED_KEYS = {"T", "x", "y", "z", "t"}


def handle_client(conn: socket.socket, addr, ser: serial.Serial) -> None:
    print(f"[+] new connection from {addr}")
    buf = ""
    try:
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            print(f"[>] data received from {addr}: {chunk.decode().strip()!r}")
            buf += chunk.decode()
            # process every newline-terminated or complete JSON object
            while True:
                buf = buf.strip()
                if not buf:
                    break
                try:
                    cmd, idx = json.JSONDecoder().raw_decode(buf)
                    buf = buf[idx:]
                except json.JSONDecodeError:
                    break

                missing = REQUIRED_KEYS - cmd.keys()
                if missing:
                    print(f"[!] ignored — missing keys: {missing}")
                    continue

                payload = json.dumps(cmd) + "\n"
                ser.write(payload.encode())
                print(f"[arm] sending to serial → x={cmd['x']} y={cmd['y']} z={cmd['z']} t={cmd['t']}")

    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        conn.close()
        print(f"[-] disconnected {addr}")


def main():
    parser = argparse.ArgumentParser(description="RoArm M2 socket relay server")
    parser.add_argument("--port",   type=int, default=LISTEN_PORT, help="TCP listen port")
    parser.add_argument("--serial", type=str, default=SERIAL_PORT, help="Serial device for arm")
    parser.add_argument("--baud",   type=int, default=BAUD_RATE)
    args = parser.parse_args()

    print(f"Opening serial {args.serial} @ {args.baud}")
    ser = serial.Serial(args.serial, args.baud, timeout=1)
    time.sleep(2)   # let ESP32 boot after serial open

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((LISTEN_HOST, args.port))
    sock.listen(5)
    print(f"Listening on {LISTEN_HOST}:{args.port}")

    try:
        while True:
            conn, addr = sock.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr, ser), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        sock.close()
        ser.close()


if __name__ == "__main__":
    main()
