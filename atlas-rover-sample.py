#!/usr/bin/env python3
"""
Send STM32 command frames over serial:

  $CMD,L=<int>,R=<int>,AUX=<int>#

L, R: -100..100
AUX : 0..100

Port : /dev/ttyACM0
Baud : 115200
Delay: 500 ms
"""

import time
import serial

PORT = "/dev/ttyACM0"
BAUD = 115200
PERIOD_S = 0.5

def clamp(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x

def make_cmd(L: int, R: int, AUX: int) -> str:
    L   = clamp(int(L),   -100, 100)
    R   = clamp(int(R),   -100, 100)
    AUX = clamp(int(AUX),    0, 100)
    return f"$CMD,L={L},R={R},AUX={AUX}#\n" 

def main():
    ser = serial.Serial(
        port=PORT,
        baudrate=BAUD,
        timeout=0.2,
        write_timeout=0.2,
    )

    time.sleep(0.2)  # let port settle (esp. after reset)
    print(f"[OK] Opened {PORT} @ {BAUD}")

    seq = [
        (0,   0,   0),
        (20,  20,  0),
        (50,  50, 10),
        (100, 100, 0),
        (50,  50,  0),
        (0,   0,   0),
        (-20,  -20,   0),
        (-30,  -30,   0),
        (-50,  -50,   0),
        (-75,  -75,   0),
        (-100,  -100,   0),
        (-30, -30,  0),  
        (0,   0,  50),  
        (-30, 30,  0),
        (-100, 100,  0),
        (0, 0,  0),
        (30, -30,  0),
        (100, -100,  0),
    ]

    try:
        i = 0
        while True:
            for j in range(0,10):
                L, R, AUX = seq[i % len(seq)]
                frame = make_cmd(L, R, AUX)
                ser.write(frame.encode("ascii"))
                ser.flush()
                print(f"[TX] {frame.strip()}")
                time.sleep(PERIOD_S)
            i += 1
                

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        try:
            ser.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()