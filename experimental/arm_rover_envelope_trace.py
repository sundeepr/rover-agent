#!/usr/bin/env python3
"""
arm_rover_envelope_trace.py — Trace the arm's reach envelope while the rover
drives along X.

The idea
────────
At a fixed height Z the arm has a radial budget

    b(Z) = √((L2+L3)² − (Z − L1)²)

shared between arm-frame X' and Y:  √(X'² + Y²) ≤ b. So the reachable Y is a
function of X':

    Y_max(X') = √(b² − X'²)

As the rover backs away from the work line, X' grows and Y_max shrinks; as it
drives forward, X' shrinks and Y_max reopens. This script holds the arm at
Y = ±Y_max(X') throughout, so the end-effector rides the *boundary* of the
workspace while the rover sweeps X' back and forth.

Two frames, two shapes — both true
──────────────────────────────────
In the ARM frame the path is a circular arc of radius b: the arm is always at
full extension, sweeping its base as X' changes.

In the WORLD frame it is a STRAIGHT LINE at constant X. The rover's motion
exactly cancels the change in X': backing up by d increases X' by d, so
world_X = rover_x + X' stays fixed. Only Y varies.

    X'    Y      rover_x   world_X
    150  422.2      0        150
    262  363.6   -112        150
    373  247.4   -224        150
    448    0.0   -298        150

That is the whole point of coupling the two: the end-effector traces a
straight world-frame line far longer than the arm alone could reach, because
the rover keeps feeding it fresh workspace.

Worked numbers at Z=350 (b = 468.1 mm, no safety margin)
────────────────────────────────────────────────────────
      X' = 100  →  Y_max = 457.2
      X' = 150  →  Y_max = 443.4
      X' = 250  →  Y_max = 395.7
      X' = 350  →  Y_max = 310.8
      X' = 468  →  Y_max =   0.0   (fully extended forward, no Y left)

Note Y does NOT reach 0 at X'=350 — it reaches 310.8 mm there. Y only closes
completely at X' = b = 468.1 mm. Starting at X'=150, that is 318 mm of backup,
not 200. Use --x-end to say where to stop explicitly.

Continuous, not sequenced
─────────────────────────
Unlike arm_rover_trace_y.py (drive, stop, sweep), here the rover drives while
the arm moves. The arm target is recomputed against the rover's estimated
position every step, so the two motions are coupled.

That estimate is OPEN-LOOP dead reckoning: position = velocity × elapsed time,
with no odometry feedback. Wheel slip, ramp-up, and carpet all make the real
X' drift from the model, and the arm has no way to notice. Expect the traced
arc to deform. Keep --rover-velocity low and the runs short.

Usage
─────
    # Dry-run: back up from X'=150 until Y closes, then return
    python experimental/arm_rover_envelope_trace.py --dry-run

    # Preview the arc and the Y_max(X') curve
    python experimental/arm_rover_envelope_trace.py --dry-run --plot

    # Back up only 200 mm (Y will reach 310.8 mm, not 0), then return
    python experimental/arm_rover_envelope_trace.py --x-end 350 --dry-run

    # Arm for real, rover stubbed out (arm assumes the rover is moving!)
    python experimental/arm_rover_envelope_trace.py --rover-noop \\
        --arm-port /dev/ttyUSB0

    # Full hardware run
    python experimental/arm_rover_envelope_trace.py \\
        --arm-port /dev/ttyUSB0 --rover-port /dev/ttyUSB1
"""

import argparse
import json
import math
import sys
import time

# ── Arm geometry (mm) — from roarm_socket_server.py ──────────────────────────
ARM_L1_LENGTH_MM = 126.06                      # base pivot height
ARM_L2_LENGTH_MM = math.hypot(236.82, 30.00)   # 238.71
ARM_L3_LENGTH_MM = math.hypot(280.15,  1.73)   # 280.16
ARM_MAX_REACH_MM = ARM_L2_LENGTH_MM + ARM_L3_LENGTH_MM   # 518.87

# Margin off the hard kinematic limit: at full extension there is no torque
# headroom and the IK solution goes singular.
REACH_SAFETY_MM = 20.0

SERVO_STEPS_PER_REV = 4096
WRIST_ANGLE = 0

# ── Rover (Roomba OI) ────────────────────────────────────────────────────────
ROOMBA_STRAIGHT_RADIUS = 0x8000
ROVER_VELOCITY_MM_S    = 60     # low: open-loop dead reckoning, error accumulates


def radial_budget(z: float) -> float:
    """Max √(X'² + Y²) reachable at height z, less the safety margin."""
    dz = z - ARM_L1_LENGTH_MM
    inner = ARM_MAX_REACH_MM ** 2 - dz ** 2
    if inner <= 0:
        return 0.0
    return math.sqrt(inner) - REACH_SAFETY_MM


def y_max_at(x_arm: float, z: float) -> float:
    """Max |Y| at arm-frame x_arm and height z. 0 when X' consumes the budget."""
    b = radial_budget(z)
    inner = b ** 2 - x_arm ** 2
    if inner <= 0:
        return 0.0
    return math.sqrt(inner)


def build_arm_command(x: float, y: float, z: float, speed: int) -> str:
    return json.dumps({
        "T":   1041,
        "x":   round(float(x), 2),
        "y":   round(float(y), 2),
        "z":   round(float(z), 2),
        "t":   WRIST_ANGLE,
        "spd": speed,
    })


def auto_speed(pts: list[tuple[float, float, float]], delay: float) -> int:
    """
    Size `spd` (servo steps/sec) so each move takes about one command period.

    spd:0 means MAXIMUM speed per the Waveshare docs — with a streamed target
    that makes the arm bolt to each point, idle, and read as jumping between
    extremes. Sizing to the step rate keeps it moving when the next target
    lands. Uses the largest per-step base rotation in the path, which for an
    envelope arc occurs where Y changes fastest.
    """
    if delay <= 0 or len(pts) < 2:
        return 0

    max_rad = 0.0
    for (x0, y0, _), (x1, y1, _) in zip(pts, pts[1:]):
        a0 = math.atan2(y0, x0)
        a1 = math.atan2(y1, x1)
        d = abs(a1 - a0)
        if d > math.pi:            # wrapped the wrong way round
            d = 2 * math.pi - d
        max_rad = max(max_rad, d)

    if max_rad <= 0:
        return 0
    spd = (max_rad / delay) / (2 * math.pi) * SERVO_STEPS_PER_REV
    return max(10, int(spd * 1.2))


def generate_path(x_start: float, x_end: float, z: float,
                  steps: int, side: int,
                  return_pass: bool) -> list[tuple[float, float, float, float]]:
    """
    Build (t_frac, x_arm, y, z) samples riding the envelope from x_start to
    x_end, optionally returning. t_frac is progress through the whole run and
    drives the rover schedule.

    x_arm is the target's position in the ARM frame, which is what the rover's
    motion changes. Backing the rover away increases x_arm.
    """
    legs = [(x_start, x_end)]
    if return_pass:
        legs.append((x_end, x_start))

    samples = []
    total = len(legs) * steps
    k = 0
    for leg_i, (a, b) in enumerate(legs):
        # Skip the duplicated junction point between legs
        first = 1 if leg_i > 0 else 0
        for i in range(first, steps + 1):
            xa = a + (b - a) * (i / steps)
            y = side * y_max_at(xa, z)
            samples.append((k / total if total else 0.0, xa, y, z))
            k += 1
    return samples


def plot_path(samples, x_start: float, x_end: float, z: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed — skipping plot")
        return

    xs = [s[1] for s in samples]
    ys = [s[2] for s in samples]
    b  = radial_budget(z)

    # Rover position that keeps world X constant, and the resulting world path.
    rover_xs = [x_start - x for x in xs]
    world_xs = [r + x for r, x in zip(rover_xs, xs)]

    fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize=(17, 5.5))

    # The traced arc in the arm frame
    ax1.plot(xs, ys, '-', color='steelblue', lw=2, label='traced path')
    th = [i * math.pi / 180 for i in range(-90, 91)]
    ax1.plot([b * math.cos(t) for t in th], [b * math.sin(t) for t in th],
             ':', color='gray', lw=1, label=f'reach envelope (b={b:.0f})')
    ax1.plot([xs[0]], [ys[0]], 'o', color='green', markersize=9, label='start')
    ax1.plot([xs[-1]], [ys[-1]], 'o', color='red', markersize=9, label='end')
    ax1.set_xlabel("Arm-frame X' (mm)"); ax1.set_ylabel('Y (mm)')
    ax1.set_title('Path in arm frame — rides the envelope')
    ax1.grid(True, alpha=0.3); ax1.set_aspect('equal'); ax1.legend(fontsize=8)
    ax1.axhline(0, color='k', lw=0.5); ax1.axvline(0, color='k', lw=0.5)

    # The same path in the WORLD frame — a straight line at constant X
    ax3.plot(world_xs, ys, '-', color='darkorange', lw=3, label='end-effector')
    ax3.plot(rover_xs, [0] * len(rover_xs), '-', color='dimgray', lw=2,
             label='rover base')
    ax3.plot([world_xs[0]], [ys[0]], 'o', color='green', markersize=9)
    ax3.plot([world_xs[-1]], [ys[-1]], 'o', color='red', markersize=9)
    for j in range(0, len(xs), max(1, len(xs) // 10)):
        ax3.plot([rover_xs[j], world_xs[j]], [0, ys[j]],
                 ':', color='gray', lw=0.8)
    ax3.set_xlabel('World X (mm)'); ax3.set_ylabel('World Y (mm)')
    ax3.set_title(f'World frame — STRAIGHT LINE at X={world_xs[0]:.0f} mm\n'
                  f'(dotted = arm, at full extension throughout)')
    ax3.grid(True, alpha=0.3); ax3.legend(fontsize=8)
    ax3.axhline(0, color='k', lw=0.5)

    # Y_max as a function of X', with the operating range marked
    xr = [i for i in range(0, int(b) + 1, 2)]
    ax2.plot(xr, [y_max_at(x, z) for x in xr], color='seagreen')
    for xv, c, lab in ((x_start, 'green', f"start X'={x_start:.0f}"),
                       (x_end,   'red',   f"end X'={x_end:.0f}")):
        ax2.axvline(xv, color=c, ls='--', lw=1)
        ax2.plot([xv], [y_max_at(xv, z)], 'o', color=c, markersize=8)
        ax2.annotate(f"  {lab}\n  Y={y_max_at(xv, z):.0f}", (xv, y_max_at(xv, z)),
                     fontsize=8, color=c, va='top')
    ax2.set_xlabel("Arm-frame X' (mm)"); ax2.set_ylabel('Max |Y| (mm)')
    ax2.set_title(f'Y_max(X\') at Z={z:.0f} — Y closes at X\'={b:.0f}')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'arm_rover_envelope_trace — Z={z:.0f} mm, '
                 f"X' {x_start:.0f} → {x_end:.0f} mm", fontsize=12)
    plt.tight_layout()
    plt.show()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Trace the arm's reach envelope while the rover drives along X")
    p.add_argument("--x-start", type=float, default=150,
                   help="Arm-frame X' at the start, where Y is widest (default: 150)")
    p.add_argument("--x-end",   type=float, default=None,
                   help="Arm-frame X' at the end. Default: the radial budget, "
                        "where Y closes to 0")
    p.add_argument("--z",       type=float, default=350,
                   help="Working height in mm (default: 350)")
    p.add_argument("--side",    choices=["positive", "negative"], default="positive",
                   help="Trace the +Y or -Y half of the envelope (default: positive)")
    p.add_argument("--steps",   type=int,   default=200,
                   help="Waypoints per leg (default: 200)")
    p.add_argument("--delay",   type=float, default=0.05,
                   help="Seconds between waypoints (default: 0.05 = 20 Hz)")
    p.add_argument("--speed",   type=int,   default=None,
                   help="Arm spd in servo steps/sec. Default: computed from the "
                        "path and --delay. Do NOT use 0 (= maximum speed)")
    p.add_argument("--no-return", action="store_true",
                   help="Stop at --x-end instead of driving back to --x-start")
    p.add_argument("--rover-velocity", type=int, default=ROVER_VELOCITY_MM_S,
                   help=f"Rover speed mm/s (default: {ROVER_VELOCITY_MM_S})")
    p.add_argument("--rover-noop", action="store_true",
                   help="Log rover commands instead of sending them; the arm "
                        "still runs the full coupled trajectory")
    p.add_argument("--arm-port",   type=str, default="/dev/ttyUSB0")
    p.add_argument("--rover-port", type=str, default="/dev/ttyUSB1")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan without touching hardware")
    p.add_argument("--plot",    action="store_true",
                   help="Show a matplotlib preview")
    args = p.parse_args()

    if args.steps < 1:
        sys.exit("--steps must be >= 1")

    b = radial_budget(args.z)
    if b <= 0:
        sys.exit(f"Z={args.z:.0f} mm is outside the arm's vertical reach")

    x_end = args.x_end if args.x_end is not None else b
    if args.x_start < 0 or x_end < 0:
        sys.exit("X' values must be >= 0")
    for name, xv in (("--x-start", args.x_start), ("--x-end", x_end)):
        if xv > b:
            sys.exit(f"{name}={xv:.0f} mm exceeds the radial budget {b:.0f} mm "
                     f"at Z={args.z:.0f} — the arm cannot reach that far.")

    side = 1 if args.side == "positive" else -1
    samples = generate_path(args.x_start, x_end, args.z, args.steps,
                            side, not args.no_return)
    pts = [(s[1], s[2], s[3]) for s in samples]
    spd = args.speed if args.speed is not None else auto_speed(pts, args.delay)

    travel = abs(x_end - args.x_start)
    legs = 1 if args.no_return else 2
    leg_time = args.steps * args.delay
    needed_v = travel / leg_time if leg_time > 0 else 0.0

    print(f"\nEnvelope trace at Z={args.z:.0f} mm  (side: {args.side} Y)")
    print(f"  radial budget b            : {b:.1f} mm "
          f"(reach {ARM_MAX_REACH_MM:.1f} − {REACH_SAFETY_MM:.0f} margin)")
    print(f"  X' {args.x_start:.0f} → {x_end:.0f} mm"
          f"   (rover backs up {travel:.0f} mm)" if x_end > args.x_start else
          f"  X' {args.x_start:.0f} → {x_end:.0f} mm"
          f"   (rover moves forward {travel:.0f} mm)")
    print(f"  Y at start                 : {y_max_at(args.x_start, args.z):.1f} mm")
    print(f"  Y at end                   : {y_max_at(x_end, args.z):.1f} mm")
    print(f"  WORLD frame                : straight line at X={args.x_start:.0f} mm, "
          f"Y {y_max_at(args.x_start, args.z):.1f} → {y_max_at(x_end, args.z):.1f} mm")
    print(f"  legs                       : {legs} "
          f"({'out and back' if legs == 2 else 'one way'})")
    print(f"  waypoints                  : {len(pts)} @ "
          f"{1/args.delay:.0f} Hz, spd={spd}")
    print(f"  leg duration               : {leg_time:.1f}s")
    print(f"  required rover speed       : {needed_v:.1f} mm/s "
          f"(set: {args.rover_velocity} mm/s)")
    if needed_v > args.rover_velocity * 1.05:
        print(f"  [warn] rover too slow for this leg time — it will lag behind "
              f"the arm and the traced arc will distort. Raise "
              f"--rover-velocity to ~{needed_v:.0f}, or raise --delay/--steps.")
    if args.rover_noop:
        print("  [note] --rover-noop: rover commands logged only, but the arm "
              "still moves as if the rover were driving. The traced shape will "
              "be WRONG unless the rover really is moving.")
    print("  [note] rover position is open-loop dead reckoning — no odometry "
          "feedback; expect drift.\n")

    if args.plot:
        plot_path(samples, args.x_start, x_end, args.z)

    # ── Hardware ──────────────────────────────────────────────────────────────
    ser = rover = rover_ctx = None
    if not args.dry_run:
        try:
            import serial
            ser = serial.Serial(args.arm_port, 115200, timeout=1)
            print(f"Arm serial opened: {args.arm_port}")
            time.sleep(2)   # allow ESP32 to boot
        except Exception as e:
            sys.exit(f"Cannot open arm port {args.arm_port}: {e}")

        if not args.rover_noop:
            try:
                import pathlib
                sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
                from roomba_controller import RoombaController
                rover = RoombaController(port=args.rover_port)
                rover_ctx = rover.connect()
                rover_ctx.__enter__()
                print(f"Rover connected: {args.rover_port}")
            except Exception as e:
                if ser:
                    ser.close()
                sys.exit(f"Cannot connect rover on {args.rover_port}: {e}")

    # Direction of rover travel per leg. Backing away from the work line
    # increases X', which on the Roomba is reverse (negative velocity).
    def leg_velocity(a: float, bb: float) -> int:
        v = args.rover_velocity
        return int(-v if bb > a else v)

    print(f"Tracing {len(pts)} waypoints...")
    start_t = time.monotonic()
    current_leg = -1
    try:
        for i, (t_frac, xa, y, z) in enumerate(samples):
            leg = 0 if (args.no_return or i <= args.steps) else 1
            if leg != current_leg:
                current_leg = leg
                a, bb = ((args.x_start, x_end) if leg == 0
                         else (x_end, args.x_start))
                vel = leg_velocity(a, bb)
                direction = "backward" if vel < 0 else "forward"
                tag = "ROVER (noop)" if args.rover_noop else "ROVER"
                print(f"  {tag} → leg {leg + 1}: drive {direction} "
                      f"at {abs(vel)} mm/s for {leg_time:.1f}s "
                      f"(X' {a:.0f} → {bb:.0f})")
                if not args.dry_run and not args.rover_noop and rover is not None:
                    rover.drive_raw(vel, ROOMBA_STRAIGHT_RADIUS)

            cmd = build_arm_command(xa, y, z, spd)
            if args.dry_run:
                if i % max(1, len(samples) // 8) == 0 or i == len(samples) - 1:
                    print(f"    [dry-run] {i:03d}  X'={xa:6.1f}  Y={y:7.1f}  → {cmd}")
            else:
                ser.write((cmd + "\n").encode())
                ser.flush()
                if ser.in_waiting:
                    ser.reset_input_buffer()

            if i < len(samples) - 1:
                sleep_for = (start_t + (i + 1) * args.delay) - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)

        if not args.dry_run and not args.rover_noop and rover is not None:
            rover.stop()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if rover is not None and not args.dry_run:
            try:
                rover.stop()
            except Exception:
                pass
            if rover_ctx is not None:
                try:
                    rover_ctx.__exit__(None, None, None)
                except Exception:
                    pass
        if ser is not None:
            ser.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
