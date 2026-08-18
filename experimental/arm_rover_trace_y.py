#!/usr/bin/env python3
"""
arm_rover_trace_y.py — Trace a long Y line at fixed world X and Z by combining
rover motion along X with arm motion in Y.

Why this exists
───────────────
With a fixed base the arm's radial budget √(X²+Y²) is shared between X and Y,
so holding X=350 at Z=350 leaves only ~±311 mm of Y. If the rover drives along
X, it can park the base *abeam* the target line (arm-frame X' = 0), handing the
entire radial budget to Y and extending the reach to ~±468 mm — a ~51% gain —
without moving the end-effector out of the same world X+Z plane.

Geometry
────────
    base pivot height        L1 = 126.06 mm
    upper + fore arm         L2 + L3 = 518.87 mm   (hard kinematic reach)
    radial budget at Z       R(Z) = √((L2+L3)² − (Z − L1)²)
    world→arm frame          X' = X_world − rover_x
    reachable                √(X'² + Y²) ≤ R(Z)

Parking the rover at rover_x = X_world gives X' = 0, so |Y| ≤ R(Z).

IMPORTANT — this reach is geometric only. It ignores servo torque, which is at
its weakest exactly here (arm near-horizontal, fully extended). Real reach will
fall short, likely by a good margin. Verify against arm_read_position.py and
lower --y-span until commanded Y matches actual Y at the sweep ends.

Sequenced, not simultaneous
───────────────────────────
The rover drives to position, stops, and only then does the arm sweep. Driving
and sweeping at once would require recomputing the arm target against live
odometry every step; sequencing avoids that entirely.

Scope limit
───────────
The rover drives along X, not Y, so it buys a *wider* Y sweep from one parking
spot — it cannot chase the line sideways to make the sweep arbitrarily long.
Max traced span is therefore 2×R(Z) ≈ 896 mm at Z=350. Longer --y-span values
are clipped, with a warning. To trace an unbounded line you would drive the
rover along Y instead and hold the arm still — a different script.

Usage
─────
    # Dry-run the default plan (X=350, Z=350, Y ±468 in 3 segments):
    python experimental/arm_rover_trace_y.py --dry-run

    # Preview the segmentation and rover stops:
    python experimental/arm_rover_trace_y.py --dry-run --plot

    # Real hardware — arm and rover on separate serial ports:
    python experimental/arm_rover_trace_y.py \\
        --arm-port /dev/ttyUSB0 --rover-port /dev/ttyUSB1

    # Arm only, no rover (falls back to the fixed-base ±311 limit):
    python experimental/arm_rover_trace_y.py --no-rover --dry-run

    # Rover stubbed out, arm runs for real at the full abeam reach.
    # Park the rover at world X first — the arm assumes it is there:
    python experimental/arm_rover_trace_y.py --rover-noop --arm-port /dev/ttyUSB0

Mode summary
────────────
    (default)      rover drives, arm moves         reach ±448 mm
    --rover-noop   rover logged only, arm moves    reach ±448 mm
    --no-rover     no rover at all, arm moves      reach ±280 mm
    --dry-run      nothing moves, plan printed     (combines with any of the above)
"""

import argparse
import json
import math
import sys
import time

# ── Arm geometry (mm) — from roarm_socket_server.py ──────────────────────────
ARM_L1_LENGTH_MM   = 126.06                      # base pivot height
ARM_L2_LENGTH_MM   = math.hypot(236.82, 30.00)   # 238.71
ARM_L3_LENGTH_MM   = math.hypot(280.15,  1.73)   # 280.16
ARM_MAX_REACH_MM   = ARM_L2_LENGTH_MM + ARM_L3_LENGTH_MM   # 518.87

# Keep a margin off the hard kinematic limit — at full extension the arm has no
# torque headroom and the IK solution becomes singular.
REACH_SAFETY_MM    = 20.0

ARM_Z_MIN, ARM_Z_MAX = 30, 500

WRIST_ANGLE = 0

# ── Rover (Roomba OI) ────────────────────────────────────────────────────────
ROOMBA_STRAIGHT_RADIUS = 0x8000   # OI special value: drive straight
ROVER_VELOCITY_MM_S    = 100      # conservative; odometry error grows with speed


def radial_budget(z: float) -> float:
    """Max √(X²+Y²) the arm can reach at height z, less the safety margin."""
    dz = z - ARM_L1_LENGTH_MM
    inner = ARM_MAX_REACH_MM ** 2 - dz ** 2
    if inner <= 0:
        return 0.0
    return math.sqrt(inner) - REACH_SAFETY_MM


def max_y_at(x_arm: float, z: float) -> float:
    """Max |Y| reachable at arm-frame x_arm and height z. 0 if out of reach."""
    budget = radial_budget(z)
    inner = budget ** 2 - x_arm ** 2
    if inner <= 0:
        return 0.0
    return math.sqrt(inner)


SERVO_STEPS_PER_REV = 4096   # 12-bit encoder: one revolution = 4096 steps


def auto_speed(pts: list[tuple[float, float, float]], delay: float) -> int:
    """
    Pick a `spd` (servo steps/sec) so each waypoint move takes roughly one
    command period.

    This matters more than it looks. `spd: 0` does NOT mean "track smoothly" —
    per the Waveshare docs it means *maximum* speed, so the arm bolts to each
    target, arrives early, and idles until the next command. With targets
    arriving every `delay` seconds that reads as slamming between extremes
    rather than tracing the line. Sizing spd to the step rate keeps the arm
    still moving when the next target lands, which is what actually blends the
    waypoints into a continuous path.
    """
    if delay <= 0 or len(pts) < 2:
        return 0

    # The Y sweep is driven mainly by base rotation, so size spd from the base
    # angle swept per step. Note the path crosses the base singularity at
    # X'=0: there atan2(y, 0) flips between ±90° and a naive difference reads
    # as a 180° jump. Guard against that by taking the total swept angle over
    # the segment and dividing, rather than differencing adjacent points.
    a_first = math.atan2(pts[0][1],  pts[0][0])
    a_last  = math.atan2(pts[-1][1], pts[-1][0])
    total_rad = abs(a_last - a_first)
    if total_rad > math.pi:          # took the wrong way round the circle
        total_rad = 2 * math.pi - total_rad

    n_steps = len(pts) - 1
    if total_rad <= 0 or n_steps <= 0:
        return 0
    # Cosine easing peaks at ~pi/2 times the mean rate; size for the peak so
    # the arm does not fall behind mid-sweep.
    max_rad_per_step = (total_rad / n_steps) * (math.pi / 2)
    rad_per_s = max_rad_per_step / delay
    spd = rad_per_s / (2 * math.pi) * SERVO_STEPS_PER_REV
    # Small headroom so the arm can absorb jitter without falling behind.
    return max(10, int(spd * 1.2))


def build_arm_command(x: float, y: float, z: float, speed: int) -> str:
    return json.dumps({
        "T":   1041,
        "x":   round(float(x), 2),
        "y":   round(float(y), 2),
        "z":   round(float(z), 2),
        "t":   WRIST_ANGLE,
        "spd": speed,
    })


def _ease(t: float, mode: str) -> float:
    """Linear progress → eased progress, for accel/decel at segment ends."""
    if mode == "cosine":
        return 0.5 * (1.0 - math.cos(math.pi * t))
    return t


def plan_segments(x_world: float, z: float,
                  y_from: float, y_to: float,
                  use_rover: bool) -> list[dict]:
    """
    Split the world-frame Y line into segments, each reachable from one rover
    stop. Returns dicts of {rover_x, y_from, y_to} in execution order.

    With the rover, each stop parks the base abeam the target (rover_x =
    x_world, so arm-frame X' = 0) and the full radial budget goes to Y. The
    line is then split into as many equal segments as that reach requires.
    """
    if not use_rover:
        # Fixed base: the arm must span X_world itself, leaving less for Y.
        reach = max_y_at(x_world, z)
        if reach <= 0:
            return []
        lo = max(min(y_from, y_to), -reach)
        hi = min(max(y_from, y_to),  reach)
        if lo >= hi:
            return []
        # Preserve the requested sweep direction
        a, b = (hi, lo) if y_from > y_to else (lo, hi)
        return [{"rover_x": 0.0, "y_from": a, "y_to": b}]

    # Rover parks abeam → arm-frame X' = 0 → full budget available for Y.
    #
    # The rover drives along X, NOT along Y, so it cannot chase the line
    # sideways. One stop is all that helps: it buys the full radial budget for
    # Y, but the reachable Y span is still capped at ±reach. A longer requested
    # span cannot be covered by adding stops — it is simply clipped here, and
    # main() reports the shortfall.
    reach = max_y_at(0.0, z)
    if reach <= 0:
        return []

    lo = max(min(y_from, y_to), -reach)
    hi = min(max(y_from, y_to),  reach)
    if lo >= hi:
        return []
    a, b = (hi, lo) if y_from > y_to else (lo, hi)
    return [{"rover_x": x_world, "y_from": a, "y_to": b}]


def segment_waypoints(seg: dict, x_world: float, z: float,
                      steps: int, ease: str) -> list[tuple[float, float, float]]:
    """Arm-frame waypoints for one segment, streamed for smooth motion."""
    x_arm = x_world - seg["rover_x"]
    pts = []
    for i in range(steps + 1):
        y = seg["y_from"] + (seg["y_to"] - seg["y_from"]) * _ease(i / steps, ease)
        pts.append((x_arm, y, z))
    return pts


def plot_plan(segs: list[dict], x_world: float, z: float, use_rover: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed — skipping plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    reach = max_y_at(0.0, z) if use_rover else max_y_at(x_world, z)

    # Top-down: world X–Y, showing each rover stop and its arm coverage
    for i, s in enumerate(segs):
        ys = [s["y_from"], s["y_to"]]
        ax1.plot([x_world] * 2, ys, '-', lw=6, alpha=0.65,
                 label=f'seg {i}' if i < 6 else None)
        ax1.plot([s["rover_x"]], [0], 'v', color='dimgray', markersize=11)
        ax1.plot([s["rover_x"], x_world], [0, sum(ys) / 2],
                 ':', color='gray', lw=1)
    ax1.set_xlabel('World X (mm)'); ax1.set_ylabel('World Y (mm)')
    ax1.set_title(f'Top-down — {len(segs)} rover stop(s)\n'
                  f'▽ = rover base, bar = arm-traced span')
    ax1.grid(True, alpha=0.3); ax1.axhline(0, color='k', lw=0.5)
    if len(segs) <= 6:
        ax1.legend(fontsize=8)

    # Reach envelope vs height, with the operating point marked
    zs = [zz for zz in range(ARM_Z_MIN, ARM_Z_MAX + 1, 5)]
    ax2.plot(zs, [max_y_at(0.0, zz) for zz in zs],
             label="rover abeam (X'=0)", color='seagreen')
    ax2.plot(zs, [max_y_at(x_world, zz) for zz in zs],
             label=f'fixed base (X={x_world:.0f})', color='indianred')
    ax2.axvline(z, color='steelblue', ls='--', lw=1, label=f'Z={z:.0f}')
    ax2.plot([z], [reach], 'o', color='steelblue', markersize=9)
    ax2.annotate(f'  ±{reach:.0f} mm', (z, reach), fontsize=10,
                 color='steelblue', va='center')
    ax2.set_xlabel('Z (mm)'); ax2.set_ylabel('Max |Y| (mm)')
    ax2.set_title('Reach envelope (geometry only — ignores torque)')
    ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)

    plt.suptitle(f'arm_rover_trace_y — world X={x_world:.0f} mm, Z={z:.0f} mm',
                 fontsize=12)
    plt.tight_layout()
    plt.show()


def drive_rover(rover, target_x: float, current_x: float,
                velocity: int, dry_run: bool, noop: bool = False) -> float:
    """
    Drive straight along X to target_x. Returns the new position.

    With `noop` the move is logged but not sent, and the travel time is not
    slept either — the rover is assumed to be parked there already, so there is
    nothing to wait for. The returned position still advances, which keeps the
    arm-frame geometry identical to a real run.
    """
    delta = target_x - current_x
    if abs(delta) < 1.0:
        return current_x

    duration = abs(delta) / velocity
    vel = int(velocity if delta > 0 else -velocity)
    tag = "ROVER (noop)" if noop else "ROVER"
    suffix = "  — assumed already in position" if noop else ""
    print(f"  {tag} → drive {delta:+.0f} mm along X "
          f"(v={vel} mm/s, {duration:.2f}s){suffix}")

    if not dry_run and not noop and rover is not None:
        rover.drive_raw(vel, ROOMBA_STRAIGHT_RADIUS)
        time.sleep(duration)
        rover.stop()
        time.sleep(0.4)   # let the chassis settle before the arm moves
    return target_x


def stream_arm(ser, pts, speed: int, delay: float, dry_run: bool) -> None:
    """Send waypoints on a fixed wall-clock schedule for continuous motion."""
    start_t = time.monotonic()
    for i, (x, y, z) in enumerate(pts):
        cmd = build_arm_command(x, y, z, speed)
        if dry_run:
            if i % max(1, len(pts) // 4) == 0 or i == len(pts) - 1:
                print(f"    [dry-run] step {i:03d}  y={y:7.1f}  →  {cmd}")
        else:
            ser.write((cmd + "\n").encode())
            ser.flush()
            if ser.in_waiting:
                ser.reset_input_buffer()

        if i < len(pts) - 1:
            sleep_for = (start_t + (i + 1) * delay) - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Trace a long Y line at fixed world X/Z using rover X motion")
    p.add_argument("--x",      type=float, default=350,
                   help="World X of the traced line in mm (default: 350)")
    p.add_argument("--z",      type=float, default=350,
                   help="Z height of the traced line in mm (default: 350)")
    p.add_argument("--y-span", type=float, default=None,
                   help="Total Y span in mm, centred on 0. Default: the maximum "
                        "the geometry allows (2×reach)")
    p.add_argument("--y-from", type=float, default=None,
                   help="Explicit Y start in mm (overrides --y-span)")
    p.add_argument("--y-to",   type=float, default=None,
                   help="Explicit Y end in mm (overrides --y-span)")
    p.add_argument("--steps",  type=int,   default=200,
                   help="Arm waypoints per segment (default: 200)")
    p.add_argument("--speed",  type=int,   default=None,
                   help="Arm speed in servo steps/sec. Default: computed from "
                        "--steps and --delay so each move takes about one "
                        "command period. Do NOT use 0 — that means MAXIMUM "
                        "speed, which makes the arm jump between extremes "
                        "instead of tracing the line.")
    p.add_argument("--delay",  type=float, default=0.05,
                   help="Seconds between arm waypoints (default: 0.05 = 20 Hz)")
    p.add_argument("--ease",   choices=["linear", "cosine"], default="cosine",
                   help="Velocity profile along each segment (default: cosine)")
    p.add_argument("--rover-velocity", type=int, default=ROVER_VELOCITY_MM_S,
                   help=f"Rover speed in mm/s (default: {ROVER_VELOCITY_MM_S})")
    p.add_argument("--no-rover", action="store_true",
                   help="Arm only — falls back to the smaller fixed-base reach")
    p.add_argument("--rover-noop", action="store_true",
                   help="Assume the rover is already parked abeam: keep the full "
                        "rover-abeam reach and run the arm for real, but log the "
                        "drive commands instead of sending them. Unlike "
                        "--no-rover the geometry is unchanged; unlike --dry-run "
                        "the arm still moves.")
    p.add_argument("--return-home", action="store_true",
                   help="Drive the rover back to its starting X when done")
    p.add_argument("--arm-port",   type=str, default="/dev/ttyUSB0",
                   help="Arm serial port (default: /dev/ttyUSB0)")
    p.add_argument("--rover-port", type=str, default="/dev/ttyUSB1",
                   help="Rover serial port (default: /dev/ttyUSB1)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan without touching hardware")
    p.add_argument("--plot",    action="store_true",
                   help="Show a matplotlib preview of the plan")
    args = p.parse_args()

    if args.steps < 1:
        sys.exit("--steps must be >= 1")

    use_rover = not args.no_rover
    budget    = radial_budget(args.z)
    if budget <= 0:
        sys.exit(f"Z={args.z:.0f} mm is outside the arm's vertical reach "
                 f"(max {ARM_L1_LENGTH_MM + ARM_MAX_REACH_MM:.0f} mm)")

    reach = max_y_at(0.0 if use_rover else args.x, args.z)
    if reach <= 0:
        sys.exit(f"X={args.x:.0f} Z={args.z:.0f} unreachable: radial budget is "
                 f"{budget:.0f} mm but X alone needs {args.x:.0f} mm. "
                 f"Lower --x/--z, or drop --no-rover.")

    # Resolve the Y range
    if args.y_from is not None or args.y_to is not None:
        y_from = args.y_from if args.y_from is not None else -reach
        y_to   = args.y_to   if args.y_to   is not None else  reach
    else:
        span   = args.y_span if args.y_span is not None else 2 * reach
        y_from, y_to = span / 2, -span / 2   # positive → negative

    segs = plan_segments(args.x, args.z, y_from, y_to, use_rover)
    if not segs:
        sys.exit("Nothing to trace — the requested Y range is empty or unreachable.")

    fixed_reach = max_y_at(args.x, args.z)
    print(f"\nPlan — world X={args.x:.0f} mm, Z={args.z:.0f} mm")
    print(f"  radial budget at Z         : {budget:.1f} mm "
          f"(reach {ARM_MAX_REACH_MM:.1f} − {REACH_SAFETY_MM:.0f} margin)")
    print(f"  max |Y|, fixed base        : {fixed_reach:.1f} mm")
    if use_rover:
        print(f"  max |Y|, rover abeam       : {reach:.1f} mm"
              + (f"  ({reach / fixed_reach:.2f}× better)" if fixed_reach > 0 else ""))
    actual_from, actual_to = segs[0]["y_from"], segs[-1]["y_to"]
    requested_span = abs(y_to - y_from)
    actual_span    = abs(actual_to - actual_from)
    print(f"  tracing Y                  : {actual_from:.1f} → {actual_to:.1f} mm "
          f"(span {actual_span:.1f} mm)")
    if actual_span < requested_span - 1.0:
        print(f"  [CLIPPED] requested {requested_span:.0f} mm but only "
              f"{actual_span:.0f} mm is reachable — the rover drives along X, "
              f"so it cannot extend the line along Y.")
    print(f"  segments / rover stops     : {len(segs)}")
    rate = f"{1 / args.delay:.0f} Hz" if args.delay > 0 else "unthrottled"
    print(f"  arm waypoints per segment  : {args.steps} @ {rate}")
    if not use_rover:
        print("  [note] --no-rover: limited to the fixed-base reach")
    elif args.rover_noop:
        print(f"  [note] --rover-noop: rover commands logged only. The arm "
              f"assumes the base is parked at X={args.x:.0f} mm — position it "
              f"there first or the traced line will be offset in X.")
    print("  [note] geometric limits only — torque will reduce real reach\n")

    if args.plot:
        plot_plan(segs, args.x, args.z, use_rover)

    # ── Hardware ──────────────────────────────────────────────────────────────
    ser = rover = rover_ctx = None
    if not args.dry_run:
        try:
            import serial
            ser = serial.Serial(args.arm_port, 115200, timeout=1)
            print(f"Arm serial opened: {args.arm_port}")
            time.sleep(2)   # allow ESP32 to boot after serial open
        except Exception as e:
            sys.exit(f"Cannot open arm port {args.arm_port}: {e}")

        if use_rover and not args.rover_noop:
            try:
                sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
                from roomba_controller import RoombaController
                rover = RoombaController(port=args.rover_port)
                rover_ctx = rover.connect()
                rover_ctx.__enter__()
                print(f"Rover connected: {args.rover_port}")
            except Exception as e:
                if ser:
                    ser.close()
                sys.exit(f"Cannot connect rover on {args.rover_port}: {e}")

    rover_x = start_x = 0.0
    try:
        for i, seg in enumerate(segs):
            print(f"Segment {i + 1}/{len(segs)}  "
                  f"Y {seg['y_from']:+.1f} → {seg['y_to']:+.1f} mm")
            if use_rover:
                rover_x = drive_rover(rover, seg["rover_x"], rover_x,
                                      args.rover_velocity, args.dry_run,
                                      args.rover_noop)
            pts = segment_waypoints(seg, args.x, args.z, args.steps, args.ease)
            spd = args.speed if args.speed is not None else auto_speed(pts, args.delay)
            print(f"  ARM   → sweep {len(pts)} waypoints at arm-frame "
                  f"X'={pts[0][0]:.1f} mm  (spd={spd})")
            if spd == 0:
                print("    [warn] spd=0 means MAXIMUM speed — the arm will jump "
                      "between targets rather than trace the line.")
            stream_arm(ser, pts, spd, args.delay, args.dry_run)

        if args.return_home and use_rover:
            print("Returning rover to start")
            drive_rover(rover, start_x, rover_x, args.rover_velocity,
                        args.dry_run, args.rover_noop)

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
