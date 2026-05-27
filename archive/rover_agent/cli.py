"""
Command-line entry point for the rover navigation agent.

Usage:
    # Anthropic Claude (default)
    python -m rover_agent.cli "navigate to the red cone"

    # Google Gemini
    python -m rover_agent.cli "navigate to the red cone" --provider gemini

    # Specific model
    python -m rover_agent.cli "find the cone" --provider gemini --model gemini-2.5-pro

    # Mock hardware for development / testing
    python -m rover_agent.cli "find the blue box" --mock --mock-images tests/fixtures/

    # Full options
    python -m rover_agent.cli --help
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# API key env-var for each provider
_PROVIDER_API_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini":    "GEMINI_API_KEY",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rover-agent",
        description="LLM-guided rover navigation agent",
    )
    p.add_argument("goal", help='High-level navigation goal, e.g. "navigate to the red cone"')

    hw = p.add_argument_group("Hardware")
    hw.add_argument(
        "--mock",
        action="store_true",
        help="Use mock hardware instead of real Raspberry Pi hardware",
    )
    hw.add_argument(
        "--mock-images",
        metavar="PATH",
        default="tests/fixtures/",
        help="Directory (or single JPEG file) for the mock camera (default: tests/fixtures/)",
    )

    llm = p.add_argument_group("LLM")
    llm.add_argument(
        "--provider",
        default="gemini",
        choices=list(_PROVIDER_API_KEY_ENV.keys()),
        help="LLM provider to use (default: gemini)",
    )
    llm.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help=(
            "Model name for the chosen provider. "
            "Anthropic examples: claude-sonnet-4-6, claude-opus-4-6. "
            "Gemini examples: gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash."
        ),
    )
    llm.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="API key override (falls back to the provider's env-var if omitted)",
    )

    cam = p.add_argument_group("Camera feed")
    cam.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the live camera feed window (useful for headless / SSH runs)",
    )

    nav = p.add_argument_group("Navigation")
    nav.add_argument("--max-steps", type=int, default=None, help="Maximum navigation steps")
    nav.add_argument(
        "--step-distance",
        type=float,
        default=None,
        metavar="METERS",
        help="Distance to drive per step (default from config)",
    )
    nav.add_argument(
        "--step-delay",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Pause between steps (default from config)",
    )

    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    p.add_argument(
        "--json-output",
        metavar="FILE",
        help="Write final NavigationState to a JSON file",
    )
    return p


def main() -> None:
    load_dotenv()

    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # ── API key ───────────────────────────────────────────────────────────────
    # For Gemini the SDK reads GEMINI_API_KEY automatically; passing None is fine.
    env_var = _PROVIDER_API_KEY_ENV[args.provider]
    api_key = args.api_key or os.environ.get(env_var) or None

    # ── Hardware ──────────────────────────────────────────────────────────────
    if args.mock:
        from rover_agent.hardware.mock_rover import make_mock_hardware
        hardware = make_mock_hardware(Path(args.mock_images))
    else:
        try:
            from rover_agent.hardware.rpi_rover import make_rpi_hardware
            hardware = make_rpi_hardware()
        except ImportError as exc:
            sys.exit(
                f"Error: RPi hardware libraries not installed ({exc}). "
                "Run with --mock for development, or install picamera2 and gpiozero on the Pi."
            )

    # ── LLM client ────────────────────────────────────────────────────────────
    from rover_agent.llm.factory import create_llm_client

    try:
        llm_client = create_llm_client(
            provider=args.provider,
            api_key=api_key,   # None → SDK reads from env var automatically
            model=args.model,
        )
    except (ValueError, ImportError) as exc:
        sys.exit(f"Error creating LLM client: {exc}")

    logging.getLogger(__name__).info(
        "Provider: %s | Model: %s", args.provider, args.model or "(default)"
    )

    # ── Live camera feed window ───────────────────────────────────────────────
    if not args.no_display:
        from rover_agent.utils.live_feed import LiveFeedCamera
        hardware.camera = LiveFeedCamera(hardware.camera)

    # ── Navigation loop ───────────────────────────────────────────────────────
    from rover_agent.navigation.loop import NavigationLoop

    loop_kwargs: dict = {
        "hardware": hardware,
        "llm_client": llm_client,
        "user_goal": args.goal,
    }
    if args.max_steps is not None:
        loop_kwargs["max_iterations"] = args.max_steps
    if args.step_distance is not None:
        loop_kwargs["step_distance_meters"] = args.step_distance
    if args.step_delay is not None:
        loop_kwargs["step_delay_seconds"] = args.step_delay

    nav_loop = NavigationLoop(**loop_kwargs)
    final_state = nav_loop.run()

    if hasattr(hardware.camera, "close"):
        hardware.camera.close()

    # ── Output ────────────────────────────────────────────────────────────────
    print(f"\nFinal status : {final_state.goal_status.value.upper()}")
    print(f"Steps taken  : {final_state.iteration_count}")
    print(f"Waypoints    : {len(final_state.waypoints_visited)}")

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.write_text(final_state.model_dump_json(indent=2), encoding="utf-8")
        print(f"State saved  : {out_path}")


if __name__ == "__main__":
    main()
