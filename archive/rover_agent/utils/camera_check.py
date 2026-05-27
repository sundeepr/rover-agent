"""
Camera validation utility.

Captures a single frame from the rover camera, optionally saves it to disk,
and displays it in a blocking tkinter window so the operator can visually
confirm the camera is working before the navigation loop starts.

Display backends (tried in order):
  1. tkinter + Pillow  — proper interactive window with Continue/Quit buttons
  2. Pillow only       — Image.show() via the system's default viewer (non-blocking)
  3. Save to file only — fallback when no display is available (headless / SSH)
"""

from __future__ import annotations

import io
import logging
import sys
from datetime import datetime
from pathlib import Path

from rover_agent.hardware.base import Camera

logger = logging.getLogger(__name__)


def capture_and_display(
    camera: Camera,
    save_dir: Path | None = None,
    window_title: str = "Rover Camera Check",
) -> bytes:
    """
    Capture one frame, display it, and return the raw JPEG bytes.

    Args:
        camera:       Camera instance to capture from.
        save_dir:     Directory to save the captured image (None = don't save).
        window_title: Title for the display window.

    Returns:
        The raw JPEG bytes of the captured frame.

    Raises:
        SystemExit: The user clicked Quit in the camera-check window.
    """
    logger.info("Camera check: capturing frame…")
    image_bytes = camera.capture_image()
    logger.info("Camera check: captured %d bytes.", len(image_bytes))

    if save_dir is not None:
        _save_image(image_bytes, save_dir)

    _display(image_bytes, window_title)
    return image_bytes


# ── Display helpers ────────────────────────────────────────────────────────────

def _display(image_bytes: bytes, title: str) -> None:
    """Try each display backend in order of preference."""
    if _try_tkinter(image_bytes, title):
        return
    if _try_pillow_show(image_bytes, title):
        logger.warning("Displayed via system viewer (non-blocking). Check the image and press Enter to continue.")
        input("  → Press Enter to continue…")
        return
    logger.warning(
        "No display available (headless environment?). "
        "Use --save-camera-check to write the image to disk and inspect it manually."
    )


def _try_tkinter(image_bytes: bytes, title: str) -> bool:
    """
    Show a blocking window with Continue / Quit buttons.
    Returns True if the window was displayed successfully.
    """
    try:
        import tkinter as tk
        from PIL import Image, ImageTk  # type: ignore[import]
    except ImportError:
        return False

    try:
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size

        root = tk.Tk()
        root.title(title)
        root.resizable(False, False)

        # ── Image label ───────────────────────────────────────────────────────
        photo = ImageTk.PhotoImage(img)
        img_label = tk.Label(root, image=photo)
        img_label.pack(padx=8, pady=8)

        # ── Info bar ──────────────────────────────────────────────────────────
        info = tk.Label(
            root,
            text=f"{w} × {h} px  |  {len(image_bytes) / 1024:.1f} KB",
            fg="grey",
        )
        info.pack()

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=8)

        result: list[str] = []  # mutable container to capture button choice

        def on_continue() -> None:
            result.append("continue")
            root.destroy()

        def on_quit() -> None:
            result.append("quit")
            root.destroy()

        tk.Button(btn_frame, text="Continue →", command=on_continue, width=14).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Quit", command=on_quit, fg="red", width=8).pack(side=tk.LEFT, padx=4)

        root.protocol("WM_DELETE_WINDOW", on_quit)
        root.mainloop()

        if result and result[0] == "quit":
            logger.info("User chose to quit at camera check.")
            sys.exit(0)

        return True

    except Exception as exc:
        logger.debug("tkinter display failed: %s", exc)
        return False


def _try_pillow_show(image_bytes: bytes, title: str) -> bool:
    """Fallback: open image with the OS default viewer (non-blocking)."""
    try:
        from PIL import Image  # type: ignore[import]
        img = Image.open(io.BytesIO(image_bytes))
        img.show(title=title)
        return True
    except Exception as exc:
        logger.debug("Pillow Image.show() failed: %s", exc)
        return False


# ── File save helper ───────────────────────────────────────────────────────────

def _save_image(image_bytes: bytes, save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = save_dir / f"camera_check_{timestamp}.jpg"
    out_path.write_bytes(image_bytes)
    logger.info("Camera check image saved: %s", out_path)
    return out_path
