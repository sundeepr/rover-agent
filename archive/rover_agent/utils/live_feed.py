"""
LiveFeedCamera — a Camera decorator that shows each captured frame in a window.

Wraps any Camera implementation. Every call to capture_image() forwards to
the inner camera, then updates the display window with the new frame.

The window runs on the main thread using non-blocking root.update() calls,
so it stays responsive while the navigation loop is running.
"""

from __future__ import annotations

import io
import logging

from rover_agent.hardware.base import Camera

logger = logging.getLogger(__name__)


class LiveFeedCamera(Camera):
    """
    Wraps a Camera and displays each captured frame in a tkinter window.

    Args:
        inner:  The real camera (MockCamera, RpiCamera, etc.)
        title:  Window title.
    """

    def __init__(self, inner: Camera, title: str = "Rover Camera Feed") -> None:
        self._inner = inner
        self._title = title
        self._root = None
        self._label = None
        self._photo = None  # must keep a reference or tkinter GCs it
        self._frame_count = 0
        self._info_var = None
        self._setup_window()

    def capture_image(self) -> bytes:
        image_bytes = self._inner.capture_image()
        self._frame_count += 1
        self._update_frame(image_bytes)
        return image_bytes

    def close(self) -> None:
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass
            self._root = None

    # ── Private ───────────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        try:
            import tkinter as tk
            from PIL import Image, ImageTk  # noqa: F401  (verify importable)
        except ImportError as exc:
            logger.warning(
                "Live feed window unavailable (%s). "
                "Install pillow to enable: pip install pillow",
                exc,
            )
            return

        self._root = tk.Tk()
        self._root.title(self._title)
        self._root.resizable(True, True)

        # Image label — filled on first capture
        self._label = tk.Label(self._root, bg="black")
        self._label.pack(fill=tk.BOTH, expand=True)

        # Status bar at the bottom
        self._info_var = tk.StringVar(value="Waiting for first frame…")
        tk.Label(
            self._root,
            textvariable=self._info_var,
            anchor="w",
            fg="grey",
            padx=6,
        ).pack(fill=tk.X, side=tk.BOTTOM)

        # Handle window close — just hide rather than crash the agent
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.update()

    def _update_frame(self, image_bytes: bytes) -> None:
        if self._root is None or self._label is None:
            return
        try:
            from PIL import Image, ImageTk

            img = Image.open(io.BytesIO(image_bytes))
            w, h = img.size

            self._photo = ImageTk.PhotoImage(img)
            self._label.configure(image=self._photo)

            self._info_var.set(
                f"Frame {self._frame_count}  |  {w} × {h} px  |  {len(image_bytes) / 1024:.1f} KB"
            )

            self._root.update()
        except Exception as exc:
            logger.debug("Live feed update failed: %s", exc)

    def _on_close(self) -> None:
        """User closed the window — hide it but keep the agent running."""
        logger.info("Camera feed window closed.")
        if self._root is not None:
            self._root.withdraw()
