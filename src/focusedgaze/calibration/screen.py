"""The full-screen canvas: the one thing calibration was missing.

:mod:`~focusedgaze.calibration.ui` collects samples and deliberately does not
draw, because drawing would make it need a GUI toolkit and the collection is the
part worth reusing. Its ``collect_pursuit_samples`` takes an ``on_frame`` hook
documented as "where a preview window or a full-screen canvas hooks in". This
module is that canvas, and it is the whole reason ``focusedgaze calibrate`` could
not run a session.

WHY THE WINDOW IS BEHIND A PROTOCOL
-----------------------------------
Drawing onto an array and putting that array on a screen are different problems.
The first is arithmetic and belongs in CI; the second needs a display and cannot
go there. So :class:`DotRenderer` composes the canvas with ordinary OpenCV calls
on a NumPy array -- which work headless, because no window is involved -- and
hands it to a :class:`WindowBackend` that is the only part needing a screen.

CI drives a recording backend and asserts on the pixels. That follows the
injection already used across this package (``providers=`` in diagnostics,
``clock=`` in ui, ``source=`` in the tracker) and it is what lets the dot's
position be a tested fact rather than something verified by looking at it.

ESCAPE IS CHECKED ON EVERY DRAW, NOT BY THE CALLER
---------------------------------------------------
Every drawing method pumps the window's event queue and raises
:class:`~focusedgaze.exceptions.CalibrationAborted` if the user pressed Escape.
Two reasons it is not left to the caller: OpenCV does not paint at all until
``waitKey`` runs, so a caller who forgets it gets a frozen black screen rather
than an obvious error; and a 45-second sweep the user cannot stop is a trap.
Making both automatic means neither can be forgotten.

RESOLUTION IS FOR THE RECORD, NOT FOR THE ARITHMETIC
-----------------------------------------------------
Targets are normalised, and a full-screen window stretches whatever canvas it is
given to fill the display. So a wrong pixel count moves no dot: it scales
uniformly and the dot still lands at the same *fraction* of the screen, which is
the only thing the fit is taught. The size matters for the aspect ratio and for
the ``screen_size`` recorded in the profile, which is why this module does not
call ``SetProcessDPIAware`` to sharpen the number -- that changes DPI handling
for the entire process, and buying metadata accuracy with a global side effect on
someone else's application is the wrong trade.
"""

from __future__ import annotations

import logging
import sys
from types import TracebackType
from typing import Final, Protocol, Self

import cv2
import numpy as np
from numpy.typing import NDArray

from ..exceptions import CalibrationAborted, CalibrationError

__all__ = [
    "ABORT_KEYS",
    "DEFAULT_SCREEN_SIZE",
    "Cv2Window",
    "DotRenderer",
    "WindowBackend",
    "screen_size_px",
    "target_to_pixels",
]

_log = logging.getLogger(__name__)

#: Used only when every way of asking the operating system has failed. Warned
#: about, because a guessed aspect ratio is the one part of a wrong size that a
#: full-screen stretch does not absorb.
DEFAULT_SCREEN_SIZE: Final[tuple[int, int]] = (1920, 1080)

#: Escape and q. Both, because Escape is the convention and q is what people who
#: have used the OpenCV demos reach for.
ABORT_KEYS: Final[frozenset[int]] = frozenset({27, ord("q"), ord("Q")})

_BLACK: Final[tuple[int, int, int]] = (0, 0, 0)
_WHITE: Final[tuple[int, int, int]] = (255, 255, 255)
_DIM: Final[tuple[int, int, int]] = (110, 110, 110)
_ACCENT: Final[tuple[int, int, int]] = (60, 60, 220)  # BGR: red centre
_FONT: Final[int] = cv2.FONT_HERSHEY_SIMPLEX

#: Dot geometry, in pixels at 1080p and scaled with the screen. Big enough to
#: follow at speed, small enough that its centre is an unambiguous fixation
#: point: the polynomial is taught the dot's centre, so a vague target is a
#: mislabelled sample.
_DOT_RADIUS_AT_1080P: Final[int] = 20
_CENTRE_RADIUS_AT_1080P: Final[int] = 5


def screen_size_px() -> tuple[int, int]:
    """The display size in pixels, as ``(width, height)``.

    Tries the OS first and tkinter second, then falls back to
    :data:`DEFAULT_SCREEN_SIZE` with a warning. Never raises: a calibration that
    refused to start because it could not measure the screen would be failing on
    the one number it can most afford to be wrong about (see the module
    docstring).
    """
    if sys.platform == "win32":  # pragma: no cover - platform-specific
        try:
            import ctypes

            user32 = ctypes.windll.user32
            width = int(user32.GetSystemMetrics(0))
            height = int(user32.GetSystemMetrics(1))
            if width > 0 and height > 0:
                return width, height
        except Exception as exc:  # noqa: BLE001 - any failure means "ask elsewhere"
            _log.debug("GetSystemMetrics failed, falling back: %s", exc)

    try:  # pragma: no cover - depends on a display being present
        import tkinter

        root = tkinter.Tk()
        try:
            width = int(root.winfo_screenwidth())
            height = int(root.winfo_screenheight())
        finally:
            root.destroy()
        if width > 0 and height > 0:
            return width, height
    except Exception as exc:  # noqa: BLE001 - headless, or no tkinter
        _log.debug("tkinter screen size failed, falling back: %s", exc)

    _log.warning(
        "Could not determine the screen size; assuming %dx%d. The dot still "
        "lands at the right fraction of the screen, but the aspect ratio may be "
        "wrong and the profile will record a guess.",
        *DEFAULT_SCREEN_SIZE,
    )
    return DEFAULT_SCREEN_SIZE


def target_to_pixels(
    target: tuple[float, float], size: tuple[int, int]
) -> tuple[int, int]:
    """Map a normalised target onto pixel coordinates, clamped to the canvas.

    Args:
        target: ``(x, y)`` with the origin at the top left, as everything else in
            this package uses.
        size: ``(width, height)`` of the canvas.

    Pure, and separate from the drawing, because "the dot was where it was
    labelled" is the single assertion that keeps the fit honest and it should be
    checkable without a screen.
    """
    width, height = size
    if width < 1 or height < 1:
        raise CalibrationError(f"a canvas needs positive dimensions, got {size}")
    x = round(float(target[0]) * (width - 1))
    y = round(float(target[1]) * (height - 1))
    return (min(max(x, 0), width - 1), min(max(y, 0), height - 1))


class WindowBackend(Protocol):
    """The part that needs a display. Everything else is arithmetic."""

    def open(self, title: str) -> None:
        """Create the window, full-screen."""

    def show(self, title: str, image: NDArray[np.uint8]) -> None:
        """Put a composed canvas on it."""

    def wait_key(self, delay_ms: int) -> int:
        """Pump the event queue; return a key code, or -1 for none."""

    def close(self, title: str) -> None:
        """Destroy the window."""


class Cv2Window:
    """The real backend: one OpenCV window, full-screen."""

    __slots__ = ()

    def open(self, title: str) -> None:
        try:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except cv2.error as exc:
            raise CalibrationError(
                "Could not open a full-screen window, so there is nowhere to draw "
                "the calibration dot.\n"
                "\n"
                "This needs a desktop session. Over SSH, in a container, or on a "
                "headless machine there is no display to use. Collect samples "
                "programmatically with focusedgaze.calibration.ui instead, then fit "
                "them with:\n"
                "    focusedgaze calibrate --from-samples samples.json\n"
                f"\nOpenCV reported: {exc}"
            ) from exc

    def show(self, title: str, image: NDArray[np.uint8]) -> None:
        cv2.imshow(title, image)

    def wait_key(self, delay_ms: int) -> int:
        # Masked to a byte: waitKey returns platform-dependent high bits and a
        # raw comparison against 27 fails on some builds when modifiers are held.
        return cv2.waitKey(delay_ms) & 0xFF

    def close(self, title: str) -> None:
        try:
            cv2.destroyWindow(title)
            # Window teardown on Windows is deferred until the event queue runs,
            # so without this the window survives the process's next blocking
            # call and sits on top of whatever the user does next.
            cv2.waitKey(1)
        except cv2.error as exc:  # pragma: no cover - teardown must not mask errors
            _log.debug("window teardown reported %s", exc)


class DotRenderer:
    """A full-screen canvas showing one dot, or a page of text.

    Args:
        size: Canvas size in pixels. Defaults to :func:`screen_size_px`.
        title: Window title. One renderer owns one window.
        backend: The display. Defaults to :class:`Cv2Window`; CI passes a fake.

    Use it as a context manager so the window is always destroyed, including when
    the sweep raises::

        with DotRenderer() as screen:
            screen.draw_dot((0.5, 0.5))

    That mirrors the camera discipline in
    :class:`~focusedgaze.capture.tracker.WebcamGazeTracker`: a full-screen window
    that outlives its failure covers the user's desktop with nothing to click.
    """

    __slots__ = ("_backend", "_height", "_open", "_title", "_width")

    def __init__(
        self,
        size: tuple[int, int] | None = None,
        *,
        title: str = "focusedgaze calibration",
        backend: WindowBackend | None = None,
    ) -> None:
        width, height = size if size is not None else screen_size_px()
        if width < 1 or height < 1:
            raise CalibrationError(f"a canvas needs positive dimensions, got {size}")
        self._width = int(width)
        self._height = int(height)
        self._title = title
        self._backend: WindowBackend = backend if backend is not None else Cv2Window()
        self._open = False

    @property
    def size(self) -> tuple[int, int]:
        """Canvas size as ``(width, height)``."""
        return (self._width, self._height)

    def _scaled(self, at_1080p: int) -> int:
        """Scale a pixel measurement taken at 1080p to this screen."""
        return max(2, round(at_1080p * self._height / 1080))

    def _canvas(self) -> NDArray[np.uint8]:
        """A black frame. Black because a bright field constricts the pupil and
        the landmarker has less to work with."""
        return np.zeros((self._height, self._width, 3), dtype=np.uint8)

    def _present(self, canvas: NDArray[np.uint8]) -> int:
        """Show a canvas, pump the queue, and honour an abort.

        Returns the key pressed, or -1. Raises
        :class:`~focusedgaze.exceptions.CalibrationAborted` on Escape or q.
        """
        if not self._open:
            raise CalibrationError(
                "DotRenderer must be used as a context manager: `with DotRenderer() as s:`"
            )
        self._backend.show(self._title, canvas)
        key = self._backend.wait_key(1)
        if key in ABORT_KEYS:
            raise CalibrationAborted("stopped by the user")
        return key

    def draw_dot(
        self,
        target: tuple[float, float],
        *,
        filled: bool = True,
        progress: float | None = None,
    ) -> int:
        """Draw the dot at a normalised target.

        Args:
            target: ``(x, y)`` in ``[0, 1]``, origin top left.
            filled: Solid while samples are being taken, hollow while waiting.
                The accuracy grid uses the distinction to show the dwell phase,
                so a user knows when they are being measured.
            progress: ``[0, 1]``, drawn as a thin bar along the bottom edge.

        No caption is drawn beside the dot on purpose: text next to the target
        pulls the eyes off it, and the sample is then labelled with a position
        the user was not looking at. Progress goes to the screen edge, far from
        anywhere the dot travels.
        """
        canvas = self._canvas()
        centre = target_to_pixels(target, self.size)
        radius = self._scaled(_DOT_RADIUS_AT_1080P)

        if filled:
            cv2.circle(canvas, centre, radius, _WHITE, -1, lineType=cv2.LINE_AA)
            cv2.circle(
                canvas, centre, self._scaled(_CENTRE_RADIUS_AT_1080P), _ACCENT, -1,
                lineType=cv2.LINE_AA,
            )
        else:
            cv2.circle(
                canvas, centre, radius, _WHITE, max(2, self._scaled(3)),
                lineType=cv2.LINE_AA,
            )

        if progress is not None:
            self._draw_progress(canvas, progress)
        return self._present(canvas)

    def _draw_progress(self, canvas: NDArray[np.uint8], fraction: float) -> None:
        """A thin bar on the bottom edge, dim enough not to attract a glance."""
        clamped = min(max(float(fraction), 0.0), 1.0)
        bar_height = self._scaled(4)
        top = self._height - bar_height
        filled_to = round(clamped * self._width)
        if filled_to > 0:
            cv2.rectangle(canvas, (0, top), (filled_to, self._height), _DIM, -1)

    def draw_message(
        self, lines: list[str], *, headline: str | None = None
    ) -> int:
        """Draw centred text. Used by the pre-flight check, never during a sweep.

        Args:
            lines: Body text, one entry per line.
            headline: Larger first line, for the state the user must act on.
        """
        canvas = self._canvas()
        head_scale = self._height / 1080 * 1.6
        body_scale = self._height / 1080 * 0.9
        thickness = max(1, self._scaled(2) // 2)

        blocks: list[tuple[str, float, tuple[int, int, int]]] = []
        if headline is not None:
            blocks.append((headline, head_scale, _WHITE))
        blocks.extend((line, body_scale, _DIM) for line in lines)

        spacing = self._scaled(58)
        total = spacing * len(blocks)
        y = (self._height - total) // 2 + spacing

        for text, scale, colour in blocks:
            (text_w, _), _ = cv2.getTextSize(text, _FONT, scale, thickness)
            x = max(0, (self._width - text_w) // 2)
            cv2.putText(
                canvas, text, (x, y), _FONT, scale, colour, thickness,
                lineType=cv2.LINE_AA,
            )
            y += spacing
        return self._present(canvas)

    def __enter__(self) -> Self:
        self._backend.open(self._title)
        self._open = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._open = False
        self._backend.close(self._title)
