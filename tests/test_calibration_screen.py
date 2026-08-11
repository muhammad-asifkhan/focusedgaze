"""The full-screen canvas.

`calibration/screen.py` splits drawing from displaying precisely so this file can
exist: the canvas is composed with ordinary OpenCV calls onto a NumPy array,
which needs no display, and only the window sits behind a backend. So every
assertion here is about pixels that were actually drawn, not about a mock having
been called.

The one thing worth testing hardest is that the dot lands where it was labelled.
A sweep teaches the polynomial "this gaze angle means that screen position", so a
dot drawn somewhere other than its target does not fail: it trains a confidently
wrong model, which is the failure mode `ui.py` is written all over to avoid.
"""

from __future__ import annotations

import numpy as np
import pytest

from focusedgaze.calibration.screen import (
    ABORT_KEYS,
    DotRenderer,
    screen_size_px,
    target_to_pixels,
)
from focusedgaze.exceptions import CalibrationAborted, CalibrationError


class _Window:
    """Records what it was asked to display, and replays scripted key presses."""

    def __init__(self, keys: list[int] | None = None) -> None:
        self.shown: list[np.ndarray] = []
        self.opened: list[str] = []
        self.closed: list[str] = []
        self._keys = list(keys or [])

    def open(self, title: str) -> None:
        self.opened.append(title)

    def show(self, title: str, image: np.ndarray) -> None:
        self.shown.append(image.copy())

    def wait_key(self, delay_ms: int) -> int:
        return self._keys.pop(0) if self._keys else -1

    def close(self, title: str) -> None:
        self.closed.append(title)


# ---------------------------------------------------------------------------
# The mapping. Pure, and the thing the fit's correctness rests on.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ((0.0, 0.0), (0, 0)),
        ((1.0, 1.0), (1919, 1079)),
        ((0.5, 0.5), (960, 540)),
        ((0.25, 0.75), (480, 809)),
    ],
)
def test_targets_map_to_the_expected_pixels(target, expected) -> None:
    assert target_to_pixels(target, (1920, 1080)) == expected


def test_the_origin_is_top_left_and_y_counts_downward() -> None:
    """Matching screens and image coordinates, not graphs. Getting this upside
    down would invert the vertical half of every calibration."""
    _, top = target_to_pixels((0.5, 0.0), (1920, 1080))
    _, bottom = target_to_pixels((0.5, 1.0), (1920, 1080))
    assert top < bottom


def test_out_of_range_targets_are_clamped_onto_the_canvas() -> None:
    assert target_to_pixels((-0.5, 2.0), (800, 600)) == (0, 599)


def test_a_canvas_with_no_area_is_rejected() -> None:
    with pytest.raises(CalibrationError):
        target_to_pixels((0.5, 0.5), (0, 100))


# ---------------------------------------------------------------------------
# Drawing.
# ---------------------------------------------------------------------------


def test_the_dot_is_drawn_at_its_target() -> None:
    """The single assertion that keeps a sweep's labels honest."""
    window = _Window()
    with DotRenderer((640, 480), backend=window) as screen:
        screen.draw_dot((0.25, 0.75))

    canvas = window.shown[0]
    x, y = target_to_pixels((0.25, 0.75), (640, 480))
    assert canvas[y, x].any(), "nothing was drawn at the target"
    assert not canvas[5, 5].any(), "the far corner should be black"


def test_the_solid_dot_carries_a_distinct_centre() -> None:
    """People fixate the centre, so the centre is what the label means.

    Sized at 1080p so the dot's geometry is its documented one: the radii scale
    with screen height, and a probe at a fixed offset would be measuring the
    scaling rather than the drawing.
    """
    window = _Window()
    with DotRenderer((1920, 1080), backend=window) as screen:
        screen.draw_dot((0.5, 0.5), filled=True)

    canvas = window.shown[0]
    x, y = target_to_pixels((0.5, 0.5), (1920, 1080))
    centre = canvas[y, x]
    ring = canvas[y, x + 12]  # outside the 5px centre, inside the 20px disc
    assert not np.array_equal(centre, ring), "centre and ring should differ"
    assert ring.min() > 200, "the ring should be near-white"


def test_hollow_and_filled_dots_differ_at_the_centre() -> None:
    """`accuracy` uses this to show the dwell phase; a user who cannot tell the
    two apart cannot know when they are being measured."""
    window = _Window()
    with DotRenderer((640, 480), backend=window) as screen:
        screen.draw_dot((0.5, 0.5), filled=True)
        screen.draw_dot((0.5, 0.5), filled=False)

    x, y = target_to_pixels((0.5, 0.5), (640, 480))
    assert window.shown[0][y, x].any(), "a filled dot has a painted centre"
    assert not window.shown[1][y, x].any(), "a hollow dot has an empty centre"


def test_progress_is_drawn_only_as_far_as_it_has_got() -> None:
    window = _Window()
    with DotRenderer((1000, 400), backend=window) as screen:
        screen.draw_dot((0.5, 0.5), progress=0.25)

    bottom = window.shown[0][399]
    assert bottom[10].any(), "the bar should be painted at the start"
    assert not bottom[900].any(), "the bar should stop at 25%"


def test_progress_is_clamped_rather_than_overflowing() -> None:
    window = _Window()
    with DotRenderer((200, 200), backend=window) as screen:
        screen.draw_dot((0.5, 0.5), progress=5.0)
    assert window.shown[0][199].any()


def test_a_message_puts_ink_on_the_canvas() -> None:
    window = _Window()
    with DotRenderer((800, 600), backend=window) as screen:
        screen.draw_message(["sit back a little"], headline="Too close")
    assert window.shown[0].any(), "the message screen should not be blank"


def test_the_sweep_screen_is_black_behind_the_dot() -> None:
    """A bright field constricts the pupil and leaves the landmarker less to
    work with, so the background is not a cosmetic choice."""
    window = _Window()
    with DotRenderer((320, 240), backend=window) as screen:
        screen.draw_dot((0.9, 0.9))
    assert window.shown[0][0:50, 0:50].sum() == 0


# ---------------------------------------------------------------------------
# The window's lifetime, and getting out.
# ---------------------------------------------------------------------------


def test_the_window_is_opened_and_destroyed() -> None:
    window = _Window()
    with DotRenderer((320, 240), backend=window):
        pass
    assert window.opened == ["focusedgaze calibration"]
    assert window.closed == ["focusedgaze calibration"]


def test_the_window_is_destroyed_even_when_the_body_raises() -> None:
    """A full-screen window that outlives its failure covers the desktop with
    nothing to click, which is the state R-12 left the camera in."""
    window = _Window()
    with pytest.raises(ZeroDivisionError), DotRenderer((320, 240), backend=window):
        raise ZeroDivisionError
    assert window.closed == ["focusedgaze calibration"]


@pytest.mark.parametrize("key", sorted(ABORT_KEYS))
def test_every_abort_key_stops_the_run(key: int) -> None:
    window = _Window(keys=[key])
    with pytest.raises(CalibrationAborted), DotRenderer((320, 240), backend=window) as screen:
        screen.draw_dot((0.5, 0.5))
    assert window.closed, "the window must still be destroyed on abort"


def test_an_ordinary_key_does_not_stop_the_run() -> None:
    window = _Window(keys=[ord("z")])
    with DotRenderer((320, 240), backend=window) as screen:
        screen.draw_dot((0.5, 0.5))
        screen.draw_dot((0.6, 0.6))
    assert len(window.shown) == 2


def test_drawing_outside_the_context_manager_is_refused() -> None:
    """Without `with`, OpenCV never pumps its queue and the screen stays black.
    Failing loudly beats a window that looks hung."""
    screen = DotRenderer((320, 240), backend=_Window())
    with pytest.raises(CalibrationError, match="context manager"):
        screen.draw_dot((0.5, 0.5))


def test_a_renderer_reports_the_size_it_was_given() -> None:
    """The profile records this, so it has to be the canvas actually used."""
    assert DotRenderer((1280, 1024), backend=_Window()).size == (1280, 1024)


def test_a_canvas_with_no_area_is_refused_at_construction() -> None:
    with pytest.raises(CalibrationError):
        DotRenderer((0, 0), backend=_Window())


def test_screen_size_is_always_usable() -> None:
    """It must never raise: a calibration that refused to start because it could
    not measure the screen would be failing on the number it can most afford to
    get wrong, since a full-screen stretch absorbs it."""
    width, height = screen_size_px()
    assert width > 0 and height > 0
