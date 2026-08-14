"""The part that needs a screen, a camera, and a person.

Referenced by `test_calibration_ui.py` since it was written; it did not exist
until the renderer did, because there was nothing to point it at.

**Deselected by default.** `pyproject.toml` sets ``addopts = -m 'not hardware'``,
so none of this runs in CI or in an ordinary `pytest` invocation. Run it
deliberately:

    pytest -m hardware -q -rs

`test_calibration_screen.py` covers the drawing against a recording backend and
is the file that runs everywhere. What is left here is exactly what a fake cannot
answer: whether a real window appears on a real display, and whether a real
camera pointed at a real face yields samples the fitter accepts. A mocked screen
would test the mock, which is the trap the `ui` module's docstring names.
"""

from __future__ import annotations

import pytest

from focusedgaze.calibration.screen import DotRenderer, screen_size_px
from focusedgaze.calibration.ui import (
    collect_pursuit_samples,
    pursuit_path,
    summarise_coverage,
)

pytestmark = pytest.mark.hardware


def test_a_real_window_opens_and_draws() -> None:
    """No person needed, but a desktop session is. The first thing to check when
    calibration "does nothing": if this fails, nothing downstream can work."""
    width, height = screen_size_px()
    assert width > 0 and height > 0

    with DotRenderer() as screen:
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            screen.draw_dot((fraction, fraction))
    # Reaching here means the window opened, painted five times and was
    # destroyed. A hung window fails by timing out rather than by asserting.


def test_the_countdown_and_guidance_screens_render() -> None:
    """The pre-flight text is the only thing standing between a user and a
    45-second sweep collected from the wrong distance."""
    with DotRenderer() as screen:
        screen.draw_message(["You are at 80 cm.", "Calibration needs 45 to 65 cm."],
                            headline="Move closer")
        screen.draw_message(["Follow the dot with your eyes"], headline="3")


def test_a_short_real_sweep_collects_usable_samples() -> None:
    """Needs a camera AND a person following the dot.

    The assertion is deliberately weak on count and strong on shape: how many
    samples a real sweep yields depends on the machine's frame rate and on
    whether the person blinked, so a tight number here would fail for reasons
    that are not defects. What must hold is that samples come back labelled with
    where the dot was.
    """
    from focusedgaze.capture import WebcamGazeTracker

    path = pursuit_path(seconds=6.0)
    with WebcamGazeTracker(profile=None) as tracker, DotRenderer() as screen:
        samples = collect_pursuit_samples(
            tracker, path, on_frame=lambda target, _r: screen.draw_dot(target)
        )

    assert samples, "no usable samples: check lighting, distance and that a face is visible"
    for pitch, yaw, x, y in samples:
        assert pitch is not None and yaw is not None
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0

    coverage = summarise_coverage(samples)
    assert coverage.total == len(samples)
    # Printed rather than asserted: a six-second sweep is too short to cover the
    # screen, and this is the number a human should look at when a calibration
    # comes out worse in one region than another.
    print("\n".join(coverage.lines()))
