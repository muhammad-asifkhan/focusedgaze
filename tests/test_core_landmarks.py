"""The frame counter that feeds MediaPipe's VIDEO-mode timestamps.

THE CRASH THIS FILE EXISTS FOR
==============================
``detect_for_video`` requires strictly increasing integer milliseconds, and the
MediaPipe landmarker keeps that state itself -- it does not know when the caller
resets anything. ``FaceLandmarker.reset`` used to zero the frame counter "so
timestamps restart cleanly", which replayed stamps MediaPipe had already seen:

    ValueError: Input timestamp must be monotonically increasing.

That is not caught anywhere. It propagates out of ``tracker.read()`` and kills
the process. The trigger is ordinary -- lose the face, find it again -- and it
took out two of three calibration attempts on a real machine, always during the
positioning step, which is precisely where someone is moving into frame.

The landmarker is faked here rather than loaded: what is under test is the
counter's arithmetic and the reset's effect on it, and a fake that enforces
MediaPipe's contract tests that far more directly than a real model would.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from focusedgaze.config import LandmarkConfig
from focusedgaze.core.landmarks import FaceLandmarker

FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


class _StrictLandmarker:
    """Enforces MediaPipe's contract: stamps must strictly increase."""

    def __init__(self) -> None:
        self.stamps: list[int] = []

    def detect_for_video(self, image: object, timestamp_ms: int) -> object:
        if self.stamps and timestamp_ms <= self.stamps[-1]:
            raise ValueError("Input timestamp must be monotonically increasing.")
        self.stamps.append(timestamp_ms)
        return SimpleNamespace(face_landmarks=[])


@pytest.fixture
def landmarker(monkeypatch):
    fake = _StrictLandmarker()
    monkeypatch.setattr(FaceLandmarker, "_build", lambda self, path: fake)
    return FaceLandmarker(config=LandmarkConfig()), fake


def test_timestamps_strictly_increase_in_ordinary_use(landmarker) -> None:
    detector, fake = landmarker
    for _ in range(10):
        detector.detect(FRAME, 0.0)
    assert fake.stamps == sorted(set(fake.stamps))


def test_a_reset_does_not_rewind_the_counter(landmarker) -> None:
    """The regression. Losing the face and finding it again calls reset, and the
    very next frame used to be stamped 0."""
    detector, fake = landmarker
    for _ in range(5):
        detector.detect(FRAME, 0.0)
    before = fake.stamps[-1]

    detector.reset()
    detector.detect(FRAME, 0.0)

    assert fake.stamps[-1] > before, (
        f"the counter rewound: {fake.stamps[-1]} follows {before}"
    )


def test_repeated_tracking_loss_never_raises(landmarker) -> None:
    """A flickering face is the normal case, not an edge case: blinks, a hand
    passing, someone leaning out of frame during the positioning step."""
    detector, fake = landmarker
    for _ in range(20):
        detector.detect(FRAME, 0.0)
        detector.reset()
    assert fake.stamps == sorted(set(fake.stamps))
    assert len(fake.stamps) == 20


def test_reset_still_clears_the_crop_smoothing(landmarker) -> None:
    """The counter must survive a reset; the smoothed box must not, or the crop
    glides in from wherever the face used to be. Narrowing what reset() forgets
    must not go so far as forgetting nothing."""
    detector, _ = landmarker
    detector._box.update(10.0, 20.0, 30.0)
    assert detector._box.started is True

    detector.reset()

    assert detector._box.started is False, "reset must still forget the crop box"
