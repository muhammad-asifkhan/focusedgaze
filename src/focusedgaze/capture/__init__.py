"""Frame sources: where images come from, kept out of the pipeline.

    from focusedgaze.capture import WebcamSource, FrameSequenceSource

    with WebcamSource() as camera:
        while (frame := camera.read()) is not None:
            ...   # frame.image is BGR, frame.timestamp is seconds

Everything here satisfies :class:`~focusedgaze.capture.base.FrameSource`, which
is a structural protocol rather than a base class, so an application that already
owns a camera can pass its own object anywhere one of these is accepted.

Which source to use:

* :class:`~focusedgaze.capture.webcam.WebcamSource` for live tracking. Captures
  on its own thread and holds only the newest frame, so slow inference never
  stalls the camera. It DROPS frames by design.
* :class:`~focusedgaze.capture.video.VideoFileSource` to replay a recording.
* :class:`~focusedgaze.capture.video.FrameSequenceSource` to replay frames
  already in memory. Needs no codec, so it is the one that behaves identically
  everywhere, and it is what the Tier 2 fixture is replayed through.

* :class:`~focusedgaze.capture.tracker.WebcamGazeTracker` composes a source with
  a :class:`~focusedgaze.core.estimator.GazeEstimator` and owns the camera. It is
  the convenience layer; the estimator is the foundation.
"""

from __future__ import annotations

from .base import Frame, FrameSource, FrameSourceBase
from .tracker import WebcamGazeTracker
from .video import DEFAULT_REPLAY_FPS, FrameSequenceSource, VideoFileSource
from .webcam import DEFAULT_READ_TIMEOUT_S, WebcamSource, resolve_backend

__all__ = [
    "DEFAULT_READ_TIMEOUT_S",
    "DEFAULT_REPLAY_FPS",
    "Frame",
    "FrameSequenceSource",
    "FrameSource",
    "FrameSourceBase",
    "VideoFileSource",
    "WebcamGazeTracker",
    "WebcamSource",
    "resolve_backend",
]
