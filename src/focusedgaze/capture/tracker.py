"""WebcamGazeTracker: the convenience layer that owns a camera.

    with WebcamGazeTracker(profile="alice") as tracker:
        for result in tracker.stream():
            if result.ok:
                move_cursor(result.x, result.y)

This is deliberately thin. :class:`~focusedgaze.core.estimator.GazeEstimator` is
the foundation and this is the convenience: it opens a
:class:`~focusedgaze.capture.webcam.WebcamSource`, feeds frames through an
estimator, and guarantees the camera is released. Anything that already has
frames should use the estimator directly and never touch this class.

MIRRORING HAPPENS HERE
----------------------
The capture layer publishes frames exactly as the camera delivered them, because
a shared frame has consumers that disagree about handedness (see
`capture/webcam.py`). Somebody has to flip, and this is the layer that knows the
frames are destined for a gaze estimator.

It is not cosmetic. The calibration polynomial was fitted through a mirrored
preview, so an unmirrored frame does not mirror the output: it makes the mapping
wrong, in a way that looks like a bad calibration rather than a bug.

THE CAMERA IS ALWAYS RELEASED
-----------------------------
Including when the consumer raises out of the middle of ``stream()``. That is
the fix for R-12, whose legacy version released a handle that was never bound and
therefore raised on every exit from the loop, leaving the device held until the
process died. On Windows nothing else can open it until then.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from types import TracebackType
from typing import TYPE_CHECKING, Self

import cv2
import numpy as np
from numpy.typing import NDArray

from ..config import GazeConfig
from ..types import GazeResult
from .base import FrameSource
from .webcam import WebcamSource

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..calibration import CalibrationProfile
    from ..core.estimator import GazeEstimator
    from ..core.positioning import FocalCalibration

__all__ = ["WebcamGazeTracker"]

_log = logging.getLogger(__name__)


class WebcamGazeTracker:
    """A webcam, an estimator, and the loop between them.

    Args:
        profile: A loaded profile, a profile name to load, or ``None`` to run
            uncalibrated (readings carry pitch and yaw with a ``NOT_CALIBRATED``
            status).
        config: Everything tunable, including which camera to open.
        focal: A measured focal length for distance estimation.
        source: A pre-built frame source. Supplying one makes this usable with a
            video file or a camera the application already owns, and the tracker
            then does **not** release it: whoever opened it owns it.
        estimator: A pre-built estimator, for tests or to share model files.

    Raises:
        CameraError: The camera could not be opened.
        ModelNotFoundError: A model file is absent. Never downloaded.
        CalibrationError: A named profile could not be loaded.
    """

    __slots__ = ("_config", "_estimator", "_owns_source", "_source")

    def __init__(
        self,
        profile: CalibrationProfile | str | None = None,
        config: GazeConfig | None = None,
        *,
        focal: FocalCalibration | None = None,
        source: FrameSource | None = None,
        estimator: GazeEstimator | None = None,
    ) -> None:
        self._config = config if config is not None else GazeConfig()

        if estimator is None:
            from ..core.estimator import GazeEstimator

            estimator = GazeEstimator(
                profile=_resolve_profile(profile), config=self._config, focal=focal
            )
        self._estimator = estimator

        # Ownership decides teardown. A source handed in by the caller belongs
        # to the caller: releasing it here would close a camera the application
        # is still using, which is worse than leaking one.
        self._owns_source = source is None
        self._source = source if source is not None else WebcamSource(self._config.camera)

    @property
    def estimator(self) -> GazeEstimator:
        """The underlying estimator, for callers who want the pure layer."""
        return self._estimator

    @property
    def source(self) -> FrameSource:
        """The frame source in use."""
        return self._source

    def read(self) -> GazeResult | None:
        """One frame to one result, or ``None`` when the source is exhausted."""
        frame = self._source.read()
        if frame is None:
            return None
        image: NDArray[np.uint8] = frame.image
        if self._config.camera.mirror:
            # Required by the calibration, not cosmetic. See the module docstring.
            image = np.ascontiguousarray(cv2.flip(image, 1), dtype=np.uint8)
        return self._estimator.process(image, frame.timestamp)

    def stream(self) -> Iterator[GazeResult]:
        """Yield a result per frame until the source ends.

        Yields every frame's result, including the unusable ones, so a caller
        can tell "no face" from "the stream stopped". Filtering on ``.ok`` is one
        line; recovering a status that was never yielded is impossible.
        """
        while (result := self.read()) is not None:
            yield result

    def close(self) -> None:
        """Release the camera and the estimator's native resources.

        Idempotent, and safe to call after an exception. The source is released
        only if this tracker opened it.
        """
        try:
            self._estimator.close()
        finally:
            if self._owns_source:
                self._source.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _resolve_profile(
    profile: CalibrationProfile | str | None,
) -> CalibrationProfile | None:
    """Accept a profile, a name, or nothing."""
    if profile is None or not isinstance(profile, str):
        return profile
    from ..calibration import CalibrationProfile as Profile

    return Profile.load(profile)
