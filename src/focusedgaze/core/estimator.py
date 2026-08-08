"""GazeEstimator: frames in, results out. No camera, no window, no network.

This is the layer that makes focusedgaze a library rather than an application.
Everything that knows how to *get* a frame lives in :mod:`focusedgaze.capture`;
everything that knows what to *do* with a reading lives above. This composes the
five stages and owns the state between them:

    frame -> landmarks + smoothed crop -> gaze model -> (pitch, yaw)
          -> calibration -> (x, y) -> One Euro filter -> GazeResult

WHAT IT OWNS, AND WHY THAT MATTERS
-----------------------------------
Three pieces of per-stream state, all of which were module-level globals in the
legacy pipeline: the crop smoothing, the two filters, and the frame counter
driving MediaPipe's video timestamps. Two estimators in one process now share
none of it. ``tests/test_core_estimator.py`` pins that, and it is the A3 test the
standing brief has carried since Phase 0.

THE FILTER RESET POLICY IS HERE, NOT IN THE FILTER
---------------------------------------------------
`core/filters.py` implements the One Euro filter and knows nothing about faces.
The *policy* of resetting it when tracking is lost is a pipeline decision and
lives here (R-7). Losing it makes the cursor glide in from a stale point when the
face comes back, which looks like drift rather than like a bug.

THE POSITIONING GATE DOES NOT SUPPRESS THE READING
---------------------------------------------------
It reports. A user leaning outside 45-65 cm gets a non-OK status with the
distance attached, not a missing reading: the legacy server had no gate at all in
its ``ok`` decision, and wiring one in so the cursor vanished would be a
behaviour change dressed as an improvement. The status is the honest middle
ground, and a caller who wants the legacy behaviour reads ``result.pitch``.
"""

from __future__ import annotations

import logging
import time
from types import TracebackType
from typing import TYPE_CHECKING, Self

import numpy as np
from numpy.typing import NDArray

from ..config import GazeConfig
from ..types import GazeResult, GazeStatus
from .filters import OneEuroFilter2D
from .landmarks import FaceLandmarker
from .model import GazeModel
from .positioning import FocalCalibration, PositioningGate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..calibration import CalibrationProfile

__all__ = ["GazeEstimator"]

_log = logging.getLogger(__name__)


class GazeEstimator:
    """Turn frames into calibrated, filtered screen coordinates.

    Args:
        profile: The calibration to apply. ``None`` is a supported state: the
            estimator still reports pitch and yaw, with status
            ``NOT_CALIBRATED``, which is what lets the setup flow show a live
            face before any calibration exists.
        config: Everything tunable. Defaults are the shipping values.
        focal: A measured focal length for distance estimation. Without one the
            gate falls back to an assumed field of view, which is rougher.
        landmarker: Pre-built landmarker, for tests or to share a model file.
        model: Pre-built gaze model, same.

    Raises:
        ModelNotFoundError: A required model file is absent. Never downloaded.
        ProviderError: No usable ONNX execution provider.

    Not thread-safe, by construction rather than by omission: MediaPipe's video
    mode needs monotonically increasing timestamps and the filters carry state
    across frames. Use one estimator per stream, which costs nothing now that
    none of the state is global.
    """

    __slots__ = ("_config", "_filter", "_gate", "_landmarker", "_model", "_profile", "_tracking")

    def __init__(
        self,
        profile: CalibrationProfile | None = None,
        config: GazeConfig | None = None,
        *,
        focal: FocalCalibration | None = None,
        landmarker: FaceLandmarker | None = None,
        model: GazeModel | None = None,
    ) -> None:
        self._config = config if config is not None else GazeConfig()
        self._profile = profile
        self._landmarker = landmarker or FaceLandmarker(config=self._config.landmarks)
        self._model = model or GazeModel(config=self._config.model)
        self._gate = PositioningGate(self._config.positioning, focal)
        filter_config = self._config.filter
        self._filter = OneEuroFilter2D(
            filter_config.min_cutoff, filter_config.beta, filter_config.d_cutoff
        )
        # Whether the previous frame produced a face. Drives the reset policy:
        # the filter is reset on the TRANSITION to lost, not on every lost
        # frame, which would be the same thing but does needless work.
        self._tracking = False

    @property
    def profile(self) -> CalibrationProfile | None:
        """The calibration in use, if any."""
        return self._profile

    @property
    def provider(self) -> str:
        """The ONNX execution provider that actually loaded."""
        return self._model.provider

    def reset(self) -> None:
        """Forget all per-stream state: crop smoothing, filters, frame counter.

        Call between independent streams. Not needed on tracking loss; that is
        handled internally.
        """
        self._landmarker.reset()
        self._filter.reset()
        self._tracking = False

    def _lose_tracking(self, timestamp: float, status: GazeStatus) -> GazeResult:
        """Report a frame with no usable reading, resetting on the transition."""
        if self._tracking:
            # R-7: forget history so the cursor does not glide in from a stale
            # point when the face returns. On the transition only.
            self._filter.reset()
            self._landmarker.reset()
            self._tracking = False
        return GazeResult.unavailable(status, timestamp)

    def process(
        self, frame: NDArray[np.uint8], timestamp: float | None = None
    ) -> GazeResult:
        """One BGR frame to one result.

        Args:
            frame: ``(height, width, 3)`` uint8 in **BGR** order. Must already be
                mirrored the same way the calibration was recorded: the
                calibration polynomial is fitted in the mirrored frame, so an
                unmirrored one does not mirror the output, it makes the mapping
                wrong.
            timestamp: Capture time in seconds. Defaults to ``time.time()``.
                Must increase across a stream: the One Euro filter
                differentiates against it.

        Returns:
            A :class:`~focusedgaze.types.GazeResult`, always. A frame with no
            face is a status, not an exception.
        """
        now = time.time() if timestamp is None else float(timestamp)

        observation = self._landmarker.detect(frame, now)
        if observation is None:
            return self._lose_tracking(now, GazeStatus.NO_FACE)

        crop = self._landmarker.crop(frame, observation.bbox)
        if crop is None:
            # A box that clamped to nothing. Same user-visible state as no face.
            return self._lose_tracking(now, GazeStatus.NO_FACE)

        pitch, yaw = self._model.predict(crop)
        self._tracking = True

        # The gate reports; it does not veto. Evaluated before calibration so a
        # rejected frame still carries its distance.
        distance_cm: float | None = None
        zone_status: GazeStatus | None = None
        position = self._gate.evaluate(observation.landmarks, frame.shape)
        if position is not None:
            distance_cm = position.distance_cm
            if not position.distance_ok:
                zone_status = GazeStatus.OUT_OF_RANGE
            elif not position.centered:
                zone_status = GazeStatus.OFF_CENTER

        if self._profile is None:
            return GazeResult.unavailable(
                GazeStatus.NOT_CALIBRATED, now,
                pitch=pitch, yaw=yaw, distance_cm=distance_cm,
            )
        if zone_status is not None:
            return GazeResult.unavailable(
                zone_status, now, pitch=pitch, yaw=yaw, distance_cm=distance_cm,
            )

        x, y = self._profile.apply(pitch, yaw)
        sx, sy = self._filter.filter(x, y, now)
        return GazeResult(
            x=sx, y=sy,
            pitch=pitch, yaw=yaw,
            distance_cm=distance_cm,
            status=GazeStatus.OK,
            timestamp=now,
        )

    def close(self) -> None:
        """Release the landmarker's native resources."""
        self._landmarker.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
