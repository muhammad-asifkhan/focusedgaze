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
import math
import time
from collections.abc import Iterable
from types import TracebackType
from typing import TYPE_CHECKING, Any, Final, Self

import numpy as np
from numpy.typing import NDArray

from ..config import GazeConfig, ModelConfig
from ..exceptions import CalibrationError
from ..types import FaceObservation, GazeResult, GazeStatus
from .filters import OneEuroFilter2D
from .landmarks import FaceLandmarker
from .model import GazeModel
from .positioning import FocalCalibration, PositioningGate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..calibration import CalibrationProfile

__all__ = ["GazeEstimator"]

_log = logging.getLogger(__name__)


#: Eye patch edge the Intel backend was exported for. Named here rather than
#: imported at module scope so selecting L2CS never pulls in the OpenVINO path.
_EYE_SIZE: Final = 60


def _build_model(config: ModelConfig) -> Any:
    """The gaze backend named by the config.

    Deferred import: a machine running L2CS must not need OpenVINO installed,
    and a machine running the Intel backend must not need an ONNX provider.
    Importing both eagerly would make each one's optional dependency mandatory.
    """
    if config.backend == "intel":
        from .intel_model import IntelGazeModel

        return IntelGazeModel()
    return GazeModel(config=config)


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

    __slots__ = (
        "_config", "_filter", "_gate", "_landmarker", "_last_observation", "_model",
        "_offset", "_profile", "_tracking",
    )

    def __init__(
        self,
        profile: CalibrationProfile | None = None,
        config: GazeConfig | None = None,
        *,
        focal: FocalCalibration | None = None,
        landmarker: FaceLandmarker | None = None,
        # Deliberately not `GazeModel | None`: the two backends have different
        # methods on purpose (`predict` takes a face crop, `predict_eyes` takes
        # two eye patches and a head pose), and narrowing this to one of them
        # would make injecting the other a type error rather than the supported
        # case it is. The `wants_eyes` flag is how they are told apart.
        model: Any = None,
    ) -> None:
        self._config = config if config is not None else GazeConfig()
        self._profile = profile
        self._landmarker = landmarker or FaceLandmarker(config=self._config.landmarks)
        self._model = model if model is not None else _build_model(self._config.model)
        self._gate = PositioningGate(self._config.positioning, focal)
        filter_config = self._config.filter
        self._filter = OneEuroFilter2D(
            filter_config.min_cutoff, filter_config.beta, filter_config.d_cutoff
        )
        # Whether the previous frame produced a face. Drives the reset policy:
        # the filter is reset on the TRANSITION to lost, not on every lost
        # frame, which would be the same thing but does needless work.
        self._tracking = False
        # Diagnostics only; see `last_observation`.
        self._last_observation: FaceObservation | None = None
        # Session correction; see `recentre`. Deliberately NOT cleared by
        # `reset`, which forgets per-stream state: an offset describes where the
        # user is sitting this session, and losing it because a face flickered
        # would undo the correction without anyone noticing.
        self._offset: tuple[float, float] = (0.0, 0.0)

    @property
    def profile(self) -> CalibrationProfile | None:
        """The calibration in use, if any."""
        return self._profile

    @property
    def provider(self) -> str:
        """Which runtime actually loaded the gaze model.

        An ONNX execution provider for the L2CS backend, or ``OpenVINO:<device>``
        for the Intel one. Recorded in accuracy reports, so a measurement can
        always be tied to what produced it.
        """
        return str(self._model.provider)

    @property
    def offset(self) -> tuple[float, float]:
        """The constant correction subtracted from every prediction.

        ``(0.0, 0.0)`` until :meth:`recentre` or :meth:`set_offset` is called.
        """
        return self._offset

    def set_offset(self, dx: float, dy: float) -> None:
        """Set the correction directly, in screen fractions.

        Raises:
            CalibrationError: If either value is not finite. A NaN here would
                silently poison every subsequent reading rather than failing.
        """
        if not (math.isfinite(dx) and math.isfinite(dy)):
            raise CalibrationError(f"offset must be finite, got ({dx!r}, {dy!r})")
        self._offset = (float(dx), float(dy))

    def clear_offset(self) -> None:
        """Forget the correction. Not done by :meth:`reset`."""
        self._offset = (0.0, 0.0)

    def recentre(
        self,
        readings: Iterable[tuple[float, float]],
        *,
        target: tuple[float, float] = (0.5, 0.5),
        distance_cm: float | None = None,
    ) -> tuple[float, float]:
        """Correct a whole session against one known point.

        WHY AN APPLICATION SHOULD DO THIS AT STARTUP
        --------------------------------------------
        A profile fixes the *shape* of the mapping, not its position. Across five
        measured runs on one machine the whole mapping shifted bodily between
        sessions -- by −0.25 to +0.34 of screen height, twice on an identical
        profile minutes apart. Posture, seat height and where the laptop lid ended
        up all move it, and no amount of calibration quality prevents that.

        Show a single dot at ``target``, collect readings while the user looks at
        it, and pass them here. The difference between where the model says they
        looked and where they were told to look is subtracted from everything
        afterwards. This is standard practice for eye trackers, and it is the
        largest single accuracy gain available to a consumer of this package.

        Args:
            readings: ``(pitch, yaw)`` pairs in radians, collected while the user
                fixated ``target``. The **median** is used, so blinks and glances
                away are tolerated without special handling.
            target: Where they were told to look, normalised. Defaults to centre.
            distance_cm: Current viewing distance, so the offset is measured in
                the same space the readings will be produced in when
                ``compensate_distance`` is enabled. Ignored otherwise.

        Returns:
            The offset now in force, as ``(dx, dy)``.

        Raises:
            CalibrationError: There is no profile, or no usable readings.
        """
        if self._profile is None:
            raise CalibrationError(
                "recentre needs a calibration profile: it corrects where a "
                "profile lands, and cannot invent one."
            )
        pairs = [
            (float(p), float(y))
            for p, y in readings
            if math.isfinite(p) and math.isfinite(y)
        ]
        if not pairs:
            raise CalibrationError(
                "recentre got no usable readings. Collect frames while the user "
                "looks at the target, and keep only those with a gaze angle."
            )
        # Median per axis, not mean: a blink or a glance away is an outlier, and
        # the whole point of this call is that it needs no supervision.
        pitch = sorted(p for p, _ in pairs)[len(pairs) // 2]
        yaw = sorted(y for _, y in pairs)[len(pairs) // 2]

        x, y = self._profile.apply(pitch, yaw)
        if self._config.positioning.compensate_distance:
            x, y = self._profile.rescaled_for(x, y, distance_cm)
        self._offset = (x - target[0], y - target[1])
        _log.info(
            "recentred: offset (%+.3f, %+.3f) from %d reading(s)",
            self._offset[0], self._offset[1], len(pairs),
        )
        return self._offset

    @property
    def last_observation(self) -> FaceObservation | None:
        """What the landmarker saw on the most recent frame, or ``None``.

        Exposed for **diagnostics**, not for the pipeline: it carries the 4x4
        head-pose matrix, which nothing in the mapping uses but which
        `focusedgaze accuracy` records so that the per-session offset can be
        correlated against head orientation. See
        :mod:`focusedgaze.core.headpose`.

        Reset to ``None`` when tracking is lost, so a stale pose cannot be read
        as the current one -- the same discipline as the filter reset, and for
        the same reason.
        """
        return self._last_observation

    def reset(self) -> None:
        """Forget all per-stream state: crop smoothing, filters, frame counter.

        Call between independent streams. Not needed on tracking loss; that is
        handled internally.
        """
        self._landmarker.reset()
        self._filter.reset()
        self._tracking = False
        self._last_observation = None

    def _lose_tracking(self, timestamp: float, status: GazeStatus) -> GazeResult:
        """Report a frame with no usable reading, resetting on the transition."""
        if self._tracking:
            # R-7: forget history so the cursor does not glide in from a stale
            # point when the face returns. On the transition only.
            self._filter.reset()
            self._landmarker.reset()
            self._tracking = False
        # Unconditionally, unlike the resets above: a reading with no face has no
        # head pose, and leaving the previous frame's would let a diagnostic
        # record a pose for a frame that never saw one.
        self._last_observation = None
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
        self._last_observation = observation

        # Two backends want different things. The flag is read from the model
        # rather than inferred from the config, so a caller injecting a model
        # directly cannot end up handing a face crop to something that expects
        # an eye -- which would produce a plausible wrong answer, not an error.
        if getattr(self._model, "wants_eyes", False):
            from .eyes import eye_crops
            from .headpose import head_angles

            pose = head_angles(observation.transform_matrix)
            # Roll is passed into the crop, not just alongside it: the model
            # expects an upright eye, and an axis-aligned crop of a tilted head
            # presents a tilted one. Measured cost of omitting it: horizontal
            # gain 0.98 -> 0.48 across a 12.6 degree head tilt.
            eyes = eye_crops(
                frame, observation.landmarks, size=_EYE_SIZE,
                roll=pose.roll if pose is not None else 0.0,
            )
            if eyes is None or pose is None:
                # No iris landmarks, an eye off the edge of the frame, or no
                # head-pose matrix. Not an error: the same recoverable state as
                # a face that could not be cropped.
                return self._lose_tracking(now, GazeStatus.NO_FACE)
            pitch, yaw = self._model.predict_eyes(eyes[0], eyes[1], pose)
        else:
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

        # Applied BEFORE the filter, not after. The filter smooths a trajectory,
        # and rescaling its output would make the smoothing fight a moving gain
        # every time the user shifts in their seat. Correcting first means the
        # filter only ever sees points in one consistent frame of reference.
        #
        # A no-op unless the config asks for it AND the profile recorded the
        # distance it was collected at, so no existing profile changes behaviour.
        if self._config.positioning.compensate_distance:
            x, y = self._profile.rescaled_for(x, y, distance_cm)

        # The session offset, last and before the filter. Last because it is
        # measured in final screen space by `recentre`, so anything applied after
        # it would move the ground it was measured against. Before the filter for
        # the same reason as the rescaling: the filter should only ever smooth
        # points in one consistent frame of reference.
        dx, dy = self._offset
        if dx or dy:
            low, high = self._profile.clamp
            x = max(low, min(high, x - dx))
            y = max(low, min(high, y - dy))

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
