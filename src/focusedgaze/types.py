"""The public data types every layer of the package speaks in.

One reading of one frame is a :class:`GazeResult`. It is deliberately a single
type for every outcome, successful or not, because the legacy pipeline used
``None`` for three different things at once: "no face in frame", "face found but
the user is out of the tracking zone", and "something is broken". A caller could
not tell them apart, so the only available response to any of them was to do
nothing (``gaze_server.py`` L368-372 does exactly that).

Here the outcome is named. :class:`GazeStatus` says which of those happened, and
genuine faults raise from :mod:`focusedgaze.exceptions` instead of being encoded
in the return value.

WHY ``ok`` IS A PROPERTY AND NOT A FIELD
----------------------------------------
The legacy wire message carries ``ok`` as its own boolean
(``gaze_server.py`` L377-378). Storing it beside ``status`` would be two
independent records of one fact, and the pair can disagree: nothing stops
``GazeResult(ok=True, status=GazeStatus.NO_FACE)`` from being constructed, and
nothing would catch it. ``ok`` is therefore derived from ``status``, which makes
that state unrepresentable rather than merely discouraged. The attribute reads
exactly the same at the call site.

Angles are radians throughout, matching the pipeline's own units
(``gaze_pipeline.py`` L65-66 converts the model's degrees to radians once, at the
boundary). Screen coordinates are normalised to [0, 1] with the origin at the
top left, which is the browser's convention and the one calibration was fitted
against.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "BBox",
    "FaceObservation",
    "Frame",
    "GazeResult",
    "GazeStatus",
    "Landmark",
    "Landmarks",
    "Matrix4x4",
    "Point2D",
]

#: A captured image as OpenCV hands it over: HxWx3, BGR channel order, uint8.
type Frame = NDArray[np.uint8]

#: A face bounding box in pixels, ``(x_min, y_min, x_max, y_max)``, as produced
#: by the smoothed square crop (``gaze_pipeline.py`` L99).
type BBox = tuple[int, int, int, int]

#: A normalised screen point, both components in [0, 1].
type Point2D = tuple[float, float]

#: MediaPipe's 4x4 facial transformation matrix, when the landmarker was asked
#: for one. Head pose is derived from it.
type Matrix4x4 = NDArray[np.float32]


class Landmark(Protocol):
    """One face landmark, normalised to the frame.

    A structural type rather than a class of ours: MediaPipe returns its own
    ``NormalizedLandmark`` objects and there is no reason to copy 478 of them
    per frame into a different shape. Anything with these three attributes will
    do, which is also what makes the pure core testable without MediaPipe
    installed.

    Attributes:
        x: Horizontal position, 0 at the left edge of the frame, 1 at the right.
        y: Vertical position, 0 at the top, 1 at the bottom.
        z: Depth relative to the face centre, in the same scale as ``x``. Roughly
            estimated by the model; treat it as ordering information, not a
            measurement.
    """

    x: float
    y: float
    z: float


#: The full landmark set for one face. Index constants live in
#: :mod:`focusedgaze.core.positioning`; the iris points require the refined
#: 478-point model.
type Landmarks = Sequence[Landmark]


class GazeStatus(StrEnum):
    """Why a reading is or is not usable.

    A :class:`StrEnum` so a member serialises straight to its own name through
    ``json.dumps`` without anyone having to remember ``.value``. The wire format
    is the one place this type is most likely to be handled carelessly, and a
    forgotten ``.value`` there would put ``"GazeStatus.OK"`` on the wire.

    Members:
        OK: A face was found, the user is in the tracking zone, and the point
            has been calibrated and filtered. The only status for which
            :attr:`GazeResult.x` and :attr:`GazeResult.y` are populated.
        NO_FACE: No face in the frame, or one too close to an edge to crop.
            Legacy equivalent: ``get_gaze_reading`` returning ``None``
            (``gaze_pipeline.py`` L147).
        OUT_OF_RANGE: A face, but at a distance outside the configured zone.
            Legacy equivalent: ``distance_ok`` false
            (``positioning_gate.py`` L112).
        OFF_CENTER: A face at a usable distance but too far from frame centre.
            Legacy equivalent: ``centered`` false (``positioning_gate.py`` L113).
        NOT_CALIBRATED: No calibration profile is loaded, so raw angles cannot be
            mapped to the screen. Legacy equivalent: the server printing an error
            and shutting the loop down (``gaze_server.py`` L316-321).
    """

    OK = "ok"
    NO_FACE = "no_face"
    OUT_OF_RANGE = "out_of_range"
    OFF_CENTER = "off_center"
    NOT_CALIBRATED = "not_calibrated"


@dataclass(frozen=True, slots=True)
class FaceObservation:
    """What the landmarker saw in one frame, before any gaze inference.

    The seam between :mod:`focusedgaze.core.landmarks` and everything that
    consumes it: the positioning gate needs the landmarks, the gaze model needs
    the crop box, and head pose needs the matrix. Passing one immutable value
    keeps them from each re-running detection.

    Args:
        landmarks: The refined 478-point set for the single tracked face.
        bbox: The smoothed square crop box in pixels. Smoothed across frames, so
            two consecutive observations of a still face are not identical.
        transform_matrix: MediaPipe's 4x4 head-pose matrix, or ``None`` when the
            landmarker was not asked for one.
        timestamp: Capture time in seconds, from the same clock as
            :attr:`GazeResult.timestamp`.

    Immutable, so it can be handed to another thread without copying.
    """

    landmarks: Landmarks
    bbox: BBox
    transform_matrix: Matrix4x4 | None
    timestamp: float


# Deliberately NOT slots=True, unlike every other dataclass in this package.
# CPython's frozen __setattr__ closes over the pre-slots class, and slots=True
# then rebuilds the class, so assigning to a NON-field attribute takes the
# `super(cls, self)` branch and fails with a confusing
# "obj is not an instance or subtype of type" TypeError instead of
# FrozenInstanceError. `ok` is a property rather than a field, so it is exactly
# that case. Measured on CPython 3.14.3; fields raise correctly either way.
@dataclass(frozen=True)
class GazeResult:
    """One frame's reading, successful or not.

    Args:
        x: Calibrated, filtered horizontal screen position in [0, 1] from the
            left edge. ``None`` unless :attr:`status` is ``OK``.
        y: The same vertically, from the top edge.
        pitch: Raw model output in radians, before calibration. Present whenever
            the gaze model ran, including when the positioning gate then
            rejected the frame, so a caller can still log or plot the raw signal.
        yaw: The same, horizontally.
        distance_cm: Estimated distance from the camera, when the positioning
            gate could measure it.
        status: Why this reading is or is not usable. See :class:`GazeStatus`.
        timestamp: Capture time in seconds since the epoch, as ``time.time()``
            reports it. This is the value the One Euro filter derived its
            sampling frequency from, so it must be the real capture time and not
            a rounded or fabricated one.

    Immutable and self-contained, so it can be published to another thread by a
    plain reference swap, which is what the shipping server relies on
    (``gaze_server.py`` L104-108).
    """

    x: float | None
    y: float | None
    pitch: float | None
    yaw: float | None
    distance_cm: float | None
    status: GazeStatus
    timestamp: float

    @property
    def ok(self) -> bool:
        """True when this reading carries a usable screen point.

        Derived from :attr:`status` rather than stored, so the two can never
        disagree. See the module docstring.
        """
        return self.status is GazeStatus.OK

    @classmethod
    def unavailable(
        cls,
        status: GazeStatus,
        timestamp: float,
        *,
        pitch: float | None = None,
        yaw: float | None = None,
        distance_cm: float | None = None,
    ) -> GazeResult:
        """Build a result with no screen point, for any non-``OK`` status.

        The keyword arguments exist because a rejected frame is not always an
        empty one: a user sitting too close still has a measured distance, and
        usually a pitch and yaw as well. Throwing that away would make the
        out-of-zone case indistinguishable from a missing face in a log.

        Args:
            status: Anything but ``GazeStatus.OK``.
            timestamp: Capture time in seconds.
            pitch: Raw pitch in radians, if the gaze model ran.
            yaw: Raw yaw in radians, if the gaze model ran.
            distance_cm: Measured distance, if the positioning gate ran.

        Raises:
            ValueError: if ``status`` is ``OK``. A successful reading has a
                screen point, so it is built with the ordinary constructor.
        """
        if status is GazeStatus.OK:
            raise ValueError(
                "GazeResult.unavailable() cannot build an OK result: "
                "construct GazeResult(...) directly with x and y"
            )
        return cls(
            x=None,
            y=None,
            pitch=pitch,
            yaw=yaw,
            distance_cm=distance_cm,
            status=status,
            timestamp=timestamp,
        )
