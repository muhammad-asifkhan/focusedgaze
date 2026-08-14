"""The Apache-2.0 gaze backend: Intel ``gaze-estimation-adas-0002``.

WHY THIS EXISTS ALONGSIDE L2CS
==============================
Not because it is more accurate -- that is unmeasured, and its published 6.95
degrees is on Intel's own validation set, not a benchmark comparable with
L2CS-Net's Gaze360 figure. It exists because of two properties that decide
whether anyone can use this library at all:

**It can be shipped.** Copyright Intel Corporation, Apache License 2.0, weights
included. The L2CS weights derive from Gaze360, whose licence names "models
trained on dataset" as covered derivative works and forbids distribution
outright, so they can never be bundled. A default backend that can live in the
wheel is the difference between ``pip install focusedgaze`` working and a
multi-step manual setup.

**It runs anywhere.** 0.139 GFLOPs against ResNet-50's ~4, and 7.5 MB against
91 MB. Measured on the development machine, an AMD integrated GPU: **2.16 ms on
CPU**, against 141.7 ms for L2CS through DirectML on the same box. That is 66x,
and it moves the bottleneck from the model to the camera. It matters for more
than throughput -- a 140 ms lag makes dwell interaction feel broken, and a
7 fps pursuit dot *steps* rather than glides, which is why calibration sweeps on
slow hardware collect noisy samples.

THREE CONVENTIONS THAT WILL NOT FAIL LOUDLY IF WRONG
=====================================================
Each of these produces a plausible gaze vector when mistaken, exactly like the
transposed output names documented in :mod:`focusedgaze.core.model`:

1. **Head pose is ``[yaw, pitch, roll]`` in DEGREES.** This package carries head
   pose as ``(pitch, yaw, roll)`` in radians, so both the order and the unit are
   converted here.
2. **The eye patches are BGR**, as OpenCV and the Open Model Zoo demos use.
3. **The output vector is (x, y, z) in camera space and is not normalised.**
   Pitch and yaw are recovered by trigonometry, not read off directly.

None of the three is verified against a person yet. They are stated so that a
disagreement between this backend and L2CS on the same face is diagnosable
rather than mysterious.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ..exceptions import ModelNotFoundError, ProviderError

__all__ = ["IntelGazeModel"]

_log = logging.getLogger(__name__)

#: Patch side the model was exported for.
EYE_SIZE: Final = 60


class IntelGazeModel:
    """``gaze-estimation-adas-0002`` behind the same seam as :class:`GazeModel`.

    Args:
        model_path: The ``.xml`` of the OpenVINO IR pair. Defaults to the
            managed cache.
        device: OpenVINO device. ``"CPU"`` everywhere; ``"GPU"`` only helps on
            Intel graphics, and at 2 ms on CPU there is nothing to gain.
        core_factory: Injection seam for tests, so the contract can be exercised
            without the runtime installed.

    Raises:
        ModelNotFoundError: The graph is absent.
        ProviderError: OpenVINO is not installed.
    """

    #: Read by :class:`~focusedgaze.core.estimator.GazeEstimator` to decide what
    #: to hand the model. A flag rather than a signature check because the two
    #: backends genuinely need different inputs, and a silent mismatch would
    #: feed a face crop to something expecting an eye.
    wants_eyes = True

    __slots__ = ("_compiled", "_device")

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "CPU",
        core_factory: Any = None,
    ) -> None:
        if model_path is None:
            from ..assets import INTEL_GAZE, asset_path

            resolved = asset_path(INTEL_GAZE)
        else:
            resolved = Path(model_path)

        if not resolved.is_file():
            raise ModelNotFoundError(
                f"the Intel gaze model is not at {resolved}.\n"
                "It is Apache-2.0 and may be fetched automatically:\n"
                "    focusedgaze download-models\n"
                "or set FOCUSEDGAZE_MODEL_DIR to a directory that already has it."
            )

        core = (core_factory or _default_core)()
        self._device = device
        self._compiled = core.compile_model(core.read_model(resolved), device)
        _log.info("gaze backend: intel gaze-estimation-adas-0002 on %s", device)

    @property
    def provider(self) -> str:
        """Named to match :attr:`GazeModel.provider`, so reports read the same."""
        return f"OpenVINO:{self._device}"

    def predict_eyes(
        self,
        left_eye: NDArray[np.uint8],
        right_eye: NDArray[np.uint8],
        head_pose: tuple[float, float, float],
    ) -> tuple[float, float]:
        """Gaze angles from two eye patches and a head pose.

        Args:
            left_eye: 60x60 BGR patch of the subject's left eye.
            right_eye: 60x60 BGR patch of the subject's right eye.
            head_pose: ``(pitch, yaw, roll)`` in **radians**, this package's
                convention. Converted here to the model's ``[yaw, pitch, roll]``
                in degrees.

        Returns:
            ``(pitch, yaw)`` in radians, matching :meth:`GazeModel.predict`, so
            everything downstream is unchanged.
        """
        pitch, yaw, roll = head_pose
        feed = {
            "left_eye_image": _as_nchw(left_eye),
            "right_eye_image": _as_nchw(right_eye),
            # Order and unit both converted. See the module docstring.
            "head_pose_angles": np.array(
                [[math.degrees(yaw), math.degrees(pitch), math.degrees(roll)]],
                dtype=np.float32,
            ),
        }
        vector = np.asarray(next(iter(self._compiled(feed).values()))).ravel()
        return _angles_from_vector(vector)

    def close(self) -> None:
        """Nothing to release; present so the backends are interchangeable."""
        return


def _default_core() -> Any:
    try:
        # Three codes because this is an optional dependency: unresolvable when
        # absent, untyped when present, and the suppression itself unused in
        # whichever of those two cases does not apply.
        import openvino as ov  # type: ignore[import-not-found, import-untyped, unused-ignore]
    except ImportError as exc:
        raise ProviderError(
            "the Intel gaze backend needs the OpenVINO runtime, which is not "
            "installed:\n"
            "    pip install openvino\n"
            "It is Apache-2.0, CPU-only by default, and runs on any x86 machine."
        ) from exc
    return ov.Core()


def _as_nchw(patch: NDArray[np.uint8]) -> NDArray[np.float32]:
    """``(60, 60, 3)`` BGR to the ``(1, 3, 60, 60)`` the graph declares.

    No normalisation: this model takes raw 0-255 values, unlike L2CS which
    expects ImageNet mean/std. Dividing here would darken every input and bias
    every reading.
    """
    return np.ascontiguousarray(
        patch.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32
    )


def _angles_from_vector(vector: NDArray[np.float32]) -> tuple[float, float]:
    """Cartesian gaze direction to ``(pitch, yaw)`` in radians.

    The model emits an unnormalised ``(x, y, z)`` in camera space: ``x`` right,
    ``y`` up, ``z`` toward the camera. Yaw is the rotation about the vertical
    axis and pitch the elevation, so:

        yaw   = atan2(-x, -z)
        pitch = asin(y / |v|)

    The negations put the result in the same handedness as the L2CS decode,
    where a positive yaw means looking to the subject's left. Getting that sign
    wrong transposes the screen horizontally, which reads as a bad calibration
    rather than a bug.
    """
    x, y, z = (float(v) for v in vector[:3])
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        return (0.0, 0.0)
    pitch = math.asin(max(-1.0, min(1.0, y / norm)))
    yaw = math.atan2(-x, -z)
    return (pitch, yaw)
