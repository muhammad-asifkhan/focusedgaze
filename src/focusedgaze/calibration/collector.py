"""Headless, scriptable calibration sample collection.

The legacy calibration routine was a single 400-line OpenCV script: it drew the
stimulus, read the camera, accumulated samples, fitted the model and saved it,
all in one loop. Nothing about it could be exercised without a screen, a webcam
and a person, so the sample-handling logic had no tests and could not be reused.

This module is the part of that job that needs none of those things. You give it
readings and the on-screen target they belong to; it validates them, keeps them,
and hands them to the fitter. A replay from a recorded session, a synthetic
sweep in a unit test, and a live routine driving a real display all use exactly
the same object, so the interactive path in ``ui.py`` becomes presentation over
tested logic rather than logic embedded in presentation.

Nothing here imports scikit-learn, OpenCV or MediaPipe. :meth:`CalibrationCollector.fit`
delegates to :mod:`focusedgaze.calibration.fitter`, which imports scikit-learn at
the moment it is called.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ..exceptions import CalibrationError
from .fitter import (
    DEFAULT_MAD_FACTOR,
    DEFAULT_MIN_KEEP,
    DEFAULT_VALIDATION_FRACTION,
    DEFAULT_VALIDATION_SEED,
    ROBUST_DEFAULT_DEGREE,
    FitResult,
    robust_fit_samples,
)
from .profile import DEFAULT_CAMERA_SIZE

__all__ = ["DEFAULT_MIN_SAMPLES", "CalibrationCollector", "CalibrationSample"]

_log = logging.getLogger("focusedgaze")

#: Fewest samples worth fitting. The same figure the shipping routine used for
#: its ``MIN_TRAIN_SAMPLES``, which is also the robust pass's ``min_keep``.
DEFAULT_MIN_SAMPLES: Final = DEFAULT_MIN_KEEP


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    """One gaze reading paired with the target the user was looking at.

    Args:
        pitch: Vertical gaze angle in radians, from the gaze model.
        yaw: Horizontal gaze angle in radians.
        target_x: Where the stimulus was, as a fraction of screen width.
        target_y: Where it was, as a fraction of screen height.

    The targets are ground truth, not predictions: the program drew the dot, so
    it knows exactly where it was. Nothing in this module ever stores a model's
    guess as if it were a label.
    """

    pitch: float
    yaw: float
    target_x: float
    target_y: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        """The ``(pitch, yaw, target_x, target_y)`` row the fitter consumes."""
        return (self.pitch, self.yaw, self.target_x, self.target_y)


class CalibrationCollector:
    """Accumulates calibration samples and fits a profile from them.

    Args:
        screen_size: ``(width, height)`` of the display being calibrated
            against, if known. Recorded on the profile, never used in the
            arithmetic: targets and predictions are screen fractions.
        camera_size: ``(width, height)`` of the capture stream.
        min_samples: Fewest samples :attr:`ready` will accept. Below this, a fit
            is technically possible and practically worthless.

    Not thread-safe: it is a mutable accumulator, so one collector belongs to one
    calibration run on one thread.
    """

    __slots__ = ("_samples", "camera_size", "min_samples", "screen_size")

    def __init__(
        self,
        *,
        screen_size: tuple[int, int] | None = None,
        camera_size: tuple[int, int] | None = DEFAULT_CAMERA_SIZE,
        min_samples: int = DEFAULT_MIN_SAMPLES,
    ) -> None:
        if min_samples < 1:
            raise CalibrationError(f"min_samples must be >= 1, got {min_samples}")
        self.screen_size = screen_size
        self.camera_size = camera_size
        self.min_samples = int(min_samples)
        self._samples: list[CalibrationSample] = []

    @staticmethod
    def _validated(
        pitch: float, yaw: float, target_x: float, target_y: float,
    ) -> CalibrationSample:
        """Check one sample and return it, or raise explaining which value is wrong."""
        checked: list[float] = []
        for label, value in zip(
            ("pitch", "yaw", "target_x", "target_y"),
            (pitch, yaw, target_x, target_y),
            strict=True,
        ):
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise CalibrationError(f"{label} must be a number, got {value!r}") from exc
            if not np.isfinite(number):
                raise CalibrationError(f"{label} must be finite, got {value!r}")
            checked.append(number)
        if not (0.0 <= checked[2] <= 1.0 and 0.0 <= checked[3] <= 1.0):
            raise CalibrationError(
                "calibration targets are fractions of the screen and must lie in [0, 1], "
                f"got ({target_x!r}, {target_y!r})"
            )
        return CalibrationSample(checked[0], checked[1], checked[2], checked[3])

    def add_sample(self, pitch: float, yaw: float, target_x: float, target_y: float) -> None:
        """Record one reading against the target it belongs to.

        Args:
            pitch: Vertical gaze angle in radians.
            yaw: Horizontal gaze angle in radians.
            target_x: Stimulus position as a fraction of screen width, in [0, 1].
            target_y: Stimulus position as a fraction of screen height, in [0, 1].

        Raises:
            CalibrationError: If any value is not a finite number, or a target
                falls outside [0, 1]. An off-screen target is a bug in whatever
                drew the stimulus, and silently accepting it would bake that bug
                into the fitted polynomial where it is far harder to find.
        """
        self._samples.append(self._validated(pitch, yaw, target_x, target_y))

    def add_samples(self, samples: Iterable[Sequence[float]]) -> int:
        """Record many ``(pitch, yaw, target_x, target_y)`` rows.

        The whole batch is validated before any of it is kept, so a bad row
        leaves the collector exactly as it was rather than holding half a batch.

        Returns:
            How many samples were added.

        Raises:
            CalibrationError: If any row is the wrong length or fails the same
                checks :meth:`add_sample` applies.
        """
        staged: list[CalibrationSample] = []
        for index, row in enumerate(samples):
            if len(row) != 4:
                raise CalibrationError(
                    f"sample {index} has {len(row)} values, expected "
                    "(pitch, yaw, target_x, target_y)"
                )
            staged.append(self._validated(row[0], row[1], row[2], row[3]))
        self._samples.extend(staged)
        return len(staged)

    def clear(self) -> None:
        """Discard everything collected so far."""
        self._samples.clear()

    @property
    def samples(self) -> tuple[CalibrationSample, ...]:
        """Everything collected, in the order it arrived."""
        return tuple(self._samples)

    @property
    def ready(self) -> bool:
        """Whether there are enough samples to be worth fitting."""
        return len(self._samples) >= self.min_samples

    def as_array(self) -> NDArray[np.float64]:
        """The samples as an ``(n, 4)`` float array, ready for the fitter."""
        if not self._samples:
            return np.empty((0, 4), dtype=np.float64)
        return np.asarray([s.as_tuple() for s in self._samples], dtype=np.float64)

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[CalibrationSample]:
        return iter(self._samples)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n={len(self._samples)}, "
            f"min_samples={self.min_samples}, ready={self.ready})"
        )

    def fit(
        self,
        *,
        degree: int = ROBUST_DEFAULT_DEGREE,
        mad_factor: float = DEFAULT_MAD_FACTOR,
        min_keep: int = DEFAULT_MIN_KEEP,
        name: str = "default",
        validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
        validation_seed: int = DEFAULT_VALIDATION_SEED,
    ) -> FitResult:
        """Fit a profile from the collected samples, via the robust path.

        The robust path is what the shipping system uses, which is where the
        effective degree of 3 comes from (see the fitter's module docstring).

        Args:
            degree: Polynomial degree.
            mad_factor: Outlier rejection threshold in robust standard deviations.
            min_keep: Refuse to refit if fewer than this many samples remain.
            name: Name for the resulting profile.
            validation_fraction: Portion held out to measure the profile's
                ``validation_error``.
            validation_seed: Seed for the hold-out split.

        Returns:
            The fit result, whose profile carries this collector's recorded
            screen and camera resolutions.

        Raises:
            CalibrationError: If there are fewer than :attr:`min_samples`
                samples, or scikit-learn is not installed.
        """
        if not self.ready:
            raise CalibrationError(
                f"only {len(self._samples)} calibration samples collected, at least "
                f"{self.min_samples} are needed. Collect more before fitting."
            )
        _log.info("fitting a degree-%d calibration from %d samples", degree, len(self._samples))
        return robust_fit_samples(
            self.as_array(),
            degree=degree,
            mad_factor=mad_factor,
            min_keep=min_keep,
            name=name,
            screen_size=self.screen_size,
            camera_size=self.camera_size,
            validation_fraction=validation_fraction,
            validation_seed=validation_seed,
        )
