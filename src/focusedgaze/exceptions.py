"""The exception hierarchy.

Every error this package raises derives from :class:`GazeError`, so a caller can
wrap the whole library in one ``except`` without catching unrelated failures
(D7). Nothing here derives from anything but ``Exception`` and its own base.

Two rules the legacy pipeline broke, and which this tree exists to enforce:

* A missing optional dependency must never surface as a bare ``ImportError``.
  The base install is deliberately provider-agnostic, so "no ONNX provider" is
  an expected state with a known remedy, not a crash. :class:`ProviderError`
  carries the remedy in its message (D1).
* A failure must never be signalled by returning ``None``. The legacy code used
  ``None`` for "no face", "out of range" and "something broke" alike, which is
  why a caller could not tell a bad frame from a bad install. Recoverable
  per-frame conditions are reported as a ``GazeStatus`` on the result; genuine
  faults raise from this tree (D7, Phase 3).
"""

from __future__ import annotations

__all__ = [
    "CalibrationError",
    "CameraError",
    "ConfigError",
    "GazeError",
    "ModelNotFoundError",
    "PositioningError",
    "ProfileVersionError",
    "ProviderError",
]


class GazeError(Exception):
    """Base class for every error raised by focusedgaze."""


class ConfigError(GazeError, ValueError):
    """A configuration value is missing, malformed, or out of range.

    Also derives from ``ValueError`` because that is what a caller validating
    their own keyword arguments will already be catching.
    """


class ModelNotFoundError(GazeError, FileNotFoundError):
    """A required model file is absent from the cache and could not be fetched.

    Raised for both the MediaPipe landmarker and the gaze weights. The gaze
    weights are never downloaded automatically (they derive from Gaze360, which
    is non-commercial research only), so this is the expected first-run state
    until the user follows the printed instructions. The message must name the
    remedy, not just the missing path.
    """


class ProviderError(GazeError, RuntimeError):
    """No usable ONNX Runtime execution provider is installed or loadable.

    The base install declares no provider on purpose so that the user chooses
    one. The message must name the extras that would fix it, so this is never
    reported as a bare ``ImportError`` (D1).
    """


class CalibrationError(GazeError):
    """Calibration could not be fitted, loaded, or applied.

    Covers too few samples, a degenerate fit, and a profile that loads but does
    not describe a usable polynomial.
    """


class ProfileVersionError(CalibrationError):
    """A calibration profile's schema version is not supported.

    Separate from :class:`CalibrationError` because the remedy differs: a
    version mismatch is migrated, whereas a bad fit is recollected.
    """


class CameraError(GazeError):
    """A frame source could not be opened, read from, or configured."""


class PositioningError(GazeError):
    """The positioning gate could not be evaluated.

    This is the *fault* case, such as a missing or unreadable focal-length
    calibration. A user who is simply sitting too close is not an error: that is
    an ``OUT_OF_RANGE`` status on the result.
    """
