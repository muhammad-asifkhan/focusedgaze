"""focusedgaze: webcam eye-gaze tracking as a library.

Pipeline: camera frame -> MediaPipe face landmarks -> smoothed square face crop
-> L2CS-Net gaze model (ONNX) -> (pitch, yaw) -> per-person polynomial
calibration -> One Euro filter -> (x, y) normalised over the screen.

Two entry points, layered:

    GazeEstimator       pure and I/O-free: you supply frames, it returns results.
                        Testable without a camera; usable with video files,
                        another capture library, or a shared camera.
    WebcamGazeTracker   convenience wrapper that owns the webcam.

Model weights are NOT distributed with this package. The gaze model derives from
the Gaze360 dataset, which restricts use to non-commercial research; see NOTICE.

STATUS, and please read it before relying on anything below.

This is 0.0.0 and the pipeline is mid-extraction. What is exported here is what
is implemented; what is missing is missing on purpose, because an ``__all__``
entry that cannot be imported is worse than an honest omission.

    Implemented and tested   config, types, exceptions, filters, positioning,
                             assets (via ``focusedgaze.assets``)
    Implemented, UNPROVEN    calibration. It has no tests yet and none of the
                             four mutation checks have been run, so the
                             pure-NumPy ``apply()`` reproducing scikit-learn's
                             term ordering is not yet demonstrated. A wrong
                             polynomial does not raise: it returns a smooth,
                             believable surface in the wrong place. Treat
                             coordinates from it as unverified.
    Not yet implemented      GazeEstimator, WebcamGazeTracker. Phases 2 and 4.

Nothing here imports scikit-learn, websockets, or an ONNX provider at import
time. Fitting a calibration needs scikit-learn, but that import is deferred to
the moment you fit, so a base install imports cleanly (D8).

The library installs no logging handlers. Logging goes to the ``focusedgaze``
logger and configuring it is the application's job.
"""

from __future__ import annotations

# Single source of truth for the version; pyproject reads it via hatch (D6).
__version__ = "0.0.0"

from .calibration import (
    CalibrationCollector,
    CalibrationProfile,
    CalibrationSample,
    FitResult,
    fit_calibration,
    list_profiles,
    migrate_pickle,
    robust_fit_samples,
)
from .config import (
    CameraConfig,
    FilterConfig,
    GazeConfig,
    LandmarkConfig,
    ModelConfig,
    PositioningConfig,
    RuntimeConfig,
)
from .exceptions import (
    CalibrationError,
    CameraError,
    ConfigError,
    GazeError,
    ModelNotFoundError,
    PositioningError,
    ProfileVersionError,
    ProviderError,
)
from .types import FaceObservation, GazeResult, GazeStatus

# The curated public surface. Deliberately narrower than the sum of the
# submodules: the asset registry and the raw fitting internals are reachable as
# ``focusedgaze.assets`` and ``focusedgaze.calibration`` but are not lifted to
# the top level, because they are tools for the CLI and for calibration time
# rather than part of the everyday API.
__all__ = [
    "CalibrationCollector",
    "CalibrationError",
    "CalibrationProfile",
    "CalibrationSample",
    "CameraConfig",
    "CameraError",
    "ConfigError",
    "FaceObservation",
    "FilterConfig",
    "FitResult",
    "GazeConfig",
    "GazeError",
    "GazeResult",
    "GazeStatus",
    "LandmarkConfig",
    "ModelConfig",
    "ModelNotFoundError",
    "PositioningConfig",
    "PositioningError",
    "ProfileVersionError",
    "ProviderError",
    "RuntimeConfig",
    "__version__",
    "fit_calibration",
    "list_profiles",
    "migrate_pickle",
    "robust_fit_samples",
]
