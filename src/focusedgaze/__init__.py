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

    The pipeline is complete: config, types, exceptions, filters, positioning,
    assets, calibration, capture, the gaze core (``GazeEstimator``), the
    convenience wrapper (``WebcamGazeTracker``), the WebSocket server and all six
    CLI commands.

    ``GazeEstimator`` is the foundation and is pure: frames in, results out, no
    camera and no network. ``WebcamGazeTracker`` owns a webcam and is the
    convenience built on top.

Calibration was the migration's highest numerical risk, because a wrong
polynomial does not raise: it returns a smooth, believable surface in the wrong
place. It is now pinned. ``apply()`` reproduces the recorded legacy output
exactly over all 169 fixture cases, term ordering is checked against a real
scikit-learn for degrees 1 to 8, and the four mutations that would produce a
plausible-but-wrong surface are each shown to fail the comparison. See
``tests/test_calibration_profile.py`` and MIGRATION_AUDIT.md section 43.

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
from .capture import (
    Frame,
    FrameSequenceSource,
    FrameSource,
    VideoFileSource,
    WebcamGazeTracker,
    WebcamSource,
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
from .core import GazeEstimator
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
    "Frame",
    "FrameSequenceSource",
    "FrameSource",
    "GazeConfig",
    "GazeError",
    "GazeEstimator",
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
    "VideoFileSource",
    "WebcamGazeTracker",
    "WebcamSource",
    "__version__",
    "fit_calibration",
    "list_profiles",
    "migrate_pickle",
    "robust_fit_samples",
]
