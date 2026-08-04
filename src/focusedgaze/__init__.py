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

STATUS: Phase 1 skeleton. The public names below are the agreed contract and are
exported as they are implemented. See MIGRATION_AUDIT.md for phase status.
"""

from __future__ import annotations

# Single source of truth for the version; pyproject reads it via hatch (D6).
__version__ = "0.0.0"

# The curated public surface. Names are declared here from the start so the
# contract is visible and reviewable, but only imported as their phase lands.
# An __all__ entry that cannot be imported is worse than an honest omission.
__all__ = [
    "__version__",
]
