"""Pure, I/O-free gaze estimation core.

    from focusedgaze.core import GazeEstimator

    est = GazeEstimator(profile=CalibrationProfile.load("alice"))
    result = est.process(frame_bgr, timestamp=t)

Frames in, results out. Nothing here opens a camera, a window or a socket, and
nothing here downloads: a missing model raises with the command that fetches it.
That is what lets the whole pipeline run in CI against a recorded fixture.

Every piece of per-stream state that was a module-level global in the legacy
pipeline is now instance state, so two estimators in one process do not interfere.

WHY THESE EXPORTS ARE LAZY
---------------------------
``focusedgaze.config`` imports :class:`~focusedgaze.core.positioning.PositioningConfig`
from this package, and :mod:`focusedgaze.core.estimator` imports ``GazeConfig``
back from ``config``. Importing the estimator eagerly here closes that loop and
fails at import time with a partially initialised module.

Resolving the names on first access (PEP 562) breaks the cycle without moving
either type, and has a second benefit: ``import focusedgaze`` no longer pays for
MediaPipe, which costs seconds and loads native libraries that most callers of
the config types never need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - for type checkers, not at runtime
    from .estimator import GazeEstimator
    from .filters import OneEuroFilter, OneEuroFilter2D
    from .landmarks import FaceLandmarker, SmoothedBox
    from .model import GazeModel, decode_angles, select_providers
    from .positioning import FocalCalibration, PositioningGate, PositioningStatus

#: Attribute name -> the submodule that defines it.
_EXPORTS: dict[str, str] = {
    "FaceLandmarker": "landmarks",
    "FocalCalibration": "positioning",
    "GazeEstimator": "estimator",
    "GazeModel": "model",
    "OneEuroFilter": "filters",
    "OneEuroFilter2D": "filters",
    "PositioningGate": "positioning",
    "PositioningStatus": "positioning",
    "SmoothedBox": "landmarks",
    "decode_angles": "model",
    "select_providers": "model",
}

# Written out rather than derived from _EXPORTS: a literal is what static tools
# read, and a computed __all__ makes every name here invisible to them.
__all__ = [
    "FaceLandmarker",
    "FocalCalibration",
    "GazeEstimator",
    "GazeModel",
    "OneEuroFilter",
    "OneEuroFilter2D",
    "PositioningGate",
    "PositioningStatus",
    "SmoothedBox",
    "decode_angles",
    "select_providers",
]


def __getattr__(name: str) -> Any:
    """Resolve a public name to its submodule on first access."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module_name}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
