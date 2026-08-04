"""Which implementation the golden tests exercise.

The point of a golden file is that the ASSERTIONS never change while the code
under them does. So the test files import from here, and only this module knows
whether "the implementation" means the legacy `gaze-detection/` scripts or the
new `focusedgaze` package.

  Phase 1     legacy only — proves the harness reproduces current behaviour
  Phase 2+    prefers focusedgaze, falls back to legacy
  Phase 8     legacy path deleted

Selection order:
  1. FOCUSEDGAZE_GOLDEN_IMPL=legacy|sdk  forces one
  2. otherwise: sdk if importable, else legacy
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Rule 4: no hard-coded absolute paths, in the harness any more than in the
# package. Environment variable first, side-by-side checkout as the fallback.
LEGACY_DIR = pathlib.Path(
    os.environ.get("FOCUSEDGAZE_LEGACY_DIR") or (_REPO_ROOT.parent / "gaze-detection")
)


class ImplUnavailable(RuntimeError):
    """Raised when neither implementation can be loaded; tests skip on this."""


def _load_legacy() -> dict[str, Any]:
    """Import the original scripts, with their working-directory assumptions.

    They resolve models via relative paths and build an ONNX session at import
    time, so this changes directory for the duration and restores it after.
    """
    if not LEGACY_DIR.is_dir():
        raise ImplUnavailable(f"legacy pipeline not found at {LEGACY_DIR}")

    prev = pathlib.Path.cwd()
    os.chdir(LEGACY_DIR)
    if str(LEGACY_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_DIR))
    try:
        import calibration_utils
        import positioning_gate

        cal_path = LEGACY_DIR / "models" / "calibration_model.pkl"
        if not cal_path.exists():
            raise ImplUnavailable(f"no calibration model at {cal_path}")
        model = calibration_utils.load_calibration(str(cal_path))

        import gaze_pipeline
        import gaze_server  # heavy: builds the ONNX session

        def make_landmarker():
            """A FaceLandmarker configured exactly as the live system does."""
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            return mp_vision.FaceLandmarker.create_from_options(
                mp_vision.FaceLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=str(LEGACY_DIR / "face_landmarker.task")
                    ),
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_facial_transformation_matrixes=True,
                )
            )

        return {
            "name": "legacy",
            "apply_calibration": lambda p, y: calibration_utils.apply_calibration(model, p, y),
            "OneEuro": gaze_server.OneEuro,
            "positioning": positioning_gate,
            "pipeline": gaze_pipeline,          # Tier 2: frames -> (pitch, yaw)
            "make_landmarker": make_landmarker,
        }
    except ImplUnavailable:
        raise
    except Exception as exc:
        raise ImplUnavailable(f"legacy import failed: {exc!r}") from exc
    finally:
        os.chdir(prev)


def _load_sdk() -> dict[str, Any]:
    """Import the new package. Not available until Phase 2."""
    try:
        from focusedgaze.core import filters, positioning  # noqa: F401
    except ImportError as exc:
        raise ImplUnavailable(f"focusedgaze core not implemented yet: {exc}") from exc
    raise ImplUnavailable("focusedgaze core is a Phase 2 stub")


_CACHE: dict[str, Any] | None = None


def get_impl() -> dict[str, Any]:
    """Return the implementation under test, loading it once per session."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    forced = os.environ.get("FOCUSEDGAZE_GOLDEN_IMPL", "").lower()
    order = {"legacy": [_load_legacy], "sdk": [_load_sdk]}.get(forced, [_load_sdk, _load_legacy])

    errors = []
    for loader in order:
        try:
            _CACHE = loader()
            return _CACHE
        except ImplUnavailable as exc:
            errors.append(str(exc))
    raise ImplUnavailable("; ".join(errors))
