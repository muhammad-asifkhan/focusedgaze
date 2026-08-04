"""Which implementation the golden tests exercise.

The point of a golden file is that the ASSERTIONS never change while the code
under them does. So the test files import from here, and only this module knows
whether "the implementation" means the legacy `gaze-detection/` scripts or the
new `focusedgaze` package.

  Phase 1     legacy only: proves the harness reproduces current behaviour
  Phase 2+    prefers focusedgaze, falls back to legacy
  Phase 8     legacy path deleted

Selection order:
  1. FOCUSEDGAZE_GOLDEN_IMPL=legacy|sdk  forces one
  2. otherwise: sdk if importable, else legacy
"""

from __future__ import annotations

import hashlib
import json
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


_FIXTURES = _REPO_ROOT / "tests" / "fixtures" / "tier1"


class ImplUnavailable(RuntimeError):
    """Raised when neither implementation can be loaded; tests skip on this."""


class FixtureModelMismatch(RuntimeError):
    """The committed calibration model is not the one the fixture was recorded from.

    Deliberately NOT an ImplUnavailable. A skip would hide this, and hiding it is
    exactly how the previous version of this problem survived undetected.
    """


def _calibration_model_path() -> pathlib.Path:
    """The committed synthetic model, verified by digest against the fixture.

    Audit section 33: this fixture used to load the live profile from the legacy
    tree, which ordinary recalibration overwrites. The expected values were
    pinned while one of the inputs was free to change underneath them, so the
    fixture began failing the first time somebody recalibrated, and reported it
    as a numeric drift of exactly 1.0 with no indication of the real cause.

    The model is now committed next to the fixture and identified by content.
    """
    model_path = _FIXTURES / "synthetic_calibration.pkl"
    fixture_path = _FIXTURES / "calibration_apply.json"

    if not model_path.exists():
        raise ImplUnavailable(
            f"missing {model_path.name}; regenerate with "
            "python tests/golden/make_synthetic_calibration.py"
        )
    if not fixture_path.exists():
        raise ImplUnavailable(f"missing {fixture_path.name}")

    expected = json.loads(fixture_path.read_text(encoding="utf-8")).get("model_sha256")
    actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if expected and actual != expected:
        raise FixtureModelMismatch(
            "calibration fixture and its model disagree.\n"
            f"  file     : {model_path}\n"
            f"  expected : {expected}   (recorded in calibration_apply.json)\n"
            f"  actual   : {actual}\n"
            "The fixture's expected values were recorded from a different model, so any\n"
            "comparison against it is meaningless. Do NOT re-record to make this pass:\n"
            "that trades a pinned baseline for whatever is on disk. Restore the model, or\n"
            "regenerate BOTH together:\n"
            "  python tests/golden/make_synthetic_calibration.py\n"
            "  python tests/golden/record_tier1.py"
        )
    return model_path


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

        model = calibration_utils.load_calibration(str(_calibration_model_path()))

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
    except (ImplUnavailable, FixtureModelMismatch):
        # FixtureModelMismatch must NOT be swallowed into ImplUnavailable below.
        # That would turn a fixture disagreeing with its own input into a skip,
        # which is precisely how the section 33 defect stayed invisible.
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
