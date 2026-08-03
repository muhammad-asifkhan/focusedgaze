"""Tier 2 golden regression: real frames -> (pitch, yaw).

Covers the one stage Tier 1 cannot: MediaPipe landmarks, the smoothed square
face crop, and L2CS ONNX inference. Needs the recorded frames, which contain a
real face and are therefore gitignored — so this is marked `hardware` and is
excluded from CI by default (rule 5).

To run it you need a fixture. Make your own:

    python tests/golden/record_tier2.py --frames 60
    pytest tests/test_golden_tier2.py -m hardware

The expected values are recorded from the pipeline at record time, so the
fixture is self-consistent whoever is in front of the camera.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from golden.adapters import ImplUnavailable, get_impl

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "tier2"
MANIFEST = FIXTURES / "manifest.json"

# Looser than Tier 1's 1e-9. Inference runs on a GPU execution provider whose
# kernels are not bit-reproducible across drivers, so demanding exactness here
# would produce flaky failures that say nothing about the refactor. 1e-4 radians
# is ~0.006 degrees — far below the model's own error, so a real regression
# still shows up loudly.
TOL_RAD = 1e-4

pytestmark = pytest.mark.hardware


@pytest.fixture(scope="module")
def fixture_data() -> dict:
    if not MANIFEST.exists():
        pytest.skip(
            f"no Tier 2 fixture at {FIXTURES}. Record one with "
            "`python tests/golden/record_tier2.py --frames 60`."
        )
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_frames_match_manifest_digest(fixture_data: dict) -> None:
    """The recording must not have been silently corrupted or swapped."""
    frames_path = FIXTURES / fixture_data["frames_file"]
    if not frames_path.exists():
        pytest.skip(f"frames file missing: {frames_path.name}")

    h = hashlib.sha256()
    with open(frames_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == fixture_data["frames_sha256"], (
        "frames.npz does not match the digest in manifest.json — re-record it"
    )


def test_pitch_yaw_matches_golden(fixture_data: dict) -> None:
    """Frames -> (pitch, yaw) must survive the extraction unchanged.

    Replays the recording through whichever implementation is selected. Face
    detection is stateful (the bounding box is smoothed across frames), so the
    frames must be fed in their original order from a clean start — which is
    exactly why this replays a sequence rather than isolated images.
    """
    try:
        impl = get_impl()
    except ImplUnavailable as exc:
        pytest.skip(f"no implementation available: {exc}")

    np = pytest.importorskip("numpy")
    frames_path = FIXTURES / fixture_data["frames_file"]
    if not frames_path.exists():
        pytest.skip(f"frames file missing: {frames_path.name}")
    frames = np.load(frames_path)["frames"]

    pipeline = impl.get("pipeline")
    if pipeline is None:
        pytest.skip("selected implementation exposes no frame pipeline yet (Phase 2)")

    pipeline.reset_bbox_smoothing()
    landmarker = impl["make_landmarker"]()

    worst = 0.0
    compared = 0
    for expected in fixture_data["expected"]:
        i = expected["frame_index"]
        pitch, yaw, _bbox = pipeline.get_gaze_reading(frames[i], landmarker, i)

        if expected["pitch"] is None:
            assert pitch is None, f"frame {i}: expected no face, got a reading"
            continue
        assert pitch is not None, f"frame {i}: expected a face, got none"
        worst = max(worst, abs(pitch - expected["pitch"]), abs(yaw - expected["yaw"]))
        compared += 1

    assert compared > 0, "fixture contained no frames with a face"
    assert worst <= TOL_RAD, (
        f"pitch/yaw drifted by {worst:.3e} rad over {compared} frames "
        f"(tolerance {TOL_RAD:.0e})"
    )
