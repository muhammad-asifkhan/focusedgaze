"""The extracted positioning gate must match the pre-refactor recording.

Phase 2 equivalence proof for `core/positioning.py`, measured against the same
Tier 1 fixture the legacy code is measured on.

Also pins the behaviour change that was the POINT of the extraction: focal
length is now an argument, so the result no longer depends on the process
working directory.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from focusedgaze.core.positioning import (
    LEFT_IRIS_CENTER,
    NOSE_TIP,
    RIGHT_IRIS_CENTER,
    FocalCalibration,
    PositioningConfig,
    PositioningGate,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "tier1"
TOL = 1e-9


class FakeLandmark:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z


def _landmarks(ipd_frac: float, nose_dx: float) -> list[FakeLandmark]:
    cx, cy = 0.5 + nose_dx, 0.5
    lm = [FakeLandmark(0.5, 0.5) for _ in range(478)]
    lm[LEFT_IRIS_CENTER] = FakeLandmark(cx - ipd_frac / 2, cy)
    lm[RIGHT_IRIS_CENTER] = FakeLandmark(cx + ipd_frac / 2, cy)
    lm[NOSE_TIP] = FakeLandmark(cx, cy)
    return lm


def _golden() -> dict:
    path = FIXTURES / "positioning_gate.json"
    if not path.exists():
        pytest.skip("fixture missing — run tests/golden/record_tier1.py")
    return json.loads(path.read_text(encoding="utf-8"))


def test_matches_legacy_recording() -> None:
    """Distance, offsets, flags and zone must all be unchanged.

    The fixture records the focal calibration that was in effect when it was
    made, so the comparison is against a known context rather than whatever the
    working directory happens to imply.
    """
    data = _golden()
    override = data.get("focal_override")
    focal = FocalCalibration(float(override[0]), int(override[1])) if override else None
    gate = PositioningGate(focal=focal)
    frame_shape = tuple(data["frame_shape"])

    for case in data["cases"]:
        st = gate.evaluate(_landmarks(case["ipd_frac"], case["nose_dx"]), frame_shape)
        label = f"ipd={case['ipd_frac']} dx={case['nose_dx']}"
        assert st is not None, label
        assert abs(st.distance_cm - case["distance_cm"]) <= TOL, label
        assert abs(st.dx - case["dx"]) <= TOL, label
        assert abs(st.dy - case["dy"]) <= TOL, label
        assert st.distance_ok == case["distance_ok"], label
        assert st.centered == case["centered"], label
        assert st.in_zone == case["in_zone"], label
        assert st.zone == case["zone"], label


def test_result_does_not_depend_on_working_directory(tmp_path, monkeypatch) -> None:
    """The bug this extraction exists to kill.

    The legacy gate loaded models/camera_focal.json via a relative path at
    construction, so identical landmarks produced 117.406 cm or 121.244 cm
    purely from where Python was launched. Focal length is now an argument, so
    the same inputs give the same answer from anywhere.
    """
    focal = FocalCalibration(900.0, 1280)
    gate = PositioningGate(focal=focal)
    lm = _landmarks(0.065, 0.0)

    monkeypatch.chdir(tmp_path)
    from_tmp = gate.evaluate(lm, (720, 1280, 3))
    monkeypatch.chdir(pathlib.Path(__file__).parent)
    from_tests = gate.evaluate(lm, (720, 1280, 3))

    assert from_tmp is not None and from_tests is not None
    assert from_tmp.distance_cm == from_tests.distance_cm


def test_without_focal_falls_back_to_assumed_fov() -> None:
    """No calibration is a supported state, not an error — but a different one."""
    lm = _landmarks(0.065, 0.0)
    measured = PositioningGate(focal=FocalCalibration(900.0, 1280))
    assumed = PositioningGate()

    a = measured.evaluate(lm, (720, 1280, 3))
    b = assumed.evaluate(lm, (720, 1280, 3))
    assert a is not None and b is not None
    assert a.distance_cm != b.distance_cm, "a measured focal should change the answer"


def test_focal_rescales_with_capture_width() -> None:
    """A focal measured at 1280 px must still be right at 640 px."""
    f = FocalCalibration(900.0, 1280)
    assert f.for_width(1280) == 900.0
    assert f.for_width(640) == 450.0


def test_focal_round_trips_through_a_file(tmp_path) -> None:
    """Save/load must preserve the calibration exactly."""
    f = FocalCalibration(912.345, 1280)
    path = tmp_path / "nested" / "camera_focal.json"
    f.save(path)
    assert FocalCalibration.load(path) == f


def test_degenerate_landmarks_return_none() -> None:
    """Eyes under a pixel apart cannot yield a distance."""
    gate = PositioningGate()
    lm = _landmarks(0.0, 0.0)          # both irises at the same point
    assert gate.evaluate(lm, (720, 1280, 3)) is None


def test_measure_focal_inverts_the_distance_formula() -> None:
    """Measuring at a known distance must reproduce that distance."""
    gate = PositioningGate()
    cfg = PositioningConfig()
    ipd_px = 120.0
    cal = gate.measure_focal(ipd_px, 1280, known_distance_cm=50.0)
    measured = PositioningGate(focal=cal)
    got = measured.focal_px(1280) * cfg.real_ipd_cm / ipd_px
    assert abs(got - 50.0) <= 1e-9
