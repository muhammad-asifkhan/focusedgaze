"""Tier 1 golden regression: numeric behaviour must not drift during extraction.

These fixtures were recorded from the unmodified `gaze-detection/` pipeline
before any refactoring (see tests/golden/record_tier1.py). Every later phase has
to reproduce them.

Nothing here needs a camera, a face, or a model file, so it runs in CI on every
push. It is the safety net rule 2 asks for.

Tolerance: 1e-9. The refactor should be arithmetically identical, not merely
close — anything looser would hide a real change. If a phase legitimately needs
a wider tolerance, that is a DEVIATION, not a quiet edit to this number.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from golden.adapters import ImplUnavailable, get_impl

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "tier1"
TOL = 1e-9


def _load(name: str) -> dict:
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"fixture missing: {path.name} — run tests/golden/record_tier1.py")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def impl() -> dict:
    try:
        return get_impl()
    except ImplUnavailable as exc:
        pytest.skip(f"no implementation available: {exc}")


def test_calibration_apply_matches_golden(impl: dict) -> None:
    """(pitch, yaw) -> (x, y) must be unchanged.

    This is the highest-risk surface in the whole migration: Phase 5 replaces a
    pickled scikit-learn pipeline with raw polynomial coefficients evaluated in
    NumPy. Identical outputs are the only proof that swap was faithful.
    """
    data = _load("calibration_apply.json")
    apply_cal = impl["apply_calibration"]

    worst = 0.0
    for case in data["cases"]:
        x, y = apply_cal(case["pitch"], case["yaw"])
        worst = max(worst, abs(x - case["x"]), abs(y - case["y"]))
    assert worst <= TOL, f"calibration drifted by {worst:.3e} (tolerance {TOL:.0e})"


def test_one_euro_matches_golden(impl: dict) -> None:
    """The filter's response to ramp, step and jitter must be unchanged."""
    data = _load("one_euro.json")
    cfg = data["config"]
    f = impl["OneEuro"](cfg["min_cutoff"], cfg["beta"], cfg["d_cutoff"])

    worst = 0.0
    for s in data["samples"]:
        got = float(f.filter(s["raw"], s["t"]))
        worst = max(worst, abs(got - s["filtered"]))
    assert worst <= TOL, f"1-euro filter drifted by {worst:.3e} (tolerance {TOL:.0e})"


@pytest.mark.parametrize("fixture", ["positioning_gate.json", "positioning_gate_nofocal.json"])
def test_positioning_gate_matches_golden(impl: dict, fixture: str) -> None:
    """Distance, centring and the zone decision must be unchanged.

    Run against BOTH focal branches. The legacy gate selects between a measured
    focal length and an assumed-HFOV fallback by whether a file exists, so a
    fixture recorded only with the file present leaves the fallback unpinned —
    and the fallback is what every user who never measured a focal length runs.

    The booleans and the zone string are compared exactly — a gate that changes
    its mind about whether you may calibrate is a behaviour change, however
    small the underlying float move.
    """
    data = _load(fixture)
    mod = impl["positioning"]
    gate = mod.PositioningGate()
    frame_shape = tuple(data["frame_shape"])

    # FINDING (Phase 1): PositioningGate loads models/camera_focal.json through a
    # RELATIVE path, so the distance it reports depends on the process working
    # directory — 117.4 cm vs 121.2 cm for identical landmarks, purely from where
    # you launched Python. The harness pins the recorded focal context here so the
    # comparison is meaningful. Phase 2 removes the ambiguity by making the focal
    # length explicit configuration instead of an implicit file lookup.
    override = data.get("focal_override")
    gate._focal_override = tuple(override) if override else None

    class FakeLandmark:
        __slots__ = ("x", "y", "z")

        def __init__(self, x: float, y: float, z: float = 0.0) -> None:
            self.x, self.y, self.z = x, y, z

    for case in data["cases"]:
        ipd, dx = case["ipd_frac"], case["nose_dx"]
        cx, cy = 0.5 + dx, 0.5
        landmarks = [FakeLandmark(0.5, 0.5) for _ in range(478)]
        if ipd > 0:
            landmarks[mod.LEFT_IRIS_CENTER] = FakeLandmark(cx - ipd / 2, cy)
            landmarks[mod.RIGHT_IRIS_CENTER] = FakeLandmark(cx + ipd / 2, cy)
        landmarks[mod.NOSE_TIP] = FakeLandmark(cx, cy)

        st = gate.evaluate(landmarks, frame_shape)
        label = f"{fixture} ipd={ipd} dx={dx}"

        if case.get("result_is_none"):
            assert st is None, f"{label}: expected no reading from degenerate geometry"
            continue
        assert st is not None, label

        if case["distance_cm"] is None:
            assert st.get("distance_cm") is None, label
        else:
            assert abs(float(st["distance_cm"]) - case["distance_cm"]) <= TOL, label
        assert bool(st.get("distance_ok")) == case["distance_ok"], label
        assert bool(st.get("centered")) == case["centered"], label
        assert bool(st.get("in_zone")) == case["in_zone"], label
        assert st.get("zone") == case["zone"], label


def test_fixtures_contain_nothing_biometric() -> None:
    """Guard rail: Tier 1 must stay numeric so it is safe to commit.

    A future contributor adding a frame dump here would leak a face into git.
    """
    for path in FIXTURES.glob("*"):
        assert path.suffix == ".json", f"unexpected fixture type committed: {path.name}"
        assert path.stat().st_size < 512_000, f"{path.name} is too large to be pure numbers"
