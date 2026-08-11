"""Phase 5's equivalence proof: the pure-NumPy polynomial must match scikit-learn.

This is the calibration counterpart of `test_core_filters.py` and
`test_core_positioning.py`. Those pin their modules against the same Tier 1
fixtures `test_golden_tier1.py` replays through the legacy code, so both
implementations are measured with one ruler. Calibration had no such file, which
made it the largest unverified surface in the migration (audit section 40.2).

WHY THIS FILE IS WRITTEN THE WAY IT IS
--------------------------------------
A wrong polynomial does not raise. It returns a smooth, believable surface in
the wrong place, and every downstream number stays plausible. So passing is not
the bar: standing rule 2 requires the comparison be shown to CATCH the specific
errors a faithful-looking refactor produces. Four are checked here, each one a
mistake somebody would actually make:

    transposed coefficient order    coefficients attached to the wrong terms
    wrong term ordering             a permuted PolynomialFeatures expansion
    swapped x/y coefficient sets    symmetric-looking, easy to miss in review
    degree mismatch                 evaluating at 2 where the model is degree 3

Measured drifts against the committed fixture when this file was written, at
tolerance 1e-9:

    (control) unmutated                  0.000e+00   exact, not merely within tolerance
    transposed coefficient order         8.318e-01
    wrong term ordering (rows 1 <-> 2)   1.000e+00
    swapped x/y coefficient sets         1.000e+00
    degree 2 instead of degree 3         3.138e-02

The control being exactly zero is the headline: `apply()` reproduces the legacy
scikit-learn pipeline bit for bit over all 169 recorded cases, not approximately.

Needs no camera, no model file and no legacy checkout. The tests that compare
against a real scikit-learn, and those that read the legacy pickle, skip when
`scikit-learn` is absent, which is correct rather than convenient: the base
install deliberately does not ship it, and the guard tests at the bottom run
either way because they are pure NumPy.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from focusedgaze.calibration.fitter import fit_calibration
from focusedgaze.calibration.profile import (
    CalibrationProfile,
    _check_powers_match_sklearn,
    migrate_pickle,
    polynomial_powers,
)
from focusedgaze.exceptions import CalibrationError

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "tier1"
TOL = 1e-9

#: Highest degree checked against a real scikit-learn. Well past the 3 the system
#: uses: the orders agree because both are built from the same construction, so
#: if that ever stops being true it will show up at some degree, not only at 3.
MAX_DEGREE = 8


# --------------------------------------------------------------------------
# Fixture loading
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def recording() -> dict:
    """The recorded (pitch, yaw) -> (x, y) cases and their provenance."""
    path = FIXTURES / "calibration_apply.json"
    if not path.exists():
        pytest.skip("fixture missing: run tests/golden/record_tier1.py")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def profile(recording: dict) -> CalibrationProfile:
    """The fixture's own model, migrated into the new format.

    The digest is ASSERTED, not skipped on. Audit section 33 records what a skip
    costs here: the fixture's expected values were pinned while one of its inputs
    was free to change underneath them, and the resulting failure reported a
    numeric drift with no hint of the real cause. A model that disagrees with the
    recording makes every comparison below meaningless, so it must stop the run
    rather than quietly remove it.
    """
    pytest.importorskip(
        "sklearn",
        reason="reading the legacy pickle needs scikit-learn; it holds live estimator objects",
    )
    path = FIXTURES / "synthetic_calibration.pkl"
    assert path.exists(), (
        f"missing {path.name}; regenerate with "
        "python tests/golden/make_synthetic_calibration.py"
    )
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == recording["model_sha256"], (
        "the committed model is not the one the fixture was recorded from.\n"
        f"  expected {recording['model_sha256']}\n"
        f"  actual   {actual}\n"
        "Do NOT re-record to make this pass: that trades a pinned baseline for "
        "whatever is on disk. Regenerate both together."
    )
    return migrate_pickle(path, name="tier1-replay")


def _worst_drift(prof: CalibrationProfile, recording: dict) -> float:
    """Largest absolute deviation from the recording, over both axes."""
    worst = 0.0
    for case in recording["cases"]:
        x, y = prof.apply(case["pitch"], case["yaw"])
        worst = max(worst, abs(x - case["x"]), abs(y - case["y"]))
    return worst


# --------------------------------------------------------------------------
# Term ordering, against a real scikit-learn
# --------------------------------------------------------------------------


def test_polynomial_powers_matches_sklearn_for_degrees_1_to_8() -> None:
    """`polynomial_powers` must reproduce `PolynomialFeatures.powers_` exactly.

    This is the trap the profile format exists to close. The coefficients come
    out of scikit-learn ordered to match its own expansion; if this package
    generates a different order, they attach to the wrong terms and nothing
    raises.

    Checked across a range of degrees rather than only the 3 the system uses,
    because the claim being made is about the construction, not about one case.
    """
    preprocessing = pytest.importorskip("sklearn.preprocessing")

    for degree in range(1, MAX_DEGREE + 1):
        theirs = preprocessing.PolynomialFeatures(degree=degree).fit(np.zeros((1, 2))).powers_
        ours = polynomial_powers(degree, 2)
        assert ours.shape == theirs.shape, (
            f"degree {degree}: {ours.shape[0]} terms generated, sklearn produced {theirs.shape[0]}"
        )
        assert np.array_equal(ours, theirs), (
            f"degree {degree}: term order diverges from scikit-learn.\n"
            f"  ours   : {ours.tolist()}\n"
            f"  sklearn: {theirs.tolist()}"
        )


def test_apply_matches_a_freshly_fitted_sklearn_prediction() -> None:
    """End to end: fit through this package, predict through scikit-learn.

    `test_apply_matches_the_legacy_recording` proves the migration path is
    faithful for one committed model. This proves the FITTING path is faithful
    too, on data neither side has seen, so a defect that only appears in
    `fit_calibration` rather than in `migrate_pickle` cannot hide.
    """
    linear_model = pytest.importorskip("sklearn.linear_model")
    preprocessing = pytest.importorskip("sklearn.preprocessing")

    rng = np.random.default_rng(20260805)
    pitch = rng.uniform(-0.7, 0.1, 200)
    yaw = rng.uniform(-0.5, 0.5, 200)
    target_x = np.clip(0.5 + 0.9 * yaw + 0.4 * yaw**2 - 0.7 * yaw**3 + 0.2 * pitch * yaw, 0.0, 1.0)
    target_y = np.clip(0.5 + 1.1 * pitch + 0.3 * pitch**2 + 0.5 * pitch**3, 0.0, 1.0)
    samples = np.column_stack([pitch, yaw, target_x, target_y])

    # validation_fraction=0: holding samples out would fit the two sides on
    # different data and measure the split, not the arithmetic.
    ours = fit_calibration(samples, degree=3, name="equivalence", validation_fraction=0.0)

    expand = preprocessing.PolynomialFeatures(degree=3)
    design = expand.fit_transform(np.column_stack([pitch, yaw]))
    reg_x = linear_model.LinearRegression().fit(design, target_x)
    reg_y = linear_model.LinearRegression().fit(design, target_y)

    probe_pitch = rng.uniform(-0.7, 0.1, 500)
    probe_yaw = rng.uniform(-0.5, 0.5, 500)
    probe_design = expand.transform(np.column_stack([probe_pitch, probe_yaw]))
    low, high = ours.clamp
    theirs_x = np.clip(reg_x.predict(probe_design), low, high)
    theirs_y = np.clip(reg_y.predict(probe_design), low, high)

    worst = 0.0
    for i in range(len(probe_pitch)):
        x, y = ours.apply(float(probe_pitch[i]), float(probe_yaw[i]))
        worst = max(worst, abs(x - theirs_x[i]), abs(y - theirs_y[i]))
    assert worst <= TOL, f"pure-NumPy apply() drifted from sklearn by {worst:.3e}"


# --------------------------------------------------------------------------
# The recorded fixture, and the mutations that must break it
# --------------------------------------------------------------------------


def test_apply_matches_the_legacy_recording(
    profile: CalibrationProfile, recording: dict
) -> None:
    """All 169 recorded cases, reproduced by pure NumPy.

    The values were recorded from the unmodified legacy pipeline, which
    evaluated a pickled scikit-learn pipeline. Reproducing them without
    scikit-learn present at apply time is the whole Phase 5 claim.

    Measured 0.000e+00 when written: bit-for-bit, not within tolerance. The
    assertion is against TOL rather than zero because BLAS may legitimately pick
    a different kernel on another machine, and a last-bit difference is not the
    defect this test is looking for.
    """
    assert len(recording["cases"]) == recording["n"]
    worst = _worst_drift(profile, recording)
    assert worst <= TOL, f"calibration drifted by {worst:.3e} (tolerance {TOL:.0e})"


def test_the_recorded_surface_is_genuinely_cubic(profile: CalibrationProfile) -> None:
    """The degree mutation only has teeth if the cubic terms carry real weight.

    This is load-bearing and easy to destroy by accident. If
    `make_synthetic_calibration.py` were ever simplified to a flatter target,
    evaluating the model at degree 2 would produce a drift near zero, the degree
    mutation below would stop failing, and the suite would go on reporting that
    it catches a degree mismatch when it no longer does.

    So the property is asserted rather than left as a comment. Measured when
    written: largest cubic coefficient 0.2805, against 0.9501 for the largest
    overall, i.e. 29.5%.
    """
    total_degree = profile.powers.sum(axis=1)
    cubic = total_degree == 3
    assert cubic.any(), "the fixture model has no cubic terms at all"

    cubic_weight = float(
        max(np.abs(profile.coef_x[cubic]).max(), np.abs(profile.coef_y[cubic]).max())
    )
    largest = float(max(np.abs(profile.coef_x).max(), np.abs(profile.coef_y).max()))
    assert cubic_weight >= 0.05 * largest, (
        f"the fixture surface is nearly flat in its cubic terms "
        f"({cubic_weight:.4f} against {largest:.4f} overall), so a degree mismatch "
        "would not be detectable and test_a_plausible_mutation_is_caught is no "
        "longer proving what it claims. Restore a genuinely cubic target in "
        "tests/golden/make_synthetic_calibration.py."
    )


# ---------------------------------------------------------------------------
# apply_raw: the same arithmetic without the clamp.
#
# `apply` clamps to the screen, which is right for a cursor and wrong for a
# diagnostic: when a session's mapping shifts bodily off one edge, every affected
# point reads as sitting exactly on that edge and the size of the shift is
# unrecoverable. An accuracy report recorded a "raw" prediction of exactly 1.0
# for three separate points, which is what alerted us that the value being stored
# had already been truncated.
# ---------------------------------------------------------------------------


def _flat_profile(x_at_zero: float = 2.5, y_at_zero: float = -1.5) -> CalibrationProfile:
    """A degree-1 profile whose output at (0, 0) is far outside the screen."""
    return CalibrationProfile(
        name="raw",
        degree=1,
        powers=np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int64),
        coef_x=np.zeros(3, dtype=np.float64),
        coef_y=np.zeros(3, dtype=np.float64),
        intercept_x=x_at_zero,
        intercept_y=y_at_zero,
    )


def test_apply_clamps_but_apply_raw_does_not() -> None:
    profile = _flat_profile()
    assert profile.apply(0.0, 0.0) == (1.0, 0.0), "apply must stay on the screen"
    assert profile.apply_raw(0.0, 0.0) == (2.5, -1.5), "apply_raw must not clamp"


def test_apply_is_exactly_apply_raw_clamped() -> None:
    """`apply` delegates rather than duplicating the arithmetic, so the two
    cannot drift apart. Bit-identical, not merely close: `robust_fit_samples`
    compares residuals against a threshold, so a one-ULP difference could change
    which samples are rejected and therefore the final coefficients."""
    profile = _flat_profile(0.4, 0.6)
    low, high = profile.clamp
    for pitch, yaw in [(0.0, 0.0), (0.13, -0.27), (-0.4, 0.4), (1e-9, -1e-9)]:
        raw = profile.apply_raw(pitch, yaw)
        expected = (max(low, min(high, raw[0])), max(low, min(high, raw[1])))
        assert profile.apply(pitch, yaw) == expected


def test_apply_raw_rejects_a_non_finite_angle() -> None:
    """The same guard as `apply`: an infinite angle must not become a silent
    coordinate."""
    profile = _flat_profile()
    for bad in (float("nan"), float("inf")):
        with pytest.raises(CalibrationError, match="finite"):
            profile.apply_raw(bad, 0.0)
        with pytest.raises(CalibrationError, match="finite"):
            profile.apply_raw(0.0, bad)


# ---------------------------------------------------------------------------
# Distance compensation.
#
# A calibration is only valid at the distance it was taught at: screen offset
# for a fixed gaze angle scales with viewing distance, so a profile learned at
# 47 cm and used at 64 cm under-reaches toward the middle of the screen.
# Measured on this project: gains of x=0.67 and y=0.61 against 0.71-0.78
# predicted by the ratio.
# ---------------------------------------------------------------------------


def test_a_profile_without_a_distance_is_never_rescaled() -> None:
    """Every profile fitted before this existed has no distance. They must be
    left exactly alone rather than guessed at."""
    profile = _flat_profile(0.4, 0.6)
    assert profile.distance_cm is None
    assert profile.distance_scale(64.0) == 1.0
    assert profile.rescaled_for(0.2, 0.8, 64.0) == (0.2, 0.8)


def test_an_unknown_current_distance_leaves_the_point_alone() -> None:
    profile = replace(_flat_profile(), distance_cm=47.0)
    assert profile.distance_scale(None) == 1.0
    assert profile.rescaled_for(0.2, 0.8, None) == (0.2, 0.8)


def test_sitting_further_away_stretches_the_prediction_outward() -> None:
    """The correction that matters. Used further away than it was taught, the
    polynomial under-reaches, so the fix is to push points away from centre."""
    profile = replace(_flat_profile(), distance_cm=47.0)
    assert profile.distance_scale(64.0) == pytest.approx(64.0 / 47.0)

    x, y = profile.rescaled_for(0.4, 0.4, 64.0)
    assert x < 0.4 and y < 0.4, "a point left of centre must move further left"
    assert x == pytest.approx(0.5 + (0.4 - 0.5) * (64.0 / 47.0))


def test_sitting_closer_pulls_the_prediction_inward() -> None:
    profile = replace(_flat_profile(), distance_cm=64.0)
    x, _ = profile.rescaled_for(0.0, 0.5, 47.0)
    assert x > 0.0, "calibrated far and used close over-reaches, so points come in"


def test_the_screen_centre_is_the_fixed_point() -> None:
    """Looking straight ahead lands in the same place at any distance."""
    profile = replace(_flat_profile(), distance_cm=47.0)
    assert profile.rescaled_for(0.5, 0.5, 64.0) == (0.5, 0.5)


def test_a_stretched_point_cannot_leave_the_screen() -> None:
    profile = replace(_flat_profile(), distance_cm=30.0)
    x, y = profile.rescaled_for(0.05, 0.95, 90.0)
    low, high = profile.clamp
    assert low <= x <= high and low <= y <= high


def test_the_correction_recovers_a_known_gain_loss() -> None:
    """The measured case, run as arithmetic. A profile taught at 47 cm and used
    at 63.7 cm reads a 0.95-of-screen target as roughly 0.74; correcting for the
    distance should put it back near where it belongs."""
    profile = replace(_flat_profile(), distance_cm=47.0)
    true_target = 0.95
    observed = 0.5 + (true_target - 0.5) * (47.0 / 63.7)   # what compression does
    corrected, _ = profile.rescaled_for(observed, 0.5, 63.7)
    assert corrected == pytest.approx(true_target, abs=1e-9)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_a_nonsense_distance_is_rejected_at_construction(bad) -> None:
    """Zero especially: it is a divisor, and a stored zero would divide by zero
    inside a running tracker."""
    with pytest.raises(CalibrationError, match="distance_cm"):
        replace(_flat_profile(), distance_cm=bad)


def test_the_distance_survives_a_round_trip() -> None:
    profile = replace(_flat_profile(), distance_cm=47.5)
    assert CalibrationProfile.from_dict(profile.to_dict()).distance_cm == 47.5


def test_a_profile_written_before_distance_existed_still_loads() -> None:
    """The field was added without bumping schema_version, because the version
    check is strict equality and a bump would reject every existing profile."""
    document = _flat_profile().to_dict()
    del document["distance_cm"]
    assert CalibrationProfile.from_dict(document).distance_cm is None


def _transposed_coefficients(prof: CalibrationProfile) -> CalibrationProfile:
    """Coefficients attached to the terms in reverse: the classic ordering slip."""
    return replace(prof, coef_x=prof.coef_x[::-1].copy(), coef_y=prof.coef_y[::-1].copy())


def _permuted_term_order(prof: CalibrationProfile) -> CalibrationProfile:
    """A feature expansion whose terms are ordered differently from sklearn's.

    Swaps the two linear terms, so `pitch` and `yaw` exchange coefficients. This
    is what a hand-rolled expansion that got the convention backwards produces.
    """
    powers = prof.powers.copy()
    powers[[1, 2]] = powers[[2, 1]]
    return replace(prof, powers=powers)


def _swapped_axes(prof: CalibrationProfile) -> CalibrationProfile:
    """The x and y coefficient sets exchanged, intercepts included."""
    return replace(
        prof,
        coef_x=prof.coef_y.copy(),
        coef_y=prof.coef_x.copy(),
        intercept_x=prof.intercept_y,
        intercept_y=prof.intercept_x,
    )


def _degree_mismatch(prof: CalibrationProfile) -> CalibrationProfile:
    """The same model evaluated as a quadratic: every cubic term dropped.

    Note the profile refuses to be built with `degree` and `powers` disagreeing,
    so this truncates both together. That validation is itself a guard against
    this mutation arriving by accident.
    """
    keep = prof.powers.sum(axis=1) <= 2
    return replace(
        prof,
        degree=2,
        powers=prof.powers[keep].copy(),
        coef_x=prof.coef_x[keep].copy(),
        coef_y=prof.coef_y[keep].copy(),
    )


MUTATIONS: list[tuple[str, Callable[[CalibrationProfile], CalibrationProfile]]] = [
    ("transposed coefficient order", _transposed_coefficients),
    ("wrong term ordering", _permuted_term_order),
    ("swapped x/y coefficient sets", _swapped_axes),
    ("degree 2 instead of degree 3", _degree_mismatch),
]


@pytest.mark.parametrize("label,mutate", MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_a_plausible_mutation_is_caught(
    profile: CalibrationProfile,
    recording: dict,
    label: str,
    mutate: Callable[[CalibrationProfile], CalibrationProfile],
) -> None:
    """Standing rule 2: a fixture that passes has not been shown to catch anything.

    Each mutation is a wrong polynomial that still evaluates cleanly and returns
    coordinates in range. If the recording did not notice one, it would not
    notice the equivalent bug in a refactor either.
    """
    worst = _worst_drift(mutate(profile), recording)
    assert worst > TOL, (
        f"the recording did not detect '{label}': worst drift {worst:.3e} is within "
        f"the {TOL:.0e} tolerance. The comparison has no teeth against this error, "
        "so passing it proves nothing."
    )


# --------------------------------------------------------------------------
# The term-order guard, which had never executed
# --------------------------------------------------------------------------
#
# `_check_powers_match_sklearn` runs on every path that reads coefficients out
# of scikit-learn, and exists to hard-fail if a future release reorders its
# feature expansion. Coverage showed its body had never been reached by any
# test. A guard that has never run is not a guard: it is a plausible-looking
# block of code with no evidence it does anything.
#
# These need no scikit-learn: the check is pure NumPy, comparing a supplied
# table against the one this package generates.


def test_the_term_order_guard_accepts_the_order_we_generate() -> None:
    """The control. A guard that rejects everything would also pass the test below."""
    for degree in range(1, MAX_DEGREE + 1):
        _check_powers_match_sklearn(polynomial_powers(degree, 2), degree)


def test_the_term_order_guard_rejects_a_permuted_table() -> None:
    """A reordered expansion must stop the fit, not be silently accepted."""
    powers = polynomial_powers(3, 2).copy()
    powers[[1, 2]] = powers[[2, 1]]
    with pytest.raises(CalibrationError, match="term order"):
        _check_powers_match_sklearn(powers, 3)


def test_the_term_order_guard_rejects_a_table_of_the_wrong_size() -> None:
    """A degree-2 expansion offered as degree 3 differs in shape, not just order."""
    with pytest.raises(CalibrationError, match="term order"):
        _check_powers_match_sklearn(polynomial_powers(2, 2), 3)


def test_the_term_order_guard_names_both_orders_when_it_fires() -> None:
    """The message has to be actionable: a bare failure here is very hard to place."""
    powers = polynomial_powers(3, 2).copy()
    powers[[1, 2]] = powers[[2, 1]]
    with pytest.raises(CalibrationError) as excinfo:
        _check_powers_match_sklearn(powers, 3)
    message = str(excinfo.value)
    assert "sklearn" in message and "expected" in message
    assert str(powers.tolist()) in message, "the rejected table is not shown"


# ---------------------------------------------------------------------------
# migrate_pickle carries the held-out error. It used to drop it.
# ---------------------------------------------------------------------------


def test_migration_carries_the_held_out_error_when_the_pickle_has_one() -> None:
    """It used to be dropped, on a documented belief that was simply false.

    The docstring asserted the legacy pickle "never stored one ... so there is
    nothing to migrate and putting a number there would be a fabrication".
    Every one of the nine legacy calibrations in the reference tree carries the
    key, and dropping it destroyed the only field that identifies which fitting
    run produced a profile. Audit 50.4 needed exactly that field to tell two
    accuracy runs apart.

    Built here rather than read from the reference tree, so the test runs
    anywhere: the point is that a value present in the dict survives migration.
    """
    pytest.importorskip("sklearn", reason="the legacy format holds live estimators")
    import pickle
    import tempfile

    from focusedgaze.calibration.profile import migrate_pickle

    source = FIXTURES / "synthetic_calibration.pkl"
    model = pickle.loads(source.read_bytes())
    model["validation_error"] = 0.24400536175553478

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "with_validation.pkl"
        path.write_bytes(pickle.dumps(model))
        migrated = migrate_pickle(path, name="carried")

    assert migrated.validation_error == pytest.approx(0.24400536175553478), (
        "the held-out error was dropped during migration"
    )


def test_migration_still_tolerates_a_pickle_without_a_held_out_error() -> None:
    """The control: absent stays absent, and must not become a fabricated 0.0."""
    pytest.importorskip("sklearn", reason="the legacy format holds live estimators")
    import pickle
    import tempfile

    from focusedgaze.calibration.profile import migrate_pickle

    source = FIXTURES / "synthetic_calibration.pkl"
    model = pickle.loads(source.read_bytes())
    model.pop("validation_error", None)

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "without_validation.pkl"
        path.write_bytes(pickle.dumps(model))
        migrated = migrate_pickle(path, name="absent")

    assert migrated.validation_error is None
