"""The accuracy report: three requirements derived from running the original.

Audit section 50 recorded two runs of the legacy tool and three defects it
exposed. Each is a test here, and each is tested in **both** directions: the bad
state is refused, and the good state is not.

Everything is pure arithmetic over measurements already taken, so all of it runs
in CI. That split is the point: two of the three defects were in the arithmetic,
and they survived because it was welded to a script nobody could run without a
camera.
"""

from __future__ import annotations

import json

import pytest

from focusedgaze.accuracy import (
    DEFAULT_SCREEN_CM,
    DEFAULT_TEST_POINTS,
    AccuracyReport,
    PointMeasurement,
    build_report,
    profile_digest,
    render,
)
from focusedgaze.exceptions import ConfigError

W, H = 34.4, 19.4

#: Run 2 from audit section 50, per-point error in cm.
RUN2_CM = {
    (0.05, 0.05): 7.8, (0.5, 0.05): 5.2, (0.95, 0.05): 4.0,
    (0.05, 0.5): 2.1,  (0.5, 0.5): 1.0,  (0.95, 0.5): 1.6,
    (0.05, 0.95): 1.9, (0.5, 0.95): 3.4, (0.95, 0.95): 3.0,
}

#: Run 1, whose failure pattern is the mirror image of run 2's.
RUN1_CM = {
    (0.05, 0.05): 1.1, (0.5, 0.05): 0.6, (0.95, 0.05): 8.2,
    (0.05, 0.5): 2.0,  (0.5, 0.5): 7.4,  (0.95, 0.5): 10.4,
    (0.05, 0.95): 3.3, (0.5, 0.95): 11.3, (0.95, 0.95): 12.0,
}


def _measurements(errors: dict, n: int = 34, fail: set | None = None) -> list[PointMeasurement]:
    """Points whose predictions sit exactly `errors[target]` cm from the target.

    Offset purely horizontally: the recorded runs captured error MAGNITUDES and
    not directions, and no figure in the report depends on the direction.
    """
    out = []
    for target, cm in errors.items():
        if fail and target in fail:
            out.append(PointMeasurement(target=target, predicted=None, n_samples=0))
        else:
            out.append(
                PointMeasurement(
                    target=target, predicted=(target[0] + cm / W, target[1]), n_samples=n
                )
            )
    return out


def _report(errors: dict = None, **kw) -> AccuracyReport:  # noqa: RUF013
    return build_report(
        _measurements(errors if errors is not None else RUN2_CM, **kw),
        screen_cm=(W, H),
        profile_digest="a" * 64,
        profile_name="run2",
        drift_offset=(-0.07, -0.04),
        provider="DmlExecutionProvider",
    )


# ---------------------------------------------------------------------------
# Requirement 1: refuse to average a partial collection.
# ---------------------------------------------------------------------------


def test_no_average_is_reported_when_any_point_collected_nothing() -> None:
    """The defect: the legacy averaged three survivors of five and reported it
    as the whole measurement, giving 24.4% for a model that measured 9.7%.

    The direction of that error is not predictable, so there is nothing useful
    to return and `None` is the honest answer.
    """
    report = _report(fail={(0.05, 0.05), (0.95, 0.95)})
    assert report.complete is False
    assert report.average_cm is None
    assert report.average_pct_width is None


def test_the_failed_points_are_named_with_their_sample_counts() -> None:
    """Which points failed is the actionable half: it says where to stand."""
    report = _report(fail={(0.05, 0.05)})
    failed = report.failed_points
    assert len(failed) == 1
    assert failed[0].label == "(5,5)"
    assert failed[0].n_samples == 0
    assert all(p.n_samples == 34 for p in report.points if p.measured)


def test_a_partial_average_is_available_but_must_be_asked_for_by_name() -> None:
    """`average_cm_over_measured` is deliberately verbose.

    A caller reaching for it is asserting they know the basis is partial, which
    is exactly what the legacy tool never made anyone say.
    """
    report = _report(fail={(0.05, 0.05)})
    assert report.average_cm is None
    partial = report.average_cm_over_measured
    assert partial is not None
    # Eight points, the 7.8 cm one removed.
    assert partial == pytest.approx((sum(RUN2_CM.values()) - 7.8) / 8, abs=1e-9)


def test_a_complete_measurement_does_report_an_average() -> None:
    """The control. A report that refused unconditionally would pass the above."""
    report = _report()
    assert report.complete is True
    assert report.average_cm == pytest.approx(3.333, abs=0.001)


def test_the_rendered_output_refuses_loudly_and_says_why() -> None:
    text = "\n".join(render(_report(fail={(0.05, 0.05), (0.5, 0.5)})))
    assert "NO AVERAGE REPORTED" in text
    assert "(5,5)" in text and "(50,50)" in text
    assert "24.4%" in text, "the message should cite what this defect once cost"
    assert "Average     :" not in text


def test_thin_sampling_is_flagged_without_suppressing_the_average() -> None:
    """Few samples is a caveat; none is a refusal. They are different."""
    report = _report(n=3)
    assert report.complete is True
    assert report.average_cm is not None
    assert len(report.thin_points) == 9
    assert "Thin sampling at" in "\n".join(render(report))


# ---------------------------------------------------------------------------
# Requirement 2: per-point and per-quadrant, never a single edge average.
# ---------------------------------------------------------------------------


def test_run2_per_point_errors_reproduce_the_recorded_figures() -> None:
    """The arithmetic, checked against real recorded numbers."""
    report = _report(RUN2_CM)
    for point in report.points:
        assert point.error_cm == pytest.approx(RUN2_CM[point.target], abs=1e-9)


def test_rows_and_columns_expose_run1s_direction() -> None:
    """Run 1 degraded to the RIGHT and toward the BOTTOM.

    An edge average showed neither: the legacy summary called it "even accuracy
    across the screen" because a good left edge cancelled a bad right one.
    """
    report = _report(RUN1_CM)
    columns = report.by_column
    rows = report.by_row
    assert columns[0.05] < columns[0.5] < columns[0.95], (
        f"the left-to-right degradation is not visible: {columns}"
    )
    assert rows[0.05] < rows[0.5] < rows[0.95], (
        f"the top-to-bottom degradation is not visible: {rows}"
    )
    assert columns[0.95] == pytest.approx(10.2, abs=0.05)


def test_rows_and_columns_expose_run2s_opposite_direction() -> None:
    """Run 2 was worst at the TOP. The same two groupings show the inverse."""
    report = _report(RUN2_CM)
    rows = report.by_row
    assert rows[0.05] > rows[0.95] > rows[0.5], (
        f"run 2's top-heavy pattern is not visible: {rows}"
    )
    assert rows[0.05] == pytest.approx(5.667, abs=0.01)


def test_the_two_runs_disagree_about_which_corner_is_worst() -> None:
    """The finding that forced a range instead of a number.

    If a future change made the report insensitive to position, this is the test
    that would notice: both runs would start looking alike.
    """
    assert _report(RUN1_CM).worst.label == "(95,95)"
    assert _report(RUN2_CM).worst.label == "(5,5)"
    assert _report(RUN1_CM).best.label == "(50,5)"


def test_the_render_leads_with_per_point_not_with_an_average() -> None:
    lines = render(_report())
    per_point = next(i for i, line in enumerate(lines) if "Per point" in line)
    average = next(i for i, line in enumerate(lines) if line.startswith("Average"))
    assert per_point < average, "the average must not be the first thing read"


def test_no_single_edge_average_is_reported_anywhere() -> None:
    """The specific shape of the legacy defect must not reappear."""
    text = "\n".join(render(_report(RUN1_CM))).lower()
    assert "edges" not in text
    assert "even accuracy" not in text


# ---------------------------------------------------------------------------
# Requirement 3: the result names its input.
# ---------------------------------------------------------------------------


def test_the_report_records_the_profile_digest_and_conditions() -> None:
    """Run 1's model was overwritten twenty minutes after it was measured.

    A result that cannot name its input cannot be reproduced, which is what run 1
    became. Audit 50.4.
    """
    data = _report().to_dict()
    assert data["profile_digest"] == "a" * 64
    assert data["profile_name"] == "run2"
    assert data["screen_cm"] == [W, H]
    assert data["drift_offset"] == [-0.07, -0.04]
    assert data["provider"] == "DmlExecutionProvider"
    assert all(p["n_samples"] == 34 for p in data["points"])


def test_the_report_is_json_serialisable() -> None:
    """It has to survive being written next to the profile it describes."""
    parsed = json.loads(json.dumps(_report().to_dict()))
    assert len(parsed["points"]) == 9


def test_the_digest_is_of_content_not_of_a_file() -> None:
    """So it survives a rename, and identifies a profile whose file was replaced."""

    class FakeProfile:
        def __init__(self, payload: str) -> None:
            self._payload = payload

        def to_json(self) -> str:
            return self._payload

    a = profile_digest(FakeProfile('{"degree": 3}'))
    b = profile_digest(FakeProfile('{"degree": 3}'))
    c = profile_digest(FakeProfile('{"degree": 2}'))
    assert a == b and a != c
    assert len(a) == 64


def test_digesting_something_that_is_not_a_profile_is_refused() -> None:
    with pytest.raises(ConfigError, match="no to_json"):
        profile_digest(object())


# ---------------------------------------------------------------------------
# Units, and the denominator being named.
# ---------------------------------------------------------------------------


def test_errors_are_reported_in_both_units_against_screen_width() -> None:
    """"% of screen" is ambiguous, and the inherited figures never said which."""
    report = _report()
    assert report.average_pct_width == pytest.approx(9.69, abs=0.01)
    centre = report.centre
    assert centre.error_cm == pytest.approx(1.0, abs=1e-9)
    assert centre.error_pct_width == pytest.approx(2.91, abs=0.01)
    assert report.to_dict()["denominator"] == "screen width"
    assert "WIDTH" in "\n".join(render(report))


def test_the_error_is_a_true_distance_using_both_screen_dimensions() -> None:
    """A purely vertical error must scale by HEIGHT, not width.

    Getting this wrong is invisible on a point that happens to be offset
    horizontally, which is every point in the reconstruction above.
    """
    report = build_report(
        [PointMeasurement(target=(0.5, 0.5), predicted=(0.5, 0.6), n_samples=30)],
        screen_cm=(W, H), profile_digest="b" * 64,
    )
    assert report.points[0].error_cm == pytest.approx(0.1 * H, abs=1e-9)


# ---------------------------------------------------------------------------
# Shape and validation.
# ---------------------------------------------------------------------------


def test_the_default_grid_is_the_legacy_one() -> None:
    """3x3, 5% in, corners included: `milestone6_test_accuracy.py:35`."""
    assert len(DEFAULT_TEST_POINTS) == 9
    assert (0.05, 0.05) in DEFAULT_TEST_POINTS
    assert (0.95, 0.95) in DEFAULT_TEST_POINTS
    assert DEFAULT_SCREEN_CM == (34.4, 19.4)


def test_an_empty_measurement_set_is_refused() -> None:
    with pytest.raises(ConfigError, match="at least one"):
        build_report([], profile_digest="c" * 64)


@pytest.mark.parametrize("bad", [(0.0, 19.4), (34.4, 0.0), (-1.0, 19.4)])
def test_a_nonsense_screen_size_is_refused(bad: tuple[float, float]) -> None:
    with pytest.raises(ConfigError, match="positive"):
        build_report(
            [PointMeasurement((0.5, 0.5), (0.5, 0.5), 10)],
            screen_cm=bad, profile_digest="d" * 64,
        )


def test_a_point_with_samples_but_no_prediction_counts_as_failed() -> None:
    """Both halves have to be present for a point to have measured anything."""
    report = build_report(
        [PointMeasurement((0.5, 0.5), None, 12)], profile_digest="e" * 64
    )
    assert report.complete is False
    assert report.points[0].error_cm is None
