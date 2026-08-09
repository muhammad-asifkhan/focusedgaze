"""The calibration sweep's geometry and coverage accounting.

`calibration/ui.py` is hardware-gated: it needs a screen and a person following
a dot, and the default suite cannot cover that. What it CAN cover is everything
in the module that is a pure function, which is deliberately most of it — the
path geometry, the coverage accounting, the median, the drift measurement.

What is **not** covered here, stated plainly rather than implied by a coverage
number: `collect_pursuit_samples` driving a real tracker while a real person
follows the dot. That is exercised by `test_calibration_ui_hardware.py` under
the `hardware` marker, and by nothing in CI.
"""

from __future__ import annotations

import pytest

from focusedgaze.calibration.ui import (
    DEFAULT_MAD_FACTOR,
    DEFAULT_MIN_KEEP,
    DEFAULT_SWEEP_DEGREE,
    REFERENCE_SAMPLE_COUNT,
    THIN_REGION_SAMPLES,
    collect_pursuit_samples,
    iter_dwell_targets,
    measure_drift,
    median_angle,
    pursuit_path,
    region_of,
    summarise_coverage,
    sweep_duration_for,
)
from focusedgaze.exceptions import CalibrationError
from focusedgaze.types import GazeResult, GazeStatus

# ---------------------------------------------------------------------------
# The shipping constants.
# ---------------------------------------------------------------------------


def test_the_sweep_defaults_are_the_shipping_ones() -> None:
    """1728 in-zone samples, degree 3, MAD 2.5, min_keep 60. Audit section 50."""
    assert REFERENCE_SAMPLE_COUNT == 1728
    assert DEFAULT_SWEEP_DEGREE == 3
    assert DEFAULT_MAD_FACTOR == 2.5
    assert DEFAULT_MIN_KEEP == 60


def test_the_sweep_defaults_agree_with_the_fitter() -> None:
    """Restating them here is only safe if they cannot drift apart."""
    from focusedgaze.calibration import fitter

    assert DEFAULT_SWEEP_DEGREE == fitter.ROBUST_DEFAULT_DEGREE
    assert DEFAULT_MAD_FACTOR == fitter.DEFAULT_MAD_FACTOR
    assert DEFAULT_MIN_KEEP == fitter.DEFAULT_MIN_KEEP


# ---------------------------------------------------------------------------
# Path geometry.
# ---------------------------------------------------------------------------


def test_the_sweep_reaches_every_corner() -> None:
    """The fit must be taught about the corners, not left to extrapolate.

    Both recorded runs found their worst error at a corner.
    """
    path = pursuit_path(rows=5, samples_per_row=10, margin=0.05)
    xs = [p[0] for p in path.points]
    ys = [p[1] for p in path.points]
    assert min(xs) == pytest.approx(0.05)
    assert max(xs) == pytest.approx(0.95)
    assert min(ys) == pytest.approx(0.05)
    assert max(ys) == pytest.approx(0.95)


def test_the_dot_never_jumps_between_rows() -> None:
    """A jump loses the user, and samples collected while they reacquire it are
    labelled with a target they were not looking at yet.

    Those mislabels are not obviously wrong, which is what makes them dangerous:
    they land in the training set looking exactly like good data. The sweep
    alternates direction so consecutive points are always adjacent.
    """
    path = pursuit_path(rows=4, samples_per_row=8)
    worst = 0.0
    for (x0, y0), (x1, y1) in zip(path.points, path.points[1:]):
        worst = max(worst, abs(x1 - x0) + abs(y1 - y0))
    # One step along a row, or one row down. Never a full-width jump.
    assert worst < 0.35, f"the dot jumps {worst:.2f} of the screen somewhere"


def test_alternate_rows_run_in_opposite_directions() -> None:
    path = pursuit_path(rows=2, samples_per_row=4, margin=0.1)
    first_row = [p[0] for p in path.points[:4]]
    second_row = [p[0] for p in path.points[4:]]
    assert first_row == sorted(first_row)
    assert second_row == sorted(second_row, reverse=True)


def test_position_is_interpolated_along_the_path() -> None:
    path = pursuit_path(rows=2, samples_per_row=2, margin=0.0)
    assert path.at(0.0) == pytest.approx((0.0, 0.0))
    assert path.at(1.0) == pytest.approx((0.0, 1.0))
    mid = path.at(0.5)
    assert 0.0 <= mid[0] <= 1.0 and 0.0 <= mid[1] <= 1.0


def test_a_fraction_outside_the_path_is_clamped_not_extrapolated() -> None:
    path = pursuit_path(rows=2, samples_per_row=2, margin=0.0)
    assert path.at(-5.0) == path.at(0.0)
    assert path.at(5.0) == path.at(1.0)


@pytest.mark.parametrize("rows,per_row", [(1, 10), (5, 1), (0, 0)])
def test_a_degenerate_sweep_is_refused(rows: int, per_row: int) -> None:
    with pytest.raises(CalibrationError, match="at least 2 rows"):
        pursuit_path(rows=rows, samples_per_row=per_row)


@pytest.mark.parametrize("margin", [-0.1, 0.5, 0.9])
def test_a_nonsense_margin_is_refused(margin: float) -> None:
    with pytest.raises(CalibrationError, match="margin"):
        pursuit_path(margin=margin)


# ---------------------------------------------------------------------------
# Coverage: the thing that actually decides accuracy.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (0.05, 0.05, (0, 0)), (0.5, 0.05, (1, 0)), (0.95, 0.05, (2, 0)),
        (0.05, 0.5, (0, 1)),  (0.5, 0.5, (1, 1)),  (0.95, 0.5, (2, 1)),
        (0.05, 0.95, (0, 2)), (0.5, 0.95, (1, 2)), (0.95, 0.95, (2, 2)),
    ],
)
def test_regions_line_up_with_the_accuracy_grid(x: float, y: float, expected: tuple) -> None:
    """Coverage during calibration and error during measurement must be reported
    over the SAME regions, or comparing them is meaningless.

    That comparison is the diagnostic: a region thin in the sweep is the region
    that will be worst in the test.
    """
    assert region_of(x, y) == expected


def test_a_full_sweep_covers_every_region() -> None:
    path = pursuit_path(rows=9, samples_per_row=27)
    samples = [(0.0, 0.0, x, y) for x, y in path.points]
    coverage = summarise_coverage(samples)
    assert coverage.empty_regions == ()
    assert coverage.total == len(path.points)


def test_a_sweep_that_missed_a_region_says_so() -> None:
    """Section 50's central finding: coverage decides accuracy, and a silently
    uncovered region is where the polynomial will extrapolate."""
    samples = [(0.0, 0.0, 0.1, 0.1)] * 200          # top-left only
    coverage = summarise_coverage(samples)
    assert (2, 2) in coverage.empty_regions
    assert len(coverage.empty_regions) == 8
    text = "\n".join(coverage.lines())
    assert "WARNING" in text and "extrapolate" in text


def test_a_thin_region_is_a_note_not_a_warning() -> None:
    """Thin and empty are different problems and get different words."""
    samples = []
    for r in range(3):
        for c in range(3):
            count = 200 if (c, r) != (2, 2) else THIN_REGION_SAMPLES - 1
            samples += [(0.0, 0.0, 0.17 + c / 3, 0.17 + r / 3)] * count
    coverage = summarise_coverage(samples)
    assert coverage.empty_regions == ()
    assert coverage.thin_regions == ((2, 2),)
    text = "\n".join(coverage.lines())
    assert "NOTE" in text and "WARNING" not in text


def test_coverage_counts_by_target_not_by_where_the_eye_went() -> None:
    """The fit is being taught about the dot's position, so that is what is
    counted. Counting by prediction would report coverage of wherever the model
    already thinks you are looking, which is circular."""
    samples = [(9.0, 9.0, 0.1, 0.1)] * 5            # angles nowhere near the target
    coverage = summarise_coverage(samples)
    assert coverage.per_region[(0, 0)] == 5


def test_a_malformed_sample_row_is_refused() -> None:
    with pytest.raises(CalibrationError, match="4 values"):
        summarise_coverage([(0.1, 0.2, 0.3)])


# ---------------------------------------------------------------------------
# Median and drift.
# ---------------------------------------------------------------------------


def test_the_median_ignores_a_blink_that_a_mean_would_absorb() -> None:
    """Matching the legacy tool's choice, and the reason for it."""
    readings = [(0.1, 0.2)] * 10 + [(9.9, -9.9)]
    pitch, yaw = median_angle(readings)
    assert pitch == pytest.approx(0.1)
    assert yaw == pytest.approx(0.2)


def test_the_median_of_nothing_is_none_rather_than_zero() -> None:
    """Zero is a plausible angle. None is not, which is the point."""
    assert median_angle([]) is None


def test_an_even_number_of_readings_averages_the_middle_two() -> None:
    assert median_angle([(0.0, 0.0), (1.0, 2.0)]) == pytest.approx((0.5, 1.0))


class _Profile:
    """Maps angles to screen positions with a fixed offset."""

    def __init__(self, offset: tuple[float, float] = (0.0, 0.0)) -> None:
        self._offset = offset

    def apply(self, pitch: float, yaw: float) -> tuple[float, float]:
        return 0.5 + self._offset[0], 0.5 + self._offset[1]


def test_drift_is_the_gap_between_where_you_looked_and_where_the_model_says() -> None:
    dx, dy = measure_drift([(0.0, 0.0)] * 5, _Profile((-0.07, -0.04)))
    assert dx == pytest.approx(-0.07)
    assert dy == pytest.approx(-0.04)


def test_no_readings_means_no_drift_rather_than_an_error() -> None:
    """An un-drifted run is the normal case and must not need special handling."""
    assert measure_drift([], _Profile()) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# The dwell/sample phase split.
# ---------------------------------------------------------------------------


def test_dwell_comes_before_collection_at_every_point() -> None:
    """Without the dwell, the samples include the saccade that got there."""
    ticks = iter([i * 0.1 for i in range(10_000)])
    phases = list(
        iter_dwell_targets(
            [(0.1, 0.1), (0.9, 0.9)],
            dwell_seconds=0.3, sample_seconds=0.3,
            clock=lambda: next(ticks),
        )
    )
    for target in ((0.1, 0.1), (0.9, 0.9)):
        this_point = [collecting for t, collecting in phases if t == target]
        assert this_point[0] is False, "collection started before the dwell"
        assert True in this_point, "no collection phase at all"
        # Once collecting starts it does not go back to dwelling.
        first_true = this_point.index(True)
        assert all(this_point[first_true:])


# ---------------------------------------------------------------------------
# The collection loop, driven by a stub rather than a person.
# ---------------------------------------------------------------------------


class _Tracker:
    """Yields scripted results. Stands in for a person following the dot."""

    def __init__(self, results: list) -> None:
        self._results = results
        self._i = 0

    def read(self):
        if self._i >= len(self._results):
            return None
        out = self._results[self._i]
        self._i += 1
        return out


def _ok(pitch: float, yaw: float) -> GazeResult:
    return GazeResult(
        x=0.5, y=0.5, pitch=pitch, yaw=yaw, distance_cm=55.0,
        status=GazeStatus.OK, timestamp=0.0,
    )


def test_collection_labels_each_sample_with_the_dot_position_at_that_moment() -> None:
    """This is the loop's whole job, and mislabelling is silent."""
    ticks = iter([i * 1.0 for i in range(100)])
    path = pursuit_path(rows=2, samples_per_row=2, margin=0.0, seconds=4.0)
    samples = collect_pursuit_samples(
        _Tracker([_ok(0.1, 0.2)] * 10), path, clock=lambda: next(ticks)
    )
    # The clock is consumed once for the start time, then once per iteration:
    # elapsed reads 1, 2, 3 before 4 ends the sweep at seconds=4.0.
    assert len(samples) == 3
    for pitch, yaw, tx, ty in samples:
        assert (pitch, yaw) == (0.1, 0.2)
        assert 0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0


def test_a_reading_with_no_angle_is_skipped_rather_than_ending_the_sweep() -> None:
    """A blink is not a failure. The sweep continues."""
    ticks = iter([i * 1.0 for i in range(100)])
    blink = GazeResult.unavailable(GazeStatus.NO_FACE, 0.0)
    path = pursuit_path(rows=2, samples_per_row=2, margin=0.0, seconds=4.0)
    samples = collect_pursuit_samples(
        _Tracker([_ok(0.1, 0.2), blink, _ok(0.3, 0.4), blink]),
        path, clock=lambda: next(ticks),
    )
    assert len(samples) == 2, "a blink ended the sweep or was recorded as data"


def test_the_sweep_stops_when_the_source_runs_out() -> None:
    ticks = iter([i * 0.01 for i in range(10_000)])
    path = pursuit_path(rows=2, samples_per_row=2, margin=0.0, seconds=60.0)
    samples = collect_pursuit_samples(
        _Tracker([_ok(0.1, 0.2)] * 3), path, clock=lambda: next(ticks)
    )
    assert len(samples) == 3


def test_the_caller_is_told_where_to_draw_the_dot() -> None:
    """Drawing lives with the caller, so this module needs no GUI toolkit."""
    seen: list[tuple[float, float]] = []
    ticks = iter([i * 1.0 for i in range(100)])
    path = pursuit_path(rows=2, samples_per_row=2, margin=0.0, seconds=4.0)
    collect_pursuit_samples(
        _Tracker([_ok(0.1, 0.2)] * 10), path,
        on_frame=lambda target, _result: seen.append(target),
        clock=lambda: next(ticks),
    )
    assert len(seen) == 3


def test_sweep_duration_scales_with_the_frame_rate() -> None:
    assert sweep_duration_for(pursuit_path(), REFERENCE_SAMPLE_COUNT, 30.0) == 58
    with pytest.raises(CalibrationError, match="fps"):
        sweep_duration_for(pursuit_path(), 100, 0.0)
