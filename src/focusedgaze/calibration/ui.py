"""Smooth-pursuit calibration and the accuracy grid: the on-screen routines.

**Hardware-gated by construction.** Everything here needs a screen to draw on
and a person to look at it. That is why it is the last stub in the package and
why its tests are marked ``hardware``: the default suite cannot cover a human
following a dot, and pretending otherwise with a mocked screen would test the
mock.

WHAT IS HERE AND WHAT DELIBERATELY IS NOT
------------------------------------------
Collection only. The fitting is :func:`~focusedgaze.calibration.fitter.robust_fit_samples`
and the error arithmetic is :mod:`focusedgaze.accuracy`, both of which are pure
and both of which are tested in CI. That split is not tidiness: audit section 50
records that two of the three defects in the legacy accuracy tool were in the
arithmetic, not in the collection, and they survived because the two were welded
together in a script nobody could run without a camera.

So the rule this module follows: **anything that can be decided without a person
in front of the screen is decided somewhere else.**

THE SWEEP
---------
Smooth pursuit rather than a grid of dwell points: the dot moves continuously and
the user follows it, which collects far more samples across far more of the
screen than asking someone to stare at nine positions. The reference sweep
collected **1728 in-zone samples**, fitted at degree 3 with MAD-based outlier
rejection at 2.5 and a floor of 60 kept samples.

Those numbers are the shipping ones and are carried as defaults, not invented
here.

COVERAGE IS THE THING THAT DECIDES ACCURACY
--------------------------------------------
Section 50's central finding: two runs of the same system by the same person
differed by a factor of two, and *which corner was worst inverted between them*,
purely because the sweep covered different parts of the screen well. So this
module reports per-region sample counts after collecting, and warns when a region
is thin, rather than silently fitting a polynomial that will extrapolate there.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from ..exceptions import CalibrationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..types import GazeResult

__all__ = [
    "DEFAULT_MAD_FACTOR",
    "DEFAULT_MIN_KEEP",
    "DEFAULT_SWEEP_DEGREE",
    "REFERENCE_SAMPLE_COUNT",
    "CoverageReport",
    "PursuitPath",
    "collect_dwell_samples",
    "grid_points",
    "pursuit_path",
    "region_of",
    "summarise_coverage",
]

_log = logging.getLogger(__name__)

#: What the reference sweep collected, in-zone. Recorded so a run producing far
#: fewer is visibly unusual rather than merely quiet.
REFERENCE_SAMPLE_COUNT: Final = 1728

#: The shipping fit for a pursuit sweep. `robust_fit_samples` defaults to these;
#: they are restated here because the sweep and the fit have to agree and a
#: reader of this module should not have to go and check.
DEFAULT_SWEEP_DEGREE: Final = 3
DEFAULT_MAD_FACTOR: Final = 2.5
DEFAULT_MIN_KEEP: Final = 60

#: Fraction of the screen edge the sweep stays inside. Matches the 5% inset the
#: accuracy grid tests at, so the fit is asked about the same region it is
#: measured on.
SWEEP_MARGIN: Final = 0.05

#: Below this many samples in a screen region, the fit is extrapolating there.
#: Derived from the reference: 1728 samples over nine regions is ~192 each, and
#: a third of that is where a region stops being meaningfully covered.
THIN_REGION_SAMPLES: Final = 64


@dataclass(frozen=True, slots=True)
class PursuitPath:
    """A parametric path for the dot to follow, in normalised screen coordinates.

    Args:
        points: The path, in order. Normalised ``(x, y)``.
        seconds: How long the full traversal should take.
    """

    points: tuple[tuple[float, float], ...]
    seconds: float

    def at(self, fraction: float) -> tuple[float, float]:
        """Position at ``fraction`` through the path, linearly interpolated."""
        if not self.points:
            raise CalibrationError("an empty pursuit path has no positions")
        clamped = min(max(fraction, 0.0), 1.0)
        if len(self.points) == 1:
            return self.points[0]
        scaled = clamped * (len(self.points) - 1)
        index = min(int(scaled), len(self.points) - 2)
        t = scaled - index
        (x0, y0), (x1, y1) = self.points[index], self.points[index + 1]
        return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)


def pursuit_path(
    *,
    rows: int = 6,
    samples_per_row: int = 64,
    margin: float = SWEEP_MARGIN,
    seconds: float = 45.0,
) -> PursuitPath:
    """A boustrophedon sweep: left to right, down, right to left, and so on.

    Chosen over a raster of separate lines because the dot never jumps: a jump
    loses the user, and the samples collected while they reacquire it are
    labelled with a target they were not yet looking at. Those samples are not
    obviously wrong, which is what makes them dangerous — they land in the
    training set as confident mislabels.

    The path reaches ``margin`` from every edge, so the fit is asked about the
    corners rather than extrapolating into them. Section 50 found the worst error
    at a corner in both recorded runs.

    **``rows`` must be a multiple of 3, and the default is 6 rather than the 5
    the reference sweep used.** Coverage is reported over the 3x3 grid of
    :func:`region_of`, and five evenly spaced rows do not divide evenly into
    three bands: they land 2-1-2, so the middle third of the screen collects
    *half* the samples of the top and bottom on every run, by every user. That is
    not a frame-rate or duration problem and no longer sweep fixes it -- more
    seconds scales all three bands and preserves the ratio. Six rows land 2-2-2.

    Measured on a real run before the change: 362 top, 187 middle, 368 bottom.
    """
    if rows < 2 or samples_per_row < 2:
        raise CalibrationError(
            f"a sweep needs at least 2 rows of 2 points, got {rows}x{samples_per_row}"
        )
    if not (0.0 <= margin < 0.5):
        raise CalibrationError(f"margin must be in [0, 0.5), got {margin}")

    low, high = margin, 1.0 - margin
    points: list[tuple[float, float]] = []
    for r in range(rows):
        y = low + (high - low) * r / (rows - 1)
        xs = [low + (high - low) * i / (samples_per_row - 1) for i in range(samples_per_row)]
        if r % 2:
            xs.reverse()
        points.extend((x, y) for x in xs)
    return PursuitPath(points=tuple(points), seconds=float(seconds))


def _has_angle(result: GazeResult | None) -> bool:
    """The default sample filter: any reading carrying a gaze angle.

    **Deliberately weaker than callers usually want**, and defined here rather
    than beside its users because it is a default argument, which is evaluated
    when the function is defined.

    An out-of-zone reading still carries pitch and yaw -- the estimator's
    positioning gate reports, it does not veto -- so this keeps samples collected
    while the user was too far away or off centre. That is why ``usable`` is a
    parameter: the CLI passes a predicate that also checks the zone, and a caller
    working from a video file with no positioning information falls back to this.
    """
    return (
        result is not None
        and result.pitch is not None
        and result.yaw is not None
    )


def grid_points(
    *,
    rows: int = 6,
    cols: int = 6,
    margin: float = SWEEP_MARGIN,
) -> tuple[tuple[float, float], ...]:
    """Static targets for a dwell calibration, in boustrophedon order.

    THE CASE FOR DWELL OVER PURSUIT
    -------------------------------
    Pursuit asks the user to follow a dot that moves at a rate set by
    ``seconds``, and it works well when the dot is redrawn smoothly. It is
    redrawn once per pipeline iteration, so on a machine running the gaze model
    at 7 fps the dot *steps* rather than glides, and a stepping target does not
    elicit smooth pursuit: the eye makes catch-up saccades instead. Slowing the
    sweep to collect more samples makes it worse, not better -- a dot taking 20
    seconds to cross the screen is effectively stationary, and the eye wanders
    off it. Both failures put noise in ``(pitch, yaw)`` while the label stays
    confident, and noise in a least-squares predictor biases the fitted gain
    toward zero. A measured run at 120 seconds lost horizontal gain (0.79 ->
    0.62) against a shorter one despite collecting three times the samples.

    Dwell has none of that. The target is stationary, so redraw rate is
    irrelevant and fixation is easy; the cost is that it collects samples at
    fewer distinct positions. On slow hardware that trade is worth taking.

    Ordering is boustrophedon so consecutive points are adjacent, which keeps the
    inter-point saccade short and the settling time honest.

    Args:
        rows: Target rows. **Multiple of 3**, for the reason in
            :func:`pursuit_path`: evenly spaced rows must divide evenly into the
            three bands :func:`region_of` reports, or coverage is skewed by
            construction.
        cols: Target columns, same constraint.
        margin: Fraction of the screen edge to stay inside.
    """
    if rows < 2 or cols < 2:
        raise CalibrationError(f"a grid needs at least 2x2 points, got {rows}x{cols}")
    if not (0.0 <= margin < 0.5):
        raise CalibrationError(f"margin must be in [0, 0.5), got {margin}")

    low, high = margin, 1.0 - margin
    points: list[tuple[float, float]] = []
    for r in range(rows):
        y = low + (high - low) * r / (rows - 1)
        xs = [low + (high - low) * c / (cols - 1) for c in range(cols)]
        if r % 2:
            xs.reverse()
        points.extend((x, y) for x in xs)
    return tuple(points)


def collect_dwell_samples(
    tracker: Any,
    points: Sequence[tuple[float, float]],
    *,
    dwell_seconds: float = 1.0,
    sample_seconds: float = 1.5,
    on_frame: Callable[[tuple[float, float], bool, GazeResult | None], None] | None = None,
    usable: Callable[[GazeResult | None], bool] = _has_angle,
    clock: Callable[[], float] = time.monotonic,
) -> list[tuple[float, float, float, float]]:
    """Hold on each target in turn, recording only once the eye has settled.

    The dwell/sample split is the whole point and is why this cannot just read
    continuously: the samples taken while the eye is still travelling to a new
    target carry the *new* target's label and the *old* target's angle, which is
    a confident mislabel of exactly the kind
    :func:`~focusedgaze.calibration.fitter.robust_fit_samples` cannot detect.
    ``collecting`` is False through the settling phase and True through the
    recording window, and only the latter is kept.

    Args:
        tracker: Anything with ``read() -> GazeResult | None``.
        points: Normalised targets, in the order they are shown.
        dwell_seconds: Settling time before recording starts.
        sample_seconds: Recording window per point.
        on_frame: ``(target, collecting, result)``. Where the canvas hooks in;
            the ``collecting`` flag is passed so the display can show the user
            when they are actually being measured.
        clock: Injection seam.

    Returns:
        ``(pitch, yaw, target_x, target_y)`` rows, as the fitter takes.
    """
    samples: list[tuple[float, float, float, float]] = []
    for target, collecting in iter_dwell_targets(
        points,
        dwell_seconds=dwell_seconds,
        sample_seconds=sample_seconds,
        clock=clock,
    ):
        result = tracker.read()
        if result is None:
            break
        if on_frame is not None:
            on_frame(target, collecting, result)
        if collecting and usable(result):
            samples.append((result.pitch, result.yaw, target[0], target[1]))
    return samples


def region_of(x: float, y: float) -> tuple[int, int]:
    """Which third-of-screen cell a normalised point falls in, as ``(col, row)``.

    A 3x3 grid matching the accuracy test points, so coverage during calibration
    and error during measurement are reported over the same regions. Comparing
    them is the whole diagnostic: a region that was thin in the sweep is the
    region that will be worst in the test.
    """
    col = 0 if x < 1 / 3 else (1 if x < 2 / 3 else 2)
    row = 0 if y < 1 / 3 else (1 if y < 2 / 3 else 2)
    return col, row


@dataclass(frozen=True)
class CoverageReport:
    """How the collected samples were spread across the screen.

    Args:
        per_region: ``(col, row) -> count``, for all nine cells.
        total: Samples collected in-zone.
    """

    per_region: dict[tuple[int, int], int]
    total: int

    @property
    def thin_regions(self) -> tuple[tuple[int, int], ...]:
        """Regions with too few samples for the fit to be interpolating there."""
        return tuple(
            cell for cell, count in sorted(self.per_region.items())
            if count < THIN_REGION_SAMPLES
        )

    @property
    def empty_regions(self) -> tuple[tuple[int, int], ...]:
        """Regions with nothing at all. The fit will extrapolate into these."""
        return tuple(
            cell for cell, count in sorted(self.per_region.items()) if count == 0
        )

    def lines(self) -> list[str]:
        """Human-readable coverage, as a grid."""
        names = ("left", "centre", "right")
        rows = ("top", "middle", "bottom")
        headline = (
            f"Collected {self.total} in-zone samples "
            f"(reference sweep: {REFERENCE_SAMPLE_COUNT})."
        )
        out = [headline]
        out.append("Coverage per screen region:")
        for r in range(3):
            cells = [f"{names[c]}={self.per_region.get((c, r), 0):>4}" for c in range(3)]
            out.append(f"  {rows[r]:<7} " + "  ".join(cells))
        if self.empty_regions:
            out.append(
                "WARNING: regions with NO samples. The fit will extrapolate there, "
                "and that is where its error will be worst."
            )
        elif self.thin_regions:
            out.append(
                f"NOTE: {len(self.thin_regions)} region(s) below "
                f"{THIN_REGION_SAMPLES} samples. Expect worse accuracy there."
            )
        return out


def summarise_coverage(samples: Sequence[Sequence[float]]) -> CoverageReport:
    """Count collected samples per screen region.

    Args:
        samples: ``(pitch, yaw, target_x, target_y)`` rows, as
            :func:`~focusedgaze.calibration.fitter.robust_fit_samples` takes.
            Counting is by **target**, i.e. where the dot was, because that is
            what the fit is being taught about.

    Pure, so it is tested in CI even though the collection that produces its
    input is not.
    """
    per_region = {(c, r): 0 for r in range(3) for c in range(3)}
    total = 0
    for row in samples:
        if len(row) < 4:
            raise CalibrationError(
                f"a calibration sample needs 4 values (pitch, yaw, x, y), got {len(row)}"
            )
        per_region[region_of(float(row[2]), float(row[3]))] += 1
        total += 1
    return CoverageReport(per_region=per_region, total=total)


def collect_pursuit_samples(
    tracker: Any,
    path: PursuitPath,
    *,
    on_frame: Callable[[tuple[float, float], GazeResult | None], None] | None = None,
    usable: Callable[[GazeResult | None], bool] = _has_angle,
    clock: Callable[[], float] = time.monotonic,
) -> list[tuple[float, float, float, float]]:
    """Run one sweep and return ``(pitch, yaw, target_x, target_y)`` rows.

    **Hardware.** Needs a tracker delivering real readings and a person following
    the dot. The caller draws: ``on_frame`` receives the dot's current position
    and the reading taken at that moment, and is where a preview window or a
    full-screen canvas hooks in. Drawing is not done here because it would make
    this function need a GUI toolkit, and the sample collection is the part worth
    reusing.

    Args:
        usable: Decides which readings become samples. Defaults to
            :func:`_has_angle`. An earlier version of this docstring claimed a
            reading taken "out of the positioning zone" simply had no angle and
            was therefore dropped for free. **That was false** -- the estimator
            returns out-of-range readings complete with angles -- so every sweep
            silently trained on samples collected from positions the calibration
            would not be valid at. Pass a stricter predicate to exclude them.

    A dropped reading is not an error: it is a blink, or a moment out of the
    zone, and the sweep continues.
    """
    started = clock()
    samples: list[tuple[float, float, float, float]] = []
    while True:
        elapsed = clock() - started
        if elapsed >= path.seconds:
            break
        target = path.at(elapsed / path.seconds)
        result = tracker.read()
        if result is None:
            break
        if on_frame is not None:
            on_frame(target, result)
        if usable(result):
            samples.append((result.pitch, result.yaw, target[0], target[1]))
    return samples


def iter_dwell_targets(
    points: Sequence[tuple[float, float]],
    *,
    dwell_seconds: float = 1.0,
    sample_seconds: float = 1.5,
    clock: Callable[[], float] = time.monotonic,
) -> Iterator[tuple[tuple[float, float], bool]]:
    """Yield ``(target, collecting)`` for a dwell-then-sample grid.

    Used by the accuracy test rather than by calibration: the grid needs the user
    settled on a static point before anything is recorded, or the samples include
    the saccade that got them there.

    ``collecting`` is False during the dwell and True during the sample window,
    which is what keeps the two phases from being confused — the legacy version
    tracked them with a boolean in a nested loop and it was the least readable
    part of the script.
    """
    for target in points:
        for duration, collecting in ((dwell_seconds, False), (sample_seconds, True)):
            phase_started = clock()
            while clock() - phase_started < duration:
                yield target, collecting


def median_angle(readings: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    """Median ``(pitch, yaw)`` over a point's readings, or ``None`` if empty.

    Median rather than mean, matching the legacy tool: a blink or a glance away
    produces an outlier that a mean would absorb into the estimate and a median
    ignores.
    """
    if not readings:
        return None
    pitches = sorted(r[0] for r in readings)
    yaws = sorted(r[1] for r in readings)

    def middle(values: list[float]) -> float:
        n = len(values)
        mid = n // 2
        return values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2.0

    return middle(pitches), middle(yaws)


def measure_drift(
    readings: Sequence[tuple[float, float]],
    profile: Any,
    centre: tuple[float, float] = (0.5, 0.5),
) -> tuple[float, float]:
    """How far the model's idea of the screen centre has drifted.

    The user looks at the centre, and whatever the model reports is compared with
    where they were actually looking. The difference is subtracted from later
    predictions.

    Returns ``(0.0, 0.0)`` when there is nothing to measure, rather than raising:
    an un-drifted run is the normal case and must not need special handling.
    """
    angle = median_angle(readings)
    if angle is None:
        return 0.0, 0.0
    x, y = profile.apply(angle[0], angle[1])
    return x - centre[0], y - centre[1]


def sweep_duration_for(path: PursuitPath, target_samples: int, fps: float) -> float:
    """How long a sweep must run to collect roughly ``target_samples``.

    Convenience for matching the reference sweep on a machine whose frame rate
    differs. At the reference ~30 fps, 1728 samples is about 58 seconds of
    perfect tracking, and rather more in practice because blinks are dropped.
    """
    if fps <= 0:
        raise CalibrationError(f"fps must be positive, got {fps}")
    return math.ceil(target_samples / fps)
