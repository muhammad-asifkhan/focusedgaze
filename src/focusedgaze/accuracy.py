"""Accuracy measurement: the port of `milestone6_test_accuracy.py`.

THE SPEC CAME FROM RUNNING THE ORIGINAL, NOT FROM READING IT
=============================================================
Audit section 50 recorded two runs of the legacy tool and three defects it
exposed. Those defects are this module's requirements, and each has a test:

**1. A figure whose basis silently narrowed is not a figure.** The legacy
held-out validation reported 24.4% for a model the 9-point test measured at
9.7%, because two of its five points collected no samples and it averaged the
three survivors. The direction of that error is not predictable, so it cannot be
corrected for after the fact. :attr:`AccuracyReport.average_cm` is therefore
``None`` whenever any point collected nothing, and the report names which points
failed and how many samples each got.

**2. Never a single edge average.** One run's summary read "even accuracy across
the screen" over a table ranging 0.6 cm to 12.0 cm, because it averaged all four
edges together and a good left edge cancelled a bad right one. This reports
per-point, per-row and per-column. Rows and columns are the groupings that
actually expose the two observed failure modes: one run degraded toward the right
and bottom, the other at the top-left, and both are invisible in an edge mean.

**3. A result whose input can be overwritten is not a measurement.** The legacy
writes every calibration to one mutable path, and the second of the two runs
destroyed the first's model twenty minutes after it was measured, so run 1 can
never be reproduced. Every report carries the profile's **digest**, along with
screen size, drift offset and per-point sample counts.

BOTH UNITS, WITH THE DENOMINATOR NAMED
---------------------------------------
Errors are reported in centimetres **and** as a percentage of **screen width**.
"% of screen" is ambiguous between width, height and diagonal, and the figures
this project inherited never said which they used, which made comparing them
guesswork. The denominator is named in the type and in the rendering.

WHAT IS PURE AND WHAT IS NOT
-----------------------------
Everything in this module is a pure function of measurements already taken.
Collecting those measurements needs a screen and a person, and lives in
:mod:`focusedgaze.calibration.ui`. That split is what lets the arithmetic — the
part that produced two of the three defects above — be tested in CI.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final

from .exceptions import ConfigError

__all__ = [
    "DEFAULT_SCREEN_CM",
    "DEFAULT_TEST_POINTS",
    "AccuracyReport",
    "PointMeasurement",
    "PointResult",
    "build_report",
    "profile_digest",
]

#: The 3x3 grid, 5% in from each edge. The corners are where extrapolation is
#: hardest and where both recorded runs found their worst point, so they are
#: measured rather than avoided. From `milestone6_test_accuracy.py:35`.
DEFAULT_TEST_POINTS: Final[tuple[tuple[float, float], ...]] = tuple(
    (x, y) for y in (0.05, 0.5, 0.95) for x in (0.05, 0.5, 0.95)
)

#: Fallback physical screen size in cm, used when the platform cannot be asked.
#: The reference machine's, and the one both recorded runs used.
DEFAULT_SCREEN_CM: Final[tuple[float, float]] = (34.4, 19.4)

#: Below this, a point's median is being taken over too few readings to mean
#: much. Both recorded runs collected 32-36 per point, so this is a floor rather
#: than a target.
MIN_USEFUL_SAMPLES: Final = 8


@dataclass(frozen=True, slots=True)
class PointMeasurement:
    """What was collected at one test point, before any error is computed.

    Args:
        target: Where the user was told to look, normalised ``(x, y)``.
        predicted: Where the model said they looked, normalised and **clamped to
            the screen**, or ``None`` when no usable reading was obtained.
        n_samples: How many gaze readings contributed to ``predicted``. Zero
            when the point failed, and recorded even then: "collected nothing"
            and "collected three" are different problems.
        raw: The same prediction **before clamping**, so a point that landed off
            the screen can still be measured. Defaults to ``predicted``.

    WHY ``raw`` EXISTS
    ------------------
    Clamping is right for the error figure -- an application cannot put a cursor
    outside the screen, so that is the error a user experiences -- but recording
    only the clamped value destroys the evidence for the largest error term this
    system has. Five runs here showed a per-session vertical offset spanning 0.41
    of screen height, and in the worst of them two points were stored as
    ``y = 0.0`` when the model had actually predicted somewhere above the screen.
    The magnitude of the offset was therefore unrecoverable from the report that
    existed to characterise it.
    """

    target: tuple[float, float]
    predicted: tuple[float, float] | None
    n_samples: int
    raw: tuple[float, float] | None = None
    #: Median head orientation while this point was measured, in radians, or
    #: ``None`` when it was not recorded. Diagnostic: nothing in the mapping
    #: consumes it. See :mod:`focusedgaze.core.headpose` for why it is being
    #: recorded before it is being used.
    head_pose: tuple[float, float, float] | None = None
    #: Median eye-to-camera distance while this point was measured, in cm.
    distance_cm: float | None = None


@dataclass(frozen=True, slots=True)
class PointResult:
    """One test point's error, in both units."""

    target: tuple[float, float]
    predicted: tuple[float, float] | None
    n_samples: int
    error_cm: float | None
    error_pct_width: float | None
    raw: tuple[float, float] | None = None
    head_pose: tuple[float, float, float] | None = None
    distance_cm: float | None = None

    @property
    def measured(self) -> bool:
        """Whether this point produced a usable reading."""
        return self.predicted is not None

    @property
    def clamped(self) -> bool:
        """Whether the prediction fell outside the screen and was pulled onto it.

        A clamped point's ``error_cm`` is a **lower bound**: clamping moves the
        prediction toward the screen, and therefore usually toward the target.
        """
        if self.predicted is None or self.raw is None:
            return False
        return self.raw != self.predicted

    @property
    def label(self) -> str:
        """The point as percentages, e.g. ``"(5,50)"``."""
        return f"({self.target[0] * 100:.0f},{self.target[1] * 100:.0f})"


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


@dataclass(frozen=True)
class AccuracyReport:
    """A complete measurement, or an explicit refusal to summarise a partial one.

    Args:
        points: Per-point results, in grid order.
        screen_cm: Physical ``(width, height)`` the errors are expressed against.
        profile_digest: SHA-256 of the calibration that produced the readings.
        profile_name: Its name, for humans.
        drift_offset: The recentring offset applied before comparison.
        provider: The ONNX execution provider used.
        measured_at: ISO-8601 UTC.
        notes: Anything the collector wants to carry through.
    """

    points: tuple[PointResult, ...]
    screen_cm: tuple[float, float]
    profile_digest: str
    profile_name: str = "unknown"
    drift_offset: tuple[float, float] = (0.0, 0.0)
    provider: str = "unknown"
    measured_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    notes: str = ""

    # -- completeness ------------------------------------------------------

    @property
    def failed_points(self) -> tuple[PointResult, ...]:
        """Points that collected no usable reading at all."""
        return tuple(p for p in self.points if not p.measured)

    @property
    def thin_points(self) -> tuple[PointResult, ...]:
        """Points that collected too few samples to be trustworthy."""
        return tuple(
            p for p in self.points if p.measured and p.n_samples < MIN_USEFUL_SAMPLES
        )

    @property
    def complete(self) -> bool:
        """Whether every point produced a reading."""
        return not self.failed_points

    # -- the headline figures ----------------------------------------------

    @property
    def average_cm(self) -> float | None:
        """Mean error across all points, or ``None`` if any point failed.

        **This is requirement 1 and it is deliberately not a number when the
        measurement is incomplete.** The legacy tool averaged the survivors of a
        partial collection and reported the result as though it were the whole
        thing, producing 24.4% for a model that measured 9.7%. There is no way
        to know which direction that error points, so there is nothing useful to
        return: use :attr:`average_cm_over_measured` if you explicitly want the
        partial figure and can say so at the call site.
        """
        if not self.complete:
            return None
        return _mean([p.error_cm for p in self.points if p.error_cm is not None])

    @property
    def average_cm_over_measured(self) -> float | None:
        """Mean over the points that *did* report, however few that is.

        Named at length on purpose. A caller reaching for this is asserting they
        know the basis is partial, which is exactly what the legacy tool failed
        to make anyone say out loud.
        """
        return _mean([p.error_cm for p in self.points if p.error_cm is not None])

    @property
    def average_pct_width(self) -> float | None:
        """:attr:`average_cm` as a percentage of screen **width**."""
        average = self.average_cm
        return None if average is None else average / self.screen_cm[0] * 100.0

    @property
    def worst(self) -> PointResult | None:
        """The point with the largest error."""
        measured = [p for p in self.points if p.error_cm is not None]
        if not measured:
            return None
        return max(measured, key=lambda p: p.error_cm or 0.0)

    @property
    def best(self) -> PointResult | None:
        measured = [p for p in self.points if p.error_cm is not None]
        if not measured:
            return None
        return min(measured, key=lambda p: p.error_cm or 0.0)

    # -- requirement 2: groupings that expose direction --------------------

    def _grouped(self, axis: int) -> dict[float, float | None]:
        """Mean error per distinct coordinate along one axis."""
        bands: dict[float, list[float]] = {}
        for point in self.points:
            if point.error_cm is None:
                continue
            bands.setdefault(point.target[axis], []).append(point.error_cm)
        return {key: _mean(values) for key, values in sorted(bands.items())}

    @property
    def by_row(self) -> dict[float, float | None]:
        """Mean error per screen row, top to bottom.

        Rows and columns rather than an edge average, because they are what make
        a *direction* visible. One recorded run degraded from 3.3 cm at the top
        to 8.9 cm at the bottom; an edge mean showed neither.
        """
        return self._grouped(1)

    @property
    def by_column(self) -> dict[float, float | None]:
        """Mean error per screen column, left to right."""
        return self._grouped(0)

    @property
    def centre(self) -> PointResult | None:
        """The centre point, which is usually the best and is worth naming."""
        for point in self.points:
            if point.target == (0.5, 0.5):
                return point
        return None

    @property
    def corners(self) -> tuple[PointResult, ...]:
        """The four corner points, where extrapolation is hardest."""
        return tuple(
            p for p in self.points if p.target[0] != 0.5 and p.target[1] != 0.5
        )

    def to_dict(self) -> dict[str, object]:
        """A JSON-serialisable record, including everything needed to reproduce it."""
        return {
            "measured_at": self.measured_at,
            "profile_name": self.profile_name,
            "profile_digest": self.profile_digest,
            "screen_cm": list(self.screen_cm),
            "drift_offset": list(self.drift_offset),
            "provider": self.provider,
            "complete": self.complete,
            "average_cm": self.average_cm,
            "average_pct_width": self.average_pct_width,
            "denominator": "screen width",
            "points": [
                {
                    "target": list(p.target),
                    "predicted": None if p.predicted is None else list(p.predicted),
                    # The unclamped prediction. Without it a point that landed
                    # off-screen is indistinguishable from one that landed on the
                    # edge, and the offset that put it there is unmeasurable.
                    "raw": None if p.raw is None else list(p.raw),
                    "clamped": p.clamped,
                    "n_samples": p.n_samples,
                    # Diagnostics, unused by the mapping: recorded so the
                    # per-session offset can be correlated against head
                    # orientation and distance before either is built into the
                    # polynomial. See focusedgaze.core.headpose.
                    "head_pose": None if p.head_pose is None else list(p.head_pose),
                    "distance_cm": p.distance_cm,
                    "error_cm": p.error_cm,
                    "error_pct_width": p.error_pct_width,
                }
                for p in self.points
            ],
            "by_row": {str(k): v for k, v in self.by_row.items()},
            "by_column": {str(k): v for k, v in self.by_column.items()},
            "notes": self.notes,
        }


def build_report(
    measurements: Sequence[PointMeasurement],
    *,
    screen_cm: tuple[float, float] = DEFAULT_SCREEN_CM,
    profile_digest: str,
    profile_name: str = "unknown",
    drift_offset: tuple[float, float] = (0.0, 0.0),
    provider: str = "unknown",
    notes: str = "",
) -> AccuracyReport:
    """Turn raw per-point measurements into a report.

    Pure: no screen, no camera, no clock beyond the timestamp. This is the half
    of the accuracy tool that can be tested, and it is the half that produced
    two of the three defects section 50 recorded.

    Raises:
        ConfigError: No measurements, or a screen dimension that is not positive.
    """
    if not measurements:
        raise ConfigError("an accuracy report needs at least one measured point")
    width, height = float(screen_cm[0]), float(screen_cm[1])
    if not (width > 0 and height > 0):
        raise ConfigError(f"screen_cm must be positive, got {screen_cm!r}")

    results = []
    for m in measurements:
        if m.predicted is None or m.n_samples <= 0:
            results.append(
                PointResult(m.target, None, max(m.n_samples, 0), None, None)
            )
            continue
        dx = (m.predicted[0] - m.target[0]) * width
        dy = (m.predicted[1] - m.target[1]) * height
        error = math.hypot(dx, dy)
        results.append(
            PointResult(
                target=m.target,
                predicted=m.predicted,
                n_samples=m.n_samples,
                error_cm=error,
                error_pct_width=error / width * 100.0,
                # Falls back to the clamped value so a caller that does not
                # supply one still produces a well-formed report; `clamped` then
                # reads False, which is the truthful answer for a measurement
                # that recorded no separate raw value.
                raw=m.raw if m.raw is not None else m.predicted,
                head_pose=m.head_pose,
                distance_cm=m.distance_cm,
            )
        )

    return AccuracyReport(
        points=tuple(results),
        screen_cm=(width, height),
        profile_digest=profile_digest,
        profile_name=profile_name,
        drift_offset=(float(drift_offset[0]), float(drift_offset[1])),
        provider=provider,
        notes=notes,
    )


def profile_digest(profile: object) -> str:
    """SHA-256 of a calibration profile's canonical JSON.

    Requirement 3. Digesting the **content** rather than a file means the digest
    survives the profile being renamed or copied, and identifies it even if the
    file it came from is later overwritten, which is precisely what happened to
    the first of the two recorded runs.
    """
    to_json = getattr(profile, "to_json", None)
    if to_json is None:
        raise ConfigError(
            f"cannot digest {type(profile).__name__}: it has no to_json(). "
            "An accuracy result must record which calibration produced it."
        )
    return hashlib.sha256(to_json().encode("utf-8")).hexdigest()


def render(report: AccuracyReport) -> list[str]:
    """The human-readable report, as lines.

    Deliberately leads with per-point rather than an average, and refuses to
    print a headline when the measurement is incomplete.
    """
    width = report.screen_cm[0]
    lines: list[str] = [
        f"Calibration : {report.profile_name}  ({report.profile_digest[:12]}...)",
        f"Screen      : {report.screen_cm[0]:.1f} x {report.screen_cm[1]:.1f} cm",
        f"Drift offset: ({report.drift_offset[0]:+.1%}, {report.drift_offset[1]:+.1%})",
        f"Provider    : {report.provider}",
        "",
        "Per point (error as cm, and as % of screen WIDTH):",
    ]
    for point in report.points:
        if not point.measured:
            lines.append(f"  {point.label:>10}   NO SAMPLES")
            continue
        # A clamped point is marked because its error is a lower bound, not a
        # measurement: the prediction landed off the screen and was pulled back
        # onto it, which moves it toward the target.
        mark = "  >off screen" if point.clamped else ""
        lines.append(
            f"  {point.label:>10}   {point.error_cm:5.1f} cm"
            f"   {point.error_pct_width:5.1f}%   n={point.n_samples}{mark}"
        )

    clamped = [p for p in report.points if p.clamped]
    if clamped:
        lines.append("")
        lines.append(
            f"  {len(clamped)} point(s) predicted off the screen. Their errors above "
            "are LOWER BOUNDS:"
        )
        lines.append(
            "  clamping moves a prediction onto the screen, and therefore toward "
            "the target. See the `raw` field for where the model actually pointed."
        )

    lines.append("")
    lines.append("By row (top to bottom) and column (left to right):")
    for name, grouping in (("row", report.by_row), ("column", report.by_column)):
        parts = [
            f"{key * 100:.0f}%={value:.1f}cm" for key, value in grouping.items()
            if value is not None
        ]
        lines.append(f"  by {name:<7} " + "  ".join(parts))

    lines.append("")
    if not report.complete:
        failed = ", ".join(p.label for p in report.failed_points)
        lines.append(
            f"NO AVERAGE REPORTED: {len(report.failed_points)} of "
            f"{len(report.points)} points collected no samples ({failed})."
        )
        lines.append(
            "  Averaging the rest would present a narrower basis as though it "
            "were the whole measurement. The original tool did that and reported "
            "24.4% for a model that measured 9.7%."
        )
        lines.append("  Fix the lighting or your position at those points and re-run.")
        return lines

    average, worst, best = report.average_cm, report.worst, report.best
    assert average is not None and worst is not None and best is not None
    lines.append(
        f"Average     : {average:.1f} cm  ({average / width * 100:.1f}% of width)"
    )
    lines.append(
        f"Best        : {best.error_cm:.1f} cm at {best.label}"
    )
    lines.append(
        f"Worst       : {worst.error_cm:.1f} cm at {worst.label}"
    )
    if report.thin_points:
        thin = ", ".join(f"{p.label} n={p.n_samples}" for p in report.thin_points)
        lines.append(f"Thin sampling at: {thin} - treat those points with caution.")
    lines.append("")
    lines.append(
        "One run is not a characterisation. Two recorded runs of this system, "
        "same person and machine twenty minutes apart, differed by a factor of "
        "two and their worst corners swapped. Measure more than once."
    )
    return lines
