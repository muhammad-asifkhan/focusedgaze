"""Dwell selection: resting your gaze on a target to choose it.

See the package docstring for why each of the three tolerances exists. All
timings are in seconds and come from the caller, so this is testable on a fake
clock and cannot drift with frame rate.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final

from ..exceptions import ConfigError

__all__ = [
    "DEFAULT_BLINK_GRACE_S",
    "DEFAULT_DWELL_S",
    "DEFAULT_HYSTERESIS",
    "DEFAULT_REARM_S",
    "DwellSelector",
    "DwellState",
    "EdgeZone",
    "Target",
    "edge_zones_for",
]

#: 1050 ms, the shipping value carried in :class:`~focusedgaze.config.RuntimeConfig`.
DEFAULT_DWELL_S: Final = 1.05

#: How far outside a target the point may stray before the dwell is cancelled,
#: in screen fractions. Roughly 1 cm on a 34 cm screen, which is under the
#: tracker's own error: the intent is to absorb jitter, not to enlarge targets.
DEFAULT_HYSTERESIS: Final = 0.03

#: How long gaze may be unavailable without cancelling. A blink is 100-400 ms,
#: and at 7 fps that is one to three frames of nothing.
DEFAULT_BLINK_GRACE_S: Final = 0.5

#: How long after a selection before the same target can fire again. Without
#: this, resting on a control re-triggers it forever.
DEFAULT_REARM_S: Final = 0.6


@dataclass(frozen=True, slots=True)
class Target:
    """A rectangular region that can be selected by looking at it.

    Args:
        name: What the caller gets back when it is chosen. Unique per selector.
        x0, y0, x1, y1: Normalised bounds, origin top-left, ``y`` downward --
            the same convention as :class:`~focusedgaze.types.GazeResult`.
        dwell_s: Per-target override. A destructive action can be made to need a
            longer look than a benign one.
    """

    name: str
    x0: float
    y0: float
    x1: float
    y1: float
    dwell_s: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("a target needs a name")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ConfigError(
                f"target {self.name!r} has no area: "
                f"({self.x0}, {self.y0}) to ({self.x1}, {self.y1})"
            )
        if self.dwell_s is not None and self.dwell_s <= 0:
            raise ConfigError(f"target {self.name!r}: dwell_s must be positive")

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Whether ``(x, y)`` is inside, optionally grown by ``margin``."""
        return (
            self.x0 - margin <= x <= self.x1 + margin
            and self.y0 - margin <= y <= self.y1 + margin
        )

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True, slots=True)
class EdgeZone:
    """A band at one screen edge, for scrolling or panning.

    Reported as "the gaze is here now", with no dwell: an edge zone is a
    continuous action, and making the user wait a second before a page starts
    scrolling reads as a broken control rather than a deliberate one.
    """

    name: str
    #: How long the gaze has been continuously inside, in seconds. Callers scale
    #: scroll speed by this so a glance does not fling the page.
    held_s: float = 0.0


@dataclass(frozen=True, slots=True)
class DwellState:
    """What the interaction looks like right now. Returned every update.

    Args:
        target: Name of the target being dwelt on, or ``None``.
        progress: ``0.0`` to ``1.0`` toward selection. Draw this: a dwell with no
            visible progress is indistinguishable from a system that has hung.
        selected: Name of a target chosen **on this update only**. It appears in
            exactly one state, so a caller can act on it without deduplicating.
        edge: The edge zone the gaze is in, or ``None``.
        tracking: Whether this update had a usable gaze point. False during a
            blink, while the dwell is still held.
    """

    target: str | None = None
    progress: float = 0.0
    selected: str | None = None
    edge: EdgeZone | None = None
    tracking: bool = True


@dataclass
class DwellSelector:
    """Feed it gaze points; it tells you what the user chose.

    ::

        selector = DwellSelector([Target("play", 0.1, 0.1, 0.4, 0.3)])
        for result in tracker.stream():
            state = selector.update(result.x, result.y, result.timestamp)
            if state.selected:
                launch(state.selected)
            draw_progress_ring(state.target, state.progress)

    ``x`` and ``y`` may be ``None`` -- pass the result through unchanged on a
    frame with no face, and the blink grace period handles it.

    Args:
        targets: The selectable regions. Overlaps resolve to the first match, so
            order them most-specific first.
        dwell_s: Default dwell time, overridable per target.
        hysteresis: Extra margin for *staying* on a target, not entering it.
        blink_grace_s: How long a lost gaze is tolerated before cancelling.
        rearm_s: Delay before the same target can be selected again.
        edges: Edge zones, from :func:`edge_zones_for` or built by hand.
    """

    targets: Sequence[Target] = ()
    dwell_s: float = DEFAULT_DWELL_S
    hysteresis: float = DEFAULT_HYSTERESIS
    blink_grace_s: float = DEFAULT_BLINK_GRACE_S
    rearm_s: float = DEFAULT_REARM_S
    edges: Sequence[Target] = ()

    _active: str | None = field(default=None, init=False, repr=False)
    _entered_at: float | None = field(default=None, init=False, repr=False)
    #: When the user was last actually seen. The blink grace is measured from
    #: here rather than from the first frame that reported nothing, because a
    #: stalled camera delivers its first empty frame long after the user was
    #: last visible, and treating that as a fresh blink would hold a dwell open
    #: across an arbitrary gap.
    _last_good_at: float | None = field(default=None, init=False, repr=False)
    _lost: bool = field(default=False, init=False, repr=False)
    _blocked_until: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _edge_since: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for value, label in (
            (self.dwell_s, "dwell_s"),
            (self.blink_grace_s, "blink_grace_s"),
            (self.rearm_s, "rearm_s"),
        ):
            if value < 0:
                raise ConfigError(f"{label} must not be negative, got {value}")
        if self.hysteresis < 0:
            raise ConfigError(f"hysteresis must not be negative, got {self.hysteresis}")
        names = [t.name for t in self.targets]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ConfigError(f"duplicate target names: {sorted(duplicates)}")

    def reset(self) -> None:
        """Forget the current dwell and every re-arm block.

        For switching screens: a dwell half-completed on a target that no longer
        exists must not complete against whatever replaced it.
        """
        self._active = None
        self._entered_at = None
        self._last_good_at = None
        self._lost = False
        self._blocked_until.clear()
        self._edge_since.clear()

    def _find(self, x: float, y: float) -> Target | None:
        """The target under the point, preferring the one already held.

        Held-first is the hysteresis: an entered target keeps the pointer until
        it is left by more than the margin, so two adjacent targets do not swap
        back and forth on jitter at the boundary.
        """
        if self._active is not None:
            for t in self.targets:
                if t.name == self._active and t.contains(x, y, self.hysteresis):
                    return t
        for t in self.targets:
            if t.contains(x, y):
                return t
        return None

    def update(
        self, x: float | None, y: float | None, timestamp: float
    ) -> DwellState:
        """Advance the interaction by one frame.

        Args:
            x: Normalised horizontal gaze, or ``None`` when unavailable.
            y: Normalised vertical gaze, or ``None``.
            timestamp: Seconds, increasing. Taken from the caller so this is
                testable and cannot drift with frame rate.
        """
        if x is None or y is None:
            return self._no_gaze(timestamp)

        if self._lost and self._last_good_at is not None and self._entered_at is not None:
            # Gaze has just come back. Push the dwell's start forward by the
            # length of the gap so the blink does not count toward it. Without
            # this the clock runs through the blackout and a well-timed blink
            # selects, which is the opposite of a deliberate action.
            self._entered_at += timestamp - self._last_good_at
        self._lost = False
        self._last_good_at = timestamp
        edge = self._edge_for(x, y, timestamp)
        target = self._find(x, y)

        if target is None:
            self._active = None
            self._entered_at = None
            return DwellState(edge=edge)

        if target.name != self._active:
            self._active = target.name
            self._entered_at = timestamp

        blocked = self._blocked_until.get(target.name)
        if blocked is not None:
            if timestamp < blocked:
                # Held over a target that was just chosen. Reported as present
                # with no progress, so the caller can show it highlighted
                # without implying another selection is coming.
                return DwellState(target=target.name, progress=0.0, edge=edge)
            del self._blocked_until[target.name]
            self._entered_at = timestamp

        needed = target.dwell_s if target.dwell_s is not None else self.dwell_s
        held = timestamp - (self._entered_at if self._entered_at is not None else timestamp)
        if needed <= 0 or held >= needed:
            self._blocked_until[target.name] = timestamp + self.rearm_s
            self._entered_at = timestamp
            return DwellState(
                target=target.name, progress=1.0, selected=target.name, edge=edge
            )

        return DwellState(
            target=target.name, progress=min(held / needed, 1.0), edge=edge
        )

    def _no_gaze(self, timestamp: float) -> DwellState:
        """A frame with no usable point: blink, or the user looked away.

        The dwell clock is **not** advanced while gaze is lost, only preserved.
        Counting a blink toward a selection would let someone select by blinking
        at the right moment, which is the opposite of deliberate.
        """
        self._lost = True
        last_seen = self._last_good_at if self._last_good_at is not None else timestamp

        if timestamp - last_seen > self.blink_grace_s:
            self._active = None
            self._entered_at = None
            return DwellState(tracking=False)

        if self._active is not None and self._entered_at is not None:
            # Progress is reported frozen at the value it had when gaze was
            # last seen -- measured from `last_seen`, not `timestamp` -- so the
            # ring the caller draws stops rather than creeping during a blink.
            needed = self.dwell_s
            for t in self.targets:
                if t.name == self._active and t.dwell_s is not None:
                    needed = t.dwell_s
            held = last_seen - self._entered_at
            return DwellState(
                target=self._active,
                progress=min(held / needed, 1.0) if needed > 0 else 1.0,
                tracking=False,
            )
        return DwellState(tracking=False)

    def _edge_for(self, x: float, y: float, timestamp: float) -> EdgeZone | None:
        for zone in self.edges:
            if zone.contains(x, y):
                since = self._edge_since.setdefault(zone.name, timestamp)
                stale = [n for n in self._edge_since if n != zone.name]
                for n in stale:
                    del self._edge_since[n]
                return EdgeZone(zone.name, held_s=timestamp - since)
        self._edge_since.clear()
        return None


def edge_zones_for(
    band: float = 0.09, *, names: Iterable[str] = ("left", "right", "top", "bottom")
) -> tuple[Target, ...]:
    """The four screen-edge bands, as targets.

    Args:
        band: Fraction of the screen at each edge. Defaults to 0.09, the
            shipping value from :class:`~focusedgaze.config.RuntimeConfig`.
        names: Which edges to build, in the order left, right, top, bottom.

    Edges are where a gaze tracker is least accurate -- both recorded accuracy
    runs on this project were worst at a corner -- which is exactly why a *band*
    works there when a small target would not.
    """
    if not (0.0 < band < 0.5):
        raise ConfigError(f"band must be in (0, 0.5), got {band}")
    wanted = set(names)
    built = []
    for name, rect in (
        ("left", (0.0, 0.0, band, 1.0)),
        ("right", (1.0 - band, 0.0, 1.0, 1.0)),
        ("top", (0.0, 0.0, 1.0, band)),
        ("bottom", (0.0, 1.0 - band, 1.0, 1.0)),
    ):
        if name in wanted:
            built.append(Target(name, *rect))
    return tuple(built)
