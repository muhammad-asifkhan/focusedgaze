"""Signal smoothing.

The One Euro filter (Casiez, Roussel & Vogel, CHI 2012) is what makes a gaze
cursor usable. A fixed low-pass filter forces a choice between jitter when you
hold still and lag when you move; this one adapts, filtering hard at low speed
and easing off as speed rises.

Extracted verbatim from the shipping implementation. The arithmetic is
unchanged: ``tests/test_golden_tier1.py::test_one_euro_matches_golden`` pins it
to within 1e-9 of the recorded pre-refactor output.
"""

from __future__ import annotations

import math

__all__ = ["OneEuroFilter", "OneEuroFilter2D"]


class OneEuroFilter:
    """Adaptive low-pass filter for a single scalar stream.

    Args:
        min_cutoff: Cutoff frequency at rest, in Hz. Lower is steadier when the
            signal is still, at the cost of a little lag on slow movement.
        beta: Speed coefficient. Higher is snappier on fast movement, at the
            cost of letting more jitter through.
        d_cutoff: Cutoff for the derivative estimate. Rarely needs changing.

    Not thread-safe: it carries per-sample state, so one instance belongs to one
    stream on one thread.
    """

    __slots__ = ("_dx_prev", "_freq", "_t_prev", "_x_prev", "beta", "d_cutoff", "min_cutoff")

    def __init__(self, min_cutoff: float = 0.7, beta: float = 0.6,
                 d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self) -> None:
        """Forget all history.

        Call when the signal is interrupted (a lost face, a mode change), so
        the filter does not glide from a stale value when it resumes.
        """
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None
        # Seeded at 30 Hz so the very first sample has a sane frequency estimate
        # before any interval has been measured.
        self._freq: float = 30.0

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self._freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, t: float) -> float:
        """Filter one sample taken at time ``t`` (seconds).

        Returns the smoothed value. Timestamps must be non-decreasing; the
        sampling rate is derived from their differences, so passing a rounded or
        fabricated ``t`` changes the output.
        """
        if self._x_prev is None:
            self._x_prev, self._t_prev = x, t
            return x

        if self._t_prev is not None and t > self._t_prev:
            self._freq = 1.0 / (t - self._t_prev)
        self._t_prev = t

        dx = (x - self._x_prev) * self._freq
        a_d = self._alpha(self.d_cutoff)
        self._dx_prev = a_d * dx + (1 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(self._dx_prev)
        a = self._alpha(cutoff)
        self._x_prev = a * x + (1 - a) * self._x_prev
        return self._x_prev

    # Convenience alias so the filter can be used as a callable.
    __call__ = filter


class OneEuroFilter2D:
    """Two independent One Euro filters, for a screen coordinate.

    The axes are filtered separately, exactly as the shipping system does: the
    horizontal and vertical components of gaze error are not correlated closely
    enough for a joint filter to be worth the complexity.
    """

    __slots__ = ("x", "y")

    def __init__(self, min_cutoff: float = 0.7, beta: float = 0.6,
                 d_cutoff: float = 1.0) -> None:
        self.x = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.y = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self) -> None:
        """Forget history on both axes."""
        self.x.reset()
        self.y.reset()

    def filter(self, x: float, y: float, t: float) -> tuple[float, float]:
        """Filter one (x, y) sample taken at time ``t``."""
        return self.x.filter(x, t), self.y.filter(y, t)

    __call__ = filter
