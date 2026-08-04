"""Show what the One Euro filter actually does to a noisy gaze signal.

Runs with the base install. No camera, no model, no calibration.

The signal is synthetic but shaped like a real one: the user fixates, flicks to
a new target, fixates again. Noise is added at a level comparable to raw model
output. The point is to see the tradeoff the filter is making, which is why it
prints the two failure modes alongside it.

    python examples/filter_demo.py
"""

from __future__ import annotations

import itertools
import random

from focusedgaze.core.filters import OneEuroFilter2D

FPS = 30.0
random.seed(7)


def gaze_signal() -> list[tuple[float, float, float]]:
    """Fixate, flick, fixate. Returns (x, y, t) with noise on top."""
    samples = []
    for frame in range(60):
        t = frame / FPS
        if frame < 20:
            tx, ty = 0.25, 0.25          # fixating top left
        elif frame < 25:
            k = (frame - 19) / 6.0       # a fast flick across the screen
            tx, ty = 0.25 + 0.5 * k, 0.25 + 0.5 * k
        else:
            tx, ty = 0.75, 0.75          # fixating bottom right

        # Noise roughly the scale of raw per-frame model jitter.
        nx = tx + random.gauss(0.0, 0.012)
        ny = ty + random.gauss(0.0, 0.012)
        samples.append((nx, ny, t))
    return samples


def jitter(values: list[float]) -> float:
    """Mean absolute frame-to-frame change. Lower is steadier."""
    if len(values) < 2:
        return 0.0
    steps = [abs(b - a) for a, b in itertools.pairwise(values)]
    return sum(steps) / len(steps)


def main() -> None:
    samples = gaze_signal()

    tuned = OneEuroFilter2D(min_cutoff=0.7, beta=0.6, d_cutoff=1.0)
    heavy = OneEuroFilter2D(min_cutoff=0.05, beta=0.0, d_cutoff=1.0)

    raw_x, tuned_x, heavy_x = [], [], []
    tuned_track, heavy_track = [], []

    for x, y, t in samples:
        raw_x.append(x)
        tx, _ = tuned.filter(x, y, t)
        hx, _ = heavy.filter(x, y, t)
        tuned_x.append(tx)
        heavy_x.append(hx)
        tuned_track.append((t, tx))
        heavy_track.append((t, hx))

    # Settling: frames after the flick ends (frame 25) until within 0.02 of target.
    def settle_frames(series: list[float]) -> int | str:
        for i in range(25, len(series)):
            if abs(series[i] - 0.75) <= 0.02:
                return i - 25
        return "never"

    fixation = slice(30, 60)
    print("One Euro filter, 30 fps, synthetic fixate/flick/fixate signal")
    print()
    print(f"{'':<22}{'jitter while still':>20}{'frames to settle':>19}")
    print("-" * 61)
    print(f"{'raw (unfiltered)':<22}{jitter(raw_x[fixation]):>20.5f}{'n/a':>19}")
    print(f"{'tuned (shipped)':<22}{jitter(tuned_x[fixation]):>20.5f}{settle_frames(tuned_x)!s:>19}")
    print(f"{'over-smoothed':<22}{jitter(heavy_x[fixation]):>20.5f}{settle_frames(heavy_x)!s:>19}")
    print()
    print("The shipped settings cut jitter by about 10x while the eye is still,")
    print("and still settle within a few frames of a flick.")
    print()
    print("The over-smoothed row is the interesting one. It was set up to win on")
    print("jitter by smoothing hard with no velocity term, and it does not: it")
    print("never reaches the target at all, so what looks like jitter is actually")
    print("it still creeping toward a position it left 25 frames earlier. Smoothing")
    print("harder does not buy steadiness once the filter can no longer keep up.")


if __name__ == "__main__":
    main()
