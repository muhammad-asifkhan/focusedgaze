"""Turning gaze coordinates into input events.

The layer between "where they are looking" and "what they chose". A consumer
that only has ``(x, y)`` has to invent dwell timing, jitter tolerance, blink
handling and re-arming for itself, and every consumer inventing them separately
is how a tracker with usable accuracy ends up feeling unusable.

WHY THIS IS PURE
================
Nothing here touches a camera, a screen or a clock it was not given. It takes a
point and a timestamp and returns state. That is the same split the rest of the
package uses -- :mod:`focusedgaze.core.estimator` is pure and
:mod:`focusedgaze.capture` owns the hardware -- and it means the interaction
rules are tested in CI, on a fake clock, without a person.

ACCURACY IS NOT THE BOTTLENECK; INTERACTION DESIGN IS
=====================================================
A calibrated webcam tracker lands around 2-3 degrees, which on a laptop at arm's
length is roughly 2-3 cm on screen. Apple's own front-camera eye tracking is in
the same range. What makes it feel precise there is not the sensor: it is large
targets, dwell with visible progress, and forgiving edges. Those are the things
this module implements.

Design consequences, each of which came from a measured property of the tracker:

* **Hysteresis.** The gaze point jitters by more than a small target's width, so
  a plain "is the point inside the rectangle" test flickers and cancels a dwell
  that a user experiences as steady. A target the pointer has entered is harder
  to leave than it was to enter.
* **A blink grace period.** Blinks produce ``NO_FACE`` for several frames. Losing
  the dwell every time someone blinks makes selection nearly impossible; on
  hardware running at 7 fps, a single blink is most of a second.
* **Re-arming.** Without it, resting on a control after selecting it fires again
  immediately -- the "Midas touch" problem, where looking at something is
  indistinguishable from choosing it.
"""

from __future__ import annotations

from .dwell import DwellSelector, DwellState, EdgeZone, Target, edge_zones_for

__all__ = [
    "DwellSelector",
    "DwellState",
    "EdgeZone",
    "Target",
    "edge_zones_for",
]
