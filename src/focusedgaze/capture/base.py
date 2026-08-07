"""What a frame source is, and what every one of them must guarantee.

The pipeline does not open cameras. :class:`~focusedgaze.core.estimator.GazeEstimator`
takes frames and returns results, and everything that knows how to *get* a frame
lives here. That split is what makes the estimator testable without hardware, and
it is why this package can be driven from a video file, a shared camera owned by
another part of an application, or a recorded fixture, without the pipeline
knowing the difference.

TIMESTAMPS ARE PART OF THE FRAME, NOT AN AFTERTHOUGHT
-----------------------------------------------------
Every frame carries the time it was captured, in **seconds**. The One Euro
filter differentiates against this value, so passing a frame counter instead of
a clock silently changes the smoothing: the filter's cutoff is expressed per
second, and feeding it "1, 2, 3" makes every gap exactly 1.0 s regardless of the
real frame rate. `docs/troubleshooting.md` lists that as a cause of jitter, which
is why the timestamp is a field of :class:`Frame` rather than something a caller
is trusted to supply correctly.

The clock is :func:`time.monotonic`, not :func:`time.time`. A wall clock can step
backwards over an NTP correction or a DST change, and a backwards step produces a
negative delta that the filter's non-monotonic guard has to absorb. Monotonic
time cannot do that.

WHY ``read()`` RETURNS ``None`` RATHER THAN RAISING
---------------------------------------------------
End of stream is an ordinary event: a video file runs out, a source is released.
It is reported as ``None``, once, and every subsequent call also returns ``None``.
A *fault* (device gone, file unreadable) raises :class:`~focusedgaze.exceptions.CameraError`.

That distinction is the same one the exception module draws for the pipeline: the
legacy code used ``None`` for "no face", "out of range" and "something broke"
alike, and a caller could not tell a finished video from a dead camera.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, Self, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = ["Frame", "FrameSource", "FrameSourceBase"]


@dataclass(frozen=True, slots=True)
class Frame:
    """One captured image and the context needed to interpret it.

    Args:
        image: The pixels, ``(height, width, 3)`` uint8 in **BGR** order, which
            is what OpenCV produces and what the rest of this package expects.
            `docs/troubleshooting.md` lists RGB-where-BGR-was-expected as a
            mistake that degrades tracking without raising, so the order is
            stated here rather than assumed.
        timestamp: Capture time in seconds from :func:`time.monotonic`. Deltas
            between frames are meaningful; the absolute value is not.
        index: Monotonically increasing sequence number from the source that
            produced it, starting at 0. Two frames with the same index are the
            same capture, which is how a caller detects that a source has
            handed back a repeat.

    Frozen because a frame travels between threads: the capture thread publishes
    it and the consumer reads it, and a mutable object crossing that boundary
    would need a lock to stay coherent. ``image`` is a NumPy array and therefore
    still mutable in principle; sources here never write into an array they have
    already published.
    """

    image: NDArray[np.uint8]
    timestamp: float
    index: int

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` in pixels, in the order OpenCV and configs use."""
        height, width = self.image.shape[:2]
        return int(width), int(height)


@runtime_checkable
class FrameSource(Protocol):
    """Anything that can hand out frames and be released.

    Structural, not inherited: an application that already owns a camera can
    satisfy this with its own object and pass it straight in. That is the point
    of the protocol. :class:`FrameSourceBase` exists for the implementations in
    this package and is not required.
    """

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` actually being delivered, not what was requested."""
        ...

    def read(self) -> Frame | None:
        """The next frame, or ``None`` once the source is exhausted or released.

        Raises:
            CameraError: The source failed in a way a caller cannot recover from
                by reading again.
        """
        ...

    def release(self) -> None:
        """Give up the underlying device or file. Safe to call more than once."""
        ...


class FrameSourceBase:
    """Shared plumbing: context management, and a release that means it.

    Subclasses implement :meth:`_read_frame` and :meth:`_release`. This class
    handles the two things every source got wrong somewhere in the legacy code:

    * **Release is idempotent and always happens.** The legacy server released
      the camera from the capture thread's own exit path, so an exception
      anywhere else left the device held until the process died. On Windows that
      means nothing else can open it, which `docs/troubleshooting.md` records as
      "it worked yesterday". Here the context manager releases on the way out
      whatever happened, and calling ``release()`` twice is not an error.
    * **A released source stays released.** ``read()`` after ``release()``
      returns ``None`` rather than reopening or raising, so a consumer loop
      unwinds naturally instead of needing to know the ordering.
    """

    __slots__ = ("_closed",)

    def __init__(self) -> None:
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether :meth:`release` has run."""
        return self._closed

    def read(self) -> Frame | None:
        if self._closed:
            return None
        return self._read_frame()

    def release(self) -> None:
        if self._closed:
            return
        # Marked closed FIRST. If _release raises, the source must not be left
        # in a state where a retry re-enters teardown on a half-released device.
        self._closed = True
        self._release()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()

    # -- subclass hooks ----------------------------------------------------

    def _read_frame(self) -> Frame | None:
        raise NotImplementedError

    def _release(self) -> None:
        raise NotImplementedError


def now() -> float:
    """Capture clock. Monotonic, for the reason given in the module docstring."""
    return time.monotonic()
