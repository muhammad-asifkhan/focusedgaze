"""A live camera behind the server's :class:`~focusedgaze.server.GazeSource`.

The server had exactly one source it could run: a replay of a recorded JSON
file. Every piece needed for a live one already existed --
:class:`~focusedgaze.capture.WebcamGazeTracker` streams results, and the source
protocol is three methods -- but nothing joined them, so a browser client could
be developed against recorded readings and never driven by a camera. This module
is that join.

WHY A THREAD
------------
``GazeSource.latest`` is called on the broadcaster tick and **must not block**.
Reading a frame does block: an inference pass is 2 ms on the Intel backend but
the camera itself paces at 30 fps, so ``read()`` waits ~33 ms for the next
frame. Calling that from the tick would couple the send rate to the capture rate
and stall the event loop for every connected client at once.

So capture runs on its own thread and publishes the newest snapshot into a slot.
``latest()`` reads the slot and returns immediately. The two are the classic
producer/consumer pair with a depth of one, which is the correct depth here: a
queue would let readings pile up behind a slow consumer and the server would
broadcast the past. Gaze has no value once it is stale -- the newest reading is
the only one anybody wants.

WHY STALENESS IS CHECKED
------------------------
A slot holding the last good reading is a trap. If the capture thread dies, or
the camera stops delivering frames, the slot keeps that reading and ``latest()``
keeps handing it out: the client sees a cursor frozen at a plausible position
and nothing anywhere reports a fault. So a snapshot older than
:data:`STALE_AFTER_S` is reported as ``ok=False`` with no coordinates, which is
the same thing the server already says for a lost face. R-6 requires ``x`` and
``y`` be ``None`` rather than stale whenever ``ok`` is false, and this is a case
of exactly that.

THE CAMERA LEASE
----------------
``pause()`` and ``resume()`` exist so another application can have the webcam
while this one is running -- the legacy system's behaviour, preserved. Pausing
closes the tracker outright rather than merely skipping frames: a webcam held
open is unavailable to everyone else regardless of whether anybody is reading
it, so "give up the camera" has to mean releasing the device. Resuming builds a
fresh tracker, which reopens it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from types import TracebackType
from typing import Any, Final, Self

from ..config import GazeConfig
from .websocket import DEFAULT_PAUSE_TIMEOUT_S, GazeSnapshot

__all__ = ["STALE_AFTER_S", "LiveGazeSource"]

_log = logging.getLogger(__name__)

#: How old a reading may be before it is reported as unusable. Three frames at
#: 30 fps, so an ordinary scheduling hiccup does not blank the cursor while a
#: genuinely stopped camera is caught in under a tenth of a second.
STALE_AFTER_S: Final = 0.1

#: How long the capture thread waits for a tracker that will not build. Only
#: reached when the camera or a model file is unavailable at resume time.
_THREAD_JOIN_GRACE_S: Final = 1.0

#: ``() -> tracker``. The seam that keeps this module testable without a webcam.
TrackerFactory = Callable[[], Any]


class LiveGazeSource:
    """A webcam, a thread, and one snapshot slot.

    Args:
        profile: Profile or profile name handed to the tracker, or ``None`` to
            run uncalibrated. An uncalibrated source produces readings whose
            ``ok`` is always ``False``: without a profile there are no screen
            coordinates to send, only raw angles, and the wire format has no
            field for those.
        config: Everything tunable, including which backend and camera.
        tracker_factory: Builds the tracker. Injected by tests; the default
            constructs a :class:`~focusedgaze.capture.WebcamGazeTracker`.
        clock: Time source, injected so staleness is testable without sleeping.

    Not started by construction. Call :meth:`start`, or use it as a context
    manager, so that failing to open a camera happens where the caller is ready
    to report it rather than inside a constructor.

    Thread safety: :meth:`latest` may be called from any thread, including the
    event loop, while capture runs. Everything else expects one caller.
    """

    __slots__ = (
        "_clock",
        "_config",
        "_factory",
        "_lock",
        "_profile",
        "_snapshot",
        "_stop",
        "_thread",
        "_tracker_open",
    )

    def __init__(
        self,
        profile: Any = None,
        config: GazeConfig | None = None,
        *,
        tracker_factory: TrackerFactory | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._profile = profile
        self._config = config if config is not None else GazeConfig()
        self._factory = tracker_factory or self._default_factory
        self._clock = clock
        self._lock = threading.Lock()
        # Starts not-ok rather than empty: a client connecting before the first
        # frame arrives must be told "no reading yet", not given a coordinate.
        self._snapshot = GazeSnapshot(ok=False, x=None, y=None, t=0.0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tracker_open = threading.Event()

    def _default_factory(self) -> Any:
        from ..capture import WebcamGazeTracker

        return WebcamGazeTracker(profile=self._profile, config=self._config)

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Open the camera and begin capturing.

        Blocks until the tracker has been built, so a camera that cannot open or
        a model that cannot load raises **here**, from the caller's thread,
        rather than disappearing into a background thread's traceback.

        Raises:
            CameraError: The camera could not be opened.
            ModelNotFoundError: A model file is absent.
            CalibrationError: The profile is unloadable, or belongs to the
                other backend.
        """
        if self._thread is not None:
            return
        # Built here, on this thread, purely so that its exceptions surface to
        # the caller. The thread then adopts it.
        tracker = self._factory()
        self._tracker_open.set()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(tracker,), name="focusedgaze-capture", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        """Stop capturing and release the camera. Idempotent."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=DEFAULT_PAUSE_TIMEOUT_S)
            if thread.is_alive():
                _log.warning(
                    "capture thread did not stop within %.1fs; the camera may "
                    "stay open until the process exits",
                    DEFAULT_PAUSE_TIMEOUT_S,
                )
        self._publish(GazeSnapshot(ok=False, x=None, y=None, t=self._clock()))

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- the capture thread -------------------------------------------------

    def _run(self, tracker: Any) -> None:
        """Read frames into the slot until asked to stop.

        Owns the tracker for its whole life and closes it on the way out,
        including on an exception: the camera must not survive this thread.
        """
        try:
            while not self._stop.is_set():
                result = tracker.read()
                if result is None:
                    # The source is exhausted. Real webcams do not end, so this
                    # means the device went away.
                    _log.warning("the frame source ended; capture is stopping")
                    break
                self._publish(_snapshot_of(result, self._clock()))
        except Exception:  # noqa: BLE001 - nothing can be raised out of a thread
            # Logged rather than swallowed silently. There is no caller to raise
            # into from here, and `latest()` going stale is what the server will
            # actually observe.
            _log.exception("capture thread failed; readings will go stale")
        finally:
            try:
                tracker.close()
            finally:
                self._tracker_open.clear()

    def _publish(self, snapshot: GazeSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    # -- GazeSource ---------------------------------------------------------

    def latest(self) -> GazeSnapshot:
        """The newest reading, or a not-ok one if it has gone stale.

        Never blocks: it takes an uncontended lock around a single attribute
        read. See the module docstring for why staleness is enforced here rather
        than trusted to the capture thread.
        """
        with self._lock:
            snapshot = self._snapshot
        if not snapshot.ok:
            return snapshot
        if self._clock() - snapshot.t > STALE_AFTER_S:
            return GazeSnapshot(ok=False, x=None, y=None, t=self._clock())
        return snapshot

    def pause(self, timeout: float = DEFAULT_PAUSE_TIMEOUT_S) -> bool:
        """Release the camera so another application can use it.

        Returns:
            Whether the device was actually released within ``timeout``. The
            server reports this to the client, so a hopeful ``True`` here would
            become a false promise there.
        """
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout)
        released = not self._tracker_open.is_set()
        if not released:
            _log.warning("camera was not released within %.1fs", timeout)
        self._publish(GazeSnapshot(ok=False, x=None, y=None, t=self._clock()))
        return released

    def resume(self) -> bool:
        """Take the camera back.

        Returns:
            Whether it reopened. A camera another application has since claimed
            will not, and that is reported rather than raised: the server is
            mid-session and a failed resume is a state to broadcast, not a crash.
        """
        if self._thread is not None:
            return True
        try:
            self.start()
        except Exception as exc:  # noqa: BLE001 - reported, not raised; see above
            _log.warning("could not reopen the camera: %s", exc)
            return False
        return True


def _snapshot_of(result: Any, now: float) -> GazeSnapshot:
    """One :class:`~focusedgaze.types.GazeResult` in wire units.

    ``x`` and ``y`` are dropped whenever the result is not usable, rather than
    being passed through as whatever the pipeline last computed. R-6: the
    client's ``if (m.ok)`` guard is what preserves its own last good position,
    and a server that sends coordinates alongside ``ok: false`` is relying on
    every present and future client to keep that guard.
    """
    if not result.ok or result.x is None or result.y is None:
        return GazeSnapshot(ok=False, x=None, y=None, t=now)
    return GazeSnapshot(ok=True, x=float(result.x), y=float(result.y), t=now)
