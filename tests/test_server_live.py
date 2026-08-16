"""The live camera source behind ``focusedgaze serve``.

Runs with no webcam: :class:`LiveGazeSource` takes a tracker factory, and these
substitute a fake that yields scripted results. That is the same seam the rest
of the suite uses for hardware, and it is what makes the threading testable at
all -- a test that needed a camera could not assert on what happens when one
stops delivering frames, which is half of what this module exists to handle.

WHAT IS ACTUALLY BEING PINNED
-----------------------------
Not "does a reading arrive", which any smoke test would catch. The failures
worth a test here are the quiet ones:

    a stale slot          a dead capture thread keeps serving its last good
                          reading, and the client shows a frozen cursor with
                          nothing reporting a fault
    coordinates on !ok    R-6: x and y must be None whenever ok is false, or a
                          client without the `if (m.ok)` guard drifts silently
    a leaked camera       pause() reports success while the device is still held
    a blocking latest()   couples the broadcast rate to the capture rate
"""

from __future__ import annotations

import threading
import time

import pytest

from focusedgaze.server import GazeSnapshot
from focusedgaze.server.live import STALE_AFTER_S, LiveGazeSource
from focusedgaze.types import GazeStatus


class _Result:
    """The half of GazeResult this module reads."""

    def __init__(self, ok: bool, x: float | None = None, y: float | None = None) -> None:
        self.ok = ok
        self.x = x
        self.y = y
        self.status = GazeStatus.OK if ok else GazeStatus.NO_FACE


class _FakeTracker:
    """Yields scripted results, then blocks so the thread stays alive.

    Blocking rather than returning ``None`` at the end matters: ``None`` means
    "the source ended", which is a distinct behaviour with its own test below.
    A real webcam does not end, so the steady state is "waiting for a frame".
    """

    def __init__(self, results, hold: bool = True) -> None:
        self._results = list(results)
        self._hold = hold
        self.closed = threading.Event()
        self.delivered = threading.Event()

    def read(self):
        if self._results:
            result = self._results.pop(0)
            if not self._results:
                self.delivered.set()
            return result
        if not self._hold:
            return None
        time.sleep(0.005)
        return _Result(False)

    def close(self) -> None:
        self.closed.set()


def _source(tracker, clock=time.time) -> LiveGazeSource:
    return LiveGazeSource(tracker_factory=lambda: tracker, clock=clock)


def _wait(predicate, timeout: float = 2.0) -> bool:
    """Poll until true. Threads make an unconditional assert a race."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


# ---------------------------------------------------------------------------
# Readings reach the slot.


def test_a_reading_reaches_latest() -> None:
    tracker = _FakeTracker([_Result(True, 0.25, 0.75)])
    with _source(tracker) as source:
        assert _wait(lambda: source.latest().ok), "no reading arrived"
        snapshot = source.latest()
    assert snapshot.x == pytest.approx(0.25)
    assert snapshot.y == pytest.approx(0.75)


def test_before_the_first_frame_the_answer_is_not_ok_rather_than_empty() -> None:
    """A client connecting early must be told "no reading yet", not given one."""
    source = LiveGazeSource(tracker_factory=lambda: _FakeTracker([]))
    snapshot = source.latest()
    assert snapshot.ok is False
    assert snapshot.x is None and snapshot.y is None


def test_an_unusable_result_carries_no_coordinates() -> None:
    """R-6. A server that sends a point alongside ok:false is relying on every
    present and future client to keep its guard."""
    tracker = _FakeTracker([_Result(False, 0.4, 0.4)])
    with _source(tracker) as source:
        assert _wait(lambda: tracker.delivered.is_set())
        snapshot = source.latest()
    assert snapshot.ok is False
    assert snapshot.x is None and snapshot.y is None


# ---------------------------------------------------------------------------
# Staleness. The frozen-cursor failure.


def test_a_stale_reading_is_reported_as_unusable() -> None:
    """The headline test. A slot that keeps its last good reading forever is
    how a dead capture thread becomes a cursor that looks alive."""
    now = [1000.0]
    tracker = _FakeTracker([_Result(True, 0.5, 0.5)])
    with _source(tracker, clock=lambda: now[0]) as source:
        assert _wait(lambda: source.latest().ok), "no reading arrived"
        now[0] += STALE_AFTER_S * 2
        snapshot = source.latest()
    assert snapshot.ok is False, "a stale reading was served as usable"
    assert snapshot.x is None and snapshot.y is None


def test_a_fresh_reading_is_not_called_stale() -> None:
    """The control: the staleness horizon must not blank a working camera."""
    now = [1000.0]
    tracker = _FakeTracker([_Result(True, 0.5, 0.5)])
    with _source(tracker, clock=lambda: now[0]) as source:
        assert _wait(lambda: source.latest().ok)
        now[0] += STALE_AFTER_S / 2
        assert source.latest().ok is True


def test_a_source_that_ends_goes_stale_rather_than_repeating_itself() -> None:
    """A webcam that is unplugged mid-session. The thread stops; the last good
    reading must not outlive it."""
    now = [1000.0]
    tracker = _FakeTracker([_Result(True, 0.5, 0.5)], hold=False)
    source = LiveGazeSource(tracker_factory=lambda: tracker, clock=lambda: now[0])
    source.start()
    assert _wait(lambda: tracker.closed.is_set()), "the tracker was not closed"
    now[0] += STALE_AFTER_S * 2
    assert source.latest().ok is False
    source.close()


# ---------------------------------------------------------------------------
# latest() must not block.


def test_latest_does_not_block_on_the_capture_rate() -> None:
    """It is called on the broadcaster tick. Coupling it to frame arrival would
    stall the event loop for every connected client at once."""
    tracker = _FakeTracker([_Result(True, 0.5, 0.5)])
    with _source(tracker) as source:
        assert _wait(lambda: source.latest().ok)
        started = time.monotonic()
        for _ in range(200):
            source.latest()
        elapsed = time.monotonic() - started
    assert elapsed < 0.2, f"200 calls took {elapsed:.3f}s; latest() is blocking"


# ---------------------------------------------------------------------------
# The camera lease.


def test_pause_releases_the_camera_and_says_so() -> None:
    tracker = _FakeTracker([_Result(True, 0.5, 0.5)])
    source = _source(tracker)
    source.start()
    assert _wait(lambda: source.latest().ok)

    assert source.pause() is True, "pause reported a camera it had not released"
    assert tracker.closed.is_set(), "the tracker was not closed"
    assert source.latest().ok is False, "readings continued after a pause"
    source.close()


def test_resume_reopens_and_readings_return() -> None:
    trackers = [
        _FakeTracker([_Result(True, 0.1, 0.1)]),
        _FakeTracker([_Result(True, 0.9, 0.9)]),
    ]
    source = LiveGazeSource(tracker_factory=lambda: trackers.pop(0))
    source.start()
    assert _wait(lambda: source.latest().ok)
    source.pause()

    assert source.resume() is True
    assert _wait(lambda: source.latest().ok), "no readings after resume"
    assert source.latest().x == pytest.approx(0.9), "the second tracker was not used"
    source.close()


def test_a_camera_that_will_not_reopen_is_reported_not_raised() -> None:
    """The server is mid-session. A failed resume is a state to broadcast."""
    first = _FakeTracker([_Result(True, 0.5, 0.5)])
    built = [first]

    def factory():
        if built:
            return built.pop()
        raise RuntimeError("device is held by another application")

    source = LiveGazeSource(tracker_factory=factory)
    source.start()
    source.pause()
    assert source.resume() is False


def test_resume_while_already_running_is_a_no_op() -> None:
    tracker = _FakeTracker([_Result(True, 0.5, 0.5)])
    source = _source(tracker)
    source.start()
    assert source.resume() is True
    assert tracker.closed.is_set() is False, "resume tore down a running capture"
    source.close()


def test_close_is_idempotent() -> None:
    tracker = _FakeTracker([_Result(True, 0.5, 0.5)])
    source = _source(tracker)
    source.start()
    source.close()
    source.close()
    assert tracker.closed.is_set()


def test_a_failure_to_open_surfaces_to_the_caller() -> None:
    """Built on the calling thread on purpose: a camera that cannot open must
    raise where somebody is ready to report it, not into a thread's traceback."""

    def factory():
        raise RuntimeError("no camera")

    with pytest.raises(RuntimeError, match="no camera"):
        LiveGazeSource(tracker_factory=factory).start()


# ---------------------------------------------------------------------------
# It is what the server will accept.


def test_it_satisfies_the_server_source_protocol() -> None:
    from focusedgaze.server import GazeSource

    assert isinstance(LiveGazeSource(tracker_factory=lambda: _FakeTracker([])), GazeSource)


def test_the_snapshot_is_the_wire_type() -> None:
    tracker = _FakeTracker([_Result(True, 0.5, 0.5)])
    with _source(tracker) as source:
        assert _wait(lambda: source.latest().ok)
        assert isinstance(source.latest(), GazeSnapshot)
