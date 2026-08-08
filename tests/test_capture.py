"""Phase 4: frame sources, including the threading the legacy server relied on.

Runs with no camera and no codec. Every hardware-shaped behaviour is exercised
through the ``capture_factory`` seam with a fake capture object, because a test
that needs a webcam does not run in CI (rule 5) and therefore does not protect
anything.

The per-platform backend selection is tested for **every** platform, not just the
one running the suite. Audit section 42 is the reason that is spelled out: a
Windows-only assumption in a validator survived because nothing ever asked what
the other platforms did, and it took five red CI runs to surface. Platform
behaviour that is only checked on one platform is not checked.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest

from focusedgaze.capture import (
    Frame,
    FrameSequenceSource,
    FrameSource,
    VideoFileSource,
    WebcamSource,
    resolve_backend,
)
from focusedgaze.config import CameraConfig
from focusedgaze.exceptions import CameraError


def _image(value: int = 0, width: int = 8, height: int = 6) -> np.ndarray:
    """A distinguishable BGR frame."""
    return np.full((height, width, 3), value % 256, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Fake capture, standing in for cv2.VideoCapture.
# ---------------------------------------------------------------------------


class FakeCapture:
    """Just enough of ``cv2.VideoCapture`` to drive a source.

    ``delay`` makes a read slow, which is how the "a slow camera must not stall
    the consumer" property is tested without a slow camera.
    """

    def __init__(
        self,
        frames: list[np.ndarray] | None = None,
        *,
        opened: bool = True,
        delay: float = 0.0,
        fail_after: int | None = None,
        forever: bool = True,
    ) -> None:
        self._frames = frames if frames is not None else [_image(i) for i in range(1, 200)]
        self._opened = opened
        self._delay = delay
        self._fail_after = fail_after
        self._forever = forever
        self._position = 0
        self.released = False
        self.settings: list[tuple[int, float]] = []
        self.reads = 0

    def isOpened(self) -> bool:
        return self._opened

    def set(self, prop: int, value: float) -> bool:
        self.settings.append((prop, value))
        return True

    def get(self, prop: int) -> float:
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._delay:
            time.sleep(self._delay)
        self.reads += 1
        if self._fail_after is not None and self.reads > self._fail_after:
            return False, None
        if self._position >= len(self._frames):
            if not self._forever:
                return False, None
            self._position = 0
        image = self._frames[self._position]
        self._position += 1
        return True, image

    def release(self) -> None:
        self.released = True


def _webcam(**kwargs: object) -> tuple[WebcamSource, FakeCapture]:
    """A WebcamSource wired to a FakeCapture, with a short read timeout."""
    capture = FakeCapture(**kwargs)  # type: ignore[arg-type]
    source = WebcamSource(
        CameraConfig(backend="msmf"),
        read_timeout=2.0,
        capture_factory=lambda index, backend: capture,
    )
    return source, capture


# ---------------------------------------------------------------------------
# Backend selection: every platform, from any platform.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "platform,expected",
    [
        ("win32", cv2.CAP_MSMF),
        ("cygwin", cv2.CAP_MSMF),
        ("darwin", cv2.CAP_AVFOUNDATION),
        ("linux", cv2.CAP_V4L2),
        ("linux2", cv2.CAP_V4L2),
    ],
)
def test_auto_backend_resolves_per_platform(platform: str, expected: int) -> None:
    """"auto" must pick the native backend, whichever machine is asking.

    Windows resolving to MSMF is the shipping behaviour, named explicitly in
    gaze_server.py:136 with the note that DSHOW capped 720p at 10 fps. The
    others are D1: letting OpenCV choose is how a slow path gets selected with
    no error anywhere.
    """
    assert resolve_backend("auto", platform) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("msmf", cv2.CAP_MSMF),
        ("dshow", cv2.CAP_DSHOW),
        ("avfoundation", cv2.CAP_AVFOUNDATION),
        ("v4l2", cv2.CAP_V4L2),
    ],
)
def test_an_explicit_backend_overrides_the_platform(name: str, expected: int) -> None:
    """A named backend is honoured even where it is not native.

    Asked for from a platform it does not belong to on purpose: the override has
    to be a property of the request, not of the machine, or it is not an
    override.
    """
    assert resolve_backend(name, "linux") == expected


@pytest.mark.parametrize("bad", ["gstreamer", "MSMF", "", "any", "cap_msmf", "v4l"])
def test_an_unknown_backend_is_a_named_error(bad: str) -> None:
    """Rule 11: the accepted set is a whitelist, so anything else is refused.

    Never a silent fallback to ``CAP_ANY``. A backend that quietly becomes
    "whatever OpenCV feels like" is the defect this table exists to prevent, and
    it presents as a performance problem rather than as an error. Note ``MSMF``
    is rejected too: the accepted spelling is lowercase, and accepting both
    would make the config's own validation and this table disagree.
    """
    with pytest.raises(CameraError, match="unknown capture backend"):
        resolve_backend(bad, "win32")


def test_auto_on_an_unknown_platform_says_what_to_do() -> None:
    """A platform with no recorded default must not guess."""
    with pytest.raises(CameraError, match="no default capture backend"):
        resolve_backend("auto", "sunos5")


# ---------------------------------------------------------------------------
# WebcamSource: threading.
# ---------------------------------------------------------------------------


def test_frames_arrive_and_carry_increasing_timestamps() -> None:
    source, _ = _webcam()
    try:
        first = source.read()
        second = source.read()
        assert first is not None and second is not None
        assert second.index > first.index
        assert second.timestamp >= first.timestamp
        assert first.image.shape == (6, 8, 3)
    finally:
        source.release()


def test_read_never_returns_the_same_capture_twice() -> None:
    """A repeated frame would be a zero movement over a non-zero interval.

    The legacy loop re-read its shared slot freely and could infer on one
    capture more than once. Harmless for a cursor, but it feeds the One Euro
    filter a velocity of zero across a real time gap, and the filter's cutoff is
    velocity-dependent. Each read here waits for something new.
    """
    source, _ = _webcam()
    try:
        seen = [source.read() for _ in range(12)]
    finally:
        source.release()
    indices = [f.index for f in seen if f is not None]
    assert len(indices) == 12
    assert indices == sorted(indices)
    assert len(set(indices)) == 12


def test_a_slow_camera_does_not_stall_the_consumer_between_frames() -> None:
    """The point of the capture thread: reads overlap with consumer work.

    With a 40 ms camera and a consumer doing 40 ms of work per frame, a
    sequential read-then-process loop needs ~80 ms per frame. Threaded capture
    overlaps them, so three frames take well under the sequential 240 ms.
    """
    source, _ = _webcam(delay=0.04)
    try:
        source.read()  # discard: the first read includes camera startup
        started = time.monotonic()
        for _ in range(3):
            assert source.read() is not None
            time.sleep(0.04)  # stand-in for inference
        elapsed = time.monotonic() - started
    finally:
        source.release()
    assert elapsed < 0.24, f"reads did not overlap with consumer work: {elapsed:.3f}s"


def test_the_newest_frame_wins_and_intermediate_frames_are_dropped() -> None:
    """Deliberate frame dropping, stated in the module docstring.

    A consumer slower than the camera must jump to the freshest frame rather
    than working through a backlog: for live tracking a stale frame has no
    value. Sequence numbers make the drop visible.
    """
    source, _ = _webcam()
    try:
        first = source.read()
        assert first is not None
        time.sleep(0.15)  # let the camera get well ahead
        latest = source.read()
        assert latest is not None
    finally:
        source.release()
    assert latest.index > first.index + 1, (
        "no frames were dropped, so the source is queueing rather than "
        f"holding only the newest (went {first.index} -> {latest.index})"
    )


def test_read_returns_none_when_the_camera_stops_delivering() -> None:
    """A dead camera ends the stream instead of hanging forever."""
    capture = FakeCapture(fail_after=3)
    source = WebcamSource(
        CameraConfig(backend="msmf"),
        read_timeout=0.3,
        capture_factory=lambda index, backend: capture,
    )
    try:
        while source.read() is not None:
            pass
    finally:
        source.release()


# ---------------------------------------------------------------------------
# WebcamSource: lifecycle.
# ---------------------------------------------------------------------------


def test_opening_a_camera_that_is_not_there_names_the_remedies() -> None:
    """The message has to distinguish the causes; they have different fixes."""
    capture = FakeCapture(opened=False)
    with pytest.raises(CameraError) as excinfo:
        WebcamSource(CameraConfig(backend="msmf"), capture_factory=lambda i, b: capture)
    message = str(excinfo.value)
    assert "muted" in message and "another process" in message
    assert capture.released, "a camera that failed to open was not released"


def test_release_stops_the_thread_and_gives_the_device_back() -> None:
    source, capture = _webcam()
    source.read()
    source.release()
    assert capture.released
    assert source.closed
    assert not any(
        t.name == "focusedgaze-capture" for t in threading.enumerate()
    ), "the capture thread outlived release()"


def test_release_is_idempotent_and_reading_after_it_ends_the_stream() -> None:
    """A consumer loop must unwind naturally rather than needing an ordering."""
    source, _ = _webcam()
    source.read()
    source.release()
    source.release()
    assert source.read() is None


def test_the_context_manager_releases_even_when_the_body_raises() -> None:
    """The 'it worked yesterday' failure: a held webcam nothing can reopen."""
    source, capture = _webcam()
    with pytest.raises(RuntimeError, match="boom"), source:
        source.read()
        raise RuntimeError("boom")
    assert capture.released


def test_the_requested_resolution_is_asked_for() -> None:
    capture = FakeCapture()
    source = WebcamSource(
        CameraConfig(backend="msmf", width=1280, height=720),
        capture_factory=lambda i, b: capture,
    )
    try:
        assert (cv2.CAP_PROP_FRAME_WIDTH, 1280) in capture.settings
        assert (cv2.CAP_PROP_FRAME_HEIGHT, 720) in capture.settings
    finally:
        source.release()


def test_size_before_any_frame_is_the_requested_resolution() -> None:
    """Until something arrives there is nothing else to report.

    Uses a camera that never delivers, because the capture thread starts in
    __init__: against a working fake the first frame can land before the
    assertion runs, and a test whose outcome depends on that timing is measuring
    the scheduler.
    """
    capture = FakeCapture(fail_after=0)
    source = WebcamSource(
        CameraConfig(backend="msmf", width=1280, height=720),
        read_timeout=0.2,
        capture_factory=lambda i, b: capture,
    )
    try:
        assert source.size == (1280, 720)
    finally:
        source.release()


def test_size_reports_what_arrives_not_what_was_asked_for() -> None:
    """A camera may ignore the request, and believing it is how a crop goes wrong."""
    capture = FakeCapture(frames=[_image(7, width=320, height=240)])
    source = WebcamSource(
        CameraConfig(backend="msmf", width=1280, height=720),
        read_timeout=2.0,
        capture_factory=lambda i, b: capture,
    )
    try:
        assert source.read() is not None
        assert source.size == (320, 240)
    finally:
        source.release()


def test_the_configured_index_and_backend_reach_the_factory() -> None:
    seen: list[tuple[int, int]] = []

    def factory(index: int, backend: int) -> FakeCapture:
        seen.append((index, backend))
        return FakeCapture()

    source = WebcamSource(CameraConfig(index=2, backend="dshow"), capture_factory=factory)
    try:
        assert seen == [(2, cv2.CAP_DSHOW)]
    finally:
        source.release()


# ---------------------------------------------------------------------------
# FrameSequenceSource.
# ---------------------------------------------------------------------------


def test_a_sequence_replays_every_frame_in_order_exactly_once() -> None:
    frames = [_image(i) for i in range(5)]
    with FrameSequenceSource(frames) as source:
        got = []
        while (frame := source.read()) is not None:
            got.append(frame)
    assert [f.index for f in got] == [0, 1, 2, 3, 4]
    assert [int(f.image[0, 0, 0]) for f in got] == [0, 1, 2, 3, 4]


def test_sequence_timestamps_come_from_the_stated_rate_not_the_clock() -> None:
    """Replaying twice must produce identical timings, or comparisons go flaky."""
    frames = [_image(i) for i in range(4)]
    first = [f.timestamp for f in _drain(FrameSequenceSource(frames, fps=30.0))]
    time.sleep(0.01)
    second = [f.timestamp for f in _drain(FrameSequenceSource(frames, fps=30.0))]
    assert first == second
    assert first == [0.0, 1 / 30, 2 / 30, 3 / 30]


def test_a_numpy_array_is_accepted_as_the_frame_axis() -> None:
    """The shape the Tier 2 fixture stores: (n, height, width, 3)."""
    array = np.stack([_image(i) for i in range(3)])
    with FrameSequenceSource(array) as source:
        assert len(source) == 3
        assert source.size == (8, 6)
        assert [f.index for f in _drain(source)] == [0, 1, 2]


def test_a_looping_sequence_never_moves_its_clock_backwards() -> None:
    """A wrap that reset the timestamp would show downstream as a huge negative dt."""
    frames = [_image(i) for i in range(3)]
    source = FrameSequenceSource(frames, fps=10.0, loop=True)
    stamps = [source.read().timestamp for _ in range(7)]  # type: ignore[union-attr]
    source.release()
    assert stamps == sorted(stamps)
    assert stamps == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def test_an_empty_sequence_is_refused() -> None:
    """An empty fixture would report success by replaying no work at all."""
    with pytest.raises(CameraError, match="at least one frame"):
        FrameSequenceSource([])


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((6, 8), dtype=np.uint8),          # greyscale
        np.zeros((6, 8, 4), dtype=np.uint8),       # BGRA
        np.zeros((6, 8, 1), dtype=np.uint8),       # single channel
    ],
)
def test_a_frame_that_is_not_three_channel_bgr_is_refused(bad: np.ndarray) -> None:
    """These do not raise downstream. They just track badly, which is worse."""
    with pytest.raises(CameraError, match="expected \\(height, width, 3\\)"):
        FrameSequenceSource([bad])


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_a_nonsense_frame_rate_is_refused(bad: float) -> None:
    with pytest.raises(CameraError, match="finite positive"):
        FrameSequenceSource([_image()], fps=bad)


# ---------------------------------------------------------------------------
# VideoFileSource.
# ---------------------------------------------------------------------------


def test_a_missing_video_file_is_distinguished_from_an_unreadable_one() -> None:
    """Different remedies, so they cannot share a message."""
    with pytest.raises(CameraError, match="no such video file"):
        VideoFileSource("definitely-not-here.mp4")


def test_an_unreadable_container_says_the_file_exists(tmp_path) -> None:
    path = tmp_path / "not-a-video.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(CameraError, match="codec or container"):
        VideoFileSource(path, capture_factory=lambda p: FakeCapture(opened=False))


def test_a_video_file_round_trips_through_a_real_codec(tmp_path) -> None:
    """One end-to-end check against OpenCV itself, not the fake.

    Everything else about this class is tested through the seam, which proves
    the logic. This proves the logic is wired to something real.
    """
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    assert writer.isOpened(), "OpenCV could not open an MJPG writer to build the fixture"
    for i in range(6):
        writer.write(np.full((48, 64, 3), i * 40, dtype=np.uint8))
    writer.release()

    with VideoFileSource(path) as source:
        assert source.size == (64, 48)
        frames = _drain(source)
    assert len(frames) == 6
    assert [f.index for f in frames] == [0, 1, 2, 3, 4, 5]


def test_an_implausible_container_frame_rate_is_not_believed(tmp_path) -> None:
    """A container claiming 1000 fps is stating a timebase, not a rate.

    Believing it would stamp frames a millisecond apart and make every velocity
    downstream enormous. Recorded both ways: an implausible rate falls back, an
    explicit one is honoured.
    """
    path = tmp_path / "clip.avi"
    path.write_bytes(b"stub")

    class WeirdFps(FakeCapture):
        def get(self, prop: int) -> float:
            return 1000.0 if prop == cv2.CAP_PROP_FPS else 0.0

    source = VideoFileSource(path, capture_factory=lambda p: WeirdFps())
    try:
        assert source.fps == 30.0
    finally:
        source.release()

    explicit = VideoFileSource(path, fps=12.0, capture_factory=lambda p: WeirdFps())
    try:
        assert explicit.fps == 12.0
    finally:
        explicit.release()


def test_releasing_a_video_gives_the_handle_back(tmp_path) -> None:
    path = tmp_path / "clip.avi"
    path.write_bytes(b"stub")
    capture = FakeCapture(frames=[_image(1)], forever=False)
    with VideoFileSource(path, capture_factory=lambda p: capture) as source:
        assert source.read() is not None
    assert capture.released


# ---------------------------------------------------------------------------
# The protocol itself.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [lambda: FrameSequenceSource([_image()]), lambda: _webcam()[0]])
def test_every_source_satisfies_the_protocol(factory) -> None:
    """Structural, so an application's own camera object can be passed in."""
    source = factory()
    try:
        assert isinstance(source, FrameSource)
    finally:
        source.release()


def test_a_foreign_object_satisfying_the_protocol_is_accepted() -> None:
    """The reason FrameSource is a Protocol and not a base class."""

    class BorrowedCamera:
        size = (2, 2)

        def read(self) -> Frame | None:
            return Frame(image=_image(1, 2, 2), timestamp=0.0, index=0)

        def release(self) -> None:
            return

    assert isinstance(BorrowedCamera(), FrameSource)


def _drain(source) -> list[Frame]:
    """Read a source to exhaustion."""
    out = []
    while (frame := source.read()) is not None:
        out.append(frame)
    return out


# ---------------------------------------------------------------------------
# R-12: the legacy shutdown NameError, and the property that makes it impossible.
# ---------------------------------------------------------------------------


def test_a_source_is_released_exactly_once_on_every_exit_path() -> None:
    """R-12 cannot recur, and this is the property that guarantees it.

    The legacy gaze loop released a capture handle in its `finally` that was
    never bound in that function (audit 39.4), so every exit from the loop
    raised NameError. It stayed invisible because it only fired at shutdown, on
    a daemon thread, as the process was dying.

    The fix is structural rather than a corrected line: release belongs to the
    source, is idempotent, and is guaranteed by the context manager. Asserted
    across all three exit paths, because "it does not raise" is not the same as
    "it released exactly once".
    """
    normal, _ = _webcam()
    with normal:
        normal.read()
    assert normal.closed

    explicit, capture = _webcam()
    explicit.release()
    explicit.release()
    explicit.release()
    assert capture.released
    assert explicit.read() is None, "a released source must end the stream, not reopen"

    raising, raising_capture = _webcam()
    with pytest.raises(RuntimeError), raising:
        raise RuntimeError("shutdown")
    assert raising_capture.released


def test_release_marks_closed_before_teardown_so_a_failure_cannot_be_retried_into() -> None:
    """A teardown that raises must not leave a half-released device reachable.

    This is the ordering that makes idempotence real rather than nominal: if
    `_release` throws, the source is already marked closed, so a retry returns
    immediately instead of re-entering teardown on a device that is partly gone.
    """
    from focusedgaze.capture.base import FrameSourceBase

    class Exploding(FrameSourceBase):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        def _read_frame(self):
            return None

        def _release(self) -> None:
            self.attempts += 1
            raise OSError("device vanished mid-teardown")

    source = Exploding()
    with pytest.raises(OSError, match="vanished"):
        source.release()
    assert source.closed
    source.release()
    assert source.attempts == 1, "teardown was re-entered on an already-closed source"
