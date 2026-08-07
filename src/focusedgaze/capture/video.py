"""Frame sources that need no camera: a video file, and an in-memory sequence.

These exist so the pipeline can be run and measured without hardware. CI has no
webcam (standing rule 5), and a regression in the pipeline should be catchable by
replaying the same frames rather than by asking somebody to sit in front of a
camera and form an opinion.

TWO SOURCES, BECAUSE THEY ANSWER DIFFERENT QUESTIONS
-----------------------------------------------------
:class:`VideoFileSource` decodes a file. It is what you point at a recording of a
real session.

:class:`FrameSequenceSource` replays arrays already in memory. It needs no codec,
no container and no temporary file, which makes it the one that can be relied on
everywhere: a codec that is present in one OpenCV wheel and absent from another
is precisely the kind of environment difference that produces a test which passes
on the machine it was written on. It is also what replays the Tier 2 fixture,
which is stored as an ``.npz`` of frames rather than as a video for the same
reason: no encoder sits between what was captured and what is replayed.

NEITHER SOURCE DROPS FRAMES
---------------------------
Unlike :class:`~focusedgaze.capture.webcam.WebcamSource`, which holds one slot
and overwrites it, these deliver every frame in order and never skip. That is the
whole point of a fixture: dropping frames under load would make a replay depend
on how fast the machine running it happens to be, and a golden comparison whose
inputs vary with load is not a golden comparison.

TIMESTAMPS ARE SYNTHETIC AND THAT IS DELIBERATE
------------------------------------------------
A replay generates timestamps from a stated frame rate rather than reading the
clock, so replaying the same frames twice produces identical timings. Wall-clock
timestamps would make the One Euro filter's output depend on how long decoding
took, which would turn any comparison against a recording into a flaky test.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
from numpy.typing import NDArray

from ..exceptions import CameraError
from .base import Frame, FrameSourceBase

__all__ = ["DEFAULT_REPLAY_FPS", "FrameSequenceSource", "VideoFileSource"]

#: Frames per second assumed when a source has no rate of its own. Matches the
#: 30 the legacy pipeline assumed when converting a frame index into MediaPipe's
#: millisecond timestamp (``gaze_pipeline.py:117``), so a replay through this
#: source produces the timestamps the recording was made with.
DEFAULT_REPLAY_FPS: Final = 30.0

#: ``(path) -> capture object``. Injection seam for tests, so the read loop and
#: the teardown can be exercised without depending on a codec being installed.
VideoCaptureFactory = Callable[[str], Any]


def _default_video_capture(path: str) -> Any:
    return cv2.VideoCapture(path)


def _validate_fps(fps: float) -> float:
    if not np.isfinite(fps) or fps <= 0:
        raise CameraError(f"fps must be a finite positive number, got {fps!r}")
    return float(fps)


class FrameSequenceSource(FrameSourceBase):
    """Replays frames already in memory, in order, exactly once each.

    Args:
        frames: The images, ``(height, width, 3)`` uint8 in BGR order. Accepts a
            sequence or any iterable, including a NumPy array whose first axis
            is the frame index, which is the shape the Tier 2 fixture stores.
        fps: Rate used to synthesise timestamps. Frame ``i`` is stamped
            ``i / fps``, so the first frame is at 0.0.
        loop: Replay from the start on exhaustion instead of ending. Timestamps
            keep advancing across the seam rather than jumping backwards, so a
            filter downstream sees a continuous clock.

    Raises:
        CameraError: ``frames`` is empty, ``fps`` is not positive, or a frame is
            not a 3-channel image.
    """

    __slots__ = ("_emitted", "_fps", "_frames", "_loop", "_position")

    def __init__(
        self,
        frames: Iterable[NDArray[np.uint8]] | NDArray[np.uint8],
        *,
        fps: float = DEFAULT_REPLAY_FPS,
        loop: bool = False,
    ) -> None:
        super().__init__()
        # list() over a NumPy array iterates its first axis, which is exactly
        # the frame axis of the Tier 2 fixture, so one spelling covers both the
        # array case and any other iterable.
        materialised: Sequence[Any] = list(frames)
        if not materialised:
            raise CameraError(
                "a frame sequence source needs at least one frame; got none. An "
                "empty fixture pins nothing and would report success by replaying "
                "no work at all."
            )
        for i, image in enumerate(materialised):
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[2] != 3:
                raise CameraError(
                    f"frame {i} has shape {array.shape}, expected (height, width, 3) "
                    "BGR. A single-channel or RGBA frame will not raise downstream, "
                    "it will just track badly."
                )
        self._frames = [np.asarray(f) for f in materialised]
        self._fps = _validate_fps(fps)
        self._loop = bool(loop)
        self._position = 0
        self._emitted = 0

    @property
    def size(self) -> tuple[int, int]:
        height, width = self._frames[0].shape[:2]
        return int(width), int(height)

    def __len__(self) -> int:
        return len(self._frames)

    def _read_frame(self) -> Frame | None:
        if self._position >= len(self._frames):
            if not self._loop:
                return None
            self._position = 0
        image = self._frames[self._position]
        # Stamped from the count EMITTED, not the position, so a looping source
        # does not hand out a timestamp that goes backwards at the wrap.
        frame = Frame(image=image, timestamp=self._emitted / self._fps, index=self._emitted)
        self._position += 1
        self._emitted += 1
        return frame

    def _release(self) -> None:
        # Nothing to give back: the frames belong to the caller and are not
        # dropped here, so a released source can still report its size.
        return


class VideoFileSource(FrameSourceBase):
    """Decodes a video file, one frame at a time.

    Args:
        path: The file to read.
        fps: Rate used to synthesise timestamps. ``None`` reads the rate from
            the container, falling back to :data:`DEFAULT_REPLAY_FPS` when the
            container does not state one or states an implausible one. Some
            containers report 0 or 1000; believing that would put every frame at
            the same instant or a millisecond apart.
        capture_factory: Injection seam for tests.

    Raises:
        CameraError: The file is missing or cannot be opened.
    """

    __slots__ = ("_capture", "_fps", "_path", "_position")

    def __init__(
        self,
        path: str | Path,
        *,
        fps: float | None = None,
        capture_factory: VideoCaptureFactory = _default_video_capture,
    ) -> None:
        super().__init__()
        self._path = Path(path)
        # Checked before opening: OpenCV returns "not opened" for a missing file
        # and for an unreadable codec alike, and those need different remedies.
        if not self._path.is_file():
            raise CameraError(f"no such video file: {self._path}")

        capture = capture_factory(str(self._path))
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"could not open {self._path} as a video. The file exists, so this "
                "is a codec or container OpenCV cannot read rather than a missing "
                "path. Re-encode it, or use FrameSequenceSource with frames you "
                "have already decoded."
            )
        self._capture = capture
        self._position = 0
        self._fps = _validate_fps(fps) if fps is not None else self._container_fps()

    def _container_fps(self) -> float:
        """The file's own frame rate, when it states a believable one."""
        try:
            declared = float(self._capture.get(cv2.CAP_PROP_FPS))
        except Exception:  # noqa: BLE001 - a backend may not support the property
            return DEFAULT_REPLAY_FPS
        # Upper bound as well as lower: a container claiming 1000 fps is stating
        # a timebase, not a rate, and stamping frames a millisecond apart would
        # make every velocity downstream enormous.
        if not np.isfinite(declared) or not (1.0 <= declared <= 480.0):
            return DEFAULT_REPLAY_FPS
        return declared

    @property
    def size(self) -> tuple[int, int]:
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    @property
    def fps(self) -> float:
        """The rate timestamps are being generated at."""
        return self._fps

    def _read_frame(self) -> Frame | None:
        ok, image = self._capture.read()
        if not ok or image is None:
            return None
        frame = Frame(image=image, timestamp=self._position / self._fps, index=self._position)
        self._position += 1
        return frame

    def _release(self) -> None:
        self._capture.release()
