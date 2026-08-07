"""Threaded webcam capture with per-platform backend selection.

CARRIED OVER FROM THE LEGACY SERVER: WHY CAPTURE HAS ITS OWN THREAD
--------------------------------------------------------------------
``gaze_server.py`` ran ``cap.read()`` on a dedicated thread that wrote the newest
frame into a shared slot, and the inference loop read that slot rather than the
camera. Its comment (L111-113) gives the reason: "Decoupling capture from
inference means the slow webcam read never stalls the GPU inference, so cursor
updates arrive as fast as frames do."

That is preserved exactly, because the alternative is worse than it looks.
``VideoCapture.read()`` blocks until the driver hands over a frame, so a
sequential read-then-infer loop runs at the *sum* of both costs and, worse, the
frame it infers on is already one camera interval old before inference starts.
With capture on its own thread, inference always begins from the freshest frame
available and the two costs overlap.

The consequence a caller must understand: **frames are dropped, deliberately.**
The slot holds one frame and the capture thread overwrites it. If inference is
slower than the camera, intermediate frames are discarded rather than queued.
That is correct for live tracking, where a stale frame has no value, and wrong
for recording, which is why :class:`~focusedgaze.capture.video.FrameSequenceSource`
exists and why the Tier 2 recorder does not use this class.

BACKEND SELECTION (D1)
----------------------
The legacy code opened ``cv2.CAP_MSMF`` unconditionally, with a note that DSHOW
capped 720p at 10 fps against MSMF's ~31. Windows keeps exactly that. The other
platforms get the backend that is native to them rather than ``CAP_ANY``,
because letting OpenCV choose is how you end up on a slow path silently:
`docs/troubleshooting.md` lists "video is smooth, gaze lags" as a backend
symptom, and a wrong-but-working backend produces no error at all.

ONE OWNER AT A TIME
-------------------
A webcam has a single owner on Windows, and reopening it before the previous
owner has let go fails. :meth:`WebcamSource.release` therefore joins the capture
thread before releasing the device and then waits a measured beat for the driver.
Those durations are empirical and are not to be rounded away.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, Final

import cv2

from ..config import CameraConfig
from ..exceptions import CameraError
from .base import Frame, FrameSourceBase, now

__all__ = ["DEFAULT_READ_TIMEOUT_S", "WebcamSource", "resolve_backend"]

_log = logging.getLogger(__name__)

#: Backend name -> OpenCV constant. An explicit table, per standing rule 11:
#: the accepted set is written out rather than inferred from whatever OpenCV
#: happens to expose, so an unknown name is a named error instead of a silent
#: fallback to ``CAP_ANY``. ``CameraConfig`` validates against the same set of
#: names, so a bad value is normally caught at configuration time; this map is
#: the second gate for a source constructed with a hand-made config.
_BACKENDS: Final[dict[str, int]] = {
    "msmf": cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "v4l2": cv2.CAP_V4L2,
}

#: Platform -> the backend "auto" resolves to. ``sys.platform`` prefixes, so
#: "linux2" and any future "linuxN" still match. Windows resolves to MSMF, which
#: is what the shipping system named explicitly (gaze_server.py:136).
_AUTO_BY_PLATFORM: Final[tuple[tuple[str, str], ...]] = (
    ("win32", "msmf"),
    ("cygwin", "msmf"),
    ("darwin", "avfoundation"),
    ("linux", "v4l2"),
)

#: Seconds to wait for the driver after releasing, before anything may reopen
#: the device. MEASURED, not chosen: the legacy server used exactly this
#: (gaze_server.py:159, "the driver needs a beat before it can be reopened").
#: Do not round it; the failure it prevents is an open() that fails on Windows.
_DRIVER_SETTLE_S: Final = 0.3

#: Seconds to wait for the capture thread to notice it should stop. Generous
#: against a blocked ``cap.read()``, which is not interruptible: the thread can
#: only exit once the driver returns.
_THREAD_JOIN_TIMEOUT_S: Final = 4.0

#: Default seconds :meth:`WebcamSource.read` waits for a frame newer than the
#: last one it returned. Comfortably longer than a frame interval at any
#: plausible rate, short enough that a dead camera is reported rather than hung.
DEFAULT_READ_TIMEOUT_S: Final = 5.0

#: ``(index, backend_flag) -> capture object``. Substituted in tests, which is
#: the only way to exercise the threading without a webcam.
CaptureFactory = Callable[[int, int], Any]


def _default_capture_factory(index: int, backend: int) -> Any:
    return cv2.VideoCapture(index, backend)


def resolve_backend(name: str, platform: str | None = None) -> int:
    """Map a configured backend name to the OpenCV constant to open with.

    Args:
        name: One of the names in :data:`_BACKENDS`, or ``"auto"``.
        platform: ``sys.platform`` value to resolve ``"auto"`` against. Defaults
            to the running platform; passed explicitly by tests, because
            per-platform behaviour that is only ever exercised on one platform
            is exactly the class of defect audit section 42 was about.

    Raises:
        CameraError: The name is not one this package knows, or ``"auto"`` was
            asked for on a platform with no native backend recorded.
    """
    if name in _BACKENDS:
        return _BACKENDS[name]
    if name != "auto":
        known = ", ".join(sorted([*_BACKENDS, "auto"]))
        raise CameraError(f"unknown capture backend {name!r}. Known backends: {known}")

    system = sys.platform if platform is None else platform
    for prefix, backend in _AUTO_BY_PLATFORM:
        if system.startswith(prefix):
            return _BACKENDS[backend]
    raise CameraError(
        f"no default capture backend is recorded for platform {system!r}. "
        "Set camera.backend explicitly to one of: " + ", ".join(sorted(_BACKENDS))
    )


class WebcamSource(FrameSourceBase):
    """A live webcam, captured on a dedicated thread.

    Args:
        config: Which device to open and how. Its ``backend`` and ``index`` are
            already validated by :class:`~focusedgaze.config.CameraConfig`.
        read_timeout: Seconds :meth:`read` waits for a new frame before deciding
            the camera has stopped delivering.
        capture_factory: Injection seam for tests. Receives ``(index, backend)``
            and returns an object with OpenCV's ``isOpened``, ``set``, ``read``
            and ``release``.
        platform: Override for ``"auto"`` backend resolution. Tests only.

    The camera is opened in ``__init__`` and the capture thread starts
    immediately, so a constructed instance is either working or has raised.
    A source that opened lazily on first ``read()`` would report a missing
    camera from somewhere the caller was not expecting it.

    ``config.mirror`` is deliberately **not** applied here. The legacy server
    published unmirrored frames and flipped a local copy inside the gaze loop
    (``read_shared_frame`` says so explicitly), because a shared frame has more
    than one consumer and they do not agree about handedness. Mirroring is a
    property of how a frame is interpreted, not of how it was captured, so it
    belongs to whatever consumes this source. Applying it here would silently
    double-flip for any consumer that also honours the setting.

    Raises:
        CameraError: The device could not be opened.
    """

    __slots__ = (
        "_capture",
        "_config",
        "_frame",
        "_last_returned",
        "_lock",
        "_new_frame",
        "_read_timeout",
        "_seq",
        "_size",
        "_stop",
        "_thread",
    )

    def __init__(
        self,
        config: CameraConfig | None = None,
        *,
        read_timeout: float = DEFAULT_READ_TIMEOUT_S,
        capture_factory: CaptureFactory = _default_capture_factory,
        platform: str | None = None,
    ) -> None:
        super().__init__()
        self._config = config if config is not None else CameraConfig()
        self._read_timeout = float(read_timeout)

        backend = resolve_backend(self._config.backend, platform)
        capture = capture_factory(self._config.index, backend)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"could not open camera index {self._config.index} using backend "
                f"{self._config.backend!r}. The device may not exist, may be muted, "
                "or may be held by another process: a crashed program can keep a "
                "webcam open until it is closed. Try a different camera.index, or "
                "set camera.backend explicitly."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)

        self._capture = capture
        self._lock = threading.Lock()
        # Condition over the same lock: a consumer waits to be told a frame
        # arrived rather than polling, so read() adds no latency of its own.
        self._new_frame = threading.Condition(self._lock)
        self._frame: Frame | None = None
        self._seq = 0
        self._last_returned = -1
        self._size = (int(self._config.width), int(self._config.height))
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._capture_loop, name="focusedgaze-capture", daemon=True
        )
        self._thread.start()

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` of the frames actually arriving.

        The requested size until the first frame lands, then the real one. A
        camera is free to ignore what it was asked for, and silently believing
        the request is how a crop ends up computed against the wrong geometry.
        """
        with self._lock:
            return self._size

    def _capture_loop(self) -> None:
        """Owns the camera. Publishes the newest frame; never queues."""
        while not self._stop.is_set():
            try:
                ok, image = self._capture.read()
            except Exception:  # noqa: BLE001 - a dying driver can raise anything
                _log.exception("capture thread: read failed, stopping")
                break
            if not ok or image is None:
                # A single dropped frame is normal; the legacy loop simply
                # continued (gaze_server.py:248). Continuing here would spin at
                # full speed against a camera that has gone away, so the stop
                # event is checked and the loop yields.
                if self._stop.wait(0.005):
                    break
                continue
            with self._new_frame:
                self._seq += 1
                height, width = image.shape[:2]
                self._size = (int(width), int(height))
                self._frame = Frame(image=image, timestamp=now(), index=self._seq - 1)
                self._new_frame.notify_all()

        # Wake anything blocked in read() so release() cannot deadlock a consumer.
        with self._new_frame:
            self._new_frame.notify_all()

    def _read_frame(self) -> Frame | None:
        """The newest frame not yet returned, waiting briefly for one to arrive.

        Deliberately waits for a frame *newer* than the last one handed out,
        rather than returning whatever is in the slot. The legacy loop re-read
        the slot freely and could process one capture twice, which is harmless
        for a cursor but produces two results with identical content and
        different timestamps. Downstream that is a zero-length movement over a
        non-zero interval, i.e. a velocity of zero fed to a filter whose cutoff
        is velocity-dependent.
        """
        deadline = time.monotonic() + self._read_timeout
        with self._new_frame:
            while True:
                frame = self._frame
                if frame is not None and frame.index > self._last_returned:
                    self._last_returned = frame.index
                    return frame
                if self._stop.is_set() or not self._thread.is_alive():
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._new_frame.wait(remaining)

    def _release(self) -> None:
        """Stop the thread, then the device, then wait for the driver.

        Strictly ordered. Releasing the capture while the thread is still inside
        ``read()`` is a use-after-free in the native layer, so the thread is
        joined first even though that costs up to a frame interval.
        """
        self._stop.set()
        with self._new_frame:
            self._new_frame.notify_all()
        self._thread.join(timeout=_THREAD_JOIN_TIMEOUT_S)
        if self._thread.is_alive():
            # Blocked in a native read that never returned. Releasing the
            # capture anyway risks the native crash described above, so the
            # device is left to the daemon thread and the process exit.
            _log.warning(
                "capture thread did not stop within %.1fs; the camera may stay "
                "held until this process exits",
                _THREAD_JOIN_TIMEOUT_S,
            )
            return
        self._capture.release()
        with self._lock:
            self._frame = None
        # Empirical, from the legacy server. The next open() fails without it.
        time.sleep(_DRIVER_SETTLE_S)
