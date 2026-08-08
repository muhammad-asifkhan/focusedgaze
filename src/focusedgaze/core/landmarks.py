"""MediaPipe face landmarks, and the smoothed square crop the gaze model eats.

THE GLOBAL THIS REPLACES
------------------------
`gaze_pipeline.py:35` held the crop's smoothing state in a module-level
``_smoothed_bbox``, reset through a module-level ``reset_bbox_smoothing()``. That
is the defect this module exists to remove: **two estimators in one process
shared one bounding box**, so each corrupted the other's crop, and the only way
to start clean was a global reset that every other user in the process also felt.

It is now instance state on :class:`FaceLandmarker`. The arithmetic is unchanged,
which is the point: the smoothing is preserved exactly and only its *ownership*
moves. `tests/test_core_landmarks.py` pins that two instances do not interfere,
which is the A3 test the standing brief has carried since Phase 0.

WHY THE CROP IS SMOOTHED AT ALL
--------------------------------
The box is a min/max over all 478 landmarks, so it jitters frame to frame even on
a still face: one landmark twitching by a pixel moves the whole box. Feeding that
straight to the model would make the crop, and therefore the predicted angle,
jitter with it. The smoothing is an exponential average over centre and half-size
with weight 0.3 on the newest frame.

Audit section 48 measured that this crop is where a landmark difference would
show first, precisely because min/max is sensitive to a single moved outlier in a
way an average is not. Across 60 real frames and two mediapipe builds, **zero of
sixty boxes differed**. Preserving this arithmetic exactly is what keeps that
true.

THE 478-POINT MODEL IS REQUIRED
--------------------------------
Not preferred: required. Distance in `focusedgaze.core.positioning` comes from
the iris landmarks, which only the refined model has. The unrefined 468-point
model loads, tracks a face, and reports distances wrong by a consistent factor,
which is a silent failure rather than an error. `focusedgaze check` verifies the
asset by digest for that reason.

NO I/O BEYOND READING THE MODEL FILE
-------------------------------------
This module never downloads. A missing landmarker raises
:class:`~focusedgaze.exceptions.ModelNotFoundError` naming the command that
fetches it. Assets are a separate, explicit step by design: a silent fetch would
break the no-network guarantee that lets the pure layer run in CI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Final, Self

import cv2
import numpy as np
from numpy.typing import NDArray

from ..config import LandmarkConfig
from ..exceptions import ModelNotFoundError
from ..types import BBox, FaceObservation

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = ["FaceLandmarker", "SmoothedBox", "smoothed_square_box"]

_log = logging.getLogger(__name__)

#: Landmarks the refined model emits. The unrefined one emits 468 and is missing
#: exactly the iris points distance estimation depends on.
REFINED_LANDMARK_COUNT: Final = 478


class SmoothedBox:
    """The crop box's exponential smoothing, as state you can own.

    Was ``_smoothed_bbox`` at module scope. The arithmetic is
    ``new = w * observed + (1 - w) * previous`` applied independently to centre x,
    centre y and half-size, with the first observation taken verbatim.

    Kept as a small class rather than three attributes on the landmarker because
    "reset this, and nothing else" is a real operation: the server resets
    smoothing when the face is lost so the crop does not glide in from wherever
    it was left.
    """

    __slots__ = ("_state", "_weight")

    def __init__(self, weight: float) -> None:
        self._weight = float(weight)
        self._state: list[float] | None = None

    @property
    def started(self) -> bool:
        """Whether any observation has been folded in yet."""
        return self._state is not None

    def reset(self) -> None:
        """Forget history. The next observation is taken verbatim."""
        self._state = None

    def update(self, cx: float, cy: float, half: float) -> tuple[float, float, float]:
        """Fold in one observation and return the smoothed centre and half-size."""
        if self._state is None:
            # First frame is taken as-is, matching `gaze_pipeline.py:82-83`.
            # Seeding with a zero would drag the first several crops toward the
            # top-left corner of the frame.
            self._state = [cx, cy, half]
        else:
            w = self._weight
            self._state[0] = w * cx + (1 - w) * self._state[0]
            self._state[1] = w * cy + (1 - w) * self._state[1]
            self._state[2] = w * half + (1 - w) * self._state[2]
        return self._state[0], self._state[1], self._state[2]


def smoothed_square_box(
    frame_shape: tuple[int, ...],
    landmarks: Sequence[Any],
    box: SmoothedBox,
    padding_ratio: float,
) -> BBox | None:
    """The square, smoothed, clamped crop box for one frame, or ``None``.

    Transcribed from `gaze_pipeline.py:70-99` with the ordering preserved
    exactly. Two details that look incidental and are not:

    * ``int()`` truncates **toward zero**, and it is applied *before* the clamp
      to the frame. Rounding instead would shift boxes by a pixel and change the
      crop, which changes the model's input.
    * The half-size is ``max(width, height) / 2 * (1 + padding)``, so the box is
      square in the *frame*, not in the face. A face wider than it is tall still
      gets a square crop, which is what the model was trained on.

    Returns ``None`` when the clamped box is empty, which happens when the face
    is far enough off-frame that the whole box lies outside it.
    """
    height, width = int(frame_shape[0]), int(frame_shape[1])
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = max(x_max - x_min, y_max - y_min) / 2 * (1 + padding_ratio)

    scx, scy, shalf = box.update(cx, cy, half)

    left = max(int(scx - shalf), 0)
    right = min(int(scx + shalf), width)
    top = max(int(scy - shalf), 0)
    bottom = min(int(scy + shalf), height)

    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


class FaceLandmarker:
    """One MediaPipe landmarker, and the crop smoothing that belongs with it.

    Args:
        model_path: The ``face_landmarker.task`` asset. Defaults to whatever
            :func:`focusedgaze.assets.asset_path` resolves, which is the managed
            cache or ``FOCUSEDGAZE_MODEL_DIR``.
        config: Detection thresholds and crop geometry.

    Raises:
        ModelNotFoundError: The asset is absent. **Never downloaded here.**

    Not thread-safe: MediaPipe's ``detect_for_video`` requires monotonically
    increasing timestamps, so two threads interleaving calls on one instance
    would produce out-of-order timestamps and be rejected. One instance per
    thread, which is now possible because nothing is shared at module scope.
    """

    __slots__ = ("_box", "_config", "_frame_index", "_landmarker")

    def __init__(
        self,
        model_path: str | Path | None = None,
        config: LandmarkConfig | None = None,
    ) -> None:
        self._config = config if config is not None else LandmarkConfig()
        self._box = SmoothedBox(self._config.bbox_smoothing)
        self._frame_index = 0
        self._landmarker = self._build(model_path)

    def _build(self, model_path: str | Path | None) -> Any:
        if model_path is None:
            from ..assets import FACE_LANDMARKER, asset_path

            resolved = asset_path(FACE_LANDMARKER)
        else:
            resolved = Path(model_path)

        if not resolved.is_file():
            raise ModelNotFoundError(
                f"the face landmarker is not at {resolved}.\n"
                "It is not downloaded automatically from here: assets are an "
                "explicit step, so the pipeline never reaches the network on its "
                "own. Fetch it with:\n"
                "    focusedgaze download-models\n"
                "or set FOCUSEDGAZE_MODEL_DIR to a directory that already has it."
            )

        # Deferred: importing mediapipe costs seconds and pulls in native
        # libraries, and nothing above this line needs it.
        from mediapipe.tasks import python as mp_python  # type: ignore[import-untyped]
        from mediapipe.tasks.python import vision as mp_vision  # type: ignore[import-untyped]

        cfg = self._config
        return mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(resolved)),
                # VIDEO, not IMAGE: the tracker carries state between frames and
                # is both faster and steadier for a stream. IMAGE mode would
                # re-detect from scratch every frame and jitter far more.
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=cfg.max_faces,
                min_face_detection_confidence=cfg.min_detection_confidence,
                min_tracking_confidence=cfg.min_tracking_confidence,
                output_facial_transformation_matrixes=cfg.output_transform_matrix,
            )
        )

    def reset(self) -> None:
        """Forget crop smoothing and the frame counter.

        Called when tracking is lost, so the crop does not glide in from where
        the face used to be, and so timestamps restart cleanly.
        """
        self._box.reset()
        self._frame_index = 0

    def detect(self, frame: NDArray[np.uint8], timestamp: float | None = None) -> FaceObservation | None:
        """Find the face in one BGR frame and return it with its crop box.

        Args:
            frame: ``(height, width, 3)`` uint8 in **BGR** order, as OpenCV
                delivers. RGB will track, badly, without raising.
            timestamp: Capture time in seconds, carried onto the observation. The
                MediaPipe video timestamp is derived from an internal frame
                counter, not from this, for the reason below.

        Returns:
            The observation, or ``None`` when there is no usable face.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Deferred for the same reason as above.
        import mediapipe as mp  # type: ignore[import-untyped]

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # `frame_index * (1000 / 30)`, from `gaze_pipeline.py:117`, preserved
        # exactly. It is a COUNTER scaled by an assumed 30 fps, not the real
        # clock, and that is deliberate rather than an oversight: MediaPipe's
        # VIDEO mode requires strictly increasing integer milliseconds, and two
        # frames arriving inside the same millisecond of wall-clock time would
        # produce a duplicate and be rejected. Deriving from a counter cannot
        # collide. The cost is that the tracker's notion of elapsed time is
        # wrong when the real rate is not 30 fps; the legacy system ran at
        # 35-44 Hz and this was never a problem in practice.
        timestamp_ms = int(self._frame_index * (1000 / self._config.video_timestamp_fps))
        self._frame_index += 1

        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        matrix = None
        matrices = getattr(result, "facial_transformation_matrixes", None)
        if matrices:
            matrix = matrices[0]

        bbox = smoothed_square_box(
            frame.shape, landmarks, self._box, self._config.crop_padding
        )
        if bbox is None:
            return None

        return FaceObservation(
            landmarks=landmarks,
            bbox=bbox,
            transform_matrix=matrix,
            timestamp=0.0 if timestamp is None else float(timestamp),
        )

    @staticmethod
    def crop(frame: NDArray[np.uint8], bbox: BBox) -> NDArray[np.uint8] | None:
        """The image inside a box, or ``None`` when it is empty.

        A separate step because the legacy code cropped twice from the same box
        (`gaze_pipeline.py:98` then `:151`) and the second one is the crop the
        model actually sees. Doing it once, explicitly, removes the question of
        whether the two could ever disagree.
        """
        left, top, right, bottom = bbox
        view: NDArray[np.uint8] = frame[top:bottom, left:right]
        if view.size == 0:
            return None
        return view

    def close(self) -> None:
        """Release the MediaPipe graph and its native resources."""
        closer = getattr(self._landmarker, "close", None)
        if closer is not None:
            closer()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
