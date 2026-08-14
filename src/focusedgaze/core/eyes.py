"""Per-eye crops, for backends that want the eyes rather than the whole face.

L2CS-Net takes one square crop of the face. Intel's ``gaze-estimation-adas-0002``
takes two 60x60 eye patches plus head pose. This module produces the second kind,
from the landmarks the pipeline already has.

WHAT IS EASY TO GET WRONG HERE
==============================
Nothing in this file can fail loudly. A crop that is slightly too tight, centred
on the wrong point, or handed over mirrored still produces a gaze vector -- a
smooth, plausible, wrong one. That is the same failure mode
:mod:`focusedgaze.core.model` warns about for its output tensor names, and the
same reason it is pinned by fixtures rather than trusted.

So the geometry is stated explicitly rather than tuned by eye:

* **Centred on the iris**, not on the eye-corner midpoint. The iris is what the
  model is reading, and on a face turned away from the camera the two differ by
  a noticeable fraction of the crop.
* **Square, sized from the inter-pupil distance**, so the crop covers the same
  proportion of the face whether the user is near or far. A fixed pixel size
  would zoom in as they lean forward.
* **Clamped to the frame**, and reported as ``None`` when the eye is partly
  outside it. A padded crop would present invented pixels as eyelid.
"""

from __future__ import annotations

import math
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from ..types import Landmarks

__all__ = ["EYE_CROP_SCALE", "LEFT_IRIS", "RIGHT_IRIS", "eye_crops"]

#: MediaPipe refined-landmark iris centres. Present only in the 478-point model;
#: the base 468-point one has no iris and would silently index an eyelid.
LEFT_IRIS: Final = 468
RIGHT_IRIS: Final = 473

#: Crop side as a fraction of the inter-pupil distance. 0.6 puts the iris in the
#: middle of a patch that reaches the eye corners without swallowing the brow,
#: which is what the Open Model Zoo demo's crop looks like.
EYE_CROP_SCALE: Final = 0.6


def eye_crops(
    frame: NDArray[np.uint8],
    landmarks: Landmarks,
    *,
    size: int = 60,
    scale: float = EYE_CROP_SCALE,
    roll: float = 0.0,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]] | None:
    """Square patches around each iris, roll-corrected and resized to ``size``.

    Args:
        frame: Full BGR frame, ``(H, W, 3)`` uint8.
        landmarks: The refined 478-point set. Fewer than 478 returns ``None``:
            without iris points there is nothing to centre on.
        size: Output side in pixels. 60 for ``gaze-estimation-adas-0002``.
        scale: Crop side as a fraction of the inter-pupil distance.
        roll: Head roll in **radians**. The patch is rotated to undo it, so the
            model always sees an upright eye.

    Returns:
        ``(left, right)`` BGR patches, or ``None`` when either eye is not fully
        inside the frame.

    **Left and right are the subject's own**, matching the landmark names, not
    the viewer's. Swapping them is not detectable downstream: the model returns
    a gaze vector either way, and it is simply wrong.

    WHY THE ROTATION IS NOT OPTIONAL
    --------------------------------
    The Open Model Zoo reference demo rotates each eye image by the head roll
    before inference (``rotateImageAroundCenter(leftEyeImage, ..., roll)``, using
    ``getRotationMatrix2D`` and ``warpAffine`` with ``BORDER_REPLICATE``). The
    model expects an upright eye; tilt the head and an axis-aligned crop presents
    a tilted one, and it has no way to know.

    This was left out of the first implementation and measured: a profile
    calibrated at roll +0.203 rad, then used at −0.016 rad — a 12.6° tilt — saw
    its **horizontal gain collapse from 0.98 to 0.48** while vertical gain stayed
    at the value the distance change alone predicted. Average error went from
    1.43 cm to 7.90 cm. The failure is silent: every reading remains plausible.
    """
    if landmarks is None or len(landmarks) <= RIGHT_IRIS:
        return None

    height, width = frame.shape[:2]
    lx, ly = landmarks[LEFT_IRIS].x * width, landmarks[LEFT_IRIS].y * height
    rx, ry = landmarks[RIGHT_IRIS].x * width, landmarks[RIGHT_IRIS].y * height

    ipd = float(np.hypot(lx - rx, ly - ry))
    if ipd <= 1.0:
        # Degenerate: both irises on one point. Happens with the 468-point model
        # or a failed detection, and would make the crop a single pixel.
        return None

    half = ipd * scale / 2.0
    patches = []
    for cx, cy in ((lx, ly), (rx, ry)):
        x0, y0 = round(cx - half), round(cy - half)
        x1, y1 = round(cx + half), round(cy + half)
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height or x1 <= x0 or y1 <= y0:
            # Partly outside. Padding would hand the model invented pixels and
            # call them eyelid.
            return None
        patch = frame[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        patches.append(
            np.ascontiguousarray(
                cv2.resize(_upright(patch, roll), (size, size),
                           interpolation=cv2.INTER_AREA),
                dtype=np.uint8,
            )
        )
    return patches[0], patches[1]


def _upright(patch: NDArray[np.uint8], roll: float) -> NDArray[np.uint8]:
    """Rotate a patch about its centre to undo head roll.

    ``BORDER_REPLICATE`` rather than a zero fill, matching the reference demo:
    the corners a rotation exposes are outside the original crop, and filling
    them with black would put a hard edge next to the eyelid that the model has
    never seen in training.

    The sign is the reference's: it rotates by ``+roll``. This package's roll
    comes from a different source (MediaPipe's transformation matrix rather than
    Open Model Zoo's head-pose net), so if a measurement shows this making
    matters worse under head tilt, the sign is the first thing to flip.
    """
    if not roll:
        return patch
    height, width = patch.shape[:2]
    centre = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, math.degrees(roll), 1.0)
    rotated: NDArray[np.uint8] = cv2.warpAffine(
        patch, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.uint8)
    return rotated
