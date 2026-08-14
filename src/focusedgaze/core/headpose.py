"""Head orientation, decomposed from MediaPipe's facial transformation matrix.

WHY THIS EXISTS, AND WHAT IT IS NOT YET
=======================================
This is **instrumentation**, not a feature of the mapping. Nothing here feeds
the calibration polynomial.

Five measured accuracy runs on one machine showed the dominant error to be a
per-session *vertical offset*: the whole mapping shifts bodily up or down
between sessions, spanning 0.41 of screen height across the five, and doing it
twice on one identical profile five minutes apart. Head tilt is the obvious
suspect, because a calibration learns ``(pitch, yaw) -> screen`` at whatever head
pose it was collected at, and the same eyeball rotation points somewhere else
once the head has moved.

Obvious is not measured. Competing explanations exist and are not excluded:
camera auto-exposure changing the crop's appearance, distance drifting inside
the 45-65 cm the gate allows, or the eye's position within the head rather than
the head's within the room. Adding head pose to the polynomial means a profile
schema change to the one part of this package protected by golden fixtures and
mutation checks, on the grounds that a wrong polynomial does not crash -- it
returns a smooth, believable surface in the wrong place. That is not a change to
make on a suspicion.

So: record the number first, correlate it with the offset, and let the data say
whether the feature is worth the risk.

THE CONVENTION, STATED BECAUSE IT MATTERS LESS THAN IT LOOKS
=============================================================
The angles are a ZYX (yaw-pitch-roll) decomposition of the matrix's rotation
block. MediaPipe's matrix maps the canonical face mesh into camera space, so
these are the head's orientation relative to the camera, in radians, and their
absolute zero is wherever MediaPipe's canonical face points.

For the question being asked -- does the offset move *with* head pose -- only
consistency matters, not the absolute reference. Two runs parameterised the same
way are comparable whatever the zero is. Do not read an absolute "your head is
tilted 5 degrees down" out of these without checking that claim separately.
"""

from __future__ import annotations

import math

import numpy as np

from ..types import Matrix4x4

__all__ = ["HeadPose", "head_angles"]

#: Below this, the ZYX decomposition is degenerate: pitch near +-90 degrees makes
#: yaw and roll describe the same rotation and their split is arbitrary. A head
#: is never actually here, so reaching it means the matrix is not a head pose.
_GIMBAL_EPS = 1e-6


class HeadPose(tuple[float, float, float]):
    """``(pitch, yaw, roll)`` in radians, with names.

    A tuple subclass rather than a dataclass so it serialises as a plain JSON
    list and compares equal to one, which is what the report format wants.
    """

    __slots__ = ()

    @property
    def pitch(self) -> float:
        """Nose up or down."""
        return self[0]

    @property
    def yaw(self) -> float:
        """Nose left or right."""
        return self[1]

    @property
    def roll(self) -> float:
        """Head tilted toward a shoulder."""
        return self[2]


def head_angles(matrix: Matrix4x4 | None) -> HeadPose | None:
    """Decompose a 4x4 facial transformation matrix into head angles.

    Args:
        matrix: MediaPipe's 4x4, or ``None`` when the landmarker was not asked
            for one.

    Returns:
        ``(pitch, yaw, roll)`` in radians, or ``None`` when there is no matrix.
        ``None`` rather than zeros: "the head was level" and "nobody measured"
        are different, and a zero would quietly become a data point.

    Raises:
        ValueError: The matrix is not 4x4 or 3x3.
    """
    if matrix is None:
        return None

    m = np.asarray(matrix, dtype=np.float64)
    if m.shape not in ((4, 4), (3, 3)):
        raise ValueError(f"a head-pose matrix must be 4x4 or 3x3, got {m.shape}")
    r = m[:3, :3]

    # Standard ZYX decomposition. The sy term is cos(pitch); when it collapses
    # the rotation is degenerate and roll is folded into yaw by convention.
    sy = math.sqrt(float(r[0, 0]) ** 2 + float(r[1, 0]) ** 2)
    if sy < _GIMBAL_EPS:
        pitch = math.atan2(-float(r[2, 0]), sy)
        yaw = math.atan2(-float(r[1, 2]), float(r[1, 1]))
        roll = 0.0
    else:
        pitch = math.atan2(-float(r[2, 0]), sy)
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
    return HeadPose((pitch, yaw, roll))
