"""Head-pose decomposition.

Pure arithmetic on a matrix, so it is fully testable without a camera. What is
*not* tested here is whether these angles predict anything: that is the open
question the numbers are being recorded to answer, and it needs a person and two
measured runs. See the module docstring for why the feature is not being built
before the correlation is known.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from focusedgaze.core.headpose import HeadPose, head_angles


def _rotation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Build a ZYX rotation matrix, i.e. the inverse of what head_angles does."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rz @ ry @ rx
    return matrix


def test_no_matrix_is_not_a_level_head() -> None:
    """`None` and "the head was level" are different, and a zero would quietly
    become a data point in the correlation this exists to serve."""
    assert head_angles(None) is None


def test_the_identity_matrix_is_all_zeros() -> None:
    pose = head_angles(np.eye(4, dtype=np.float32))
    assert pose == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


@pytest.mark.parametrize(
    ("pitch", "yaw", "roll"),
    [
        (0.0, 0.0, 0.0),
        (0.20, 0.0, 0.0),
        (0.0, -0.35, 0.0),
        (0.0, 0.0, 0.15),
        (0.10, 0.25, -0.30),
        (-0.40, 0.10, 0.05),
    ],
)
def test_angles_round_trip_through_a_rotation_matrix(pitch, yaw, roll) -> None:
    """The decomposition must invert the composition, or the recorded numbers
    describe a different rotation from the one the head was in."""
    pose = head_angles(_rotation(pitch, yaw, roll))
    assert pose == pytest.approx((pitch, yaw, roll), abs=1e-6)


def test_the_axes_are_named_and_in_order() -> None:
    pose = head_angles(_rotation(0.1, 0.2, 0.3))
    assert isinstance(pose, HeadPose)
    assert pose.pitch == pytest.approx(0.1, abs=1e-6)
    assert pose.yaw == pytest.approx(0.2, abs=1e-6)
    assert pose.roll == pytest.approx(0.3, abs=1e-6)


def test_a_pose_compares_equal_to_a_plain_tuple() -> None:
    """It is serialised into a JSON list, so it must behave like one."""
    assert head_angles(np.eye(4, dtype=np.float32)) == (0.0, 0.0, 0.0)


def test_a_three_by_three_rotation_is_accepted() -> None:
    assert head_angles(_rotation(0.1, 0.0, 0.0)[:3, :3]) == pytest.approx(
        (0.1, 0.0, 0.0), abs=1e-6
    )


@pytest.mark.parametrize("shape", [(2, 2), (4, 3), (16,)])
def test_a_matrix_of_the_wrong_shape_is_rejected(shape) -> None:
    with pytest.raises(ValueError, match="4x4 or 3x3"):
        head_angles(np.zeros(shape, dtype=np.float32))


def test_the_degenerate_case_does_not_raise() -> None:
    """Pitch at +-90 degrees makes yaw and roll describe the same rotation. A
    head is never there, so this means the matrix is not a head pose -- but it
    must still return numbers rather than blow up a measurement run."""
    pose = head_angles(_rotation(math.pi / 2, 0.3, 0.4))
    assert pose is not None
    assert all(math.isfinite(a) for a in pose)
