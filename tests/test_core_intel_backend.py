"""The Apache-2.0 gaze backend, and the eye crops that feed it.

Everything here is a convention that produces a plausible wrong answer when
mistaken -- crop geometry, channel order, angle order, angle units, vector
handedness -- which is the same class of defect as the transposed output tensor
names documented in `core/model.py`. None of it raises when wrong. So it is
pinned rather than trusted.

The OpenVINO runtime is faked throughout: what is under test is the conversions
either side of the graph, not the graph.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from focusedgaze.core.eyes import LEFT_IRIS, RIGHT_IRIS, eye_crops
from focusedgaze.core.intel_model import (
    IntelGazeModel,
    _angles_from_vector,
    _as_nchw,
)


class Landmark:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x, self.y = x, y


def _landmarks(left=(0.4, 0.5), right=(0.6, 0.5), count=478):
    marks = [Landmark(0.5, 0.5) for _ in range(count)]
    if count > RIGHT_IRIS:
        marks[LEFT_IRIS] = Landmark(*left)
        marks[RIGHT_IRIS] = Landmark(*right)
    return marks


def _frame(w=640, h=480):
    """A gradient, so a crop taken from the wrong place is distinguishable."""
    x = np.linspace(0, 255, w, dtype=np.uint8)
    return np.repeat(np.tile(x, (h, 1))[:, :, None], 3, axis=2).copy()


# ---------------------------------------------------------------------------
# Eye crops.
# ---------------------------------------------------------------------------


def test_crops_come_back_at_the_requested_size() -> None:
    left, right = eye_crops(_frame(), _landmarks(), size=60)
    assert left.shape == (60, 60, 3) and right.shape == (60, 60, 3)
    assert left.dtype == np.uint8


def test_the_base_468_point_model_is_refused() -> None:
    """It has no iris landmarks, so indexing 468 would silently return an
    eyelid and the crop would be centred on the wrong thing."""
    assert eye_crops(_frame(), _landmarks(count=468)) is None


def test_both_irises_on_one_point_is_refused() -> None:
    """A failed detection. The crop side is derived from the inter-pupil
    distance, so this would ask for a one-pixel patch."""
    assert eye_crops(_frame(), _landmarks(left=(0.5, 0.5), right=(0.5, 0.5))) is None


def test_an_eye_outside_the_frame_is_refused_rather_than_padded() -> None:
    """Padding would hand the model invented pixels and call them eyelid."""
    assert eye_crops(_frame(), _landmarks(left=(0.01, 0.5), right=(0.2, 0.5))) is None


def test_the_left_crop_is_the_subjects_left_landmark() -> None:
    """Swapping the two is undetectable downstream: the model returns a gaze
    vector either way, and it is simply wrong."""
    frame = _frame()
    left, right = eye_crops(frame, _landmarks(left=(0.25, 0.5), right=(0.75, 0.5)))
    # The frame is a left-to-right gradient, so the patch taken from x=0.25 must
    # be darker than the one from x=0.75.
    assert left.mean() < right.mean()


def test_the_crop_scales_with_the_inter_pupil_distance() -> None:
    """A fixed pixel size would zoom in as the user leans forward, changing what
    the model sees for no reason other than distance."""
    frame = _frame()
    near = eye_crops(frame, _landmarks(left=(0.3, 0.5), right=(0.7, 0.5)))
    far = eye_crops(frame, _landmarks(left=(0.45, 0.5), right=(0.55, 0.5)))
    # Both resize to 60x60, so compare the spread of the source region: a wider
    # face means a wider crop and therefore more of the gradient inside it.
    assert near[0].std() > far[0].std()


# ---------------------------------------------------------------------------
# Tensor layout.
# ---------------------------------------------------------------------------


def test_the_patch_is_reshaped_to_nchw() -> None:
    out = _as_nchw(np.zeros((60, 60, 3), dtype=np.uint8))
    assert out.shape == (1, 3, 60, 60)
    assert out.dtype == np.float32


def test_the_patch_is_not_normalised() -> None:
    """This model takes raw 0-255, unlike L2CS which expects ImageNet mean/std.
    Dividing here would darken every input and bias every reading."""
    patch = np.full((60, 60, 3), 200, dtype=np.uint8)
    assert _as_nchw(patch).max() == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Vector to angles.
# ---------------------------------------------------------------------------


def test_looking_straight_at_the_camera_is_zero_zero() -> None:
    pitch, yaw = _angles_from_vector(np.array([0.0, 0.0, -1.0], dtype=np.float32))
    assert pitch == pytest.approx(0.0, abs=1e-9)
    assert yaw == pytest.approx(0.0, abs=1e-9)


def test_looking_up_is_a_positive_pitch() -> None:
    pitch, _ = _angles_from_vector(np.array([0.0, 0.5, -1.0], dtype=np.float32))
    assert pitch > 0


def test_looking_down_is_a_negative_pitch() -> None:
    pitch, _ = _angles_from_vector(np.array([0.0, -0.5, -1.0], dtype=np.float32))
    assert pitch < 0


def test_the_yaw_sign_matches_the_l2cs_handedness() -> None:
    """Getting this backwards transposes the screen horizontally, which reads as
    a bad calibration rather than as a bug."""
    _, yaw = _angles_from_vector(np.array([-0.5, 0.0, -1.0], dtype=np.float32))
    assert yaw > 0, "a gaze toward -x must give a positive yaw"


def test_a_zero_vector_does_not_divide_by_zero() -> None:
    assert _angles_from_vector(np.zeros(3, dtype=np.float32)) == (0.0, 0.0)


def test_the_vector_need_not_be_unit_length() -> None:
    """The model's own documentation says the output is not normalised."""
    short = _angles_from_vector(np.array([0.1, 0.2, -0.3], dtype=np.float32))
    long = _angles_from_vector(np.array([1.0, 2.0, -3.0], dtype=np.float32))
    assert short == pytest.approx(long)


# ---------------------------------------------------------------------------
# The head-pose contract, through a fake runtime.
# ---------------------------------------------------------------------------


class _Recording:
    """Captures what the graph was actually fed."""

    def __init__(self) -> None:
        self.feed = None

    def read_model(self, path):
        return "model"

    def compile_model(self, model, device):
        return self

    def __call__(self, feed):
        self.feed = feed
        return {"gaze_vector": np.array([[0.0, 0.0, -1.0]], dtype=np.float32)}


def _model(tmp_path):
    xml = tmp_path / "gaze-estimation-adas-0002.xml"
    xml.write_text("<net/>")
    core = _Recording()
    return IntelGazeModel(model_path=xml, core_factory=lambda: core), core


def test_head_pose_is_sent_as_yaw_pitch_roll_in_degrees(tmp_path) -> None:
    """This package carries (pitch, yaw, roll) in radians; the model declares
    [yaw, pitch, roll] in degrees. Both the order and the unit are converted,
    and neither mistake would raise."""
    model, core = _model(tmp_path)
    eye = np.zeros((60, 60, 3), dtype=np.uint8)

    model.predict_eyes(eye, eye, (0.1, 0.2, 0.3))     # pitch, yaw, roll (rad)

    sent = core.feed["head_pose_angles"].ravel()
    assert sent[0] == pytest.approx(math.degrees(0.2)), "first slot must be YAW"
    assert sent[1] == pytest.approx(math.degrees(0.1)), "second slot must be PITCH"
    assert sent[2] == pytest.approx(math.degrees(0.3)), "third slot must be ROLL"
    assert abs(sent[0]) > 1.0, "degrees, not radians"


def test_both_eyes_reach_the_graph_in_their_own_slots(tmp_path) -> None:
    model, core = _model(tmp_path)
    left = np.full((60, 60, 3), 10, dtype=np.uint8)
    right = np.full((60, 60, 3), 200, dtype=np.uint8)

    model.predict_eyes(left, right, (0.0, 0.0, 0.0))

    assert core.feed["left_eye_image"].max() == pytest.approx(10.0)
    assert core.feed["right_eye_image"].max() == pytest.approx(200.0)


def test_the_backend_announces_that_it_wants_eyes(tmp_path) -> None:
    """The estimator reads this to decide what to hand over. Without it a face
    crop would be fed to something expecting an eye."""
    model, _ = _model(tmp_path)
    assert model.wants_eyes is True
    assert "OpenVINO" in model.provider


def test_a_missing_model_names_the_command_that_fetches_it(tmp_path) -> None:
    from focusedgaze.exceptions import ModelNotFoundError

    with pytest.raises(ModelNotFoundError, match="download-models"):
        IntelGazeModel(model_path=tmp_path / "absent.xml")
