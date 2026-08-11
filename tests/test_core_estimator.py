"""Phase 2: the crop smoothing, and the A3 isolation the brief has carried since Phase 0.

The headline test here is that two estimators in one process do not interfere.
That is the proof the module-level `_smoothed_bbox` global is really gone, and it
is the one the standing brief has listed since Phase 0.

None of this needs MediaPipe, a model file or a camera: the landmarker and the
gaze model are injected, and the crop smoothing is exercised directly. Every
number is compared against the legacy arithmetic transcribed from
`gaze_pipeline.py:70-99`.
"""

from __future__ import annotations

import numpy as np
import pytest

from focusedgaze.config import GazeConfig, LandmarkConfig
from focusedgaze.core.estimator import GazeEstimator
from focusedgaze.core.landmarks import SmoothedBox, smoothed_square_box
from focusedgaze.exceptions import CalibrationError
from focusedgaze.types import FaceObservation, GazeStatus


class Landmark:
    """A MediaPipe normalised landmark: x, y in [0, 1]."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z


def _face(cx: float, cy: float, size: float = 0.2, count: int = 478) -> list[Landmark]:
    """A square blob of landmarks centred on (cx, cy) in normalised coords."""
    half = size / 2
    marks = [Landmark(cx, cy) for _ in range(count)]
    marks[0] = Landmark(cx - half, cy - half)
    marks[1] = Landmark(cx + half, cy + half)
    return marks


# ---------------------------------------------------------------------------
# The smoothing arithmetic, against the legacy transcription.
# ---------------------------------------------------------------------------


def test_the_first_observation_is_taken_verbatim() -> None:
    """`gaze_pipeline.py:82-83`. Seeding from zero would drag the first crops
    toward the top-left corner for several frames."""
    box = SmoothedBox(0.3)
    assert not box.started
    assert box.update(100.0, 200.0, 50.0) == (100.0, 200.0, 50.0)
    assert box.started


def test_smoothing_is_the_legacy_exponential_average() -> None:
    """new = 0.3 * observed + 0.7 * previous, per axis and on the half-size."""
    box = SmoothedBox(0.3)
    box.update(100.0, 200.0, 50.0)
    cx, cy, half = box.update(200.0, 100.0, 100.0)
    assert cx == pytest.approx(0.3 * 200 + 0.7 * 100)
    assert cy == pytest.approx(0.3 * 100 + 0.7 * 200)
    assert half == pytest.approx(0.3 * 100 + 0.7 * 50)


def test_reset_makes_the_next_observation_verbatim_again() -> None:
    box = SmoothedBox(0.3)
    box.update(100.0, 200.0, 50.0)
    box.reset()
    assert not box.started
    assert box.update(10.0, 20.0, 5.0) == (10.0, 20.0, 5.0)


def test_the_default_smoothing_weight_is_the_shipping_one() -> None:
    """0.3, from `gaze_pipeline.py:18`. Changing it retunes the whole system."""
    assert LandmarkConfig().bbox_smoothing == 0.3
    assert LandmarkConfig().crop_padding == 0.3


def test_the_box_is_square_and_padded_the_legacy_way() -> None:
    """half = max(width, height) / 2 * (1 + padding), square in the FRAME."""
    box = SmoothedBox(1.0)  # weight 1 = no smoothing, so the geometry is visible
    # A face 0.2 wide and 0.1 tall in a 1000x1000 frame: 200 x 100 px.
    marks = [Landmark(0.5, 0.5) for _ in range(478)]
    marks[0] = Landmark(0.4, 0.45)
    marks[1] = Landmark(0.6, 0.55)
    bbox = smoothed_square_box((1000, 1000, 3), marks, box, 0.3)
    assert bbox is not None
    left, top, right, bottom = bbox
    # max(200, 100) / 2 * 1.3 = 130 half-size, centred on (500, 500).
    assert (left, right) == (370, 630)
    assert (top, bottom) == (370, 630)
    assert (right - left) == (bottom - top), "the crop must be square"


def test_the_box_is_clamped_to_the_frame_and_truncated_not_rounded() -> None:
    """int() truncates toward zero and is applied BEFORE the clamp.

    Rounding instead would move boxes by a pixel, which changes the crop and
    therefore the model's input. Audit 48 measured zero of sixty boxes differing
    across two mediapipe builds; that only stays true if this is exact.
    """
    box = SmoothedBox(1.0)
    marks = _face(0.02, 0.02, size=0.2)          # mostly off the top-left
    bbox = smoothed_square_box((480, 640, 3), marks, box, 0.3)
    assert bbox is not None
    left, top, right, bottom = bbox
    assert left == 0 and top == 0, "the box must clamp to the frame edge"
    assert right <= 640 and bottom <= 480


def test_a_box_entirely_off_frame_returns_none() -> None:
    """Same user-visible state as no face, and it must not raise."""
    box = SmoothedBox(1.0)
    marks = _face(-2.0, -2.0, size=0.1)
    assert smoothed_square_box((480, 640, 3), marks, box, 0.3) is None


# ---------------------------------------------------------------------------
# A3: the isolation test the brief has carried since Phase 0.
# ---------------------------------------------------------------------------


def test_two_smoothed_boxes_do_not_share_state() -> None:
    """The narrow version: the state object itself is per-instance."""
    a, b = SmoothedBox(0.3), SmoothedBox(0.3)
    a.update(100.0, 100.0, 50.0)
    a.update(900.0, 900.0, 200.0)
    assert b.update(10.0, 20.0, 5.0) == (10.0, 20.0, 5.0), (
        "the second box inherited the first's history"
    )


class StubLandmarker:
    """A landmarker whose detections are scripted. Owns its own smoothing."""

    def __init__(self, faces: list[list[Landmark] | None]) -> None:
        self._faces = faces
        self._i = 0
        self._box = SmoothedBox(0.3)
        self.resets = 0

    def detect(self, frame, timestamp=None):
        marks = self._faces[min(self._i, len(self._faces) - 1)]
        self._i += 1
        if marks is None:
            return None
        bbox = smoothed_square_box(frame.shape, marks, self._box, 0.3)
        if bbox is None:
            return None
        return FaceObservation(
            landmarks=marks, bbox=bbox, transform_matrix=None,
            timestamp=0.0 if timestamp is None else float(timestamp),
        )

    @staticmethod
    def crop(frame, bbox):
        left, top, right, bottom = bbox
        view = frame[top:bottom, left:right]
        return None if view.size == 0 else view

    def reset(self) -> None:
        self.resets += 1
        self._box.reset()

    def close(self) -> None:
        return


class StubModel:
    """Returns a fixed angle, and records every crop it was handed."""

    def __init__(self, pitch: float = -0.2, yaw: float = 0.1) -> None:
        self.pitch, self.yaw = pitch, yaw
        self.crops: list[tuple[int, int]] = []

    @property
    def provider(self) -> str:
        return "StubExecutionProvider"

    def predict(self, crop):
        self.crops.append((crop.shape[1], crop.shape[0]))
        return self.pitch, self.yaw


def _estimator(faces, model=None, profile=None):
    return GazeEstimator(
        profile=profile,
        config=GazeConfig(),
        landmarker=StubLandmarker(faces),
        model=model or StubModel(),
    )


# ---------------------------------------------------------------------------
# Distance compensation, at the estimator.
#
# This changes the runtime mapping, so the thing worth pinning hardest is that
# it does nothing at all unless it is asked for.
# ---------------------------------------------------------------------------


def _distance_profile(distance_cm=None):
    """A profile mapping any angle to a fixed off-centre point, so the only
    thing that can move the output is the rescaling."""
    from focusedgaze.calibration import CalibrationProfile

    return CalibrationProfile(
        name="d", degree=1,
        powers=np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int64),
        coef_x=np.zeros(3), coef_y=np.zeros(3),
        intercept_x=0.25, intercept_y=0.5,
        distance_cm=distance_cm,
    )


def _compensating_config(on: bool) -> GazeConfig:
    from dataclasses import replace

    base = GazeConfig()
    return replace(base, positioning=replace(base.positioning, compensate_distance=on))


def test_compensation_is_off_by_default() -> None:
    assert GazeConfig().positioning.compensate_distance is False


def test_a_default_estimator_does_not_rescale(monkeypatch) -> None:
    """The guarantee that makes this safe to add: existing behaviour is
    untouched unless the flag is set."""
    est = GazeEstimator(
        profile=_distance_profile(47.0), config=GazeConfig(),
        landmarker=StubLandmarker([_face(0.5, 0.5)]), model=StubModel(),
    )
    result = est.process(np.zeros((480, 640, 3), dtype=np.uint8), timestamp=0.0)
    assert result.ok
    assert result.x == pytest.approx(0.25), "the prediction was altered with the flag off"


def test_a_profile_with_no_distance_is_never_rescaled() -> None:
    """Even with compensation on. Nothing already on disk changes behaviour."""
    est = GazeEstimator(
        profile=_distance_profile(None), config=_compensating_config(True),
        landmarker=StubLandmarker([_face(0.5, 0.5)]), model=StubModel(),
    )
    result = est.process(np.zeros((480, 640, 3), dtype=np.uint8), timestamp=0.0)
    assert result.ok
    assert result.x == pytest.approx(0.25)


def _face_with_eyes(cx=0.5, cy=0.5, ipd=0.10):
    """A face whose iris centres are actually apart, so the positioning gate can
    estimate a distance. `_face` puts every landmark on one point, which gives an
    inter-pupil distance of zero and therefore no distance at all.

    The default separation is chosen to land inside the 45-65 cm zone: distance
    is inversely proportional to the pupil separation in pixels, and a narrower
    face reads as further away and gets rejected as OUT_OF_RANGE before any
    correction is reached.
    """
    from focusedgaze.core.positioning import LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER

    marks = _face(cx, cy)
    marks[LEFT_IRIS_CENTER] = Landmark(cx - ipd / 2, cy)
    marks[RIGHT_IRIS_CENTER] = Landmark(cx + ipd / 2, cy)
    return marks


def test_compensation_stretches_the_prediction_when_enabled() -> None:
    """With a distance recorded and the flag on, a reading taken further away
    than the calibration is pushed back out toward where it belongs."""
    face = _face_with_eyes()
    measured = GazeEstimator(
        profile=_distance_profile(None), config=_compensating_config(True),
        landmarker=StubLandmarker([face]), model=StubModel(),
    ).process(np.zeros((480, 640, 3), dtype=np.uint8), timestamp=0.0)
    assert measured.distance_cm is not None, "the stub face yields no distance"

    est = GazeEstimator(
        profile=_distance_profile(47.0), config=_compensating_config(True),
        landmarker=StubLandmarker([face]), model=StubModel(),
    )
    result = est.process(np.zeros((480, 640, 3), dtype=np.uint8), timestamp=0.0)
    assert result.ok

    expected = 0.5 + (0.25 - 0.5) * (measured.distance_cm / 47.0)
    expected = max(0.0, min(1.0, expected))
    assert result.x == pytest.approx(expected, abs=1e-9)
    assert result.x != pytest.approx(0.25), "the correction did nothing"


# ---------------------------------------------------------------------------
# Session recentring.
#
# A profile fixes the shape of the mapping, not its position. Across five
# measured runs the whole mapping shifted bodily between sessions -- by -0.25 to
# +0.34 of screen height, twice on an identical profile minutes apart.
# ---------------------------------------------------------------------------


def _centred_estimator(profile=None, config=None):
    return GazeEstimator(
        profile=profile if profile is not None else _distance_profile(),
        config=config or GazeConfig(),
        landmarker=StubLandmarker([_face_with_eyes()]),
        model=StubModel(),
    )


def _read(est):
    return est.process(np.zeros((480, 640, 3), dtype=np.uint8), timestamp=0.0)


def test_a_fresh_estimator_has_no_offset() -> None:
    assert _centred_estimator().offset == (0.0, 0.0)


def test_recentring_moves_the_reading_onto_the_target() -> None:
    """The whole promise: look at the centre, say so, and the centre is where
    the tracker now reports you looking.

    Two estimators rather than one read before and one after: the One Euro
    filter is stateful, so a second reading through the same estimator is
    smoothed toward the first and lands between them.
    """
    plain = _centred_estimator()        # maps every angle to (0.25, 0.5)
    assert _read(plain).x == pytest.approx(0.25)

    est = _centred_estimator()
    est.recentre([(StubModel().pitch, StubModel().yaw)])

    after = _read(est)
    assert after.x == pytest.approx(0.5, abs=1e-9)
    assert after.y == pytest.approx(0.5, abs=1e-9)


def test_recentring_can_target_any_point() -> None:
    est = _centred_estimator()
    est.recentre([(StubModel().pitch, StubModel().yaw)], target=(0.1, 0.9))
    result = _read(est)
    assert (result.x, result.y) == pytest.approx((0.1, 0.9), abs=1e-9)


def test_the_median_is_used_so_a_blink_does_not_matter() -> None:
    """No supervision is expected of the caller, so an outlier must not shift
    the correction."""
    est = _centred_estimator()
    good = (StubModel().pitch, StubModel().yaw)
    est.recentre([good, good, (5.0, -5.0), good, good])
    assert _read(est).x == pytest.approx(0.5, abs=1e-9)


def test_an_offset_survives_tracking_loss() -> None:
    """`reset` forgets per-stream state. The offset describes where the user is
    sitting, and losing it because a face flickered would silently undo the
    correction."""
    est = _centred_estimator()
    est.recentre([(StubModel().pitch, StubModel().yaw)])
    offset = est.offset
    est.reset()
    assert est.offset == offset


def test_clearing_the_offset_restores_the_raw_mapping() -> None:
    est = _centred_estimator()
    est.recentre([(StubModel().pitch, StubModel().yaw)])
    est.clear_offset()
    assert est.offset == (0.0, 0.0)
    assert _read(est).x == pytest.approx(0.25)


def test_recentring_without_a_profile_is_refused() -> None:
    """It corrects where a profile lands. With no profile there is nothing to
    correct, and silently doing nothing would look like it worked."""
    est = GazeEstimator(
        profile=None, config=GazeConfig(),
        landmarker=StubLandmarker([_face_with_eyes()]), model=StubModel(),
    )
    with pytest.raises(CalibrationError, match="profile"):
        est.recentre([(0.1, 0.2)])


def test_recentring_with_no_usable_readings_is_refused() -> None:
    est = _centred_estimator()
    with pytest.raises(CalibrationError, match="no usable readings"):
        est.recentre([])
    with pytest.raises(CalibrationError, match="no usable readings"):
        est.recentre([(float("nan"), 0.1), (0.2, float("inf"))])


def test_a_non_finite_offset_is_refused() -> None:
    """A NaN here would poison every later reading rather than failing."""
    est = _centred_estimator()
    with pytest.raises(CalibrationError, match="finite"):
        est.set_offset(float("nan"), 0.0)


def test_a_recentred_reading_cannot_leave_the_screen() -> None:
    est = _centred_estimator()
    est.set_offset(-0.9, -0.9)
    result = _read(est)
    assert 0.0 <= result.x <= 1.0 and 0.0 <= result.y <= 1.0


def test_two_estimators_in_one_process_do_not_interfere() -> None:
    """A3, the test the standing brief has carried since Phase 0.

    The legacy pipeline kept the crop box in a module-level global, so two
    estimators shared one bounding box and each corrupted the other's crop. The
    only way to start clean was a global reset that every other user in the
    process also felt.

    Driven here to the point where it would actually show: one estimator is fed
    twenty frames of a face in the top-left, moving its smoothed box a long way,
    while the other is fed a single frame of a face in the bottom-right. If any
    state were shared, the second estimator's very first crop would be dragged
    toward the first's box.
    """
    far_left = [_face(0.2, 0.2, size=0.15)] * 20
    bottom_right = [_face(0.8, 0.8, size=0.15)]

    model_a, model_b = StubModel(), StubModel()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    a = _estimator(far_left, model_a)
    b = _estimator(bottom_right, model_b)

    for i in range(20):
        a.process(frame, timestamp=float(i))

    # `b` has seen nothing. Its first frame must be smoothed from scratch, so
    # its crop is exactly the unsmoothed geometry of a bottom-right face.
    b.process(frame, timestamp=0.0)

    reference = _estimator(bottom_right, StubModel())
    reference_model = reference._model
    reference.process(frame, timestamp=0.0)

    assert model_b.crops == reference_model.crops, (
        "the second estimator's first crop was influenced by the first "
        f"estimator's history: {model_b.crops} vs a clean {reference_model.crops}"
    )


def test_a_second_estimator_is_unaffected_by_the_first_being_reset() -> None:
    """The other half of A3: reset must be local too.

    `reset_bbox_smoothing()` was module-level, so one caller clearing its own
    state cleared everybody's.
    """
    faces = [_face(0.3, 0.3, size=0.15)] * 5
    a, b = _estimator(faces), _estimator(faces)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for i in range(5):
        a.process(frame, timestamp=float(i))
        b.process(frame, timestamp=float(i))

    a.reset()
    assert a._landmarker.resets == 1
    assert b._landmarker.resets == 0, "resetting one estimator reset the other"


# ---------------------------------------------------------------------------
# process(), end to end with stubs.
# ---------------------------------------------------------------------------


def test_no_face_is_a_status_not_an_exception() -> None:
    est = _estimator([None])
    result = est.process(np.zeros((480, 640, 3), dtype=np.uint8), timestamp=1.0)
    assert result.status is GazeStatus.NO_FACE
    assert result.ok is False
    assert result.x is None and result.y is None
    assert result.timestamp == 1.0


def test_without_a_profile_the_angles_are_still_reported() -> None:
    """NOT_CALIBRATED is a supported state: the setup flow shows a live face
    before any calibration exists."""
    est = _estimator([_face(0.5, 0.5)], StubModel(pitch=-0.3, yaw=0.2))
    result = est.process(np.zeros((480, 640, 3), dtype=np.uint8), timestamp=1.0)
    assert result.status is GazeStatus.NOT_CALIBRATED
    assert result.pitch == pytest.approx(-0.3)
    assert result.yaw == pytest.approx(0.2)
    assert result.x is None


def test_the_filter_is_reset_on_the_transition_to_lost_not_every_frame() -> None:
    """R-7. The policy is the pipeline's, not the filter's.

    Losing it makes the cursor glide in from a stale point when the face
    returns, which reads as drift rather than as a bug.
    """
    faces = [_face(0.5, 0.5), None, None, None]
    est = _estimator(faces)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(4):
        est.process(frame, timestamp=float(i))
    # One reset for the transition, not one per lost frame.
    assert est._landmarker.resets == 1


def test_a_calibrated_reading_is_filtered_and_ok() -> None:
    pytest.importorskip("sklearn", reason="fitting a profile needs scikit-learn")
    from focusedgaze.calibration import robust_fit_samples

    rng = np.random.default_rng(5)
    pitch = rng.uniform(-0.5, 0.1, 150)
    yaw = rng.uniform(-0.4, 0.4, 150)
    samples = np.column_stack(
        [pitch, yaw, np.clip(0.5 + yaw, 0, 1), np.clip(0.5 - pitch, 0, 1)]
    )
    profile = robust_fit_samples(samples, name="est").profile

    est = _estimator([_face(0.5, 0.5)] * 3, StubModel(pitch=-0.2, yaw=0.1), profile=profile)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = [est.process(frame, timestamp=i / 30) for i in range(3)]

    assert all(r.status is GazeStatus.OK for r in results)
    for r in results:
        assert r.x is not None and 0.0 <= r.x <= 1.0
        assert r.y is not None and 0.0 <= r.y <= 1.0


def test_the_provider_is_reported_through_the_estimator() -> None:
    est = _estimator([_face(0.5, 0.5)])
    assert est.provider == "StubExecutionProvider"


def test_the_context_manager_closes_the_landmarker() -> None:
    closed: list[bool] = []

    class Closing(StubLandmarker):
        def close(self) -> None:
            closed.append(True)

    est = GazeEstimator(landmarker=Closing([_face(0.5, 0.5)]), model=StubModel())
    with est:
        pass
    assert closed == [True]
