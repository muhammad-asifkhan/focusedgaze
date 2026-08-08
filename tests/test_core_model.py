"""Phase 2: the gaze model's decode, provider selection, and refusal to fetch.

Three things are pinned here, and each is a defect that does not raise:

* the first output tensor is named `pitch_bins` and holds **yaw**;
* the angle is a softmax-weighted **expectation**, not an argmax;
* a missing model is an error, never a download.

None of these needs the real 91 MB graph: the session is injected, so the decode
and the wiring are testable in CI with no model file and no network.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from focusedgaze.config import ModelConfig
from focusedgaze.core.model import GazeModel, decode_angles, select_providers
from focusedgaze.exceptions import ModelNotFoundError, ProviderError


def _logits(peak_bin: int, bins: int = 90, sharpness: float = 40.0) -> np.ndarray:
    """A sharply unimodal distribution centred on `peak_bin`."""
    x = np.arange(bins, dtype=np.float32)
    return (-((x - peak_bin) ** 2) / sharpness).reshape(1, bins).astype(np.float32)


class FakeSession:
    """Stands in for an ONNX InferenceSession. Records what it was asked."""

    def __init__(self, first: np.ndarray, second: np.ndarray, provider: str = "CPUExecutionProvider") -> None:
        self._outputs = (first, second)
        self._provider = provider
        self.inputs_seen: list[np.ndarray] = []

    def get_providers(self) -> list[str]:
        return [self._provider]

    def get_inputs(self):
        class _In:
            name = "input"

        return [_In()]

    def run(self, _outputs, feed: dict[str, np.ndarray]):
        self.inputs_seen.append(next(iter(feed.values())))
        return list(self._outputs)


def _model(session: FakeSession, tmp_path: pathlib.Path, config: ModelConfig | None = None) -> GazeModel:
    path = tmp_path / "l2cs_gaze360.onnx"
    path.write_bytes(b"not a real graph; the session is injected")
    return GazeModel(path, config=config, session_factory=lambda *a: session)


# ---------------------------------------------------------------------------
# Defect 1: the tensor names lie.
# ---------------------------------------------------------------------------


def test_the_first_output_tensor_is_yaw_despite_its_name(tmp_path) -> None:
    """The graph names output 0 `pitch_bins` and it contains YAW.

    L2CS-Net's forward() returns (pre_yaw_gaze, pre_pitch_gaze). The export
    labelled position 0 `pitch_bins` anyway, and the labels are strings attached
    at export time that do not change what the tensors hold.

    Getting this backwards does not raise. It transposes every reading, so the
    cursor moves sideways when you look up, which reads as a bad calibration
    rather than as a decode bug. This test is what makes that impossible to do
    accidentally: the two peaks are at different bins, so a swap is unambiguous.
    """
    # Output 0 peaks at bin 60, output 1 at bin 20.
    session = FakeSession(_logits(60), _logits(20))
    model = _model(session, tmp_path)

    pitch, yaw = model.predict(np.zeros((64, 64, 3), dtype=np.uint8))

    # bin 60 -> 60*4 - 180 = +60 deg, bin 20 -> 20*4 - 180 = -100 deg.
    assert math.degrees(yaw) == pytest.approx(60.0, abs=0.5), (
        "the FIRST tensor is yaw; if this reads -100 the unpack was swapped"
    )
    assert math.degrees(pitch) == pytest.approx(-100.0, abs=0.5)


def test_decode_angles_names_both_ends_explicitly() -> None:
    """The helper takes yaw first and returns pitch first, on purpose.

    The asymmetry is confined to one signature that names both ends, rather than
    spread across call sites where it would be invisible.
    """
    pitch, yaw = decode_angles(_logits(60), _logits(20), ModelConfig())
    assert math.degrees(yaw) == pytest.approx(60.0, abs=0.5)
    assert math.degrees(pitch) == pytest.approx(-100.0, abs=0.5)


# ---------------------------------------------------------------------------
# Defect 2: expectation, not argmax.
# ---------------------------------------------------------------------------


def test_the_decode_is_an_expectation_not_an_argmax() -> None:
    """A bimodal distribution is where the two disagree, and it is the case
    that matters.

    Two equal peaks at bins 20 and 60: the expectation lands between them, at
    bin 40, i.e. -20 degrees. An argmax returns whichever peak wins the tie,
    -100 or +60. Both are smooth, plausible and wrong.

    Audit section 7 described this as an argmax once and was corrected. This
    test is what stops that description becoming true.
    """
    bins = 90
    x = np.arange(bins, dtype=np.float32)
    bimodal = np.maximum(-((x - 20) ** 2) / 20.0, -((x - 60) ** 2) / 20.0)
    logits = bimodal.reshape(1, bins).astype(np.float32)

    pitch, _ = decode_angles(logits, logits, ModelConfig())
    degrees = math.degrees(pitch)

    assert degrees == pytest.approx(-20.0, abs=1.0), (
        f"expected the weighted mean of bins 20 and 60 (-20 deg), got {degrees:.1f}. "
        "An argmax would give -100 or +60."
    )
    assert not (-101 < degrees < -99), "this is the argmax answer, not the expectation"
    assert not (59 < degrees < 61), "this is the argmax answer, not the expectation"


def test_the_decode_moves_smoothly_rather_than_jumping_a_bin() -> None:
    """Smoothness is why the Tier 2 tolerance is choosable at all.

    Audit 48 measured a 1e-06 rad difference between ONNX providers precisely
    because a small perturbation moves a weighted mean slightly. An argmax would
    jump a full 4-degree bin on a near-tie and make that measurement meaningless.
    """
    config = ModelConfig()
    base = _logits(45)
    nudged = base + np.random.default_rng(3).normal(0, 1e-4, base.shape).astype(np.float32)
    a, _ = decode_angles(base, base, config)
    b, _ = decode_angles(nudged, nudged, config)
    assert abs(math.degrees(a - b)) < 0.1, "a tiny logit change moved the angle a long way"


@pytest.mark.parametrize("peak,expected_deg", [(30, -60.0), (45, 0.0), (60, 60.0)])
def test_a_sharp_interior_peak_decodes_to_that_bin_centre(peak: int, expected_deg: float) -> None:
    """The control: on a symmetric interior peak the expectation is the peak.

    Interior on purpose. A peak at bin 0 or 89 is truncated by the end of the
    array, so the distribution is one-sided and the mean is pulled inward. That
    is correct, and it is the subject of the next test.
    """
    logits = _logits(peak, sharpness=4.0)
    pitch, _ = decode_angles(logits, logits, ModelConfig())
    assert math.degrees(pitch) == pytest.approx(expected_deg, abs=0.5)


@pytest.mark.parametrize("peak,limit_deg,inward", [(0, -180.0, 1), (89, 176.0, -1)])
def test_a_peak_at_the_edge_is_pulled_inward(peak: int, limit_deg: float, inward: int) -> None:
    """An expectation cannot reach the extreme bin; an argmax lands exactly on it.

    A Gaussian centred on bin 0 has no mass to its left, so the weighted mean
    sits strictly inside the range. This is a third way the two decodes differ,
    found while writing the control above: the first version of that test
    asserted -180.0 for a peak at bin 0 and failed, because it was asserting
    argmax behaviour by accident.
    """
    logits = _logits(peak, sharpness=4.0)
    pitch, _ = decode_angles(logits, logits, ModelConfig())
    degrees = math.degrees(pitch)

    assert degrees != pytest.approx(limit_deg, abs=0.1), (
        "the decode landed exactly on the extreme bin, which is what an argmax "
        "would do and what an expectation cannot"
    )
    assert (degrees - limit_deg) * inward > 0, "pulled the wrong way"
    assert abs(degrees - limit_deg) < 10.0, "pulled implausibly far for a sharp peak"


def test_extreme_logits_do_not_produce_nan() -> None:
    """The max-subtraction in the softmax is load-bearing, not decorative.

    Without it `exp` overflows to inf, inf/inf is nan, and a nan coordinate
    propagates all the way to the wire without raising anywhere.
    """
    huge = np.full((1, 90), 10_000.0, dtype=np.float32)
    huge[0, 30] = 20_000.0
    pitch, yaw = decode_angles(huge, huge, ModelConfig())
    assert math.isfinite(pitch) and math.isfinite(yaw)


# ---------------------------------------------------------------------------
# Preprocessing.
# ---------------------------------------------------------------------------


def test_preprocess_produces_the_tensor_the_graph_expects(tmp_path) -> None:
    """448x448, NCHW, float32, ImageNet-normalised, from a BGR crop."""
    session = FakeSession(_logits(45), _logits(45))
    model = _model(session, tmp_path)
    tensor = model.preprocess(np.full((97, 61, 3), 128, dtype=np.uint8))

    assert tensor.shape == (1, 3, 448, 448)
    assert tensor.dtype == np.float32
    # 128/255 normalised per channel, in RGB order after the colour conversion.
    expected = [(128 / 255 - m) / s for m, s in zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    for channel, value in enumerate(expected):
        assert tensor[0, channel].mean() == pytest.approx(value, abs=1e-4)


def test_the_crop_is_converted_from_bgr_not_assumed_rgb(tmp_path) -> None:
    """A pure-blue BGR crop must land in the BLUE channel of the RGB tensor.

    Passing RGB where BGR is expected does not raise; it just tracks worse.
    """
    session = FakeSession(_logits(45), _logits(45))
    model = _model(session, tmp_path)
    blue_bgr = np.zeros((32, 32, 3), dtype=np.uint8)
    blue_bgr[:, :, 0] = 255                      # BGR channel 0 is blue

    tensor = model.preprocess(blue_bgr)
    # After BGR->RGB, blue is channel 2.
    assert tensor[0, 2].mean() > tensor[0, 0].mean()


# ---------------------------------------------------------------------------
# Providers.
# ---------------------------------------------------------------------------


def test_the_preference_order_is_honoured() -> None:
    chosen = select_providers(
        ("DmlExecutionProvider", "CPUExecutionProvider"),
        ("CPUExecutionProvider", "DmlExecutionProvider", "AzureExecutionProvider"),
    )
    assert chosen[0] == "DmlExecutionProvider", "the preferred provider was not chosen"


def test_an_unavailable_preference_falls_back_to_cpu() -> None:
    """A machine without DirectML still runs, slowly. Legacy behaviour."""
    assert select_providers(("DmlExecutionProvider",), ("CPUExecutionProvider",)) == [
        "CPUExecutionProvider"
    ]


def test_no_provider_at_all_is_a_named_error_naming_the_extras() -> None:
    """D8: never a bare ImportError, and the message carries the remedy."""
    with pytest.raises(ProviderError) as excinfo:
        select_providers(("DmlExecutionProvider",), ())
    message = str(excinfo.value)
    assert "focusedgaze[directml]" in message
    assert "focusedgaze[cpu]" in message


def test_the_loaded_provider_is_reported_once(tmp_path, caplog) -> None:
    """Exactly one line, naming what ACTUALLY loaded.

    A silent CPU fallback is 104 ms per frame against 15 ms, and nothing else in
    the pipeline reports it: the user sees five updates a second and reports a
    tracking bug.
    """
    session = FakeSession(_logits(45), _logits(45), provider="DmlExecutionProvider")
    with caplog.at_level("INFO", logger="focusedgaze.core.model"):
        model = _model(session, tmp_path)
    lines = [r.getMessage() for r in caplog.records if "running on" in r.getMessage()]
    assert len(lines) == 1, f"expected exactly one provider line, got {lines}"
    assert "DmlExecutionProvider" in lines[0]
    assert model.provider == "DmlExecutionProvider"


# ---------------------------------------------------------------------------
# The no-I/O constraint.
# ---------------------------------------------------------------------------


def test_a_missing_model_raises_and_does_not_download(tmp_path, monkeypatch) -> None:
    """The pure layer must never reach the network.

    Assets are a separate explicit step by design. A silent fetch would break
    the guarantee that makes this layer testable in CI, and for the gaze weights
    specifically it would breach the Gaze360 licence position in NOTICE.

    Enforced rather than asserted from reading: every download entry point is
    replaced with something that fails the test if it is called at all.
    """
    # Reached through sys.modules, the one spelling that cannot be shadowed.
    # `from focusedgaze.assets import download` binds the FUNCTION, because
    # assets/__init__ re-exports a function of that name over its own submodule
    # (audit 40.3). Both obvious spellings give the callable.
    import sys

    import focusedgaze.assets.download  # noqa: F401

    download_module = sys.modules["focusedgaze.assets.download"]

    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("the estimator attempted a download")

    monkeypatch.setattr(download_module, "download", forbidden)
    monkeypatch.setattr(download_module, "ensure", forbidden)
    monkeypatch.setattr(download_module, "ensure_all", forbidden)
    monkeypatch.setattr(download_module, "urllib_transport", forbidden)

    with pytest.raises(ModelNotFoundError) as excinfo:
        GazeModel(tmp_path / "absent.onnx")

    message = str(excinfo.value)
    assert "does not download it, ever" in message
    assert "export-onnx" in message, "the message must name the way to obtain it"


def test_the_missing_model_message_explains_the_licence_position(tmp_path) -> None:
    """A user hitting this on first run needs to know it is deliberate."""
    with pytest.raises(ModelNotFoundError, match="non-commercial"):
        GazeModel(tmp_path / "absent.onnx")
