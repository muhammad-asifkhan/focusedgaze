"""The L2CS-Net gaze model: ONNX session, provider selection, angle decode.

TWO THINGS HERE LOOK WRONG AND ARE CORRECT
===========================================

1. THE OUTPUT TENSOR NAMES LIE
------------------------------
The exported graph names its outputs ``pitch_bins`` and ``yaw_bins``, in that
order. **The first tensor contains YAW.**

L2CS-Net's ``forward()`` returns ``(pre_yaw_gaze, pre_pitch_gaze)``, yaw first.
The export script labelled position 0 ``pitch_bins`` anyway, and the labels are
only strings attached at export time: they do not change what the tensors hold.
The legacy decode unpacked them in the true order despite the labels
(`gaze_pipeline.py:59-60`, whose comment reads "CONFIRMED FIX: model outputs are
(yaw, pitch) order"), and so does this.

**If you ever "fix" the names in `focusedgaze export-onnx`, you must change the
unpack order here in the same commit**, or every reading silently transposes its
axes. Transposed axes do not raise: the cursor simply moves sideways when you
look up, which reads as a bad calibration rather than as a bug in the decode.
Pinned by ``test_the_first_output_tensor_is_yaw_despite_its_name``.

2. THE ANGLE IS AN EXPECTATION, NOT AN ARGMAX
----------------------------------------------
The decode is::

    softmax(logits) . arange(bins) * bin_width_deg - angle_offset_deg

a **probability-weighted expectation over all 90 bins**, then degrees to radians.
It is not ``argmax``. Audit section 7 described it as an argmax once and was
corrected.

The difference is not cosmetic. An argmax agrees with the expectation only when
the distribution is sharply unimodal; on anything flatter or bimodal it returns a
different answer, and that answer is smooth and plausible and wrong. It is also
**discontinuous**: a near-tie between two adjacent bins makes the output jump a
full 4 degrees, where the expectation moves smoothly.

Audit section 48 leans on that smoothness twice: the residual difference between
two ONNX providers stayed at 1e-06 rad precisely because a small perturbation in
the logits moves a weighted mean slightly rather than flipping a maximum. An
argmax implementation would have been unmeasurably noisy across providers and
would have made the Tier 2 tolerance impossible to choose.

NO I/O, AND NO DOWNLOADING
==========================
Loading a model reads a file. It never fetches one. A missing model raises
:class:`~focusedgaze.exceptions.ModelNotFoundError` naming the command, because
the weights derive from Gaze360 and this package does not fetch them at all, and
because a silent download would break the no-network guarantee that lets the pure
layer run in CI.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
from numpy.typing import NDArray

from ..config import ModelConfig
from ..exceptions import ModelNotFoundError, ProviderError

__all__ = ["GazeModel", "decode_angles", "select_providers"]

_log = logging.getLogger(__name__)

#: The provider ONNX Runtime always has. Used as the last resort so a machine
#: with no accelerator still runs, matching `gaze_pipeline.py:29`.
_CPU: Final = "CPUExecutionProvider"


def select_providers(preferred: tuple[str, ...], available: tuple[str, ...]) -> list[str]:
    """Filter a preference list down to what this install actually has.

    Order is preserved, so the first entry is the most preferred provider that is
    present. Falls back to CPU when nothing in the list is available, which is
    the legacy behaviour: a machine without DirectML still runs, slowly.

    Raises:
        ProviderError: Nothing at all is available, including CPU. That means
            onnxruntime is installed but broken, which is a different problem
            from it being absent and needs saying differently.
    """
    usable = [p for p in preferred if p in available]
    if usable:
        return usable
    if _CPU in available:
        return [_CPU]
    raise ProviderError(
        "ONNX Runtime reports no usable execution provider, not even CPU. "
        f"Preferred: {list(preferred)}; available: {list(available)}. "
        "The runtime is installed but cannot execute anything, which usually "
        "means a broken or mismatched build. Reinstall one of: "
        "focusedgaze[directml], focusedgaze[cuda], focusedgaze[cpu]."
    )


def decode_angles(
    yaw_logits: NDArray[np.float32],
    pitch_logits: NDArray[np.float32],
    config: ModelConfig,
) -> tuple[float, float]:
    """Turn the two bin distributions into ``(pitch_rad, yaw_rad)``.

    Note the argument order: **yaw first**, matching the tensor order the graph
    actually emits, and the return order is **pitch first**, matching every
    caller in this package. The asymmetry is deliberate and is the safest place
    to put it: the confusing part is confined to one function signature that
    names both ends explicitly, rather than spread across the call sites.
    """
    bins = np.arange(config.bins, dtype=np.float32)

    def expectation(logits: NDArray[np.float32]) -> float:
        # Subtract the row max before exponentiating. Standard, and load-bearing:
        # the raw logits can be large enough that exp overflows to inf, and
        # inf/inf is nan, which would propagate silently to a nan coordinate.
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        degrees = float(np.sum(probs * bins, axis=1)[0]) * config.bin_width_deg
        return degrees - config.angle_offset_deg

    pitch_deg = expectation(pitch_logits)
    yaw_deg = expectation(yaw_logits)
    return pitch_deg * math.pi / 180.0, yaw_deg * math.pi / 180.0


class GazeModel:
    """The ONNX gaze model, loaded once and run per crop.

    Args:
        model_path: The ``.onnx`` graph. Defaults to whatever
            :func:`focusedgaze.assets.asset_path` resolves.
        config: Input geometry, normalisation and provider preference.
        session_factory: Injection seam for tests. Receives
            ``(path, providers, intra_op_num_threads)``.

    Raises:
        ModelNotFoundError: The graph is absent. **Never downloaded.**
        ProviderError: ONNX Runtime is missing or has no usable provider.
    """

    __slots__ = ("_config", "_input_name", "_provider", "_session")

    def __init__(
        self,
        model_path: str | Path | None = None,
        config: ModelConfig | None = None,
        session_factory: Any = None,
    ) -> None:
        self._config = config if config is not None else ModelConfig()

        if model_path is None:
            from ..assets import GAZE_MODEL, asset_path

            resolved = asset_path(GAZE_MODEL)
        else:
            resolved = Path(model_path)

        if not resolved.is_file():
            raise ModelNotFoundError(
                f"the gaze model is not at {resolved}.\n"
                "focusedgaze does not download it, ever: it derives from the "
                "Gaze360 dataset, whose authors restrict it to non-commercial "
                "research. Obtain the PyTorch weights yourself and convert once:\n"
                "    focusedgaze export-onnx --weights <L2CSNet_gaze360.pkl>\n"
                "See NOTICE. Run `focusedgaze check` to confirm what is missing."
            )

        self._session = (session_factory or self._default_session)(
            resolved, self._config.providers, self._config.intra_op_num_threads
        )
        self._provider = self._session.get_providers()[0]
        self._input_name = self._session.get_inputs()[0].name

        # Exactly one line, naming the provider that ACTUALLY loaded rather than
        # the one that was asked for. A CPU fallback is the difference between
        # ~15 ms and ~104 ms per frame, and nothing else in the system reports
        # it: the user experiences it as five updates a second and reports a
        # tracking bug. `focusedgaze check` reports the same fact up front.
        _log.info("gaze model running on %s", self._provider)

    @staticmethod
    def _default_session(path: Path, preferred: tuple[str, ...], threads: int) -> Any:
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ProviderError(
                "no ONNX Runtime is installed, so the gaze model cannot run. "
                "The base install deliberately declares no execution provider so "
                "the choice is yours. Install exactly one:\n"
                "    pip install 'focusedgaze[directml]'   # Windows GPU\n"
                "    pip install 'focusedgaze[cuda]'       # NVIDIA\n"
                "    pip install 'focusedgaze[cpu]'        # anywhere, slower"
            ) from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        providers = select_providers(preferred, tuple(ort.get_available_providers()))
        return ort.InferenceSession(str(path), sess_options=options, providers=providers)

    @property
    def provider(self) -> str:
        """The execution provider that actually loaded."""
        return str(self._provider)

    def preprocess(self, crop_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
        """BGR crop to the normalised NCHW tensor the graph expects.

        Transcribed from `gaze_pipeline.py:49-54`. The order matters: colour
        conversion, then resize, then scale to [0, 1], then ImageNet
        normalisation, then HWC to CHW. Normalising before the resize would
        interpolate normalised values rather than pixels and give a different
        tensor.
        """
        cfg = self._config
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        size = cfg.input_size
        resized = cv2.resize(rgb, (size, size)).astype(np.float32) / 255.0
        mean = np.asarray(cfg.imagenet_mean, dtype=np.float32)
        std = np.asarray(cfg.imagenet_std, dtype=np.float32)
        normalised = (resized - mean) / std
        chw = np.transpose(normalised, (2, 0, 1))
        tensor: NDArray[np.float32] = np.expand_dims(chw, axis=0).astype(np.float32)
        return tensor

    def predict(self, crop_bgr: NDArray[np.uint8]) -> tuple[float, float]:
        """One crop to ``(pitch_rad, yaw_rad)``.

        The unpack below is the defect described at the top of this module: the
        graph's first output is named ``pitch_bins`` and holds **yaw**.
        """
        tensor = self.preprocess(crop_bgr)
        yaw_logits, pitch_logits = self._session.run(None, {self._input_name: tensor})
        return decode_angles(yaw_logits, pitch_logits, self._config)
