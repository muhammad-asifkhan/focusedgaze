"""Every tunable in the pipeline, as frozen dataclasses.

The legacy system spread its tuning across module-level constants in three
files, so changing one meant editing source, and running two differently-tuned
pipelines in one process was impossible. Worse, some of those constants were not
constants: ``gaze_pipeline.py`` L15-16 computed a model path from
``pathlib.Path.cwd()`` at import time, which is why the server had to
``os.chdir()`` before importing it (``gaze_server.py`` L40).

Here configuration is an argument. Nothing in this module is mutable, nothing is
global, and nothing is read from the filesystem unless you ask for it by path.

DEFAULTS ARE THE SHIPPING VALUES, EXACTLY
-----------------------------------------
Every default below reproduces the value the pre-extraction system ran with, and
carries the legacy ``file:line`` it was taken from. A default that drifts is a
behaviour change wearing a refactor's clothes, so
``tests/test_config.py::test_defaults_match_the_legacy_pipeline`` pins the whole
set. Where a value also appears as a default in already-extracted code, such as
:class:`focusedgaze.core.filters.OneEuroFilter`, the two are asserted equal
rather than trusted to stay that way.

Line numbers refer to the legacy tree reachable through ``FOCUSEDGAZE_LEGACY_DIR``
(see ``CONTEXT_HANDOFF.md``), at the revision the migration was recorded from.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
* **Model and asset paths.** The legacy defaults were relative paths resolved
  against the working directory, which is the F1 defect: identical input gave a
  different answer depending on where Python was launched. Asset resolution is
  Phase 6's registry, not a string with a plausible-looking default.
* **Calibration settings.** Phase 5 owns the profile format and the fit.
* **Serialisation of a reading.** The wire format belongs to the server extra
  (Phase 7). Putting a payload shape here would make it two sources of truth.
"""

from __future__ import annotations

import json
import logging
import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Final

from .core.positioning import PositioningConfig
from .exceptions import ConfigError

__all__ = [
    "CameraConfig",
    "FilterConfig",
    "GazeConfig",
    "LandmarkConfig",
    "ModelConfig",
    "PositioningConfig",
    "RuntimeConfig",
]

_log = logging.getLogger("focusedgaze")

#: Section names of :class:`GazeConfig`, in declaration order. Used for
#: serialisation and to reject unknown top-level keys.
_SECTIONS: Final[tuple[str, ...]] = (
    "camera",
    "landmarks",
    "model",
    "filter",
    "positioning",
    "runtime",
)

#: Capture backends the package knows how to ask OpenCV for. "auto" lets the
#: capture layer pick per platform (D1), which on Windows means MSMF, the
#: backend the shipping system named explicitly.
_CAMERA_BACKENDS: Final[tuple[str, ...]] = ("auto", "msmf", "dshow", "avfoundation", "v4l2")


# --------------------------------------------------------------------------
# Validation helpers
#
# Every one of these raises ConfigError, never TypeError, because a value that
# arrived from a TOML file is user input and deserves an error naming the field.
# They also normalise: TOML has no float literal for `1`, so an integer written
# where a float is expected must not leave an int sitting in a float field.
# --------------------------------------------------------------------------


def _number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_min: bool = False,
    integer: bool = False,
) -> float:
    """Return ``value`` as a validated number.

    Args:
        value: The raw value, from a constructor argument or a parsed file.
        label: Fully qualified field name, used in the error message.
        minimum: Lower bound, inclusive unless ``exclusive_min``.
        maximum: Upper bound, inclusive.
        exclusive_min: Treat ``minimum`` as a strict bound.
        integer: Require a whole number and return an ``int``.

    Raises:
        ConfigError: if the value is not a finite number, or is out of bounds.
    """
    # bool is a subclass of int, so `width = true` would otherwise pass as 1.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a number, got {value!r}")
    if not math.isfinite(value):
        # NaN compares False against every bound, so an unchecked NaN would pass
        # every range test below and then poison the filter state silently.
        raise ConfigError(f"{label} must be finite, got {value!r}")
    if integer:
        if isinstance(value, float) and not value.is_integer():
            raise ConfigError(f"{label} must be a whole number, got {value!r}")
        number: float = int(value)
    else:
        number = float(value)
    if minimum is not None:
        if exclusive_min and number <= minimum:
            raise ConfigError(f"{label} must be greater than {minimum}, got {number!r}")
        if not exclusive_min and number < minimum:
            raise ConfigError(f"{label} must be at least {minimum}, got {number!r}")
    if maximum is not None and number > maximum:
        raise ConfigError(f"{label} must be at most {maximum}, got {number!r}")
    return number


def _set_number(
    obj: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_min: bool = False,
    integer: bool = False,
) -> None:
    """Validate one numeric field of a frozen config and write the result back."""
    label = f"{type(obj).__name__}.{name}"
    checked = _number(
        getattr(obj, name),
        label,
        minimum=minimum,
        maximum=maximum,
        exclusive_min=exclusive_min,
        integer=integer,
    )
    object.__setattr__(obj, name, checked)


def _set_flag(obj: object, name: str) -> None:
    """Validate one boolean field of a frozen config."""
    value = getattr(obj, name)
    if not isinstance(value, bool):
        raise ConfigError(f"{type(obj).__name__}.{name} must be true or false, got {value!r}")


def _set_text(obj: object, name: str, *, allowed: tuple[str, ...] | None = None) -> None:
    """Validate one string field of a frozen config, optionally against a whitelist."""
    label = f"{type(obj).__name__}.{name}"
    value = getattr(obj, name)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{label} must be a non-empty string, got {value!r}")
    if allowed is not None and value not in allowed:
        raise ConfigError(f"{label} must be one of {', '.join(allowed)}, got {value!r}")


def _set_triple(obj: object, name: str, *, nonzero: bool = False) -> None:
    """Validate a three-number field and normalise it to a tuple of floats.

    Accepts a list so a value parsed from TOML or JSON, where there are no
    tuples, round-trips back to an equal config.
    """
    label = f"{type(obj).__name__}.{name}"
    value = getattr(obj, name)
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ConfigError(f"{label} must be a list of three numbers, got {value!r}")
    if len(value) != 3:
        raise ConfigError(f"{label} must have exactly three entries, got {len(value)}")
    numbers = tuple(_number(v, f"{label}[{i}]") for i, v in enumerate(value))
    if nonzero and any(n == 0.0 for n in numbers):
        raise ConfigError(f"{label} is a divisor and must not contain zero, got {value!r}")
    object.__setattr__(obj, name, numbers)


def _set_string_tuple(obj: object, name: str) -> None:
    """Validate a non-empty sequence of strings and normalise it to a tuple."""
    label = f"{type(obj).__name__}.{name}"
    value = getattr(obj, name)
    # A bare string is a sequence of characters, and silently splitting one into
    # single-letter entries is exactly the kind of no-op-shaped bug that hides.
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ConfigError(f"{label} must be a list of strings, got {value!r}")
    if not value:
        raise ConfigError(f"{label} must not be empty")
    for entry in value:
        if not isinstance(entry, str) or not entry:
            raise ConfigError(f"{label} entries must be non-empty strings, got {entry!r}")
    object.__setattr__(obj, name, tuple(value))


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Which camera to open and how to configure it.

    Args:
        index: Device index passed to OpenCV.
        width: Requested capture width in pixels.
        height: Requested capture height in pixels.
        backend: Capture backend, or "auto" to let the capture layer choose per
            platform. See :data:`_CAMERA_BACKENDS` for the accepted values.
        mirror: Flip each frame horizontally before inference. Required, not
            cosmetic: calibration was recorded through a mirrored preview, so an
            unmirrored frame inverts left and right against the fitted model.

    Frozen, so a running capture cannot be reconfigured underneath itself.
    """

    index: int = 0                       # gaze_server.py:91  CAM_INDEX = 0
    width: int = 1280                    # gaze_server.py:92  CAM_W = 1280
    height: int = 720                    # gaze_server.py:92  CAM_H = 720
    # gaze_server.py:136 opens cv2.CAP_MSMF unconditionally, with the note that
    # DSHOW capped 720p at 10 fps. "auto" resolves to MSMF on Windows, so the
    # tested platform keeps the shipping behaviour while D1's per-platform
    # backend selection stays possible elsewhere.
    backend: str = "auto"
    mirror: bool = True                  # gaze_server.py:362  cv2.flip(frame, 1)

    def __post_init__(self) -> None:
        _set_number(self, "index", minimum=0, integer=True)
        _set_number(self, "width", minimum=1, integer=True)
        _set_number(self, "height", minimum=1, integer=True)
        _set_text(self, "backend", allowed=_CAMERA_BACKENDS)
        _set_flag(self, "mirror")


@dataclass(frozen=True, slots=True)
class LandmarkConfig:
    """MediaPipe FaceLandmarker options, and the face crop derived from them.

    Args:
        max_faces: Faces to track. The pipeline is single-user by design.
        min_detection_confidence: MediaPipe's detection threshold.
        min_tracking_confidence: MediaPipe's tracking threshold.
        output_transform_matrix: Ask for the 4x4 head-pose matrix. Head-pose
            features depend on it.
        video_timestamp_fps: Frame rate assumed when converting a frame index
            into the millisecond timestamp MediaPipe's VIDEO mode requires.
        bbox_smoothing: Weight of the newest box in the exponential smoothing of
            the face crop. Higher follows faster and jitters more.
        crop_padding: Fraction of the face box added on each side before
            cropping, so the model sees some context.

    ``bbox_smoothing`` was module-level mutable state in the legacy pipeline
    (``gaze_pipeline.py`` L35), which meant two estimators in one process
    corrupted each other's crops. The value is preserved; the global is not.
    """

    max_faces: int = 1                          # gaze_server.py:302  num_faces=1
    min_detection_confidence: float = 0.5       # gaze_server.py:303
    min_tracking_confidence: float = 0.5        # gaze_server.py:304
    output_transform_matrix: bool = True        # gaze_server.py:305
    # gaze_pipeline.py:117  timestamp_ms = int(frame_idx * (1000 / 30))
    video_timestamp_fps: float = 30.0
    bbox_smoothing: float = 0.3                 # gaze_pipeline.py:18  BBOX_SMOOTHING
    crop_padding: float = 0.3                   # gaze_pipeline.py:70  padding_ratio=0.3

    def __post_init__(self) -> None:
        _set_number(self, "max_faces", minimum=1, integer=True)
        _set_number(self, "min_detection_confidence", minimum=0.0, maximum=1.0)
        _set_number(self, "min_tracking_confidence", minimum=0.0, maximum=1.0)
        _set_flag(self, "output_transform_matrix")
        _set_number(self, "video_timestamp_fps", minimum=0.0, exclusive_min=True)
        _set_number(self, "bbox_smoothing", minimum=0.0, maximum=1.0)
        _set_number(self, "crop_padding", minimum=0.0)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """How the L2CS-Net ONNX graph is fed and how its output is decoded.

    Args:
        bins: Number of angle bins the model emits per axis.
        input_size: Square input edge in pixels.
        imagenet_mean: Per-channel mean subtracted after scaling to [0, 1].
        imagenet_std: Per-channel standard deviation divided out. A divisor, so
            it may not contain zero.
        bin_width_deg: Degrees per bin.
        angle_offset_deg: Degrees subtracted after scaling, centring the range
            on zero.
        providers: ONNX Runtime execution providers in preference order.
            Unavailable ones are skipped, and the one that loaded is logged.
        intra_op_num_threads: ONNX Runtime intra-op thread count. 0 lets the
            runtime decide.

    The decode is a softmax-weighted expectation over the bins, not an argmax
    (``gaze_pipeline.py`` L61-64): ``sum(softmax(logits) * bin_index) *
    bin_width_deg - angle_offset_deg``. The distinction matters and is not
    cosmetic. An argmax agrees with an expectation only when the distribution is
    sharply unimodal and diverges by a whole bin near a tie, which is a
    plausible, smooth, wrong answer rather than a crash. Audit 32.6b.
    """

    bins: int = 90                                                  # gaze_pipeline.py:17
    input_size: int = 448                                           # gaze_pipeline.py:51
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)  # gaze_pipeline.py:20
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)   # gaze_pipeline.py:21
    bin_width_deg: float = 4.0                                      # gaze_pipeline.py:63-64
    angle_offset_deg: float = 180.0                                 # gaze_pipeline.py:63-64
    # gaze_pipeline.py:28  _preferred = ["DmlExecutionProvider", "CPUExecutionProvider"]
    providers: tuple[str, ...] = ("DmlExecutionProvider", "CPUExecutionProvider")
    intra_op_num_threads: int = 4                                   # gaze_pipeline.py:25

    def __post_init__(self) -> None:
        _set_number(self, "bins", minimum=1, integer=True)
        _set_number(self, "input_size", minimum=1, integer=True)
        _set_triple(self, "imagenet_mean")
        _set_triple(self, "imagenet_std", nonzero=True)
        _set_number(self, "bin_width_deg", minimum=0.0, exclusive_min=True)
        _set_number(self, "angle_offset_deg")
        _set_string_tuple(self, "providers")
        _set_number(self, "intra_op_num_threads", minimum=0, integer=True)


@dataclass(frozen=True, slots=True)
class FilterConfig:
    """One Euro filter settings for the screen point.

    Args:
        min_cutoff: Cutoff frequency at rest, in Hz. Lower is steadier when
            still, at the cost of lag on slow movement.
        beta: Speed coefficient. Higher is snappier on fast movement, at the
            cost of letting more jitter through.
        d_cutoff: Cutoff for the derivative estimate. Rarely worth changing.
        enabled: Filter the point at all. Off is not a shipping configuration;
            it exists so accuracy can be measured on the unsmoothed signal.

    These are the same three numbers :class:`focusedgaze.core.filters.OneEuroFilter`
    defaults to. The test suite asserts the two agree rather than assuming it,
    because a default that exists in two places drifts in one of them.
    """

    min_cutoff: float = 0.7      # gaze_server.py:100  ONE_EURO_MIN_CUTOFF
    beta: float = 0.6            # gaze_server.py:101  ONE_EURO_BETA
    d_cutoff: float = 1.0        # gaze_server.py:102  ONE_EURO_DCUTOFF
    # No legacy equivalent: the shipping pipeline always filtered.
    enabled: bool = True

    def __post_init__(self) -> None:
        _set_number(self, "min_cutoff", minimum=0.0, exclusive_min=True)
        _set_number(self, "beta", minimum=0.0)
        _set_number(self, "d_cutoff", minimum=0.0, exclusive_min=True)
        _set_flag(self, "enabled")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """How the running system behaves once the pipeline itself is settled.

    Args:
        send_hz: Nominal broadcast rate for the server extra. Nominal because
            Windows timer granularity measured it out at 35-44 Hz.
        host: Interface the server binds to. Loopback by default: this is a
            local input device, not a network service.
        port: Port the server binds to.
        dwell_ms: How long a gaze must rest on a target before it counts as a
            click, for consumers that implement dwell selection.
        edge_zone: Fraction of the screen at each edge that triggers panning or
            scrolling, for consumers that implement edge navigation.

    ``dwell_ms`` and ``edge_zone`` describe the behaviour of a consumer, not of
    the tracker. They are carried here because the shipping system's values are
    tuned to this tracker's accuracy and would otherwise be lost, and because
    two consumers of the same tracker disagreeing about dwell time is a bug that
    already exists downstream. See the note in ``tests/test_config.py``.
    """

    send_hz: float = 60.0        # gaze_server.py:93  SEND_HZ = 60
    host: str = "localhost"      # gaze_server.py:69  HOST
    port: int = 8765             # gaze_server.py:70  PORT
    # NOT from gaze_server.py: these two live in the browser client.
    dwell_ms: float = 1050.0     # game/workingGameTemplate/gaze-client.js:39  DWELL_MS
    edge_zone: float = 0.09      # game/workingGameTemplate/gaze-client.js:42  EDGE_ZONE

    def __post_init__(self) -> None:
        _set_number(self, "send_hz", minimum=0.0, exclusive_min=True)
        _set_text(self, "host")
        _set_number(self, "port", minimum=1, maximum=65535, integer=True)
        _set_number(self, "dwell_ms", minimum=0.0, exclusive_min=True)
        # Half the screen from each side would leave no middle at all.
        _set_number(self, "edge_zone", minimum=0.0, maximum=0.5)


def _check_positioning(config: PositioningConfig) -> None:
    """Validate a :class:`PositioningConfig`, which carries no validation itself.

    That class was extracted in Phase 2 and lives in
    :mod:`focusedgaze.core.positioning`, next to the gate that consumes it. It is
    re-exported from here rather than redefined, because two classes of the same
    name holding the same six numbers is the defect this module exists to
    remove. The cost is that its checks live here instead of in its own
    ``__post_init__``, so a ``PositioningConfig`` built directly is unvalidated
    until it reaches a :class:`GazeConfig`.

    Raises:
        ConfigError: on a non-numeric field, an impossible bound, or a zone with
            no width.
    """
    label = "PositioningConfig"
    minimum = _number(config.min_distance_cm, f"{label}.min_distance_cm",
                      minimum=0.0, exclusive_min=True)
    maximum = _number(config.max_distance_cm, f"{label}.max_distance_cm",
                      minimum=0.0, exclusive_min=True)
    if minimum >= maximum:
        raise ConfigError(
            f"{label}.min_distance_cm ({minimum}) must be less than "
            f"max_distance_cm ({maximum}): the tracking zone would be empty"
        )
    _number(config.warn_margin_cm, f"{label}.warn_margin_cm", minimum=0.0)
    _number(config.center_tolerance, f"{label}.center_tolerance",
            minimum=0.0, maximum=0.5, exclusive_min=True)
    _number(config.real_ipd_cm, f"{label}.real_ipd_cm", minimum=0.0, exclusive_min=True)
    _number(config.assumed_hfov_deg, f"{label}.assumed_hfov_deg",
            minimum=0.0, maximum=180.0, exclusive_min=True)


# --------------------------------------------------------------------------
# Composite
# --------------------------------------------------------------------------


def _field_names(cls: type[Any]) -> tuple[str, ...]:
    """The declared field names of a dataclass, in order."""
    return tuple(f.name for f in fields(cls))


def _reject_unknown(data: Mapping[str, Any], known: tuple[str, ...], label: str) -> None:
    """Raise if ``data`` carries keys the target does not declare.

    Ignoring an unrecognised key would make a typo in a config file a silent
    no-op, which is the failure mode this project has been bitten by more than
    any other. It fails loudly instead, naming what was accepted.
    """
    unknown = sorted(k for k in data if k not in known)
    if unknown:
        raise ConfigError(
            f"unknown key(s) in [{label}]: {', '.join(unknown)}. "
            f"Known keys: {', '.join(known)}"
        )


def _section_kwargs(data: Mapping[str, Any], name: str, cls: type[Any]) -> dict[str, Any]:
    """Pull one section out of a parsed mapping as validated constructor kwargs."""
    raw = data.get(name)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"[{name}] must be a table of settings, got {type(raw).__name__}")
    _reject_unknown(raw, _field_names(cls), name)
    return dict(raw)


def _section_to_dict(section: Any) -> dict[str, Any]:
    """One section as plain JSON- and TOML-writable values."""
    out: dict[str, Any] = {}
    for f in fields(section):
        value = getattr(section, f.name)
        # Neither JSON nor TOML has a tuple. Lists come back through
        # __post_init__ as tuples again, so the round-trip is lossless.
        out[f.name] = list(value) if isinstance(value, tuple) else value
    return out


@dataclass(frozen=True, slots=True)
class GazeConfig:
    """The whole configuration, one section per stage of the pipeline.

    Args:
        camera: Capture device and format.
        landmarks: Face detection and the crop handed to the gaze model.
        model: ONNX input preparation, output decode and execution providers.
        filter: Smoothing of the final screen point.
        positioning: The zone the user must sit in.
        runtime: Broadcast rate, bind address, and consumer-facing interaction
            timings.

    Build it any of four ways::

        GazeConfig()                                    # shipping defaults
        GazeConfig(camera=CameraConfig(index=1))        # kwargs
        GazeConfig.from_dict({"camera": {"index": 1}})  # a mapping
        GazeConfig.from_file(path)                      # a .toml or .json file

    Every one of them validates. Frozen throughout, so a configuration handed to
    a running estimator cannot change underneath it, and one instance can be
    shared between threads.
    """

    camera: CameraConfig = field(default_factory=CameraConfig)
    landmarks: LandmarkConfig = field(default_factory=LandmarkConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    positioning: PositioningConfig = field(default_factory=PositioningConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self) -> None:
        expected: tuple[tuple[str, type[Any]], ...] = (
            ("camera", CameraConfig),
            ("landmarks", LandmarkConfig),
            ("model", ModelConfig),
            ("filter", FilterConfig),
            ("positioning", PositioningConfig),
            ("runtime", RuntimeConfig),
        )
        for name, cls in expected:
            value = getattr(self, name)
            if not isinstance(value, cls):
                raise ConfigError(
                    f"GazeConfig.{name} must be a {cls.__name__}, "
                    f"got {type(value).__name__}"
                )
        # Each section validated itself on construction. This one cannot, so it
        # is checked on the way in rather than left as the single unchecked hole.
        _check_positioning(self.positioning)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GazeConfig:
        """Build from a nested mapping, one table per section.

        Missing sections take their defaults. Unknown sections and unknown keys
        within a section are errors, not ignored: a silently discarded setting
        looks identical to one that was applied.

        Args:
            data: ``{"camera": {"index": 1}, "filter": {"beta": 0.4}, ...}``.

        Raises:
            ConfigError: on an unknown key, a section that is not a table, or
                any value that fails its field's validation.
        """
        if not isinstance(data, Mapping):
            raise ConfigError(f"configuration must be a mapping, got {type(data).__name__}")
        _reject_unknown(data, _SECTIONS, "config")
        return cls(
            camera=CameraConfig(**_section_kwargs(data, "camera", CameraConfig)),
            landmarks=LandmarkConfig(**_section_kwargs(data, "landmarks", LandmarkConfig)),
            model=ModelConfig(**_section_kwargs(data, "model", ModelConfig)),
            filter=FilterConfig(**_section_kwargs(data, "filter", FilterConfig)),
            positioning=PositioningConfig(
                **_section_kwargs(data, "positioning", PositioningConfig)
            ),
            runtime=RuntimeConfig(**_section_kwargs(data, "runtime", RuntimeConfig)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> GazeConfig:
        """Load from a TOML or JSON file, chosen by suffix.

        TOML is parsed with the standard library's :mod:`tomllib`, so reading a
        configuration file adds no dependency.

        Args:
            path: A ``.toml`` or ``.json`` file laid out as :meth:`from_dict`
                expects. There is no default location and no search: the path is
                always explicit, because a config discovered relative to the
                working directory is the F1 defect in a new place.

        Raises:
            ConfigError: if the file is missing, unreadable, has an unsupported
                suffix, does not parse, or contains an invalid setting. A
                missing file is a ``ConfigError`` rather than a
                ``FileNotFoundError`` so that one ``except GazeError`` still
                covers the whole package.
        """
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix not in (".toml", ".json"):
            raise ConfigError(
                f"{p} has an unsupported configuration format {suffix!r}: "
                "expected .toml or .json"
            )
        try:
            raw = p.read_bytes()
        except OSError as exc:
            raise ConfigError(f"could not read configuration {p}: {exc}") from exc
        try:
            data = tomllib.loads(raw.decode("utf-8")) if suffix == ".toml" else json.loads(raw)
        except (UnicodeDecodeError, tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
            raise ConfigError(f"{p} is not valid {suffix.lstrip('.')}: {exc}") from exc
        config = cls.from_dict(data)
        _log.debug("loaded configuration from %s", p)
        return config

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Serialise to plain types, ready for ``json.dump`` or a TOML writer.

        Round-trips: ``GazeConfig.from_dict(cfg.to_dict()) == cfg``. Tuples come
        out as lists, because neither format has a tuple, and are restored on the
        way back in.
        """
        return {name: _section_to_dict(getattr(self, name)) for name in _SECTIONS}
