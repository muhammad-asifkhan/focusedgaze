"""The configuration defaults ARE the shipping behaviour, so they are pinned.

Phase 3's correctness property is narrow and total: every default in
`focusedgaze.config` must reproduce the value the pre-extraction pipeline ran
with. A refactor that silently retunes the system is indistinguishable from one
that preserves it until somebody measures accuracy again, which in this project
means a person sitting in front of a camera. That is far too late to find out.

The expected values below are written out literally rather than read back from
`focusedgaze.config`, which would only prove the module agrees with itself. Each
carries the legacy `file:line` it was harvested from. Two of them do NOT come
from the Python pipeline at all; see `test_dwell_and_edge_zone_come_from_the_browser_client`.

The suite runs everywhere, including CI, because it needs no legacy checkout: it
is the frozen record, not a comparison against a tree that may be absent. A
version that skipped without one would be worth nothing on the machine that
matters most.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from focusedgaze import config as config_module
from focusedgaze.config import (
    CameraConfig,
    FilterConfig,
    GazeConfig,
    LandmarkConfig,
    ModelConfig,
    PositioningConfig,
    RuntimeConfig,
)
from focusedgaze.core import positioning as positioning_module
from focusedgaze.core.filters import OneEuroFilter
from focusedgaze.exceptions import ConfigError, GazeError

# (section, field, expected value, legacy source)
LEGACY_DEFAULTS: list[tuple[str, str, object, str]] = [
    ("camera", "index", 0, "gaze_server.py:91"),
    ("camera", "width", 1280, "gaze_server.py:92"),
    ("camera", "height", 720, "gaze_server.py:92"),
    ("camera", "mirror", True, "gaze_server.py:362"),

    ("landmarks", "max_faces", 1, "gaze_server.py:302"),
    ("landmarks", "min_detection_confidence", 0.5, "gaze_server.py:303"),
    ("landmarks", "min_tracking_confidence", 0.5, "gaze_server.py:304"),
    ("landmarks", "output_transform_matrix", True, "gaze_server.py:305"),
    ("landmarks", "video_timestamp_fps", 30.0, "gaze_pipeline.py:117"),
    ("landmarks", "bbox_smoothing", 0.3, "gaze_pipeline.py:18"),
    ("landmarks", "crop_padding", 0.3, "gaze_pipeline.py:70"),

    ("model", "bins", 90, "gaze_pipeline.py:17"),
    ("model", "input_size", 448, "gaze_pipeline.py:51"),
    ("model", "imagenet_mean", (0.485, 0.456, 0.406), "gaze_pipeline.py:20"),
    ("model", "imagenet_std", (0.229, 0.224, 0.225), "gaze_pipeline.py:21"),
    ("model", "bin_width_deg", 4.0, "gaze_pipeline.py:63-64"),
    ("model", "angle_offset_deg", 180.0, "gaze_pipeline.py:63-64"),
    ("model", "providers", ("DmlExecutionProvider", "CPUExecutionProvider"),
     "gaze_pipeline.py:28"),
    ("model", "intra_op_num_threads", 4, "gaze_pipeline.py:25"),

    ("filter", "min_cutoff", 0.7, "gaze_server.py:100"),
    ("filter", "beta", 0.6, "gaze_server.py:101"),
    ("filter", "d_cutoff", 1.0, "gaze_server.py:102"),

    ("positioning", "min_distance_cm", 45.0, "positioning_gate.py:30"),
    ("positioning", "max_distance_cm", 65.0, "positioning_gate.py:31"),
    ("positioning", "warn_margin_cm", 5.0, "positioning_gate.py:32"),
    ("positioning", "center_tolerance", 0.12, "positioning_gate.py:33"),
    ("positioning", "real_ipd_cm", 6.3, "positioning_gate.py:35"),
    ("positioning", "assumed_hfov_deg", 60.0, "positioning_gate.py:41"),

    ("runtime", "send_hz", 60.0, "gaze_server.py:93"),
    ("runtime", "host", "localhost", "gaze_server.py:69"),
    ("runtime", "port", 8765, "gaze_server.py:70"),
    ("runtime", "dwell_ms", 1050.0, "gaze-client.js:39"),
    ("runtime", "edge_zone", 0.09, "gaze-client.js:42"),
]

SECTION_CLASSES = [
    CameraConfig,
    LandmarkConfig,
    ModelConfig,
    FilterConfig,
    PositioningConfig,
    RuntimeConfig,
]


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("section", "name", "expected", "source"), LEGACY_DEFAULTS)
def test_defaults_match_the_legacy_pipeline(
    section: str, name: str, expected: object, source: str
) -> None:
    """Every default reproduces the value the shipping system ran with."""
    got = getattr(getattr(GazeConfig(), section), name)
    assert got == expected, f"{section}.{name} drifted from {source}: {got!r} != {expected!r}"
    # 1280 == 1280.0 and True == 1, so equality alone would let an int/float or
    # bool/int swap through. The types are part of the contract too.
    assert type(got) is type(expected), (
        f"{section}.{name} changed type: {type(got).__name__} not {type(expected).__name__}"
    )


def test_every_field_of_every_section_is_pinned() -> None:
    """A new default must be added to LEGACY_DEFAULTS, not slipped in unnoticed.

    Without this, someone can add a field with an invented default and the table
    above still passes, which is the whole class of bug this file exists to stop.
    Fields that genuinely have no legacy counterpart are listed here explicitly.
    """
    no_legacy_counterpart = {
        # "auto" resolves to MSMF on Windows, which is what gaze_server.py:136
        # opened. See the deviation recorded in the Phase 3 audit notes.
        ("camera", "backend"),
        # The shipping pipeline always filtered; there was no off switch.
        ("filter", "enabled"),
    }
    pinned = {(section, name) for section, name, _, _ in LEGACY_DEFAULTS}
    config = GazeConfig()
    declared = {
        (section.name, f.name)
        for section in dataclasses.fields(config)
        for f in dataclasses.fields(getattr(config, section.name))
    }
    missing = declared - pinned - no_legacy_counterpart
    assert not missing, f"unpinned configuration fields: {sorted(missing)}"


def test_filter_defaults_agree_with_the_already_extracted_filter() -> None:
    """FilterConfig and OneEuroFilter must not each hold their own idea of 0.7.

    `core/filters.py` was extracted in Phase 2 with the tuning hard-coded as
    argument defaults. This config now states the same three numbers. Two copies
    of one value drift, and the drift shows up as a slightly different feel
    rather than as a failure, so the agreement is asserted rather than assumed.
    """
    live = OneEuroFilter()
    configured = FilterConfig()
    assert (live.min_cutoff, live.beta, live.d_cutoff) == (
        configured.min_cutoff,
        configured.beta,
        configured.d_cutoff,
    )


def test_positioning_config_is_the_one_from_the_core_and_not_a_copy() -> None:
    """There must be exactly one PositioningConfig in the package.

    `core/positioning.py` already defined it, so this module re-exports that
    class instead of declaring a second one with the same name and the same six
    numbers. If someone later redefines it here, the two would diverge silently
    and `isinstance` checks would start failing in confusing places.
    """
    assert PositioningConfig is positioning_module.PositioningConfig


def test_dwell_and_edge_zone_come_from_the_browser_client() -> None:
    """Documents where these two values actually live, because it is not obvious.

    They are consumer-interaction timings, and the only place they exist in the
    legacy system is the browser overlay `gaze-client.js` (L39 and L42), not
    `gaze_server.py`. A sibling file in the same directory, `forest.js` L434,
    declares its own DWELL_MS of 1000 ms for the same behaviour, so the legacy
    system disagrees with itself about the dwell time. 1050 is the documented
    one and is the value carried here.

    This test asserts nothing the table above does not. It exists so the next
    reader looking for DWELL_MS in the Python sources does not conclude it was
    invented.
    """
    assert RuntimeConfig().dwell_ms == 1050.0
    assert RuntimeConfig().edge_zone == 0.09


# --------------------------------------------------------------------------
# Immutability
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", SECTION_CLASSES)
def test_sections_are_frozen(cls: type) -> None:
    """A running estimator's configuration cannot be changed underneath it."""
    instance = cls()
    name = dataclasses.fields(instance)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, name, getattr(instance, name))


def test_composite_is_frozen() -> None:
    """Including the composite, not only its sections."""
    config = GazeConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.camera = CameraConfig()


def test_tuple_fields_cannot_be_mutated_in_place() -> None:
    """Frozen is not enough on its own: a list field would still be editable."""
    model = ModelConfig()
    assert isinstance(model.providers, tuple)
    assert isinstance(model.imagenet_mean, tuple)
    assert isinstance(model.imagenet_std, tuple)


def test_module_holds_no_mutable_state() -> None:
    """No lists, dicts or sets at module level (rule: no global mutable state).

    The legacy pipeline kept a module-level smoothing box and a module-level
    LATEST dict, which is why two estimators in one process corrupted each
    other. Nothing in the configuration layer gets to repeat that.
    """
    offenders = {
        name: type(value).__name__
        for name, value in vars(config_module).items()
        if not name.startswith("__") and isinstance(value, (list, dict, set, bytearray))
    }
    assert not offenders, f"mutable module-level state in config.py: {offenders}"


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (CameraConfig, {"index": -1}),
        (CameraConfig, {"width": 0}),
        (CameraConfig, {"height": 0}),
        (CameraConfig, {"width": 1.5}),
        (CameraConfig, {"width": float("nan")}),
        (CameraConfig, {"width": "1280"}),
        (CameraConfig, {"backend": "opengl"}),
        (CameraConfig, {"backend": ""}),
        (CameraConfig, {"mirror": "yes"}),
        (LandmarkConfig, {"max_faces": 0}),
        (LandmarkConfig, {"min_detection_confidence": 1.5}),
        (LandmarkConfig, {"min_tracking_confidence": -0.1}),
        (LandmarkConfig, {"video_timestamp_fps": 0}),
        (LandmarkConfig, {"bbox_smoothing": 1.1}),
        (LandmarkConfig, {"crop_padding": -0.1}),
        (LandmarkConfig, {"output_transform_matrix": 1}),
        (ModelConfig, {"bins": 0}),
        (ModelConfig, {"input_size": 0}),
        (ModelConfig, {"bin_width_deg": 0}),
        (ModelConfig, {"intra_op_num_threads": -1}),
        (ModelConfig, {"imagenet_mean": (0.485, 0.456)}),
        (ModelConfig, {"imagenet_mean": "0.485"}),
        (ModelConfig, {"imagenet_std": (0.229, 0.0, 0.225)}),
        (ModelConfig, {"providers": ()}),
        (ModelConfig, {"providers": "DmlExecutionProvider"}),
        (ModelConfig, {"providers": ("",)}),
        (FilterConfig, {"min_cutoff": 0.0}),
        (FilterConfig, {"beta": -0.1}),
        (FilterConfig, {"d_cutoff": 0.0}),
        (FilterConfig, {"enabled": "true"}),
        (RuntimeConfig, {"send_hz": 0}),
        (RuntimeConfig, {"host": ""}),
        (RuntimeConfig, {"port": 0}),
        (RuntimeConfig, {"port": 70000}),
        (RuntimeConfig, {"dwell_ms": 0}),
        (RuntimeConfig, {"edge_zone": 0.6}),
        (RuntimeConfig, {"edge_zone": -0.01}),
    ],
)
def test_out_of_range_values_are_rejected(cls: type, kwargs: dict[str, object]) -> None:
    """Bad input fails loudly at construction, not later as a strange reading."""
    with pytest.raises(ConfigError):
        cls(**kwargs)


def test_config_error_is_catchable_as_one_package_error() -> None:
    """ConfigError stays inside the package's own tree (D7)."""
    with pytest.raises(GazeError):
        CameraConfig(width=0)
    with pytest.raises(ValueError, match="width"):
        CameraConfig(width=0)


def test_an_empty_tracking_zone_is_rejected() -> None:
    """min >= max would let the gate accept nothing at all, silently."""
    with pytest.raises(ConfigError, match="tracking zone"):
        GazeConfig(positioning=PositioningConfig(min_distance_cm=70.0, max_distance_cm=65.0))


def test_positioning_values_are_validated_through_the_composite() -> None:
    """PositioningConfig carries no validation of its own, so GazeConfig does it."""
    with pytest.raises(ConfigError):
        GazeConfig(positioning=PositioningConfig(assumed_hfov_deg=0.0))
    with pytest.raises(ConfigError):
        GazeConfig(positioning=PositioningConfig(real_ipd_cm=0.0))
    with pytest.raises(ConfigError):
        GazeConfig(positioning=PositioningConfig(center_tolerance=0.9))


def test_a_section_of_the_wrong_type_is_rejected() -> None:
    """A dict where a section belongs would otherwise fail much later."""
    with pytest.raises(ConfigError, match="CameraConfig"):
        GazeConfig(camera={"index": 0})  # type: ignore[arg-type]


def test_values_are_normalised_to_their_declared_type() -> None:
    """TOML has no float literal for 1, so an int must not stay an int."""
    assert type(FilterConfig(d_cutoff=1).d_cutoff) is float
    assert ModelConfig(imagenet_mean=[0.1, 0.2, 0.3]).imagenet_mean == (0.1, 0.2, 0.3)
    assert isinstance(ModelConfig(providers=["CPUExecutionProvider"]).providers, tuple)


# --------------------------------------------------------------------------
# Round trips
# --------------------------------------------------------------------------


def test_dict_round_trip_preserves_everything() -> None:
    """to_dict then from_dict must return an equal configuration."""
    original = GazeConfig()
    assert GazeConfig.from_dict(original.to_dict()) == original


def test_dict_round_trip_preserves_overrides() -> None:
    """Round-tripping the defaults alone would not prove much."""
    original = GazeConfig(
        camera=CameraConfig(index=2, width=640, height=480, backend="dshow", mirror=False),
        filter=FilterConfig(min_cutoff=0.25, beta=1.5, enabled=False),
        model=ModelConfig(providers=("CPUExecutionProvider",), intra_op_num_threads=1),
        positioning=PositioningConfig(min_distance_cm=40.0, max_distance_cm=80.0),
        runtime=RuntimeConfig(port=9000, send_hz=30.0),
    )
    assert GazeConfig.from_dict(original.to_dict()) == original


def test_partial_dict_takes_defaults_for_the_rest() -> None:
    """A config file naming one setting must not blank out the others."""
    config = GazeConfig.from_dict({"camera": {"index": 3}})
    assert config.camera.index == 3
    assert config.camera.width == GazeConfig().camera.width
    assert config.filter == FilterConfig()


def test_to_dict_is_json_serialisable() -> None:
    """Tuples are not JSON, so they must already be lists on the way out."""
    text = json.dumps(GazeConfig().to_dict())
    assert GazeConfig.from_dict(json.loads(text)) == GazeConfig()


@pytest.mark.parametrize("suffix", [".json", ".toml"])
def test_file_round_trip(tmp_path, suffix: str) -> None:
    """Both supported formats load to the same configuration.

    The JSON file is written from `to_dict`. The TOML one is written by hand,
    because the standard library reads TOML but does not write it, and adding a
    writer would add a dependency this package will not take for a config file.
    """
    original = GazeConfig(
        camera=CameraConfig(index=1, width=640, height=480),
        filter=FilterConfig(min_cutoff=0.4),
    )
    path = tmp_path / f"focusedgaze{suffix}"
    if suffix == ".json":
        path.write_text(json.dumps(original.to_dict()), encoding="utf-8")
    else:
        path.write_text(
            "[camera]\n"
            "index = 1\n"
            "width = 640\n"
            "height = 480\n"
            "\n"
            "[filter]\n"
            "min_cutoff = 0.4\n",
            encoding="utf-8",
        )
    assert GazeConfig.from_file(path) == original


def test_toml_integers_load_as_floats(tmp_path) -> None:
    """`beta = 1` in a config file must not leave an int in a float field."""
    path = tmp_path / "focusedgaze.toml"
    path.write_text("[filter]\nbeta = 1\n", encoding="utf-8")
    loaded = GazeConfig.from_file(path)
    assert loaded.filter.beta == 1.0
    assert type(loaded.filter.beta) is float


def test_from_file_accepts_a_string_path(tmp_path) -> None:
    """Callers should not have to build a Path first."""
    path = tmp_path / "focusedgaze.json"
    path.write_text("{}", encoding="utf-8")
    assert GazeConfig.from_file(str(path)) == GazeConfig()


# --------------------------------------------------------------------------
# Loud failure
# --------------------------------------------------------------------------


def test_an_unknown_key_is_an_error_not_a_no_op() -> None:
    """A misspelled setting must fail, not be quietly dropped.

    A silently ignored key looks exactly like one that was applied, which is the
    single most repeated failure mode in this migration.
    """
    with pytest.raises(ConfigError, match="widht"):
        GazeConfig.from_dict({"camera": {"widht": 640}})


def test_an_unknown_section_is_an_error() -> None:
    with pytest.raises(ConfigError, match="smoothing"):
        GazeConfig.from_dict({"smoothing": {"beta": 0.5}})


def test_a_section_that_is_not_a_table_is_an_error() -> None:
    with pytest.raises(ConfigError, match="camera"):
        GazeConfig.from_dict({"camera": 5})


def test_from_dict_rejects_a_non_mapping() -> None:
    with pytest.raises(ConfigError):
        GazeConfig.from_dict([("camera", {})])  # type: ignore[arg-type]


def test_an_unsupported_suffix_is_an_error(tmp_path) -> None:
    path = tmp_path / "focusedgaze.yaml"
    path.write_text("camera: {}", encoding="utf-8")
    with pytest.raises(ConfigError, match="expected .toml or .json"):
        GazeConfig.from_file(path)


def test_a_missing_file_raises_config_error(tmp_path) -> None:
    """Not FileNotFoundError, so one `except GazeError` still covers the package."""
    with pytest.raises(ConfigError):
        GazeConfig.from_file(tmp_path / "absent.toml")


@pytest.mark.parametrize(
    ("suffix", "text"),
    [(".toml", "[camera\nindex = 0\n"), (".json", "{not json}")],
)
def test_malformed_files_report_the_path(tmp_path, suffix: str, text: str) -> None:
    path = tmp_path / f"broken{suffix}"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="broken"):
        GazeConfig.from_file(path)


def test_a_file_with_a_bad_value_reports_the_field(tmp_path) -> None:
    """Validation applies to loaded files exactly as it does to kwargs."""
    path = tmp_path / "focusedgaze.toml"
    path.write_text("[filter]\nmin_cutoff = -1.0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="min_cutoff"):
        GazeConfig.from_file(path)
