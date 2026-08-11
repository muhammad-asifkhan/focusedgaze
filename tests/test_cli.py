"""Phase 6: every CLI command, actually run.

``main`` takes its output stream as an argument, so these call it directly and
assert on what it wrote rather than capturing file descriptors. That matters
more than it sounds: a command whose only test is "it did not raise" passes just
as happily when it prints nothing useful, and the *content* of these messages is
the deliverable. ``download-models`` refusing to fetch the gaze weights is only
correct if it says why.

Exit codes are asserted everywhere. They are the part a script depends on.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from focusedgaze.assets import FACE_LANDMARKER, GAZE_MODEL
from focusedgaze.assets.download import AssetReport
from focusedgaze.cli import main
from focusedgaze.diagnostics import CheckResult


def run(*argv: str) -> tuple[int, str]:
    """Run the CLI and return ``(exit_code, output)``."""
    buffer = io.StringIO()
    code = main(list(argv), out=buffer)
    return code, buffer.getvalue()


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def test_version_prints_the_package_version() -> None:
    from focusedgaze import __version__

    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    # argparse writes --version to stdout itself and exits; the value is what
    # matters and it is single-sourced from __init__ (D6).
    assert __version__ == "0.0.0"


def test_no_arguments_prints_help_and_succeeds() -> None:
    code, output = run()
    assert code == 0
    assert "download-models" in output and "check" in output


COMMANDS = (
    "download-models", "setup", "check", "calibrate", "export-onnx", "serve", "demo",
    "accuracy",
)


def test_the_command_set_is_complete() -> None:
    """All eight commands exist.

    `demo` and the live path of `serve` needed ``GazeEstimator``, which landed
    with Phase 2. `accuracy` is the Phase 8 port of the milestone script. Until
    each existed they were absent rather than stubbed, because a subcommand that
    parses and apologises reads as a feature in --help.

    `setup` is the eighth: the four things that must be true before the pipeline
    runs were previously discovered one failure at a time.
    """
    _, output = run()
    for command in COMMANDS:
        assert command in output, f"{command} is missing from --help"


@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_is_reachable(command: str) -> None:
    _, output = run()
    assert command in output


def test_an_unknown_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["nonsense"])
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# download-models
# ---------------------------------------------------------------------------


def test_download_models_reports_each_asset(monkeypatch, tmp_path) -> None:
    from focusedgaze import assets

    reports = (
        AssetReport(FACE_LANDMARKER, "downloaded", tmp_path / FACE_LANDMARKER.filename),
        AssetReport(GAZE_MODEL, "present", tmp_path / GAZE_MODEL.filename),
    )
    monkeypatch.setattr(assets, "ensure_all", lambda **kw: reports)
    code, output = run("download-models")
    assert code == 0
    assert "downloaded" in output and "already present" in output


def test_download_models_does_not_call_a_manual_asset_a_failure(monkeypatch) -> None:
    """The licence split working as designed must not make the command red.

    A permanently failing command on a correct installation trains people to
    ignore it, and this one carries the instructions that matter.
    """
    from focusedgaze import assets

    reports = (
        AssetReport(FACE_LANDMARKER, "present", None),
        AssetReport(GAZE_MODEL, "manual", None, GAZE_MODEL.instructions),
    )
    monkeypatch.setattr(assets, "ensure_all", lambda **kw: reports)
    code, output = run("download-models")
    assert code == 0, "a manual asset was treated as a failure"
    assert "non-commercial research" in output
    assert "not fetched automatically" in output


def test_download_models_fails_when_a_fetchable_asset_could_not_be_fetched(monkeypatch) -> None:
    """The control: a real failure must still be a failure."""
    from focusedgaze import assets

    reports = (AssetReport(FACE_LANDMARKER, "failed", None, "connection refused"),)
    monkeypatch.setattr(assets, "ensure_all", lambda **kw: reports)
    code, output = run("download-models")
    assert code == 1
    assert "connection refused" in output


def test_download_models_can_report_without_fetching(monkeypatch) -> None:
    seen: dict[str, object] = {}

    from focusedgaze import assets

    def fake(**kwargs: object) -> tuple[AssetReport, ...]:
        seen.update(kwargs)
        return ()

    monkeypatch.setattr(assets, "ensure_all", fake)
    assert run("download-models", "--no-download")[0] == 0
    assert seen["allow_download"] is False


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def _results(*statuses: str) -> tuple[CheckResult, ...]:
    return tuple(
        CheckResult(name=f"c{i}", status=s, summary=f"summary {i}", remedy=f"remedy {i}")  # type: ignore[arg-type]
        for i, s in enumerate(statuses)
    )


def test_check_runs_and_reports_every_line(monkeypatch) -> None:
    from focusedgaze import diagnostics

    monkeypatch.setattr(diagnostics, "run_checks", lambda **kw: _results("ok", "warn"))
    code, output = run("check")
    assert code == 0
    assert "[ ok ] c0" in output and "[warn] c1" in output


def test_a_warning_does_not_fail_the_command(monkeypatch) -> None:
    """"Slower than it could be" is not a failed run.

    A CI job that treated it as one would be unusable, and a user would learn to
    pass a flag that silences the thing worth reading.
    """
    from focusedgaze import diagnostics

    monkeypatch.setattr(diagnostics, "run_checks", lambda **kw: _results("ok", "warn"))
    code, output = run("check")
    assert code == 0
    assert "not as good as it should be" in output


def test_a_failure_fails_the_command(monkeypatch) -> None:
    from focusedgaze import diagnostics

    monkeypatch.setattr(diagnostics, "run_checks", lambda **kw: _results("ok", "fail"))
    code, output = run("check")
    assert code == 1
    assert "[FAIL]" in output


def test_remedies_are_shown_for_problems_and_not_for_healthy_lines(monkeypatch) -> None:
    from focusedgaze import diagnostics

    monkeypatch.setattr(diagnostics, "run_checks", lambda **kw: _results("ok", "fail"))
    _, output = run("check")
    assert "remedy 1" in output
    assert "remedy 0" not in output


def test_check_json_is_parseable(monkeypatch) -> None:
    """Machine-readable output has to survive a Path in a detail field."""
    from focusedgaze import diagnostics

    monkeypatch.setattr(
        diagnostics,
        "run_checks",
        lambda **kw: (
            CheckResult("c", "ok", "fine", detail={"path": __import__("pathlib").Path("/x")}),
        ),
    )
    code, output = run("check", "--json")
    assert code == 0
    parsed = json.loads(output)
    assert parsed[0]["name"] == "c" and parsed[0]["status"] == "ok"


def test_no_camera_is_passed_through(monkeypatch) -> None:
    seen: dict[str, object] = {}

    from focusedgaze import diagnostics

    def fake(**kwargs: object) -> tuple[CheckResult, ...]:
        seen.update(kwargs)
        return ()

    monkeypatch.setattr(diagnostics, "run_checks", fake)
    run("check", "--no-camera")
    assert seen["camera"] is False


def test_check_runs_end_to_end_without_a_camera() -> None:
    """No mocking at all: the real checks, on this machine, headless."""
    code, output = run("check", "--no-camera")
    assert code in (0, 1)
    assert "interpreter" in output
    assert "camera check skipped" in output


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


def _samples_file(tmp_path):
    rng = np.random.default_rng(11)
    pitch = rng.uniform(-0.5, 0.1, 120)
    yaw = rng.uniform(-0.4, 0.4, 120)
    rows = [
        [float(p), float(y), float(np.clip(0.5 + y, 0, 1)), float(np.clip(0.5 - p, 0, 1))]
        for p, y in zip(pitch, yaw)
    ]
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_calibrate_with_no_action_starts_a_session(monkeypatch, tmp_path) -> None:
    """`calibrate` with no flags now runs a real session rather than refusing.

    This replaces a test asserting the old "Phase 2 is not implemented yet"
    message. That message outlived the thing it described: Phase 2 shipped, and
    the actual blocker was that nothing drew a dot. The run gets as far as
    needing the gaze model and then explains which piece is missing -- the
    correct failure, and a different one from declining to try.

    FOCUSEDGAZE_MODEL_DIR is pointed at an empty directory deliberately. Without
    it this test reads the developer's real model cache, and on a machine that
    has a model it opens the webcam and runs a live calibration inside the unit
    suite.
    """
    monkeypatch.setenv("FOCUSEDGAZE_MODEL_DIR", str(tmp_path / "empty"))
    code, output = run("calibrate", "--directory", str(tmp_path))
    assert code == 1
    assert "Calibrating profile" in output
    assert "Phase 2" not in output, "the stale phase message is gone"


def test_calibrate_list_on_an_empty_directory(tmp_path) -> None:
    code, output = run("calibrate", "--list", "--directory", str(tmp_path))
    assert code == 0
    assert "No calibration profiles" in output


def test_calibrate_fits_from_samples(tmp_path) -> None:
    pytest.importorskip("sklearn", reason="fitting needs scikit-learn")
    samples = _samples_file(tmp_path)
    code, output = run(
        "calibrate", "--from-samples", str(samples), "--name", "alice",
        "--directory", str(tmp_path),
    )
    assert code == 0
    assert (tmp_path / "alice.json").is_file()
    assert "degree" in output and "outlier" in output


def test_calibrate_lists_and_marks_the_active_profile(tmp_path) -> None:
    pytest.importorskip("sklearn", reason="fitting needs scikit-learn")
    samples = _samples_file(tmp_path)
    run("calibrate", "--from-samples", str(samples), "--name", "alice", "--directory", str(tmp_path))
    run("calibrate", "--from-samples", str(samples), "--name", "bob", "--directory", str(tmp_path))

    code, output = run("calibrate", "--activate", "bob", "--directory", str(tmp_path))
    assert code == 0

    code, output = run("calibrate", "--list", "--directory", str(tmp_path))
    assert code == 0
    assert "* bob" in output
    assert "  alice" in output


def test_calibrate_deletes_and_reports_a_missing_profile(tmp_path) -> None:
    pytest.importorskip("sklearn", reason="fitting needs scikit-learn")
    samples = _samples_file(tmp_path)
    run("calibrate", "--from-samples", str(samples), "--name", "alice", "--directory", str(tmp_path))

    assert run("calibrate", "--delete", "alice", "--directory", str(tmp_path))[0] == 0
    code, output = run("calibrate", "--delete", "alice", "--directory", str(tmp_path))
    assert code == 1
    assert "No profile named" in output


def test_calibrate_migrates_a_legacy_pickle(tmp_path) -> None:
    """The committed synthetic model, through the CLI rather than the API."""
    pytest.importorskip("sklearn", reason="reading the legacy pickle needs scikit-learn")
    import pathlib

    pkl = pathlib.Path(__file__).parent / "fixtures" / "tier1" / "synthetic_calibration.pkl"
    if not pkl.is_file():
        pytest.skip(f"fixture missing: {pkl.name}")

    code, output = run(
        "calibrate", "--migrate", str(pkl), "--name", "legacy", "--directory", str(tmp_path)
    )
    assert code == 0
    assert (tmp_path / "legacy.json").is_file()
    assert "does not execute code" in output


@pytest.mark.parametrize(
    "content,expected",
    [
        ("not json at all", "not valid JSON"),
        ("[]", "holds no samples"),
        ("[[1, 2, 3]]", "not 4 numbers"),
        ('[[1, 2, 3, "x"]]', "non-number"),
    ],
)
def test_a_malformed_samples_file_is_reported_not_traced(
    tmp_path, content: str, expected: str
) -> None:
    """Every failure here is a GazeError, so one handler renders it (D7)."""
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    code, output = run("calibrate", "--from-samples", str(path), "--directory", str(tmp_path))
    assert code == 1
    assert output.startswith("error: ")
    assert expected in output


def test_a_missing_samples_file_is_reported(tmp_path) -> None:
    code, output = run(
        "calibrate", "--from-samples", str(tmp_path / "nope.json"), "--directory", str(tmp_path)
    )
    assert code == 1
    assert "could not read" in output


def test_activating_a_profile_that_does_not_exist_is_an_error(tmp_path) -> None:
    code, output = run("calibrate", "--activate", "ghost", "--directory", str(tmp_path))
    assert code == 1
    assert output.startswith("error: ")


# ---------------------------------------------------------------------------
# export-onnx
# ---------------------------------------------------------------------------


def _no_export_deps(monkeypatch) -> None:
    """Force the missing-dependency state.

    These two tests used to rely on torch genuinely being absent, on the grounds
    that it is not a runtime dependency. That held until someone installed the
    `export` extra to actually convert a model, at which point both tests started
    exercising a different branch and failing. The state under test is now stated
    rather than inherited from whatever happens to be installed.
    """
    from focusedgaze import cli

    monkeypatch.setattr(
        cli, "_missing_export_dependencies", lambda: ["torch", "onnx", "l2cs"]
    )


def test_export_onnx_prints_the_manual_l2cs_install_line(monkeypatch) -> None:
    """It cannot declare `l2cs`, so it has to say it.

    PyPI rejects direct-URL dependencies in any dependency list, extras
    included, so declaring the git URL would make the wheel unpublishable
    (DEV-1). This message is the entire mitigation, which is why the exact
    install line is asserted rather than just "some instructions appeared".
    """
    _no_export_deps(monkeypatch)
    code, output = run("export-onnx")
    assert code == 1
    assert "pip install git+https://github.com/Ahmednull/L2CS-Net.git" in output
    assert "focusedgaze[export]" in output
    assert "unpublishable" in output


def test_export_onnx_names_which_dependencies_are_missing(monkeypatch) -> None:
    _no_export_deps(monkeypatch)
    code, output = run("export-onnx")
    assert code == 1
    assert "Cannot export: missing" in output
    assert "torch" in output or "l2cs" in output


def test_export_onnx_reports_a_missing_checkpoint(monkeypatch, tmp_path) -> None:
    """Past the dependency gate, the next thing that can be wrong."""
    from focusedgaze import cli

    monkeypatch.setattr(cli, "_missing_export_dependencies", list)
    code, output = run("export-onnx", "--weights", str(tmp_path / "absent.pkl"))
    assert code == 1
    assert "No PyTorch checkpoint" in output


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def test_serve_without_a_source_says_what_to_do_rather_than_starting() -> None:
    """The live source is Phase 2. Saying so beats binding a port with no feed.

    R-10: the launcher treats "port is listening" as "ready". A server that
    opened the port with nothing behind it would report success for a system
    that can never produce a reading.
    """
    code, output = run("serve")
    assert code == 1
    assert "Phase 2" in output
    assert "--replay" in output


def test_serve_rejects_a_malformed_replay_file(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[[1, 2]]", encoding="utf-8")
    code, output = run("serve", "--replay", str(path))
    assert code == 1
    assert output.startswith("error: ")
    assert "[ok, x, y]" in output


def test_serve_rejects_an_out_of_range_port(tmp_path) -> None:
    """Rule 11 reaches the CLI: the port is validated before anything binds."""
    path = tmp_path / "r.json"
    path.write_text(json.dumps([[True, 0.5, 0.5]]), encoding="utf-8")
    code, output = run("serve", "--replay", str(path), "--port", "70000")
    assert code == 1
    assert "port must be between" in output


def test_serve_rejects_a_host_outside_the_character_class(tmp_path) -> None:
    path = tmp_path / "r.json"
    path.write_text(json.dumps([[True, 0.5, 0.5]]), encoding="utf-8")
    code, output = run("serve", "--replay", str(path), "--host", "bad host;rm")
    assert code == 1
    assert "not permitted in a host" in output


def test_a_replay_reading_keeps_null_coordinates_when_not_ok(tmp_path) -> None:
    """R-6 survives the JSON round trip: false must carry null, not 0."""
    from focusedgaze.cli import _load_readings

    path = tmp_path / "r.json"
    path.write_text(json.dumps([[False, None, None], [True, 0.25, 0.75]]), encoding="utf-8")
    readings = _load_readings(path)
    assert readings[0] == (False, None, None)
    assert readings[1] == (True, 0.25, 0.75)


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------


def test_accuracy_without_a_profile_explains_why_it_cannot_measure() -> None:
    """Raw angles are not screen positions, so there is nothing to compare."""
    code, output = run("accuracy")
    assert code == 2, "usage error, not a failed measurement"
    assert "--profile" in output
    assert "not screen positions" in output


def test_accuracy_can_re_render_a_saved_result(tmp_path) -> None:
    """The reporting half is exercised without a person in front of a camera."""
    from focusedgaze.accuracy import PointMeasurement, build_report

    W = 34.4
    errors = {
        (0.05, 0.05): 7.8, (0.5, 0.05): 5.2, (0.95, 0.05): 4.0,
        (0.05, 0.5): 2.1,  (0.5, 0.5): 1.0,  (0.95, 0.5): 1.6,
        (0.05, 0.95): 1.9, (0.5, 0.95): 3.4, (0.95, 0.95): 3.0,
    }
    report = build_report(
        [
            PointMeasurement(t, (t[0] + cm / W, t[1]), 34)
            for t, cm in errors.items()
        ],
        screen_cm=(W, 19.4), profile_digest="f" * 64, profile_name="run2",
    )
    path = tmp_path / "run2.json"
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    code, output = run("accuracy", "--from-json", str(path))
    assert code == 0
    assert "run2" in output
    assert "3.3 cm" in output
    assert "(5,5)" in output and "7.8 cm" in output


def test_accuracy_exits_nonzero_on_an_incomplete_saved_result(tmp_path) -> None:
    """A partial grid is a failure to measure, and the exit code says so."""
    from focusedgaze.accuracy import PointMeasurement, build_report

    report = build_report(
        [
            PointMeasurement((0.5, 0.5), (0.5, 0.5), 30),
            PointMeasurement((0.05, 0.05), None, 0),
        ],
        profile_digest="g" * 64,
    )
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    code, output = run("accuracy", "--from-json", str(path))
    assert code == 1
    assert "NO AVERAGE REPORTED" in output


@pytest.mark.parametrize(
    "content,expected",
    [
        ("not json", "not valid JSON"),
        ('{"points": []}', "not a focusedgaze accuracy report"),
        ('{"points": [{"target": [0.5]}]}', "not a focusedgaze accuracy report"),
    ],
)
def test_a_malformed_saved_result_is_reported_not_traced(
    tmp_path, content: str, expected: str
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    code, output = run("accuracy", "--from-json", str(path))
    assert code == 1
    assert output.startswith("error: ")
    assert expected in output


# ---------------------------------------------------------------------------
# Interactive calibration and the accuracy grid.
#
# Both drive a real collection loop against a scripted tracker and a recording
# renderer. The camera and the screen are the only things faked: the sweep path,
# the sample labelling, the coverage accounting and the fit are the shipping
# code, so these tests fail if any of them stops agreeing with the others.
# ---------------------------------------------------------------------------


class _Screen:
    """A renderer that records targets instead of painting them."""

    def __init__(self) -> None:
        self.dots: list[tuple[tuple[float, float], bool]] = []
        self.messages: list[str] = []
        self.size = (1920, 1080)
        self.closed = False

    def draw_dot(self, target, *, filled=True, progress=None):
        self.dots.append((target, filled))
        return -1

    def draw_message(self, lines, *, headline=None):
        self.messages.append(headline or "")
        return -1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True


class _Tracker:
    """A camera that reports the same well-positioned face forever.

    ``limit`` exists to cut a stream short on purpose, which is how the
    thin-coverage case is produced. Unlimited by default: the collection loop is
    bounded by its own clock, and a fake fast enough to exhaust a counter before
    that clock runs out would end the sweep early and silently.
    """

    class _Estimator:
        """Mirrors the parts of GazeEstimator that `accuracy` reads.

        `provider` goes into the report; `last_observation` carries the head-pose
        matrix it records as a diagnostic. A fake missing either produces an
        AttributeError rather than a test failure, so keep it in step with the
        real class.
        """

        provider = "FakeExecutionProvider"
        last_observation = None

    def __init__(self, result_factory, limit: int | None = None) -> None:
        self._factory = result_factory
        self._left = limit
        self.closed = False
        self.estimator = self._Estimator()

    def read(self):
        if self._left is not None:
            if self._left <= 0:
                return None
            self._left -= 1
        return self._factory()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True


def _reading(pitch: float, yaw: float, status=None, distance_cm=55.0):
    from focusedgaze.types import GazeResult, GazeStatus

    return GazeResult(
        x=None, y=None, pitch=pitch, yaw=yaw, distance_cm=distance_cm,
        status=status or GazeStatus.NOT_CALIBRATED, timestamp=0.0,
    )


def _out_of_range_reading(distance_cm=69.0):
    """What the estimator ACTUALLY returns for a badly positioned user.

    Two things here are easy to get wrong and both were:

    * The reading **carries pitch and yaw**. The estimator's gate reports rather
      than vetoes, so a check for "has an angle" accepts it.
    * While calibrating there is no profile, and the estimator returns
      ``NOT_CALIBRATED`` *before* it consults the zone -- so ``OUT_OF_RANGE``
      never appears on that path at all. Only ``distance_cm`` betrays it.

    A fake that returns ``OUT_OF_RANGE`` here would be testing a state the
    calibration path cannot produce, which is how the no-op gate passed its
    tests while letting a real run be measured at 69.4 cm.
    """
    from focusedgaze.types import GazeStatus

    return _reading(0.1, 0.2, GazeStatus.NOT_CALIBRATED, distance_cm=distance_cm)


def _install_fakes(monkeypatch, screen, tracker) -> None:
    """Swap the camera and the window, leaving every decision to real code."""
    from focusedgaze import capture
    from focusedgaze.calibration import screen as screen_module

    monkeypatch.setattr(screen_module, "DotRenderer", lambda *a, **k: screen)
    monkeypatch.setattr(capture, "WebcamGazeTracker", lambda *a, **k: tracker)


def _spread_readings():
    """Angles that vary, so the fit is not asked to invert a constant."""
    state = {"n": 0}

    def next_reading():
        state["n"] += 1
        n = state["n"]
        return _reading(0.05 * ((n % 17) - 8), 0.05 * ((n % 23) - 11))

    return next_reading


def test_calibrate_runs_a_sweep_and_writes_a_profile(monkeypatch, tmp_path) -> None:
    """The whole point: a session that ends with a loadable profile on disk."""
    from focusedgaze.calibration import CalibrationProfile

    screen = _Screen()
    _install_fakes(monkeypatch, screen, _Tracker(_spread_readings()))

    code, output = run(
        "calibrate", "--name", "tester", "--directory", str(tmp_path),
        "--seconds", "2", "--no-preflight", "--force",
    )
    assert code == 0, output
    assert "Fitted" in output
    assert screen.dots, "the sweep drew no dot"
    assert screen.closed, "the window was not destroyed"

    profile = CalibrationProfile.load("tester", directory=tmp_path)
    x, y = profile.apply(0.1, -0.2)
    assert isinstance(x, float) and isinstance(y, float)


def test_the_sweep_draws_the_dot_across_the_whole_screen(monkeypatch, tmp_path) -> None:
    """Coverage is the documented top cause of bad accuracy, and it is decided
    by where the dot actually went."""
    screen = _Screen()
    _install_fakes(monkeypatch, screen, _Tracker(_spread_readings()))
    run("calibrate", "--name", "t", "--directory", str(tmp_path),
        "--seconds", "2", "--no-preflight", "--force")

    xs = [target[0] for target, _ in screen.dots]
    ys = [target[1] for target, _ in screen.dots]
    assert min(xs) < 0.1 and max(xs) > 0.9, "the dot never reached the left/right edges"
    assert min(ys) < 0.1 and max(ys) > 0.9, "the dot never reached the top/bottom"


def test_grid_calibration_writes_a_profile(monkeypatch, tmp_path) -> None:
    """The dwell path. On hardware running the model at ~7 fps the pursuit dot
    steps rather than glides, which defeats smooth pursuit; a stationary target
    does not care about redraw rate."""
    from focusedgaze.calibration import CalibrationProfile

    screen = _Screen()
    _install_fakes(monkeypatch, screen, _Tracker(_spread_readings()))

    code, output = run(
        "calibrate", "--grid", "--name", "gridtest", "--directory", str(tmp_path),
        "--rows", "6", "--dwell", "0.001", "--sample", "0.004", "--no-preflight",
        "--force",
    )
    assert code == 0, output
    assert "Look at each of 36 dots" in output
    CalibrationProfile.load("gridtest", directory=tmp_path)


def test_grid_calibration_saves_without_force(monkeypatch, tmp_path) -> None:
    """The path a real user takes. Every other grid test passes --force, which
    skips the coverage check, so this is the one that proves an ordinary run
    reaches `profile.save` at all.

    A 6x6 grid puts 4 targets in each of the 9 regions region_of reports, so a
    complete run cannot leave one empty and must not be refused.
    """
    from focusedgaze.calibration import CalibrationProfile

    screen = _Screen()
    _install_fakes(monkeypatch, screen, _Tracker(_spread_readings()))

    code, output = run(
        "calibrate", "--grid", "--name", "noforce", "--directory", str(tmp_path),
        "--rows", "6", "--dwell", "0.001", "--sample", "0.004", "--no-preflight",
    )
    assert code == 0, output
    assert "Not fitting" not in output, output
    CalibrationProfile.load("noforce", directory=tmp_path)


def test_a_region_starved_of_samples_blocks_the_fit_and_says_which(
    monkeypatch, tmp_path
) -> None:
    """The most likely reason a real run writes no profile, so the message has to
    name the regions rather than just refusing."""
    from focusedgaze.calibration.ui import region_of

    spread = _spread_readings()

    def reading_unless_bottom_right():
        # Out of zone for a stretch of the sweep, which is what leaning back or
        # away during part of a 90-second grid actually looks like.
        return spread()

    screen = _Screen()
    # Every reading out of zone: nothing is collected anywhere.
    _install_fakes(monkeypatch, screen, _Tracker(lambda: _out_of_range_reading(69.0)))
    code, output = run(
        "calibrate", "--grid", "--name", "starved", "--directory", str(tmp_path),
        "--rows", "6", "--dwell", "0.001", "--sample", "0.004", "--no-preflight",
    )
    assert code == 1
    assert "No usable samples" in output, output
    assert not (tmp_path / "starved.json").exists()
    assert region_of(0.5, 0.5) == (1, 1)  # the grid regions are the reported ones


def test_a_failed_run_says_what_it_discarded_and_why(monkeypatch, tmp_path) -> None:
    """A refusal is only actionable if it names the cause. Both non-saving exits
    previously looked identical whether the user was absent, off centre, or three
    centimetres too far away."""
    screen = _Screen()
    _install_fakes(monkeypatch, screen, _Tracker(lambda: _out_of_range_reading(69.0)))

    code, output = run(
        "calibrate", "--grid", "--name", "why", "--directory", str(tmp_path),
        "--rows", "6", "--dwell", "0.001", "--sample", "0.004", "--no-preflight",
    )
    assert code == 1
    assert "Discarded" in output, output
    assert "too far" in output, output
    assert "69 cm" in output, output
    assert "move closer" in output, output


def test_the_discard_breakdown_counts_each_reading_once(monkeypatch, tmp_path) -> None:
    """The renderer asks the same zone question the collector does, to decide
    whether to fill the dot. Counting both doubled every figure: a real run
    reported "Discarded 180 of 954" for a sweep that saw 477 and dropped 90."""
    reads = {"n": 0}
    spread = _spread_readings()

    def counted():
        reads["n"] += 1
        # Every third reading is out of zone, so the ratio is known exactly.
        return _out_of_range_reading(69.0) if reads["n"] % 3 == 0 else spread()

    screen = _Screen()
    _install_fakes(monkeypatch, screen, _Tracker(counted))
    _, output = run(
        "calibrate", "--grid", "--name", "counted", "--directory", str(tmp_path),
        "--rows", "6", "--dwell", "0.001", "--sample", "0.004", "--no-preflight",
        "--force",
    )

    line = next(ln for ln in output.splitlines() if ln.startswith("Discarded"))
    seen = int(line.split(" of ")[1].split()[0])
    fitted = next(ln for ln in output.splitlines() if ln.startswith("Fitted"))
    kept = int(fitted.split()[1])
    dropped = int(line.split()[1])
    assert seen == kept + dropped, (
        f"{line!r} does not reconcile with {fitted!r}: "
        f"{kept} kept + {dropped} dropped != {seen} seen"
    )
    assert seen <= reads["n"], (
        f"tallied {seen} readings but the camera only delivered {reads['n']}"
    )


def test_a_clean_run_reports_no_discards(monkeypatch, tmp_path) -> None:
    """The breakdown must stay silent when there is nothing to explain."""
    screen = _Screen()
    _install_fakes(monkeypatch, screen, _Tracker(_spread_readings()))
    code, output = run(
        "calibrate", "--grid", "--name", "clean", "--directory", str(tmp_path),
        "--rows", "6", "--dwell", "0.001", "--sample", "0.004", "--no-preflight",
    )
    assert code == 0, output
    assert "Discarded" not in output, output


def test_the_grid_shows_every_target_and_marks_the_recording_phase(
    monkeypatch, tmp_path
) -> None:
    screen = _Screen()
    _install_fakes(monkeypatch, screen, _Tracker(_spread_readings()))
    run("calibrate", "--grid", "--name", "g", "--directory", str(tmp_path),
        "--rows", "6", "--dwell", "0.001", "--sample", "0.004", "--no-preflight",
        "--force")

    from focusedgaze.calibration.ui import grid_points

    assert {t for t, _ in screen.dots} == set(grid_points(rows=6, cols=6))
    assert any(filled for _, filled in screen.dots), "no recording phase shown"
    assert any(not filled for _, filled in screen.dots), "no settling phase shown"


def test_the_default_sweep_covers_the_three_bands_evenly(monkeypatch, tmp_path) -> None:
    """Regression guard for the 2:1:2 skew: a real run collected 362 top, 187
    middle, 368 bottom, and no sweep duration could have fixed it."""
    from focusedgaze.calibration.ui import region_of

    screen = _Screen()
    _install_fakes(monkeypatch, screen, _Tracker(_spread_readings()))
    run("calibrate", "--name", "b", "--directory", str(tmp_path),
        "--seconds", "2", "--no-preflight", "--force")

    bands = [sum(1 for t, _ in screen.dots if region_of(*t)[1] == r) for r in range(3)]
    assert 0 not in bands, f"a band got no dot at all: {bands}"
    assert max(bands) / min(bands) < 1.5, f"bands are skewed: {bands}"


def test_a_sweep_that_saw_nothing_is_reported_not_fitted(monkeypatch, tmp_path) -> None:
    from focusedgaze.types import GazeStatus

    screen = _Screen()
    blind = _Tracker(lambda: _reading(None, None, GazeStatus.NO_FACE))
    _install_fakes(monkeypatch, screen, blind)

    code, output = run("calibrate", "--name", "t", "--directory", str(tmp_path),
                       "--seconds", "2", "--no-preflight")
    assert code == 1
    assert "No usable samples" in output
    assert "focusedgaze check" in output


def test_the_dot_goes_hollow_while_nothing_is_being_recorded(monkeypatch, tmp_path) -> None:
    """Turns a silently wasted 45 seconds into something visible while it is
    still running."""
    from focusedgaze.types import GazeStatus

    screen = _Screen()
    _install_fakes(monkeypatch, screen,
                   _Tracker(lambda: _reading(None, None, GazeStatus.NO_FACE)))
    run("calibrate", "--name", "t", "--directory", str(tmp_path),
        "--seconds", "2", "--no-preflight")

    assert screen.dots, "the sweep drew no dot"
    assert not any(filled for _, filled in screen.dots)


def _fast_preflight(monkeypatch) -> None:
    """Shrink the gate's real-time waits so tests do not sit through them."""
    from focusedgaze import cli

    monkeypatch.setattr(cli, "_PREFLIGHT_HOLD_FRAMES", 3)
    monkeypatch.setattr(cli, "_COUNTDOWN_SECONDS", 0.01)


def test_preflight_waits_for_a_usable_position(monkeypatch, tmp_path) -> None:
    """A sweep started on someone leaning out of range costs 45 seconds to
    discover."""
    spread = _spread_readings()
    state = {"i": 0}

    def next_reading():
        if state["i"] < 5:
            state["i"] += 1
            return _out_of_range_reading()
        return spread()

    _fast_preflight(monkeypatch)
    screen = _Screen()
    _install_fakes(monkeypatch, screen, _Tracker(next_reading))
    run("calibrate", "--name", "t", "--directory", str(tmp_path),
        "--seconds", "1", "--force")

    assert any("Move" in m or "distance" in m.lower() for m in screen.messages), (
        f"the user was never told how to reposition: {screen.messages[:5]}"
    )


def test_a_reading_with_an_angle_is_not_proof_of_position(monkeypatch) -> None:
    """The bug this file previously missed, stated directly.

    The estimator's gate reports rather than vetoes, so OUT_OF_RANGE and
    OFF_CENTER readings arrive complete with pitch and yaw. A pre-flight that
    only checked for an angle therefore accepted every state it existed to
    reject, and a real accuracy run was collected at 69.4 cm against a 65 cm
    limit.
    """
    from focusedgaze.cli import _has_angle, _is_in_zone
    from focusedgaze.config import GazeConfig
    from focusedgaze.types import GazeStatus

    config = GazeConfig()
    too_far = _out_of_range_reading(69.0)
    assert _has_angle(too_far) is True, "the reading does carry an angle"
    assert _is_in_zone(too_far, config) is False, "but it is not a held position"

    for status in (GazeStatus.OUT_OF_RANGE, GazeStatus.OFF_CENTER):
        flagged = _reading(0.1, 0.2, status)
        assert _has_angle(flagged) is True
        assert _is_in_zone(flagged, config) is False, f"{status.name} passed the gate"

    assert _is_in_zone(_reading(0.1, 0.2), config) is True, "a good reading must pass"


@pytest.mark.parametrize("distance", [30.0, 44.9, 65.1, 69.4, 90.0])
def test_a_distance_outside_the_configured_range_never_counts_as_held(
    distance: float,
) -> None:
    """Checked against the bounds directly, because the calibration path reports
    NOT_CALIBRATED and never OUT_OF_RANGE -- the status cannot be trusted there."""
    from focusedgaze.cli import _is_in_zone
    from focusedgaze.config import GazeConfig

    assert _is_in_zone(_out_of_range_reading(distance), GazeConfig()) is False


def test_out_of_zone_readings_are_not_collected_as_samples(
    monkeypatch, tmp_path
) -> None:
    """A sample taken from outside the zone is a plausible angle with a confident
    label, which is undetectable after the fact. Every sweep collected them until
    the gate was fixed."""
    _fast_preflight(monkeypatch)
    screen = _Screen()
    _install_fakes(monkeypatch, screen, _Tracker(lambda: _out_of_range_reading(69.0)))

    code, output = run("calibrate", "--name", "t", "--directory", str(tmp_path),
                       "--seconds", "1", "--no-preflight")
    assert code == 1
    assert "No usable samples" in output, output


def test_an_empty_screen_region_blocks_the_fit(monkeypatch, tmp_path) -> None:
    """Fitting a polynomial with a region it never saw is the documented top
    cause of bad accuracy, so it must be refused rather than warned about."""
    screen = _Screen()
    # A sweep that dies after a handful of frames covers the top-left only.
    _install_fakes(monkeypatch, screen, _Tracker(_spread_readings(), limit=6))

    code, output = run("calibrate", "--name", "t", "--directory", str(tmp_path),
                       "--seconds", "30", "--no-preflight")
    assert code == 1
    assert "no samples at all" in output
    assert "--force" in output


def test_escape_during_a_sweep_saves_nothing(monkeypatch, tmp_path) -> None:
    from focusedgaze.calibration import list_profiles
    from focusedgaze.exceptions import CalibrationAborted

    class _Aborting(_Screen):
        def draw_dot(self, target, *, filled=True, progress=None):
            raise CalibrationAborted("stopped by the user")

    _install_fakes(monkeypatch, _Aborting(), _Tracker(_spread_readings()))
    code, output = run("calibrate", "--name", "t", "--directory", str(tmp_path),
                       "--seconds", "5", "--no-preflight")
    assert code == 1
    assert "Stopped" in output
    assert not output.startswith("error: "), "a deliberate stop is not a fault"
    assert list_profiles(tmp_path) == []


def test_accuracy_draws_each_target_and_marks_the_sampling_phase(
    monkeypatch, tmp_path
) -> None:
    """`iter_dwell_targets` yields `collecting` for exactly this, and before the
    renderer existed the command told users to look at dots it never drew."""
    from focusedgaze.accuracy import DEFAULT_TEST_POINTS
    from focusedgaze.calibration import robust_fit_samples

    samples = [
        (0.05 * ((i % 17) - 8), 0.05 * ((i % 23) - 11), (i % 3) / 2, (i // 3 % 3) / 2)
        for i in range(400)
    ]
    robust_fit_samples(samples, name="acc").profile.save(directory=tmp_path)

    screen = _Screen()
    _install_fakes(monkeypatch, screen, _Tracker(_spread_readings()))
    monkeypatch.setenv("FOCUSEDGAZE_PROFILE_DIR", str(tmp_path))

    code, output = run("accuracy", "--profile", "acc", "--dwell", "0.01",
                       "--sample", "0.01", "--no-preflight")
    assert code in (0, 1), output
    drawn = {target for target, _ in screen.dots}
    assert drawn == set(DEFAULT_TEST_POINTS), "not every test point was shown"
    assert any(filled for _, filled in screen.dots), "no sampling phase was shown"
    assert any(not filled for _, filled in screen.dots), "no dwell phase was shown"


def _saved_profile(tmp_path, name="acc"):
    """A real fitted profile on disk, for the accuracy command to load."""
    from focusedgaze.calibration import robust_fit_samples

    samples = [
        (0.05 * ((i % 17) - 8), 0.05 * ((i % 23) - 11), (i % 3) / 2, (i // 3 % 3) / 2)
        for i in range(400)
    ]
    robust_fit_samples(samples, name=name).profile.save(directory=tmp_path)


def test_accuracy_gates_on_position_before_measuring(monkeypatch, tmp_path) -> None:
    """A profile is fitted from a held position, so measuring it from a different
    one measures the posture difference as much as the profile. Two runs on this
    project showed a vertical offset of -0.246 and -0.272 -- near-identical, which
    is what a gate on one command and not the other produces."""
    from focusedgaze.types import GazeStatus

    _saved_profile(tmp_path)
    spread = _spread_readings()
    state = {"i": 0}

    def next_reading():
        if state["i"] < 4:
            state["i"] += 1
            # WITH angles. `accuracy` runs with a profile loaded, so the
            # estimator reaches its zone check and returns OUT_OF_RANGE complete
            # with pitch and yaw. A fake returning None angles here would be a
            # state the real pipeline never emits, and would exercise the
            # no-face branch instead of the one under test.
            return _reading(0.1, 0.2, GazeStatus.OUT_OF_RANGE, distance_cm=69.0)
        return spread()

    _fast_preflight(monkeypatch)
    screen = _Screen()
    _install_fakes(monkeypatch, screen, _Tracker(next_reading))
    monkeypatch.setenv("FOCUSEDGAZE_PROFILE_DIR", str(tmp_path))

    run("accuracy", "--profile", "acc", "--dwell", "0.01", "--sample", "0.01")
    assert screen.messages, "accuracy measured without ever gating on position"
    assert any("Move" in m or "distance" in m.lower() for m in screen.messages)


def test_accuracy_preflight_can_be_skipped(monkeypatch, tmp_path) -> None:
    _saved_profile(tmp_path)
    _fast_preflight(monkeypatch)
    screen = _Screen()
    _install_fakes(monkeypatch, screen, _Tracker(_spread_readings()))
    monkeypatch.setenv("FOCUSEDGAZE_PROFILE_DIR", str(tmp_path))

    run("accuracy", "--profile", "acc", "--dwell", "0.01", "--sample", "0.01",
        "--no-preflight")
    assert screen.messages == [], "--no-preflight still ran the gate"


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


def _empty_model_and_profile_dirs(monkeypatch, tmp_path) -> None:
    """Isolate setup from the developer's real cache.

    `setup` reports on the machine it runs on, so without this these tests pass
    or fail according to whether whoever runs them happens to have converted a
    model and made a profile.
    """
    monkeypatch.setenv("FOCUSEDGAZE_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("FOCUSEDGAZE_PROFILE_DIR", str(tmp_path / "profiles"))


def test_setup_reports_the_gaze_model_and_names_the_next_step(
    monkeypatch, tmp_path
) -> None:
    from focusedgaze import assets

    _empty_model_and_profile_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        assets, "ensure_all",
        lambda **kw: (AssetReport(FACE_LANDMARKER, "present", None),),
    )
    code, output = run("setup")
    assert code == 1
    assert "gaze-model" in output
    assert "setup --weights" in output
    assert "non-commercial research only" in output


def test_setup_does_not_print_the_licence_block_twice(monkeypatch) -> None:
    """download-models already prints it in full. Repeating it in one run is how
    people learn to skim past the part that matters."""
    from focusedgaze import assets

    monkeypatch.setattr(
        assets, "ensure_all",
        lambda **kw: (
            AssetReport(FACE_LANDMARKER, "present", None),
            AssetReport(GAZE_MODEL, "manual", None, detail="Gaze360 restriction..."),
        ),
    )
    _, output = run("setup")
    assert output.count("Gaze360 restriction") == 0


def test_setup_names_both_install_lines_when_conversion_is_impossible(
    monkeypatch, tmp_path
) -> None:
    """`l2cs` is git-only, so it cannot be an extra and must be a printed step."""
    from focusedgaze import assets, cli

    _empty_model_and_profile_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        assets, "ensure_all",
        lambda **kw: (AssetReport(FACE_LANDMARKER, "present", None),),
    )
    monkeypatch.setattr(cli, "_missing_export_dependencies",
                        lambda: ["torch", "onnx", "l2cs"])
    code, output = run("setup", "--weights", str(tmp_path / "weights.pkl"))
    assert code == 1
    assert "focusedgaze[export]" in output
    assert "git+https://github.com/Ahmednull/L2CS-Net.git" in output


def test_setup_offers_both_ways_to_supply_the_graph(monkeypatch, tmp_path) -> None:
    """The conversion is done once by one person; everybody else is handed the
    ONNX, so the message has to name that route too."""
    from focusedgaze import assets

    _empty_model_and_profile_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        assets, "ensure_all",
        lambda **kw: (AssetReport(FACE_LANDMARKER, "present", None),),
    )
    _, output = run("setup")
    assert "--onnx" in output
    assert "--weights" in output


def test_weights_and_onnx_are_mutually_exclusive() -> None:
    """Two answers to one question would silently pick a winner."""
    with pytest.raises(SystemExit) as excinfo:
        main(["setup", "--weights", "a.pkl", "--onnx", "b.onnx"])
    assert excinfo.value.code == 2


def test_installing_a_missing_onnx_is_reported(monkeypatch, tmp_path) -> None:
    from focusedgaze import assets

    monkeypatch.setattr(
        assets, "ensure_all",
        lambda **kw: (AssetReport(FACE_LANDMARKER, "present", None),),
    )
    monkeypatch.setenv("FOCUSEDGAZE_MODEL_DIR", str(tmp_path))
    code, output = run("setup", "--onnx", str(tmp_path / "absent.onnx"))
    assert code == 1
    assert "no file at" in output


def test_a_file_that_is_not_a_gaze_model_never_reaches_the_cache(
    monkeypatch, tmp_path
) -> None:
    """Validation happens before the copy on purpose. A wrong file must fail
    while it is still the user's file, not once it is sitting in the cache under
    the name the runtime trusts."""
    from focusedgaze import assets
    from focusedgaze.assets import GAZE_MODEL

    monkeypatch.setattr(
        assets, "ensure_all",
        lambda **kw: (AssetReport(FACE_LANDMARKER, "present", None),),
    )
    monkeypatch.setenv("FOCUSEDGAZE_MODEL_DIR", str(tmp_path))

    impostor = tmp_path / "not-a-model.onnx"
    impostor.write_bytes(b"this is not an ONNX graph")

    code, output = run("setup", "--onnx", str(impostor))
    assert code == 1
    assert "did not load as a gaze model" in output
    assert not (tmp_path / GAZE_MODEL.filename).exists(), (
        "a rejected file was copied into the cache anyway"
    )


def test_a_valid_graph_is_placed_where_the_runtime_reads(monkeypatch, tmp_path) -> None:
    """The accept path. `GazeModel` is injected rather than shipping a 91 MB
    fixture: what is being tested here is the placement, and whether a real graph
    loads is `GazeModel`'s own subject in test_core_model.py."""
    from focusedgaze import assets
    from focusedgaze.assets import GAZE_MODEL
    from focusedgaze.core import model as model_module

    class _Loads:
        """Mirrors GazeModel's real surface. The method is `predict`; an earlier
        version of this fake invented `estimate`, which let a typo in the command
        pass every test and be caught only by mypy."""

        def __init__(self, model_path=None, **kw):
            self.path = model_path

        def predict(self, crop_bgr):
            return (0.01, -0.02)

    monkeypatch.setattr(
        assets, "ensure_all",
        lambda **kw: (AssetReport(FACE_LANDMARKER, "present", None),),
    )
    monkeypatch.setattr(model_module, "GazeModel", _Loads)
    monkeypatch.setenv("FOCUSEDGAZE_MODEL_DIR", str(tmp_path))

    handed_over = tmp_path / "from-a-colleague.onnx"
    handed_over.write_bytes(b"pretend graph")

    code, output = run("setup", "--onnx", str(handed_over))
    installed = tmp_path / GAZE_MODEL.filename
    assert installed.exists(), output
    assert installed.read_bytes() == b"pretend graph"
    assert "installed to" in output
    assert code in (0, 1)  # 1 only because no calibration exists yet


def test_installing_the_graph_already_in_place_is_a_no_op(monkeypatch, tmp_path) -> None:
    """Re-running setup pointed at the cache's own copy must not try to copy a
    file onto itself."""
    from focusedgaze import assets
    from focusedgaze.assets import GAZE_MODEL

    monkeypatch.setattr(
        assets, "ensure_all",
        lambda **kw: (AssetReport(FACE_LANDMARKER, "present", None),),
    )
    monkeypatch.setenv("FOCUSEDGAZE_MODEL_DIR", str(tmp_path))
    installed = tmp_path / GAZE_MODEL.filename
    installed.write_bytes(b"placeholder")

    # Present already, so setup reports it rather than reinstalling.
    _, output = run("setup", "--onnx", str(installed))
    assert "present at" in output or "already in place" in output


def test_export_defaults_into_the_directory_the_runtime_reads(monkeypatch) -> None:
    """The old default was a bare filename, i.e. the working directory, so a
    correct export left `check` still reporting the model missing."""
    from focusedgaze.assets import GAZE_MODEL, asset_path
    from focusedgaze.cli import _build_parser

    args = _build_parser().parse_args(["export-onnx", "--weights", "w.pkl"])
    assert args.output is None
    assert asset_path(GAZE_MODEL).name == GAZE_MODEL.filename
