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


def test_demo_is_absent_rather_than_stubbed() -> None:
    """A subcommand that parses and apologises reads as a feature in --help.

    `serve` landed in Phase 7 and is listed. `demo` still needs the pipeline.
    """
    _, output = run()
    assert "demo" not in output


@pytest.mark.parametrize(
    "command", ["download-models", "check", "calibrate", "export-onnx", "serve"]
)
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


def test_calibrate_with_no_action_says_what_works_and_fails(tmp_path) -> None:
    """Interactive capture needs the Phase 2 pipeline and says so plainly."""
    code, output = run("calibrate", "--directory", str(tmp_path))
    assert code == 1
    assert "Phase 2" in output
    assert "--from-samples" in output and "--list" in output


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


def test_export_onnx_prints_the_manual_l2cs_install_line() -> None:
    """It cannot declare `l2cs`, so it has to say it.

    PyPI rejects direct-URL dependencies in any dependency list, extras
    included, so declaring the git URL would make the wheel unpublishable
    (DEV-1). This message is the entire mitigation, which is why the exact
    install line is asserted rather than just "some instructions appeared".
    """
    code, output = run("export-onnx")
    assert code == 1
    assert "pip install git+https://github.com/Ahmednull/L2CS-Net.git" in output
    assert "focusedgaze[export]" in output
    assert "unpublishable" in output


def test_export_onnx_names_which_dependencies_are_missing() -> None:
    code, output = run("export-onnx")
    assert code == 1
    assert "Cannot export: missing" in output
    # torch is not a runtime dependency of this package, so it is genuinely
    # absent in the dev environment and this exercises the real code path.
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
