"""Phase 6: the checks behind ``focusedgaze check``.

Every one of these conditions produces a *working* system that is quietly worse
than the user expects, which is why they are worth a command at all. So each is
tested in both directions: the bad state is detected, and the good state is not
reported as bad. A check that fires on everything is as useless as one that fires
on nothing.

No camera, no GPU, no model files. Everything arrives through an injection seam.
"""

from __future__ import annotations

import numpy as np
import pytest

from focusedgaze.capture import Frame, FrameSequenceSource
from focusedgaze.diagnostics import (
    BRIGHTNESS_FLOOR,
    CheckResult,
    check_calibration,
    check_camera,
    check_models,
    check_onnx_provider,
    run_checks,
    worst_status,
)
from focusedgaze.exceptions import CameraError


def _frames(values: list[int], size: int = 8) -> FrameSequenceSource:
    return FrameSequenceSource(
        [np.full((size, size, 3), v, dtype=np.uint8) for v in values], fps=30.0
    )


# ---------------------------------------------------------------------------
# ONNX provider: the silent CPU fallback.
# ---------------------------------------------------------------------------


def test_a_missing_onnxruntime_names_the_extras() -> None:
    def absent() -> list[str]:
        raise ImportError("No module named 'onnxruntime'")

    result = check_onnx_provider(absent)
    assert result.status == "fail"
    assert "directml" in result.remedy and "cuda" in result.remedy and "cpu" in result.remedy


def test_cpu_only_is_a_warning_with_the_measured_cost() -> None:
    """The whole reason this check exists: nothing else reports it.

    Inference goes from ~15 ms to ~104 ms and the system keeps working, so the
    user sees "roughly five updates a second" and reports a tracking bug.
    """
    result = check_onnx_provider(lambda: ["CPUExecutionProvider"])
    assert result.status == "warn"
    assert "104" in result.remedy and "15" in result.remedy


def test_an_accelerated_provider_is_not_reported_as_a_problem() -> None:
    """The control. A check that warned unconditionally would pass the test above."""
    result = check_onnx_provider(lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    assert result.status == "ok"
    assert "Dml" in result.summary


def test_no_providers_at_all_is_a_failure_not_a_warning() -> None:
    assert check_onnx_provider(list).status == "fail"


def test_a_broken_onnxruntime_is_reported_rather_than_crashing_the_command() -> None:
    def broken() -> list[str]:
        raise RuntimeError("DLL load failed")

    result = check_onnx_provider(broken)
    assert result.status == "fail"
    assert "DLL load failed" in result.summary


# ---------------------------------------------------------------------------
# Models: presence, and the 468-vs-478 landmark trap.
# ---------------------------------------------------------------------------


def _model_env(tmp_path) -> dict[str, str]:
    return {"FOCUSEDGAZE_MODEL_DIR": str(tmp_path)}


def test_a_missing_auto_downloadable_model_says_to_run_download_models(tmp_path) -> None:
    results = {r.name: r for r in check_models(_model_env(tmp_path))}
    landmarker = results["model:face_landmarker"]
    assert landmarker.status == "fail"
    assert "download-models" in landmarker.remedy


def test_a_missing_manual_model_carries_the_licence_instructions(tmp_path) -> None:
    """Not a generic 'run download-models': this one is never fetched."""
    results = {r.name: r for r in check_models(_model_env(tmp_path))}
    gaze = results["model:gaze_model"]
    assert gaze.status == "fail"
    assert "non-commercial" in gaze.remedy
    assert "download-models" not in gaze.remedy


def test_a_wrong_landmark_model_is_caught_by_its_digest(tmp_path) -> None:
    """The 468-point model loads fine and tracks a face. Only the bytes differ.

    `docs/troubleshooting.md` records the symptom as "distance is always wrong by
    a similar factor", because distance is derived from the iris points that only
    the refined 478-point model has. Nothing downstream raises, so the digest is
    the only place this is catchable.
    """
    (tmp_path / "face_landmarker.task").write_bytes(b"the unrefined 468-point model")
    results = {r.name: r for r in check_models(_model_env(tmp_path))}
    landmarker = results["model:face_landmarker"]
    assert landmarker.status == "fail"
    assert "478" in landmarker.remedy and "468" in landmarker.remedy
    assert landmarker.detail["expected"] != landmarker.detail["actual"]


def test_the_model_directory_override_is_reported(tmp_path) -> None:
    """Which directory is in use is the first thing to establish when files 'vanish'."""
    results = {r.name: r for r in check_models(_model_env(tmp_path))}
    assert results["model-dir"].detail["override"] is True
    assert str(tmp_path) in results["model-dir"].summary


# ---------------------------------------------------------------------------
# Calibration.
# ---------------------------------------------------------------------------


def _profile(tmp_path, name: str):
    from focusedgaze.calibration import robust_fit_samples

    rng = np.random.default_rng(7)
    pitch = rng.uniform(-0.5, 0.1, 120)
    yaw = rng.uniform(-0.4, 0.4, 120)
    samples = np.column_stack(
        [pitch, yaw, np.clip(0.5 + yaw, 0, 1), np.clip(0.5 - pitch, 0, 1)]
    )
    result = robust_fit_samples(samples, name=name)
    return result.profile.save(directory=tmp_path)


def test_no_calibration_is_a_warning_that_explains_the_symptom(tmp_path) -> None:
    result = check_calibration(str(tmp_path))
    assert result.status == "warn"
    assert "wrong place" in result.remedy


def test_a_profile_with_none_active_is_a_warning(tmp_path) -> None:
    pytest.importorskip("sklearn", reason="fitting a profile needs scikit-learn")
    _profile(tmp_path, "alice")
    result = check_calibration(str(tmp_path))
    assert result.status == "warn"
    assert "--activate" in result.remedy


def test_an_active_profile_is_reported_as_fine(tmp_path) -> None:
    """The control, so the two warnings above are not vacuous."""
    pytest.importorskip("sklearn", reason="fitting a profile needs scikit-learn")
    from focusedgaze.calibration import set_active_profile

    _profile(tmp_path, "alice")
    set_active_profile("alice", tmp_path)
    result = check_calibration(str(tmp_path))
    assert result.status == "ok"
    assert result.detail["active"] == "alice"


def test_an_active_profile_that_does_not_exist_is_a_failure(tmp_path) -> None:
    """A dangling pointer, which the library itself will not create.

    `set_active_profile` refuses a name with no file, and `delete_profile`
    clears the pointer on its way out, so both ordinary doors are already shut.
    The state is therefore built out of band, by removing the file directly:
    it is reachable by editing the profile directory by hand or by a partial
    sync, and `check` is exactly where somebody would find out.
    """
    pytest.importorskip("sklearn", reason="fitting a profile needs scikit-learn")
    from focusedgaze.calibration import set_active_profile

    _profile(tmp_path, "alice")
    _profile(tmp_path, "bob")
    set_active_profile("alice", tmp_path)
    (tmp_path / "alice.json").unlink()

    result = check_calibration(str(tmp_path))
    assert result.status == "fail"
    assert "alice" in result.summary


# ---------------------------------------------------------------------------
# Camera: opening, darkness, and the exposure ramp.
# ---------------------------------------------------------------------------


def test_a_camera_that_will_not_open_names_the_likely_causes() -> None:
    def refuse():
        raise CameraError("could not open camera index 0")

    results = check_camera(refuse)
    assert results[0].status == "fail"
    assert "another process" in results[0].remedy


def test_a_camera_that_opens_but_delivers_nothing_is_a_failure() -> None:
    class Silent:
        size = (640, 480)

        def read(self) -> Frame | None:
            return None

        def release(self) -> None:
            return

    results = check_camera(Silent)
    assert results[0].status == "fail"
    assert "no frames" in results[0].summary


def test_a_dark_room_is_reported_with_the_muted_camera_alternative() -> None:
    """Both causes produce frames, so the remedy has to name both.

    A muted camera returns frames too, which is why this cannot just say
    "turn a light on".
    """
    results = check_camera(lambda: _frames([3] * 40), settle=False)
    brightness = next(r for r in results if r.name == "camera-brightness")
    assert brightness.status == "warn"
    assert "muted" in brightness.remedy
    assert brightness.detail["mean_brightness"] < BRIGHTNESS_FLOOR


def test_a_well_lit_room_is_not_reported_as_dark() -> None:
    """The control for the darkness check."""
    results = check_camera(lambda: _frames([120] * 40), settle=False)
    brightness = next(r for r in results if r.name == "camera-brightness")
    assert brightness.status == "ok"


def test_exposure_is_not_declared_settled_while_it_is_still_climbing() -> None:
    """Audit 41.3, pinned.

    A camera ramping from 52/255 to 100/255 over several seconds moves a
    fraction of a percent between consecutive frames, so a stability test that
    compares neighbours reports "settled" almost immediately and at the wrong
    value. The comparison must span a window comparable to the ramp.

    The clock is injected so this measures the algorithm rather than the machine.
    """
    ramp = list(range(50, 130))  # a steady climb, never flat
    ticks = iter([i * 0.1 for i in range(500)])
    last = [0.0]

    def clock() -> float:
        last[0] = next(ticks)
        return last[0]

    results = check_camera(lambda: _frames(ramp), settle=True, clock=clock)
    brightness = next(r for r in results if r.name == "camera-brightness")
    assert brightness.detail["settled"] is False, (
        "a monotonically climbing image was reported as settled, which is the "
        "defect audit 41.3 describes: the stability window was shorter than the "
        "transient it was watching"
    )
    assert brightness.status == "warn"
    assert "not settled" in brightness.summary or "not reflect" in brightness.remedy


def test_a_steady_image_is_declared_settled() -> None:
    """The control: the settle check must be able to finish."""
    ticks = iter([i * 0.1 for i in range(500)])

    def clock() -> float:
        return next(ticks)

    results = check_camera(lambda: _frames([100] * 200), settle=True, clock=clock)
    brightness = next(r for r in results if r.name == "camera-brightness")
    assert brightness.detail["settled"] is True
    assert brightness.status == "ok"


def test_the_camera_is_released_even_when_a_check_finds_a_problem() -> None:
    """A diagnostic that holds the webcam makes the next thing you try fail."""
    source = _frames([2] * 10)
    check_camera(lambda: source, settle=False)
    assert source.closed


# ---------------------------------------------------------------------------
# The whole run.
# ---------------------------------------------------------------------------


def test_a_headless_run_skips_the_camera_rather_than_failing(tmp_path) -> None:
    """CI has no webcam (rule 5), and absence there is expected, not diagnostic."""
    results = run_checks(
        camera=False,
        env=_model_env(tmp_path),
        providers=lambda: ["CPUExecutionProvider"],
        profile_directory=str(tmp_path),
    )
    camera = next(r for r in results if r.name == "camera")
    assert camera.status == "ok"
    assert camera.detail["skipped"] is True


def test_the_checks_come_in_a_deliberate_order(tmp_path) -> None:
    """Interpreter first, because it frames everything below it."""
    results = run_checks(
        camera=False,
        env=_model_env(tmp_path),
        providers=lambda: ["DmlExecutionProvider"],
        profile_directory=str(tmp_path),
    )
    assert results[0].name == "interpreter"
    assert results[1].name == "onnx-provider"


@pytest.mark.parametrize(
    "statuses,expected",
    [
        (["ok", "ok"], "ok"),
        (["ok", "warn"], "warn"),
        (["warn", "fail"], "fail"),
        (["fail", "ok"], "fail"),
    ],
)
def test_worst_status_picks_the_most_severe(statuses: list[str], expected: str) -> None:
    results = [CheckResult(name=str(i), status=s, summary="") for i, s in enumerate(statuses)]  # type: ignore[arg-type]
    assert worst_status(results) == expected
