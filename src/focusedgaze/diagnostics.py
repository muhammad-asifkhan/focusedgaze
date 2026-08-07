"""Environment checks: what ``focusedgaze check`` actually inspects.

`docs/troubleshooting.md` is mostly a list of environment problems that produce a
working-but-wrong system: a muted camera returns frames, a dark room returns
frames with no face in them, a missing ONNX provider falls back to CPU and merely
gets slow, an absent calibration still yields coordinates. **None of them raise.**
That is what makes them expensive: the symptom is always "tracking is bad", and
the cause is somewhere else entirely.

This module turns that page into a program. One command reports every one of
those conditions with the remedy attached, so a user who is confused has one
thing to run rather than a document to work through.

WHY THE LOGIC IS HERE AND NOT IN ``cli.py``
--------------------------------------------
Every check is a pure function of injected inputs: the environment mapping, a
frame-source factory, a provider lister. That is what makes them testable without
a camera, without a GPU and without model files, which is the only way they get
tested at all (rule 5). ``cli.py`` renders these results and chooses an exit code;
it contains no diagnosis of its own.

SEVERITY IS A DECISION, NOT A DESCRIPTION
------------------------------------------
``fail`` means the pipeline cannot work. ``warn`` means it will run and be worse
than you expect, which is the category this whole module exists for. ``ok``
means checked and fine. Anything that cannot be determined says so rather than
guessing, because a check that reports "probably fine" is worse than no check.
"""

from __future__ import annotations

import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np

from . import __version__
from .assets import GAZE_MODEL, asset_path, model_dir, model_dir_override, runtime_assets
from .assets.registry import sha256_file
from .calibration.profile import active_profile_name, list_profiles, profiles_dir
from .capture import Frame, FrameSource
from .config import CameraConfig
from .exceptions import CameraError, GazeError

__all__ = [
    "BRIGHTNESS_FLOOR",
    "CheckResult",
    "Status",
    "run_checks",
]

Status = Literal["ok", "warn", "fail"]

#: Mean pixel value below which a frame is too dark for face detection to be
#: reliable. MEASURED, not chosen: audit section 41 records a recording attempt
#: that produced no faces at a mean brightness of 2.8/255, and a settled indoor
#: value on the reference webcam of about 100/255. 40 sits well above the failing
#: case and well below the working one.
BRIGHTNESS_FLOOR: Final = 40.0

#: How long to let the camera's auto-exposure settle before judging brightness.
#: Audit section 41: the reference webcam sits near 52/255 for the first 3.5 s
#: and only reaches ~100/255 by 6.5 s. A check that sampled immediately would
#: report a dark room on a perfectly lit one.
_SETTLE_WINDOW_S: Final = 1.5
_SETTLE_MAX_S: Final = 6.0

#: Brightness climb, over a window of _SETTLE_WINDOW_S, below which exposure is
#: considered settled. Compared against a reading a full window earlier, never
#: against the previous frame: at ~33 ms apart a slow ramp looks flat, and a
#: stability test whose window is shorter than the transient it is watching will
#: always report stable (audit 41.3).
_SETTLE_DELTA: Final = 2.0


@dataclass(frozen=True)
class CheckResult:
    """One diagnosis.

    Args:
        name: Short stable identifier, e.g. ``"onnx-provider"``. Stable because
            a user pastes it into an issue.
        status: See the module docstring. ``warn`` is the interesting one.
        summary: One line, stating what was found rather than what was expected.
        remedy: What to do about it. Empty when there is nothing to do.
        detail: Supporting numbers, shown when asked for.
    """

    name: str
    status: Status
    summary: str
    remedy: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


#: ``() -> sequence of provider names``. Injection seam.
ProviderLister = Callable[[], Sequence[str]]
#: ``() -> FrameSource``. Injection seam; the real one opens a webcam.
SourceFactory = Callable[[], FrameSource]


def _list_providers() -> Sequence[str]:
    # Deferred: the base install declares no provider on purpose, so importing
    # this at module scope would make `focusedgaze check` the one command that
    # cannot run on the install it exists to diagnose.
    import onnxruntime  # type: ignore[import-untyped]

    providers: Sequence[str] = onnxruntime.get_available_providers()
    return providers


def _open_webcam() -> FrameSource:
    # Deferred so a --no-camera run does not pay for the capture import.
    from .capture import WebcamSource

    return WebcamSource(CameraConfig())


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_interpreter() -> CheckResult:
    """Informational. Always first, because it frames everything below it."""
    return CheckResult(
        name="interpreter",
        status="ok",
        summary=f"focusedgaze {__version__} on Python {platform.python_version()} ({sys.platform})",
        detail={
            "focusedgaze": __version__,
            "python": platform.python_version(),
            "platform": sys.platform,
            "machine": platform.machine(),
        },
    )


def check_onnx_provider(providers: ProviderLister = _list_providers) -> CheckResult:
    """Is an execution provider installed, and is it the fast one?

    The CPU fallback is the case worth catching. It is not an error and nothing
    reports it: inference simply goes from ~15 ms to ~104 ms, which a user
    experiences as "roughly five updates a second" and reports as a tracking
    bug (`docs/troubleshooting.md`).
    """
    try:
        available = list(providers())
    except ImportError:
        return CheckResult(
            name="onnx-provider",
            status="fail",
            summary="onnxruntime is not installed, so the gaze model cannot run at all",
            remedy=(
                "Install exactly one provider: pip install 'focusedgaze[directml]' on "
                "Windows with a GPU, [cuda] with an NVIDIA GPU elsewhere, or [cpu] to "
                "run without one."
            ),
        )
    except Exception as exc:  # noqa: BLE001 - a broken install can raise anything
        return CheckResult(
            name="onnx-provider",
            status="fail",
            summary=f"onnxruntime is installed but did not report its providers: {exc}",
            remedy="Reinstall the provider extra; the installation looks damaged.",
        )

    accelerated = [p for p in available if p != "CPUExecutionProvider"]
    if not available:
        return CheckResult(
            name="onnx-provider",
            status="fail",
            summary="onnxruntime reports no execution providers at all",
            remedy="Reinstall one of the provider extras: [directml], [cuda] or [cpu].",
            detail={"providers": available},
        )
    if not accelerated:
        return CheckResult(
            name="onnx-provider",
            status="warn",
            summary="only the CPU provider is available, so inference will be slow",
            remedy=(
                "Expect roughly 5 updates per second rather than 30. Install "
                "'focusedgaze[directml]' on Windows or [cuda] with an NVIDIA GPU: "
                "measured 104 ms per inference on CPU against 15 ms on DirectML."
            ),
            detail={"providers": available},
        )
    return CheckResult(
        name="onnx-provider",
        status="ok",
        summary=f"accelerated provider available: {accelerated[0]}",
        detail={"providers": available},
    )


def check_models(env: Mapping[str, str] | None = None) -> list[CheckResult]:
    """Are the model files present, and are they the right files?

    The digest check is what catches the 468-point landmark model. That one is
    listed in `docs/troubleshooting.md` as "distance is always wrong by a similar
    factor", because distance is derived from the iris points and the
    unrefined model does not have them. It loads perfectly and tracks a face, so
    nothing downstream notices.
    """
    results: list[CheckResult] = []
    override = model_dir_override(env)
    where = model_dir(env)
    results.append(
        CheckResult(
            name="model-dir",
            status="ok",
            summary=f"model directory: {where}"
            + (" (from FOCUSEDGAZE_MODEL_DIR)" if override is not None else " (managed cache)"),
            detail={"path": str(where), "override": override is not None},
        )
    )

    for asset in runtime_assets():
        path = asset_path(asset, env)
        if not path.exists():
            manual = not asset.auto_download
            results.append(
                CheckResult(
                    name=f"model:{asset.name}",
                    status="fail",
                    summary=f"{asset.filename} is missing",
                    remedy=(
                        asset.instructions
                        if manual
                        else "Run: focusedgaze download-models"
                    ),
                    detail={"path": str(path), "auto_download": asset.auto_download},
                )
            )
            continue

        if asset.sha256 is None:
            # The ONNX graph is exported locally, so its bytes differ per
            # machine and no digest can honestly be enforced. Presence is all
            # that can be checked; reference_sha256 is informational.
            results.append(
                CheckResult(
                    name=f"model:{asset.name}",
                    status="ok",
                    summary=f"{asset.filename} present ({path.stat().st_size / 1e6:.1f} MB)",
                    detail={"path": str(path), "digest_enforced": False},
                )
            )
            continue

        actual = sha256_file(path)
        if actual != asset.sha256:
            results.append(
                CheckResult(
                    name=f"model:{asset.name}",
                    status="fail",
                    summary=f"{asset.filename} is present but is not the expected file",
                    remedy=(
                        "Delete it and run: focusedgaze download-models. If this is the "
                        "face landmarker, the most likely cause is the unrefined "
                        "468-point model: distance comes from the iris points, which "
                        "only the refined 478-point model has, so it will track a face "
                        "and report distances that are wrong by a consistent factor."
                    ),
                    detail={"path": str(path), "expected": asset.sha256, "actual": actual},
                )
            )
            continue

        results.append(
            CheckResult(
                name=f"model:{asset.name}",
                status="ok",
                summary=f"{asset.filename} present and verified",
                detail={"path": str(path), "sha256": actual},
            )
        )
    return results


def check_calibration(directory: str | None = None) -> CheckResult:
    """Is there a calibration, and is one selected?

    Without one the pipeline still produces coordinates. They are simply the raw
    model output rather than anything mapped to this person's screen, which
    presents as "the cursor is in the wrong place" rather than as an error.
    """
    try:
        names = list_profiles(directory)
        active = active_profile_name(directory)
        where = profiles_dir(directory)
    except GazeError as exc:
        return CheckResult(
            name="calibration",
            status="fail",
            summary=f"could not read the profile directory: {exc}",
            remedy="Check permissions on the configuration directory.",
        )

    if not names:
        return CheckResult(
            name="calibration",
            status="warn",
            summary="no calibration profile found",
            remedy=(
                "Gaze is per person and does not transfer. Without a profile you get "
                "raw model output, not screen coordinates, which looks like a cursor "
                "that is simply in the wrong place. Run: focusedgaze calibrate"
            ),
            detail={"directory": str(where)},
        )
    if active is None:
        return CheckResult(
            name="calibration",
            status="warn",
            summary=f"{len(names)} profile(s) exist but none is active",
            remedy="Select one: focusedgaze calibrate --activate NAME",
            detail={"profiles": names, "directory": str(where)},
        )
    if active not in names:
        return CheckResult(
            name="calibration",
            status="fail",
            summary=f"the active profile {active!r} does not exist",
            remedy="Select an existing one: focusedgaze calibrate --activate NAME",
            detail={"active": active, "profiles": names},
        )
    return CheckResult(
        name="calibration",
        status="ok",
        summary=f"active profile: {active} ({len(names)} available)",
        detail={"active": active, "profiles": names, "directory": str(where)},
    )


def _mean_brightness(frame: Frame) -> float:
    return float(np.asarray(frame.image, dtype=np.float64).mean())


def check_camera(
    source_factory: SourceFactory = _open_webcam,
    *,
    settle: bool = True,
    clock: Callable[[], float] = time.monotonic,
) -> list[CheckResult]:
    """Does a camera open, does it deliver frames, and is the room lit?

    Waits for auto-exposure before judging brightness, comparing against a
    reading a full window earlier rather than against the previous frame. Audit
    section 41 records why both halves matter: a flat 1.0 s sleep captured at
    roughly half the settled brightness and the resulting dark image was
    diagnosed twice as an unlit room, and the first replacement compared
    consecutive frames and declared success at 1.4 s, part-way up the ramp.
    """
    try:
        source = source_factory()
    except CameraError as exc:
        return [
            CheckResult(
                name="camera",
                status="fail",
                summary=f"could not open a camera: {exc}",
                remedy=(
                    "Check the device exists and is not held by another process; a "
                    "crashed program can keep a webcam open. Set camera.index if you "
                    "have more than one."
                ),
            )
        ]

    results: list[CheckResult] = []
    try:
        first = source.read()
        if first is None:
            return [
                CheckResult(
                    name="camera",
                    status="fail",
                    summary="the camera opened but delivered no frames",
                    remedy="The device is present but not producing video. Replug it.",
                )
            ]
        results.append(
            CheckResult(
                name="camera",
                status="ok",
                summary=f"camera delivering {first.size[0]}x{first.size[1]} frames",
                detail={"width": first.size[0], "height": first.size[1]},
            )
        )
        results.append(_brightness_result(source, first, settle=settle, clock=clock))
    finally:
        source.release()
    return results


def _brightness_result(
    source: FrameSource,
    first: Frame,
    *,
    settle: bool,
    clock: Callable[[], float],
) -> CheckResult:
    started = clock()
    brightness = _mean_brightness(first)
    settled = not settle
    window_mark = started
    window_value = brightness
    frames = 1

    while settle and clock() - started < _SETTLE_MAX_S:
        frame = source.read()
        if frame is None:
            break
        frames += 1
        brightness = _mean_brightness(frame)
        if clock() - window_mark >= _SETTLE_WINDOW_S:
            if abs(brightness - window_value) < _SETTLE_DELTA:
                settled = True
                break
            window_mark = clock()
            window_value = brightness

    detail = {
        "mean_brightness": round(brightness, 1),
        "settled": settled,
        "frames_sampled": frames,
        "seconds": round(clock() - started, 2),
    }
    if brightness < BRIGHTNESS_FLOOR:
        return CheckResult(
            name="camera-brightness",
            status="warn",
            summary=f"frames are very dark: mean {brightness:.1f}/255",
            remedy=(
                "Face detection needs a lit subject. Turn a light on, or check the "
                "camera is not muted: a muted camera still returns frames, so this "
                "looks like a tracking failure rather than a device problem. A "
                "previous recording attempt failed here at 2.8/255."
            ),
            detail=detail,
        )
    if not settled:
        return CheckResult(
            name="camera-brightness",
            status="warn",
            summary=f"exposure had not settled after {_SETTLE_MAX_S:.0f}s (mean {brightness:.1f}/255)",
            remedy=(
                "The image is bright enough, but still changing. Readings taken now "
                "may not reflect steady-state behaviour; give the camera a few "
                "seconds before calibrating."
            ),
            detail=detail,
        )
    return CheckResult(
        name="camera-brightness",
        status="ok",
        summary=f"lighting is adequate: mean {brightness:.1f}/255",
        detail=detail,
    )


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------


def run_checks(
    *,
    camera: bool = True,
    env: Mapping[str, str] | None = None,
    providers: ProviderLister = _list_providers,
    source_factory: SourceFactory = _open_webcam,
    profile_directory: str | None = None,
    settle: bool = True,
) -> tuple[CheckResult, ...]:
    """Every check, in the order a reader should meet them.

    Args:
        camera: Probe the webcam. ``False`` for a headless or CI run, where
            "no camera" is expected rather than diagnostic.

    Ordered deliberately: interpreter, then the things that stop it working, then
    the things that make it worse than expected. A user reads until something is
    not ``ok``.
    """
    results: list[CheckResult] = [check_interpreter(), check_onnx_provider(providers)]
    results.extend(check_models(env))
    results.append(check_calibration(profile_directory))
    if camera:
        results.extend(check_camera(source_factory, settle=settle))
    else:
        results.append(
            CheckResult(
                name="camera",
                status="ok",
                summary="camera check skipped (--no-camera)",
                detail={"skipped": True},
            )
        )
    return tuple(results)


def worst_status(results: Sequence[CheckResult]) -> Status:
    """The most severe status present. Drives the exit code."""
    if any(r.status == "fail" for r in results):
        return "fail"
    if any(r.status == "warn" for r in results):
        return "warn"
    return "ok"


# Re-exported so the CLI does not import the registry directly for one constant.
GAZE_MODEL_NAME: Final = GAZE_MODEL.name
