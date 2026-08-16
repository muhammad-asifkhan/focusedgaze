"""The ``focusedgaze`` console entry point.

Eight commands:

    download-models   fetch what may be fetched, explain what may not   (Phase 6)
    setup             fresh install to usable, in one command           (Phase 6)
    check             diagnose the environment                          (Phase 6)
    calibrate         run a session, or manage profiles                 (Phase 6)
    export-onnx       convert PyTorch L2CS weights to the ONNX graph    (Phase 6)
    serve             run the gaze WebSocket server                     (Phase 7)
    demo              print live readings from the camera               (Phase 2)
    accuracy          measure calibration accuracy across the screen    (Phase 8)

``demo`` and the live path of ``serve`` both needed ``GazeEstimator``, which
landed with Phase 2. ``serve`` now defaults to the camera through
:class:`~focusedgaze.server.LiveGazeSource`; ``--replay`` remains, because
replaying recorded readings is how the wire format is tested without one.

``calibrate`` and ``accuracy`` draw a full-screen dot through
:mod:`focusedgaze.calibration.screen`. Both were previously unrunnable for the
same reason: the collection loops existed, and nothing put a target on the
screen for the user to look at.

EXIT CODES
----------
``0`` fine, ``1`` something is wrong, ``2`` the command was used incorrectly
(argparse's own convention). ``check`` maps its worst finding to this: a ``warn``
is still ``0``, because "slower than it could be" is not a failed run and a CI
job that treated it as one would be unusable.

THIS MODULE DECIDES NOTHING
---------------------------
Parsing, rendering and exit codes only. Every diagnosis lives in
:mod:`focusedgaze.diagnostics` and every fetch in :mod:`focusedgaze.assets`,
because logic reachable only through ``main(["check"])`` is logic that gets
tested through string matching on stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, TextIO

from . import __version__
from .accuracy import AccuracyReport
from .exceptions import CalibrationAborted, GazeError
from .server import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_SEND_HZ

__all__ = ["main"]

#: Marks each check line. ASCII on purpose: this output gets pasted into issues
#: and terminals, and a Windows console in a legacy code page renders anything
#: else as replacement characters. The project has already fixed one bug of
#: exactly that shape (commit "Fix em-dash in CLI console output").
_MARK: Final[dict[str, str]] = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]"}

#: Selects the gaze backend for every command, so somebody who wants the
#: non-default one does not have to type ``--backend`` on every invocation.
#: Read by the CLI only; see :func:`_config_for` for why not by ``config.py``.
BACKEND_ENV: Final = "FOCUSEDGAZE_BACKEND"

#: The backend names the flag and the environment variable both accept. Spelled
#: out rather than imported from ``config`` so that building the parser does not
#: import the config module; ``test_cli`` pins that the two agree.
_BACKEND_NAMES: Final[tuple[str, ...]] = ("l2cs", "intel")

#: Consecutive usable frames before the sweep starts. At ~30 fps this is about a
#: second of the user actually holding the position, which is long enough to
#: reject a face that drifted through the zone and short enough not to nag.
_PREFLIGHT_HOLD_FRAMES: Final[int] = 30

#: Seconds between "you are positioned" and the dot moving, so the user can get
#: their eyes to it before it is carrying a label.
_COUNTDOWN_SECONDS: Final[float] = 3.0


def _print_reports(reports: Sequence[Any], out: TextIO) -> int:
    """Render asset reports. Returns the number of genuine failures."""
    failures = 0
    for report in reports:
        asset = report.asset
        if report.state == "present":
            print(f"{_MARK['ok']} {asset.name}: already present at {report.path}", file=out)
        elif report.state == "downloaded":
            print(f"{_MARK['ok']} {asset.name}: downloaded to {report.path}", file=out)
        elif report.state == "manual":
            # NOT a failure. This is the licence split working as designed, and
            # counting it as an error would make the command permanently red on
            # a correct installation.
            print(f"{_MARK['warn']} {asset.name}: not fetched automatically", file=out)
            for line in report.detail.splitlines():
                print(f"       {line}", file=out)
        else:
            failures += 1
            print(f"{_MARK['fail']} {asset.name}: {report.detail}", file=out)
    return failures


def _add_backend_arg(parser: argparse.ArgumentParser) -> None:
    """The ``--backend`` flag, on every command that loads or checks a model.

    Both backends are supported and neither is going away. Intel is the default
    because it is the only one a fresh install can actually reach: its weights
    are Apache-2.0 and fetched automatically, where the L2CS weights may not be
    distributed by this package at all. It also measured faster and slightly
    more accurate here -- 2.0 ms against 142, 1.43 cm against 1.96.

    ``default=None`` rather than the name itself, so that "not given" stays
    distinguishable from "given, and happens to match the default". Two things
    need that distinction: ``_cmd_calibrate``'s ``--from-samples`` branch, and
    the :data:`BACKEND_ENV` fallback in :func:`_config_for`, which must not be
    overridden by a default argparse invented.
    """
    parser.add_argument(
        "--backend",
        choices=_BACKEND_NAMES,
        default=None,
        help=f"which gaze model to use. intel (default) is Apache-2.0, ~70x faster "
             f"here, and fetched automatically; l2cs is the original and needs a "
             f"model you supply yourself. Set {BACKEND_ENV} to change the default "
             f"without typing this flag every time",
    )


def _config_for(args: argparse.Namespace, env: Mapping[str, str] | None = None) -> Any:
    """A :class:`GazeConfig` with the command line's backend applied.

    Precedence, highest first: ``--backend``, then :data:`BACKEND_ENV`, then the
    declared default in :class:`~focusedgaze.config.ModelConfig`. The flag wins
    because it is the more specific instruction: somebody who typed a backend on
    this command line means it for this command, whatever their shell says.

    The environment is read **here and not in ``config.py``**, which reads no
    environment at all. ``GazeConfig()`` is a library constructor, and a library
    whose declared default silently changes with an ambient variable is the
    "located rather than declared" trap ``assets/registry.py`` records this
    project being bitten by three times. A CLI is the opposite case: configuring
    it through the environment is what an environment is for, and the effect
    stops at the process the user launched.

    Raises:
        ConfigError: if :data:`BACKEND_ENV` holds a name that is not a backend.
            Named explicitly, because a bad value there is invisible on the
            command line and the failure would otherwise be unattributable.
    """
    import os
    from dataclasses import replace

    from .config import GazeConfig
    from .exceptions import ConfigError

    config = GazeConfig()
    backend = getattr(args, "backend", None)
    if not backend:
        source = env if env is not None else os.environ
        # An empty or whitespace-only value is what an unset shell variable
        # expands to in a script. Treated as unset, per the same reasoning in
        # `registry.model_dir_override`.
        from_env = source.get(BACKEND_ENV, "").strip()
        if from_env:
            if from_env not in _BACKEND_NAMES:
                raise ConfigError(
                    f"{BACKEND_ENV}={from_env!r} is not a gaze backend; "
                    f"expected one of {', '.join(_BACKEND_NAMES)}"
                )
            backend = from_env
    if backend:
        config = replace(config, model=replace(config.model, backend=backend))
    return config


def _backend_of(args: argparse.Namespace) -> str:
    """The backend name a command should report and fetch assets for."""
    return str(_config_for(args).model.backend)


def _profile_backend_error(args: argparse.Namespace, name: str | None) -> str | None:
    """The message to print instead of running, or ``None`` to carry on.

    ``GazeEstimator`` refuses a mismatched profile on its own, and that check is
    the one that matters because it also covers library callers. This runs first
    so the CLI can say more than the pure layer honestly can: it knows where
    profiles live, so it can name one that would work instead of only naming the
    problem. Returns the text rather than printing it, so the caller keeps
    control of the exit code.
    """
    if not name:
        return None
    from .calibration.profile import CalibrationProfile, profiles_for_backend
    from .exceptions import CalibrationError

    backend = _backend_of(args)
    try:
        profile = CalibrationProfile.load(name, directory=getattr(args, "directory", None))
    except CalibrationError:
        # Not this function's business: loading it again in a moment will raise
        # with the message that actually describes the problem.
        return None

    complaint = profile.backend_complaint(backend)
    if complaint is None or complaint[0] != "fail":
        return None
    message = complaint[1]
    usable = [
        n
        for n in profiles_for_backend(backend, getattr(args, "directory", None))
        if n != name
    ]
    if usable:
        message += (
            f"\n\nAlready calibrated for {backend}: {', '.join(usable)}.\n"
            f"    focusedgaze {args.command} --profile {usable[0]}"
        )
    return message


def _cmd_download_models(args: argparse.Namespace, out: TextIO) -> int:
    from .assets import ensure_all

    reports = ensure_all(
        allow_download=not args.no_download, backend=_backend_of(args)
    )
    failures = _print_reports(reports, out)
    manual = [r for r in reports if r.state == "manual"]
    if manual:
        print(
            "\nOne or more models must be installed by hand. That is deliberate: "
            "they derive from a dataset restricted to non-commercial research, so "
            "this package does not distribute, mirror or fetch them.",
            file=out,
        )
    return 1 if failures else 0


def _cmd_check(args: argparse.Namespace, out: TextIO) -> int:
    from .diagnostics import run_checks, worst_status

    results = run_checks(camera=not args.no_camera, backend=_backend_of(args))

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "name": r.name,
                        "status": r.status,
                        "summary": r.summary,
                        "remedy": r.remedy,
                        "detail": dict(r.detail),
                    }
                    for r in results
                ],
                indent=2,
                default=str,
            ),
            file=out,
        )
    else:
        for result in results:
            print(f"{_MARK[result.status]} {result.name}: {result.summary}", file=out)
            if result.remedy and result.status != "ok":
                for line in _wrap(result.remedy):
                    print(f"       {line}", file=out)
        worst = worst_status(results)
        if worst == "ok":
            print("\nEverything checked out.", file=out)
        elif worst == "warn":
            print(
                "\nUsable, but not as good as it should be. See the notes above.",
                file=out,
            )
        else:
            print("\nSomething is wrong. Fix the [FAIL] lines first.", file=out)

    # A warning is not a failure: it means slower or less accurate, not broken.
    return 1 if worst_status(results) == "fail" else 0


def _cmd_calibrate(args: argparse.Namespace, out: TextIO) -> int:
    from .calibration import (
        CalibrationProfile,
        active_profile_name,
        delete_profile,
        list_profiles,
        migrate_pickle,
        robust_fit_samples,
        set_active_profile,
    )

    if args.list:
        names = list_profiles(args.directory)
        active = active_profile_name(args.directory)
        if not names:
            print("No calibration profiles. Create one with --from-samples.", file=out)
            return 0
        for name in names:
            print(f"{'*' if name == active else ' '} {name}", file=out)
        return 0

    if args.activate:
        set_active_profile(args.activate, args.directory)
        print(f"Active profile is now {args.activate!r}.", file=out)
        return 0

    if args.delete:
        removed = delete_profile(args.delete, args.directory)
        print(
            f"Deleted {args.delete!r}." if removed else f"No profile named {args.delete!r}.",
            file=out,
        )
        return 0 if removed else 1

    if args.migrate:
        migrated = migrate_pickle(args.migrate, name=args.name)
        path = migrated.save(directory=args.directory)
        print(f"Migrated {args.migrate} -> {path}", file=out)
        print(
            "The pickle is not needed any more. Keep the JSON; it loads without "
            "scikit-learn and does not execute code.",
            file=out,
        )
        return 0

    if args.from_samples:
        samples = _load_samples(args.from_samples)
        # getattr, not _backend_of: these samples were collected by something
        # else, at some earlier time, and this process has no way to know which
        # model produced the angles in the file. Stamping the default onto them
        # would manufacture a provenance record out of nothing, which is worse
        # than leaving it unknown -- an unknown backend is warned about, a wrong
        # one is trusted. The user asserts it with --backend or not at all.
        result = robust_fit_samples(
            samples, name=args.name, backend=getattr(args, "backend", None)
        )
        profile: CalibrationProfile = result.profile
        path = profile.save(directory=args.directory)
        print(f"Fitted {len(samples)} samples -> {path}", file=out)
        print(
            f"       degree {profile.degree}, "
            f"{result.n_dropped} outlier(s) dropped, "
            f"fit error {_fmt(profile.fit_error)}, "
            f"held-out error {_fmt(profile.validation_error)}",
            file=out,
        )
        return 0

    # Interactive capture: a camera, a screen, and a person following the dot.
    return _run_interactive_calibration(args, out)


def _run_interactive_calibration(args: argparse.Namespace, out: TextIO) -> int:
    """Position the user, sweep the dot, report coverage, fit, save.

    Every step but the drawing already existed. The sweep path, the sample
    collection, the coverage accounting and the robust fit are
    :mod:`focusedgaze.calibration.ui` and
    :mod:`focusedgaze.calibration.fitter`, both pure and both covered in CI; this
    function is the wiring plus the one thing that needs a display.
    """
    from .calibration import active_profile_name, robust_fit_samples, set_active_profile
    from .calibration.screen import DotRenderer
    from .calibration.ui import (
        collect_dwell_samples,
        collect_pursuit_samples,
        grid_points,
        pursuit_path,
        summarise_coverage,
    )
    from .capture import WebcamGazeTracker

    config = _config_for(args)

    print(
        f"Calibrating profile {args.name!r} on the {config.model.backend} backend.",
        file=out,
    )
    if args.grid:
        points = grid_points(rows=args.rows, cols=args.rows)
        total = len(points) * (args.dwell + args.sample)
        print(
            f"Look at each of {len(points)} dots in turn, about "
            f"{total:.0f} seconds in total. The dot is hollow while you settle on "
            "it and solid while you are being measured.",
            file=out,
        )
    else:
        path = pursuit_path(rows=args.rows, seconds=args.seconds)
        print(
            f"Follow the dot with your eyes for {args.seconds:.0f} seconds. "
            "Keep your head still and move only your eyes.",
            file=out,
        )
    print("Escape or q stops at any point.\n", file=out)

    # The camera is opened before the window so its warm-up happens against a
    # terminal the user can read, not a black full-screen canvas that looks hung.
    tracker = WebcamGazeTracker(profile=None, config=config)
    samples: list[tuple[float, float, float, float]] = []
    with tracker, DotRenderer() as screen:
        if not args.no_preflight:
            _run_preflight(tracker, screen, config)

        # The same predicate the gate uses. A sample collected from outside the
        # zone teaches the polynomial a mapping that does not hold there, and it
        # is not detectable afterwards: it is a plausible angle with a confident
        # label. `filled` shows the user which frames are actually counting.
        #
        # Tallied, because "no samples" and "not fitting" are the two ways this
        # command declines to write a profile and neither previously said what
        # was actually being thrown away. A run that discards everything for
        # being 3 cm too far away should say so, not just refuse.
        tally = _ZoneTally(config)

        # Two callers, one of which must NOT tally. `tally.accepts` goes to the
        # collector, which asks once per candidate reading; the renderer asks the
        # same question again to decide whether to fill the dot, and counting
        # that too doubles every figure in the breakdown. Measured on a real run:
        # "Discarded 180 of 954" for a sweep that saw 477 readings and dropped 90.
        def _shown_as_counting(result: Any) -> bool:
            return _is_in_zone(result, config)

        if args.grid:
            shown: list[tuple[float, float]] = []

            def _on_dwell(
                target: tuple[float, float], collecting: bool, result: Any
            ) -> None:
                if not shown or shown[-1] != target:
                    shown.append(target)
                screen.draw_dot(
                    target,
                    filled=collecting and _shown_as_counting(result),
                    progress=(len(shown) - 1) / max(len(points) - 1, 1),
                )

            samples = collect_dwell_samples(
                tracker, points,
                dwell_seconds=args.dwell,
                sample_seconds=args.sample,
                on_frame=_on_dwell,
                usable=tally.accepts,
            )
        else:
            started = time.monotonic()

            def _on_frame(target: tuple[float, float], result: Any) -> None:
                # Hollow means "this frame is not being recorded". It costs
                # nothing -- `filled` already exists for the accuracy grid -- and
                # it turns a silently wasted sweep into something the user can
                # see and correct while it is still running.
                screen.draw_dot(
                    target,
                    filled=_shown_as_counting(result),
                    progress=(time.monotonic() - started) / max(path.seconds, 1e-9),
                )

            samples = collect_pursuit_samples(
                tracker, path, on_frame=_on_frame, usable=tally.accepts
            )

    coverage = summarise_coverage(samples)
    print(file=out)
    for line in coverage.lines():
        print(line, file=out)
    for line in tally.lines():
        print(line, file=out)
    print(file=out)

    if not samples:
        print(
            "No usable samples. Nothing the camera saw was inside the positioning "
            "zone for the whole sweep.\n"
            "The discard breakdown above says which. Run `focusedgaze check` to test "
            "the camera and the lighting.",
            file=out,
        )
        return 1

    if coverage.empty_regions and not args.force:
        cells = ", ".join(f"({c},{r})" for c, r in coverage.empty_regions)
        print(
            f"Not fitting: {len(coverage.empty_regions)} screen region(s) got no "
            f"samples at all -- {cells}.\n"
            "\n"
            "A polynomial fitted without them extrapolates into those regions, and "
            "that is the single largest cause of bad accuracy in this system. Run it "
            "again and make sure your eyes actually reach the edges and corners.\n"
            "\n"
            "Use --force to fit anyway.",
            file=out,
        )
        return 1

    fit = robust_fit_samples(
        samples, name=args.name, screen_size=screen.size,
        distance_cm=_median(tally.distances),
        backend=_backend_of(args),
    )
    profile = fit.profile
    saved = profile.save(directory=args.directory)
    print(f"Fitted {len(samples)} samples -> {saved}", file=out)
    print(
        f"       degree {profile.degree}, "
        f"{fit.n_dropped} outlier(s) dropped, "
        f"fit error {_fmt(profile.fit_error)}, "
        f"held-out error {_fmt(profile.validation_error)}",
        file=out,
    )
    if profile.distance_cm:
        print(
            f"       collected at {profile.distance_cm:.0f} cm. This profile is only "
            f"accurate near that distance;\n"
            f"       sitting further away makes it under-reach toward the middle of "
            f"the screen.",
            file=out,
        )

    if active_profile_name(args.directory) is None:
        set_active_profile(args.name, args.directory)
        print(f"       {args.name!r} is now the active profile.", file=out)

    print(
        f"\nCheck it: focusedgaze accuracy --profile {args.name}",
        file=out,
    )
    return 0


class _ZoneTally:
    """Counts why readings were discarded during a calibration run.

    A refusal to fit is only actionable if it says what was thrown away. Both
    non-saving exits -- "no usable samples" and an empty screen region -- looked
    identical whether the user was absent, off centre, or three centimetres
    beyond the distance limit. This turns them into a number and a remedy.
    """

    __slots__ = (
        "_config", "_far", "_near", "distances", "kept", "no_face", "off_centre", "seen",
    )

    def __init__(self, config: Any) -> None:
        self._config = config
        self.seen = 0
        self.kept = 0
        self.no_face = 0
        self.off_centre = 0
        self._near: list[float] = []
        self._far: list[float] = []
        #: Distances of the readings that were KEPT. The profile records their
        #: median, because that is the distance the polynomial is actually valid
        #: at -- recording only the rejected ones, as an earlier version did,
        #: left the calibration distance inferrable but not known.
        self.distances: list[float] = []

    def accepts(self, result: Any) -> bool:
        """The predicate the collector calls, recording as it decides."""
        from .types import GazeStatus

        self.seen += 1
        if _is_in_zone(result, self._config):
            self.kept += 1
            distance = getattr(result, "distance_cm", None)
            if distance is not None:
                self.distances.append(float(distance))
            return True

        if not _has_angle(result):
            self.no_face += 1
            return False
        distance = getattr(result, "distance_cm", None)
        bounds = self._config.positioning
        if distance is not None and distance < bounds.min_distance_cm:
            self._near.append(distance)
        elif distance is not None and distance > bounds.max_distance_cm:
            self._far.append(distance)
        elif getattr(result, "status", None) is GazeStatus.OFF_CENTER:
            self.off_centre += 1
        else:
            self.off_centre += 1
        return False

    def lines(self) -> list[str]:
        """Human-readable breakdown, empty when nothing was discarded."""
        dropped = self.seen - self.kept
        if dropped <= 0:
            return []
        bounds = self._config.positioning
        out = [f"Discarded {dropped} of {self.seen} readings:"]
        if self.no_face:
            out.append(f"  {self.no_face:>5}  no face detected")
        for label, values, remedy in (
            ("too close", self._near, "move back"),
            ("too far", self._far, "move closer"),
        ):
            if values:
                middle = sorted(values)[len(values) // 2]
                out.append(
                    f"  {len(values):>5}  {label} (median {middle:.0f} cm; "
                    f"this needs {bounds.min_distance_cm:.0f}-"
                    f"{bounds.max_distance_cm:.0f} cm, so {remedy})"
                )
        if self.off_centre:
            out.append(f"  {self.off_centre:>5}  off centre")
        if dropped > self.kept:
            out.append(
                "  Most of the run was discarded. Fix the reason above and re-run: "
                "a profile fitted from what is left will be worse than no profile, "
                "because it looks like it worked."
            )
        return out


def _has_angle(result: Any) -> bool:
    """Whether a reading carries a gaze angle at all.

    ``NOT_CALIBRATED`` counts, and that is the subtlety worth stating: during
    calibration there is by definition no profile yet, so a perfectly positioned
    user reports ``NOT_CALIBRATED`` and never ``OK``. Gating on ``OK`` would wait
    forever for a state that cannot occur until the thing being created already
    exists.

    This is **not** enough to decide the user is positioned. See
    :func:`_is_in_zone`.
    """
    return (
        result is not None
        and getattr(result, "pitch", None) is not None
        and getattr(result, "yaw", None) is not None
    )


def _is_in_zone(result: Any, config: Any) -> bool:
    """Whether a reading was taken from a position the calibration is valid at.

    TWO TRAPS, BOTH OF WHICH THIS FUNCTION EXISTED WITHOUT AND WAS WRONG
    --------------------------------------------------------------------
    1. **An out-of-range reading still carries pitch and yaw.** The estimator's
       gate reports, it does not veto: ``OUT_OF_RANGE`` and ``OFF_CENTER``
       results are returned complete with angles. So a check for "has an angle"
       passes every state it exists to reject, and the pre-flight became a test
       for "is a face visible". Measured consequence: an accuracy run collected
       at 69.4 cm, 4.4 cm beyond the 65 cm limit the gate advertises.

    2. **The calibration path never reports a zone status at all.** With no
       profile the estimator returns ``NOT_CALIBRATED`` *before* it consults the
       zone, so ``OUT_OF_RANGE`` cannot appear while calibrating -- which is
       precisely when the gate matters most. ``distance_cm`` survives that path,
       so it is checked directly against the configured bounds rather than
       trusting a status that is not populated.
    """
    from .types import GazeStatus

    if not _has_angle(result):
        return False
    if getattr(result, "status", None) in (
        GazeStatus.NO_FACE, GazeStatus.OUT_OF_RANGE, GazeStatus.OFF_CENTER
    ):
        return False
    distance = getattr(result, "distance_cm", None)
    if distance is not None:
        bounds = config.positioning
        if not (bounds.min_distance_cm <= distance <= bounds.max_distance_cm):
            return False
    return True


def _run_preflight(tracker: Any, screen: Any, config: Any) -> None:
    """Hold until the user is positioned, then count down into the sweep.

    A sweep collected from someone leaning out of range produces samples the fit
    cannot use, and the cost is discovered 45 seconds later. This is the cheap
    check that stops that happening.

    Raises:
        CalibrationAborted: The user pressed Escape, or the camera stopped.
    """
    from .exceptions import CalibrationAborted

    steady = 0
    while steady < _PREFLIGHT_HOLD_FRAMES:
        result = tracker.read()
        if result is None:
            raise CalibrationAborted("the camera stopped delivering frames")
        if _is_in_zone(result, config):
            steady += 1
            remaining = _PREFLIGHT_HOLD_FRAMES - steady
            # The distance is shown even when it is acceptable, which is the
            # point: the zone is 20 cm deep, and a profile is only accurate near
            # the distance it was collected at. Someone who can see they are at
            # 47 cm can choose to sit where they normally sit instead of merely
            # somewhere legal. A run calibrated at ~47 cm and used at 64 cm lost
            # a third of its reach.
            distance = getattr(result, "distance_cm", None)
            body = [f"Hold still... {remaining}"]
            if distance is not None:
                body.append(f"{distance:.0f} cm -- sit where you normally would")
            screen.draw_message(body, headline="Good position")
        else:
            steady = 0
            headline, body = _preflight_guidance(result, config)
            screen.draw_message(body, headline=headline)

    countdown_started = time.monotonic()
    while True:
        elapsed = time.monotonic() - countdown_started
        if elapsed >= _COUNTDOWN_SECONDS:
            break
        left = int(_COUNTDOWN_SECONDS - elapsed) + 1
        screen.draw_message(["Follow the dot with your eyes"], headline=str(left))


def _preflight_guidance(result: Any, config: Any) -> tuple[str, list[str]]:
    """What to tell the user about a reading that is not usable yet."""
    from .types import GazeStatus

    status = getattr(result, "status", None)
    if status is GazeStatus.NO_FACE or not _has_angle(result):
        return (
            "No face detected",
            ["Sit in front of the camera.", "Check the lens is not covered and the room is lit."],
        )

    distance = getattr(result, "distance_cm", None)
    low = config.positioning.min_distance_cm
    high = config.positioning.max_distance_cm
    out_of_range = status is GazeStatus.OUT_OF_RANGE or (
        # Checked independently of the status, because the calibration path
        # reports NOT_CALIBRATED and never OUT_OF_RANGE. Without this the user
        # gets "Getting a usable reading..." forever while sitting too far away.
        distance is not None and not (low <= distance <= high)
    )
    if out_of_range:
        if distance is not None:
            direction = "Move back" if distance < low else "Move closer"
            return (
                direction,
                [f"You are at {distance:.0f} cm.", f"This needs {low:.0f} to {high:.0f} cm."],
            )
        return ("Adjust your distance", [f"This needs {low:.0f} to {high:.0f} cm."])

    if status is GazeStatus.OFF_CENTER:
        return (
            "Move to the centre",
            ["Line your face up with the middle of the camera's view."],
        )
    return ("Waiting for the camera", ["Getting a usable reading..."])


def _cmd_accuracy(args: argparse.Namespace, out: TextIO) -> int:
    """Measure where the model thinks you are looking against where you were told.

    The port of `milestone6_test_accuracy.py`, with the three requirements audit
    section 50 derived from running the original. The measurement itself needs a
    person; ``--from-json`` re-renders a saved result, which is how the reporting
    is exercised without one.
    """
    from .accuracy import (
        DEFAULT_TEST_POINTS,
        PointMeasurement,
        build_report,
        profile_digest,
        render,
    )

    if args.from_json:
        report = _report_from_json(args.from_json)
        for line in render(report):
            print(line, file=out)
        return 0 if report.complete else 1

    from .calibration import CalibrationProfile
    from .calibration.screen import DotRenderer
    from .calibration.ui import iter_dwell_targets, measure_drift, median_angle
    from .capture import WebcamGazeTracker

    if not args.profile:
        print(
            "focusedgaze accuracy needs a calibration to measure.\n"
            "\n"
            "    focusedgaze accuracy --profile alice\n"
            "\n"
            "It compares where you are told to look against where that profile says\n"
            "you looked, at nine points across the screen. Without a profile there is\n"
            "nothing to measure: the raw angles are not screen positions.",
            file=out,
        )
        return 2

    from dataclasses import replace

    config = _config_for(args)
    if args.compensate_distance:
        config = replace(
            config,
            positioning=replace(config.positioning, compensate_distance=True),
        )
    mismatch = _profile_backend_error(args, args.profile)
    if mismatch is not None:
        print(f"error: {mismatch}", file=out)
        return 1

    profile = CalibrationProfile.load(args.profile)
    screen = (args.screen_width_cm, args.screen_height_cm)
    tracker = WebcamGazeTracker(profile=profile, config=config)

    print(
        f"Measuring {len(DEFAULT_TEST_POINTS)} points against profile "
        f"{args.profile!r} on a {screen[0]:.1f} x {screen[1]:.1f} cm screen.",
        file=out,
    )
    print(
        "Look at each dot as it appears. It is hollow while you settle on it and "
        "solid while you are being measured.",
        file=out,
    )
    if profile.distance_cm:
        print(
            f"This profile was calibrated at {profile.distance_cm:.0f} cm. Sit at "
            "roughly that distance,\n"
            "or pass --compensate-distance to correct for the difference.",
            file=out,
        )
    print("Escape, q or Ctrl+C aborts.\n", file=out)

    from .core.headpose import head_angles

    collected: dict[tuple[float, float], list[tuple[float, float]]] = {
        point: [] for point in DEFAULT_TEST_POINTS
    }
    # Diagnostics, alongside the readings rather than inside them: nothing in the
    # mapping consumes head pose, and the point of recording it is to find out
    # whether it should.
    poses: dict[tuple[float, float], list[tuple[float, float, float]]] = {
        point: [] for point in DEFAULT_TEST_POINTS
    }
    distances: dict[tuple[float, float], list[float]] = {
        point: [] for point in DEFAULT_TEST_POINTS
    }
    try:
        # NOT named `screen`: that name is already bound above to the display's
        # size in centimetres, and the report's error arithmetic reads it.
        with tracker, DotRenderer(title="focusedgaze accuracy") as display:
            # The same gate `calibrate` runs, for the same reason and now for a
            # second one. A profile is fitted from a held position, so measuring
            # it from a different one measures the difference between the two
            # postures as much as the profile. Two runs on this project showed a
            # vertical offset of -0.246 and -0.272 -- near-identical, so not a
            # user shifting about but a reproducible consequence of gating one
            # command and not the other.
            if not args.no_preflight:
                _run_preflight(tracker, display, config)
            for target, collecting in iter_dwell_targets(
                DEFAULT_TEST_POINTS,
                dwell_seconds=args.dwell,
                sample_seconds=args.sample,
            ):
                # Drawn before the reading is taken, so the dot the user is being
                # measured against is already on the screen. `collecting` is what
                # `iter_dwell_targets` yields the flag for: hollow through the
                # dwell, solid through the sample window, so nobody has to guess
                # which phase they are in.
                display.draw_dot(target, filled=collecting)
                result = tracker.read()
                if result is None:
                    break
                if collecting and result.pitch is not None and result.yaw is not None:
                    collected[target].append((result.pitch, result.yaw))
                    observation = tracker.estimator.last_observation
                    pose = (
                        head_angles(observation.transform_matrix)
                        if observation is not None else None
                    )
                    if pose is not None:
                        poses[target].append((pose.pitch, pose.yaw, pose.roll))
                    if result.distance_cm is not None:
                        distances[target].append(result.distance_cm)
            drift = measure_drift(collected[(0.5, 0.5)], profile) if args.drift else (0.0, 0.0)
    except (KeyboardInterrupt, CalibrationAborted):
        print("\nAborted. No report: a partial grid is not a measurement.", file=out)
        return 1

    measurements = []
    for target in DEFAULT_TEST_POINTS:
        readings = collected[target]
        angle = median_angle(readings)
        if angle is None:
            measurements.append(PointMeasurement(target, None, 0))
            continue
        # apply_raw, not apply: `apply` clamps to the screen before returning, so
        # reading its output as "raw" records a value that has already been
        # truncated and reveals nothing about a prediction that landed outside.
        x, y = profile.apply_raw(angle[0], angle[1])
        raw = (x - drift[0], y - drift[1])
        measurements.append(
            PointMeasurement(
                target=target,
                # Clamped, because an application cannot put a cursor off the
                # screen and that is the error a user actually experiences.
                predicted=(min(max(raw[0], 0.0), 1.0), min(max(raw[1], 0.0), 1.0)),
                n_samples=len(readings),
                # Unclamped, because clamping a prediction that landed off the
                # screen moves it toward the target and hides how far out it was.
                raw=raw,
                head_pose=_median_pose(poses[target]),
                distance_cm=_median(distances[target]),
            )
        )

    report = build_report(
        measurements,
        screen_cm=screen,
        profile_digest=profile_digest(profile),
        profile_name=args.profile,
        drift_offset=drift,
        provider=tracker.estimator.provider,
    )
    for line in render(report):
        print(line, file=out)

    if args.save:
        # NOT named `target`: that name is bound to a screen point in the loop
        # above, and reusing it here would read as the same thing.
        destination = Path(args.save)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nSaved to {destination}", file=out)

    return 0 if report.complete else 1


def _median(values: Sequence[float]) -> float | None:
    """Middle value, or ``None`` for nothing.

    Median rather than mean for the same reason the gaze angles use one: a blink
    or a moment of bad tracking is an outlier, not a contribution.
    """
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _median_pose(
    poses: Sequence[tuple[float, float, float]],
) -> tuple[float, float, float] | None:
    """Per-axis median of head angles. ``None`` when nothing was recorded."""
    if not poses:
        return None
    axes = [_median([p[i] for p in poses]) for i in range(3)]
    if any(a is None for a in axes):
        return None
    return (float(axes[0]), float(axes[1]), float(axes[2]))  # type: ignore[arg-type]


def _report_from_json(path: str | Path) -> AccuracyReport:
    """Rebuild a report from a saved result, for re-rendering and comparison."""
    from .accuracy import PointMeasurement, build_report

    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GazeError(f"could not read {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GazeError(f"{source} is not valid JSON: {exc}") from exc

    try:
        measurements = [
            PointMeasurement(
                target=(float(p["target"][0]), float(p["target"][1])),
                predicted=None if p["predicted"] is None
                else (float(p["predicted"][0]), float(p["predicted"][1])),
                n_samples=int(p["n_samples"]),
                # Optional: reports written before `raw` existed do not carry it,
                # and must still re-render rather than being rejected as
                # malformed. build_report falls back to the clamped value.
                raw=None if p.get("raw") is None
                else (float(p["raw"][0]), float(p["raw"][1])),
            )
            for p in data["points"]
        ]
        return build_report(
            measurements,
            screen_cm=(float(data["screen_cm"][0]), float(data["screen_cm"][1])),
            profile_digest=str(data["profile_digest"]),
            profile_name=str(data.get("profile_name", "unknown")),
            drift_offset=(float(data["drift_offset"][0]), float(data["drift_offset"][1])),
            provider=str(data.get("provider", "unknown")),
        )
    except (KeyError, TypeError, IndexError, ValueError) as exc:
        raise GazeError(f"{source} is not a focusedgaze accuracy report: {exc}") from exc


def _cmd_demo(args: argparse.Namespace, out: TextIO) -> int:
    """Open the camera and print live readings until interrupted.

    The quickest way to confirm the whole chain works on a given machine. Prints
    rather than drawing a window: a preview needs a GUI toolkit, and the thing
    worth confirming is that coordinates arrive and move.
    """
    from .capture import WebcamGazeTracker
    from .types import GazeStatus

    mismatch = _profile_backend_error(args, args.profile)
    if mismatch is not None:
        print(f"error: {mismatch}", file=out)
        return 1

    tracker = WebcamGazeTracker(profile=args.profile, config=_config_for(args))
    print(f"Camera {tracker.source.size[0]}x{tracker.source.size[1]}, "
          f"provider {tracker.estimator.provider}. Ctrl+C to stop.", file=out)
    if args.profile is None:
        print("No profile given, so coordinates are unavailable: this reports "
              "raw pitch and yaw only. Pass --profile NAME for screen positions.",
              file=out)

    seen = 0
    try:
        with tracker:
            for result in tracker.stream():
                seen += 1
                if args.frames and seen > args.frames:
                    break
                if result.ok:
                    print(f"  x={result.x:.4f} y={result.y:.4f} "
                          f"dist={_fmt_cm(result.distance_cm)}", file=out)
                elif result.status is GazeStatus.NOT_CALIBRATED:
                    print(f"  pitch={result.pitch:+.4f} yaw={result.yaw:+.4f} rad "
                          f"dist={_fmt_cm(result.distance_cm)}", file=out)
                else:
                    print(f"  {result.status.value}", file=out)
    except KeyboardInterrupt:
        print("\nStopped.", file=out)
    return 0


def _fmt_cm(value: float | None) -> str:
    return "?" if value is None else f"{value:.0f}cm"


def _cmd_serve(args: argparse.Namespace, out: TextIO) -> int:
    """Run the gaze WebSocket server.

    Two sources. The camera is the default; ``--replay`` walks a recorded JSON
    file instead, which is how the wire format is exercised without hardware and
    how a browser client is developed on a machine with no webcam.
    """
    import asyncio

    from .server import GazeServer, GazeSnapshot

    if not args.replay:
        return _serve_live(args, out)

    readings = _load_readings(args.replay)
    print(f"Replaying {len(readings)} readings at {args.hz:g} Hz.", file=out)

    class ReplaySource:
        """Walks the recording, holding the last reading once it runs out."""

        def __init__(self) -> None:
            self._i = 0

        def latest(self) -> GazeSnapshot:
            ok, x, y = readings[min(self._i, len(readings) - 1)]
            self._i += 1
            # A fresh timestamp per reading, so the broadcaster's dedup key
            # changes and the gaze feed actually paces (R-5).
            return GazeSnapshot(ok=ok, x=x, y=y, t=time.time())

        def pause(self, timeout: float = 4.0) -> bool:
            return True

        def resume(self) -> bool:
            return True

    server = GazeServer(ReplaySource(), host=args.host, port=args.port, send_hz=args.hz)
    print(f"Serving at ws://{args.host}:{args.port} - Ctrl+C to stop.", file=out)
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        print("\nStopped.", file=out)
    return 0


def _serve_live(args: argparse.Namespace, out: TextIO) -> int:
    """Serve readings from the camera.

    A profile is required and its absence is refused rather than worked around.
    Without one the pipeline yields raw pitch and yaw, and the wire format has
    no field for those -- every message would carry ``ok: false`` and a client
    would see a server that connects, paces correctly and never reports a
    position. That is a worse failure than declining to start.
    """
    import asyncio

    from .server import GazeServer, LiveGazeSource

    if not args.profile:
        print(
            "focusedgaze serve needs a calibration profile.\n"
            "\n"
            "    focusedgaze serve --profile NAME\n"
            "\n"
            "The wire format carries screen coordinates, and those come from a\n"
            "profile: without one the pipeline has only raw angles, so every\n"
            "message would say ok: false. Run 'focusedgaze calibrate' first, or\n"
            "serve a recording instead with --replay.",
            file=out,
        )
        return 1

    mismatch = _profile_backend_error(args, args.profile)
    if mismatch is not None:
        print(f"error: {mismatch}", file=out)
        return 1

    source = LiveGazeSource(profile=args.profile, config=_config_for(args))
    # Started before the server so a camera or model failure is reported here,
    # with its own remedy, instead of surfacing once a client has connected.
    source.start()
    server = GazeServer(source, host=args.host, port=args.port, send_hz=args.hz)
    print(
        f"Serving live gaze at ws://{args.host}:{args.port} "
        f"(backend {_backend_of(args)}, profile {args.profile!r}) - Ctrl+C to stop.",
        file=out,
    )
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        print("\nStopped.", file=out)
    finally:
        source.close()
    return 0


def _load_readings(path: str | Path) -> list[tuple[bool, float | None, float | None]]:
    """Read ``[ok, x, y]`` rows from JSON.

    Raises:
        GazeError: unreadable, not JSON, or not a list of 3-element rows.
    """
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GazeError(f"could not read {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GazeError(f"{source} is not valid JSON: {exc}") from exc

    rows = raw.get("readings") if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or not rows:
        raise GazeError(
            f"{source} holds no readings. Expected a JSON list of [ok, x, y] rows, "
            'or an object with a "readings" key holding one.'
        )
    out: list[tuple[bool, float | None, float | None]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise GazeError(f"{source}: row {i} is not [ok, x, y]: {row!r}")
        ok = bool(row[0])
        # x and y must stay None when ok is false (R-6): the client's guard is
        # what preserves its last good position.
        x = None if row[1] is None else float(row[1])
        y = None if row[2] is None else float(row[2])
        out.append((ok, x, y))
    return out


def _cmd_export_onnx(args: argparse.Namespace, out: TextIO) -> int:
    """Convert the PyTorch checkpoint to the ONNX graph the runtime loads.

    ``l2cs`` is installable only from a git URL, and PyPI rejects direct-URL
    dependencies in **any** dependency list including extras, so declaring it
    would make the wheel unpublishable (DEV-1). The instruction is therefore
    printed rather than declared, which is the whole reason this command has a
    prerequisites step at all.
    """
    missing = _missing_export_dependencies()
    if missing:
        print("Cannot export: missing " + ", ".join(missing) + ".\n", file=out)
        print("Install the conversion dependencies:\n", file=out)
        print("    pip install 'focusedgaze[export]'", file=out)
        print(
            "    pip install git+https://github.com/Ahmednull/L2CS-Net.git\n",
            file=out,
        )
        print(
            "The second line is separate on purpose. `l2cs` is only installable "
            "from a git URL, and PyPI rejects direct-URL dependencies in any "
            "dependency list, extras included, so this package cannot declare it "
            "without becoming unpublishable.",
            file=out,
        )
        return 1

    # Checked before importing torch, which takes seconds. A typo in a path
    # should be reported at once rather than after a framework has loaded.
    weights = Path(args.weights)
    if not weights.is_file():
        print(f"No PyTorch checkpoint at {weights}.", file=out)
        print(
            "Pass --weights with the path to L2CSNet_gaze360.pkl from the official "
            "L2CS-Net distribution.",
            file=out,
        )
        return 1

    # Both ignore lists carry `unused-ignore` because these modules come from the
    # `export` extra: absent on most machines, where the import is unresolvable,
    # and present on any machine that has actually converted a model, where a
    # bare ignore then becomes an error in its own right. The suppression has to
    # be correct in both environments or the type check fails for whoever has the
    # other one.
    import torch  # type: ignore[import-not-found, unused-ignore]
    from l2cs import getArch  # type: ignore[import-not-found, import-untyped, unused-ignore]

    from .assets import GAZE_MODEL, MODEL_DIR_ENV, asset_path

    # Defaults to the directory the runtime actually reads. The previous default
    # was the bare filename, i.e. the current working directory, so a user who
    # followed the printed instructions exactly ended up with a correct export
    # that `check` still reported as missing, with nothing to connect the two.
    output = Path(args.output) if args.output else asset_path(GAZE_MODEL)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = getArch("ResNet50", args.bins)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()

    # WARNING - these output NAMES are wrong, deliberately left as-is.
    # L2CS-Net's forward() returns (pre_yaw_gaze, pre_pitch_gaze) - yaw FIRST -
    # but this export labels tensor[0] "pitch_bins". The labels are only strings
    # attached at export time; they do not change what the tensors contain. The
    # decode step reads tensor[0] as YAW, which is correct. If you ever "fix" the
    # names here you MUST also change the unpack order there, or every gaze
    # reading silently transposes its axes.
    torch.onnx.export(
        model,
        # A 1-tuple, not a bare tensor. Both trace identically -- torch wraps a
        # lone tensor into `(tensor,)` itself -- but the documented signature
        # takes a tuple of positional arguments, and the bare form is only a type
        # error rather than a behaviour change. The exported graph is unaffected.
        (torch.randn(1, 3, args.input_size, args.input_size),),
        str(output),
        input_names=["input"],
        output_names=["pitch_bins", "yaw_bins"],
        dynamic_axes={"input": {0: "batch_size"}},
        opset_version=12,
        dynamo=False,  # the older, more stable exporter; avoids onnxscript
    )
    print(f"Exported {weights} -> {output}", file=out)
    if args.output:
        expected = asset_path(GAZE_MODEL)
        if Path(output).resolve() != Path(expected).resolve():
            print(
                f"\nNote: the runtime loads {expected}, not this path. Move it there, "
                f"or set {MODEL_DIR_ENV} to {output.parent}.",
                file=out,
            )
    else:
        print("This is where the runtime looks, so nothing else needs moving.", file=out)
    print(
        "Note: the output tensors are named pitch_bins/yaw_bins but hold yaw "
        "first. That mislabelling is upstream and is relied upon by the decode "
        "step; do not correct it here alone.",
        file=out,
    )
    return 0


def _cmd_setup(args: argparse.Namespace, out: TextIO) -> int:
    """Take a fresh install to a usable one, in a single command.

    Four things must be true before the pipeline runs, and before this command
    existed a user discovered them one failure at a time: the landmarker, the
    gaze graph, a provider, and a calibration. This reports all four and does the
    parts it is allowed to do.

    It never installs anything. The conversion needs torch and a git-only
    package, and a CLI that quietly pulls 2.5 GB into whichever environment
    happens to be active is not a convenience. The exact commands are printed and
    the user runs them.
    """
    from .assets import GAZE_MODEL, asset_path, ensure_all
    from .diagnostics import check_runtime

    backend = _backend_of(args)
    print(f"focusedgaze setup  (backend: {backend})\n", file=out)

    # 1. Whatever may be fetched, fetched. Only the auto-downloadable assets are
    #    reported here: the gaze model is step 3's subject, and printing its
    #    licence block twice in one run trains people to skim past it.
    reports = ensure_all(backend=backend)
    failures = _print_reports([r for r in reports if r.asset.auto_download], out)

    # 2. The runtime, because a correct model on no runtime fails at the first
    #    frame with a message about ONNX or OpenVINO rather than about setup.
    #    Which runtime that is depends on the backend; see `check_runtime`.
    runtime = check_runtime(backend)
    print(f"{_MARK[runtime.status]} {runtime.name}: {runtime.summary}", file=out)
    if runtime.remedy and runtime.status != "ok":
        for line in _wrap(runtime.remedy):
            print(line, file=out)
    if runtime.status == "fail":
        failures += 1

    # On the Intel backend the gaze model is one of those downloads, so the
    # whole of step 3 is already done and there is nothing to obtain by hand.
    if backend == "intel":
        ready = all(r.state != "failed" for r in reports)
        if ready and failures == 0:
            print(
                "\nThe Intel gaze model is Apache-2.0 and was fetched "
                "automatically.\nNothing to install by hand.",
                file=out,
            )
        return _setup_calibration_step(args, out, failures)

    # 3. The gaze graph: the one thing that is not fetched, and the one thing
    #    this command can genuinely shorten.
    destination = asset_path(GAZE_MODEL)
    if destination.is_file():
        print(f"{_MARK['ok']} gaze-model: present at {destination}", file=out)
    elif args.onnx:
        result = _install_onnx(args.onnx, destination, out)
        if result != 0:
            return result
    elif args.weights:
        result = _export_for_setup(args, destination, out)
        if result != 0:
            return result
    else:
        print(f"{_MARK['fail']} gaze-model: {GAZE_MODEL.filename} is missing", file=out)
        print(
            "       focusedgaze cannot fetch it. Two ways in:\n"
            "\n"
            "       If somebody has already converted it for you:\n"
            "           focusedgaze setup --onnx <path to l2cs_gaze360.onnx>\n"
            "\n"
            "       Otherwise obtain L2CSNet_gaze360.pkl from the official L2CS-Net\n"
            "       distribution and convert it here (needs the export extra):\n"
            "           focusedgaze setup --weights <path to L2CSNet_gaze360.pkl>\n"
            "\n"
            "       It derives from Gaze360, which is non-commercial research only.\n"
            "       See NOTICE.",
            file=out,
        )

    if not destination.is_file():
        failures += 1
    return _setup_calibration_step(args, out, failures)


def _setup_calibration_step(
    args: argparse.Namespace, out: TextIO, failures: int
) -> int:
    """The last step of setup, shared by both backends.

    A missing calibration is a warning rather than a failure: it is the next
    thing to do, not a broken install, and it needs a person.
    """
    from .calibration import list_profiles

    profiles = list_profiles()
    if profiles:
        print(f"{_MARK['ok']} calibration: {len(profiles)} profile(s): {', '.join(profiles)}",
              file=out)
    else:
        print(f"{_MARK['warn']} calibration: none yet", file=out)

    print(file=out)
    if failures:
        print("Not ready yet. Fix the [FAIL] lines above.", file=out)
        return 1
    if not profiles:
        print("Models and provider are ready. Next: focusedgaze calibrate", file=out)
        return 0
    print("Ready. Try: focusedgaze demo", file=out)
    return 0


def _install_onnx(source: str | Path, destination: Path, out: TextIO) -> int:
    """Validate an already-converted graph and place it where the runtime reads.

    This exists so the conversion is done **once, by one person**. The export
    needs torch and a git-only package for a job whose output is a portable ONNX
    graph: the execution provider is chosen at load time, not baked in, so one
    person's export serves colleagues on other GPUs and other operating systems.
    Without this they would be copying a file into a cache directory by hand.

    The graph is loaded and run before it is copied, not after. A file that turns
    out to be the wrong model should fail while it is still the user's file, not
    once it is sitting in the cache under the name the runtime trusts.
    """
    import shutil

    from .assets import GAZE_MODEL, sha256_file
    from .core.model import GazeModel

    candidate = Path(source)
    if not candidate.is_file():
        print(f"{_MARK['fail']} gaze-model: no file at {candidate}", file=out)
        return 1

    if candidate.resolve() == destination.resolve():
        print(f"{_MARK['ok']} gaze-model: already in place at {destination}", file=out)
        return 0

    print(f"       checking {candidate} is a usable gaze graph...", file=out)
    try:
        import numpy as np

        model = GazeModel(model_path=candidate)
        pitch, yaw = model.predict(np.zeros((64, 64, 3), dtype=np.uint8))
    except GazeError:
        raise
    except Exception as exc:  # noqa: BLE001 - any load failure means "not this file"
        print(f"{_MARK['fail']} gaze-model: {candidate.name} did not load as a gaze model",
              file=out)
        print(
            "       It must be an ONNX export of L2CS-Net: one image input and two\n"
            "       bin tensors out. A different model, or a truncated download, fails\n"
            f"       exactly here.\n       ONNX Runtime reported: {exc}",
            file=out,
        )
        return 1

    if not (math.isfinite(pitch) and math.isfinite(yaw)):
        print(f"{_MARK['fail']} gaze-model: {candidate.name} loaded but produced "
              f"{pitch}, {yaw}", file=out)
        return 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, destination)
    print(f"{_MARK['ok']} gaze-model: installed to {destination}", file=out)

    # Informational only, never enforced: the graph is produced on whichever
    # machine ran the export, so its bytes legitimately differ between installs.
    # See assets/registry.py on why this asset pins no digest.
    digest = sha256_file(destination)
    if GAZE_MODEL.reference_sha256 and digest == GAZE_MODEL.reference_sha256:
        print("       matches the reference export byte for byte.", file=out)
    else:
        print(f"       sha256 {digest[:16]}... (differs from the reference export, "
              "which is normal: ONNX bytes depend on the exporting machine)", file=out)
    return 0


def _export_for_setup(
    args: argparse.Namespace, destination: Path, out: TextIO
) -> int:
    """Run the ONNX conversion as part of setup, into the cache the runtime reads."""
    missing = _missing_export_dependencies()
    if missing:
        print(f"{_MARK['fail']} gaze-model: cannot convert, missing "
              f"{', '.join(missing)}", file=out)
        print(
            "       Install the conversion dependencies, then re-run this command:\n"
            "           pip install 'focusedgaze[export]'\n"
            "           pip install git+https://github.com/Ahmednull/L2CS-Net.git\n"
            "       The second line is separate because `l2cs` is only installable from\n"
            "       a git URL, which PyPI forbids this package from declaring.",
            file=out,
        )
        return 1

    print(f"       converting {args.weights} -> {destination}", file=out)
    print("       (loading torch, this takes a moment)", file=out)
    export_args = argparse.Namespace(
        weights=args.weights,
        output=str(destination),
        bins=args.bins,
        input_size=args.input_size,
    )
    return _cmd_export_onnx(export_args, out)


def _missing_export_dependencies() -> list[str]:
    """Which of the conversion dependencies are absent, in install order."""
    missing = []
    for module, label in (("torch", "torch"), ("onnx", "onnx"), ("l2cs", "l2cs")):
        try:
            __import__(module)
        except ImportError:
            missing.append(label)
    return missing


def _load_samples(path: str | Path) -> list[list[float]]:
    """Read ``(pitch, yaw, target_x, target_y)`` rows from JSON.

    Raises:
        GazeError: The file is unreadable or is not a list of 4-number rows.
    """
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GazeError(f"could not read {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GazeError(f"{source} is not valid JSON: {exc}") from exc

    rows = raw.get("samples") if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or not rows:
        raise GazeError(
            f"{source} holds no samples. Expected a JSON list of "
            "[pitch, yaw, target_x, target_y] rows, or an object with a "
            '"samples" key holding one.'
        )
    out: list[list[float]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise GazeError(f"{source}: row {i} is not 4 numbers: {row!r}")
        try:
            out.append([float(v) for v in row])
        except (TypeError, ValueError) as exc:
            raise GazeError(f"{source}: row {i} holds a non-number: {row!r}") from exc
    return out


def _fmt(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.4f}"


def _wrap(text: str, width: int = 72) -> list[str]:
    """Wrap a remedy for the terminal, without importing textwrap for one call."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="focusedgaze",
        description="Webcam eye-gaze tracking.",
    )
    parser.add_argument("--version", action="version", version=f"focusedgaze {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    download = sub.add_parser(
        "download-models",
        help="fetch the models that may be fetched, explain the ones that may not",
    )
    download.add_argument(
        "--no-download",
        action="store_true",
        help="report what is present without fetching anything",
    )

    setup = sub.add_parser(
        "setup", help="take a fresh install to a usable one, in one command"
    )
    # One source for the graph or the other, never both: they are two ways to
    # answer the same question and supplying each would silently pick a winner.
    source = setup.add_mutually_exclusive_group()
    source.add_argument(
        "--weights",
        help="path to L2CSNet_gaze360.pkl; converts it into the model cache",
    )
    source.add_argument(
        "--onnx",
        help="path to an already-converted l2cs_gaze360.onnx; validates and installs it",
    )
    setup.add_argument("--bins", type=int, default=90, help="gaze bins the model was trained with")
    setup.add_argument("--input-size", type=int, default=448, help="model input resolution")

    check = sub.add_parser("check", help="diagnose the environment")
    check.add_argument(
        "--no-camera",
        action="store_true",
        help="skip the camera probe (headless machines and CI)",
    )
    check.add_argument("--json", action="store_true", help="machine-readable output")

    calibrate = sub.add_parser(
        "calibrate", help="run a calibration session, or manage saved profiles"
    )
    calibrate.add_argument("--list", action="store_true", help="list profiles")
    calibrate.add_argument("--activate", metavar="NAME", help="make a profile the active one")
    calibrate.add_argument("--delete", metavar="NAME", help="delete a profile")
    calibrate.add_argument(
        "--from-samples",
        metavar="FILE",
        help="fit a profile from recorded [pitch, yaw, x, y] rows in JSON",
    )
    calibrate.add_argument(
        "--migrate", metavar="PKL", help="convert a legacy pickled calibration"
    )
    calibrate.add_argument("--name", default="default", help="profile name to write")
    calibrate.add_argument("--directory", help="profile directory (default: user config dir)")
    calibrate.add_argument(
        "--seconds", type=float, default=45.0, help="how long the pursuit sweep runs"
    )
    calibrate.add_argument(
        "--grid",
        action="store_true",
        help="dwell on static dots instead of following a moving one; steadier on "
             "slow hardware, where a stepping dot defeats smooth pursuit",
    )
    calibrate.add_argument(
        "--rows",
        type=int,
        default=6,
        help="rows in the sweep or grid. A multiple of 3 keeps coverage even "
             "across the three screen bands (default: 6)",
    )
    calibrate.add_argument(
        "--dwell", type=float, default=1.0,
        help="--grid only: seconds to settle on each dot before recording",
    )
    calibrate.add_argument(
        "--sample", type=float, default=1.5,
        help="--grid only: seconds of recording per dot",
    )
    calibrate.add_argument(
        "--no-preflight",
        action="store_true",
        help="skip the positioning check and start the sweep immediately",
    )
    calibrate.add_argument(
        "--force",
        action="store_true",
        help="fit even when a screen region collected no samples",
    )

    serve = sub.add_parser("serve", help="run the gaze WebSocket server")
    serve.add_argument(
        "--host", default=DEFAULT_HOST,
        help=f"bind address (default: {DEFAULT_HOST}; keep it loopback, the stream "
             "is unauthenticated)",
    )
    serve.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port")
    serve.add_argument("--hz", type=float, default=DEFAULT_SEND_HZ, help="broadcast tick rate")
    serve.add_argument(
        "--profile",
        help="calibration profile to serve. Required for the camera, ignored by "
             "--replay, whose coordinates are already in the recording",
    )
    serve.add_argument(
        "--replay", metavar="FILE",
        help="serve recorded [ok, x, y] readings from JSON instead of a camera",
    )

    demo = sub.add_parser("demo", help="print live gaze readings from the camera")
    demo.add_argument("--profile", help="calibration profile name")
    demo.add_argument(
        "--frames", type=int, default=0,
        help="stop after N frames (0 runs until interrupted)",
    )

    accuracy = sub.add_parser(
        "accuracy", help="measure calibration accuracy across the screen"
    )
    accuracy.add_argument("--profile", help="calibration profile to measure")
    accuracy.add_argument(
        "--screen-width-cm", type=float, default=34.4, help="physical screen width"
    )
    accuracy.add_argument(
        "--screen-height-cm", type=float, default=19.4, help="physical screen height"
    )
    accuracy.add_argument("--dwell", type=float, default=1.0, help="seconds before sampling")
    accuracy.add_argument("--sample", type=float, default=1.5, help="seconds of sampling")
    accuracy.add_argument(
        "--drift", action="store_true",
        help="subtract the centre-point drift offset before comparing. Note this "
             "forces the centre point's error to exactly zero, because that is "
             "where the offset is measured",
    )
    accuracy.add_argument(
        "--no-preflight",
        action="store_true",
        help="skip the positioning check and start measuring immediately",
    )
    accuracy.add_argument(
        "--compensate-distance",
        action="store_true",
        help="rescale predictions for the difference between where you are sitting "
             "and where the profile was calibrated. Needs a profile that recorded "
             "its distance",
    )
    accuracy.add_argument("--save", metavar="FILE", help="write the result as JSON")
    accuracy.add_argument(
        "--from-json", metavar="FILE", help="re-render a saved result instead of measuring"
    )

    # Every command that loads a model, checks for one, or fetches one. Not
    # export-onnx, which converts L2CS weights specifically and has no meaning
    # for another backend.
    for command in (download, setup, check, calibrate, serve, demo, accuracy):
        _add_backend_arg(command)

    export = sub.add_parser("export-onnx", help="convert PyTorch L2CS weights to ONNX")
    export.add_argument(
        "--weights", default="L2CSNet_gaze360.pkl", help="the PyTorch checkpoint to convert"
    )
    export.add_argument(
        "--output",
        default=None,
        help="where to write the graph (default: the managed model cache, where the runtime looks)",
    )
    export.add_argument("--bins", type=int, default=90, help="gaze bins the model was trained with")
    export.add_argument("--input-size", type=int, default=448, help="model input resolution")

    return parser


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    """Entry point for the ``focusedgaze`` console script.

    Args:
        argv: Arguments, defaulting to ``sys.argv[1:]``.
        out: Where to write. Injected by tests so output is asserted on directly
            rather than through captured file descriptors.
    """
    stream = out if out is not None else sys.stdout
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help(stream)
        return 0

    handlers = {
        "download-models": _cmd_download_models,
        "setup": _cmd_setup,
        "check": _cmd_check,
        "calibrate": _cmd_calibrate,
        "export-onnx": _cmd_export_onnx,
        "serve": _cmd_serve,
        "demo": _cmd_demo,
        "accuracy": _cmd_accuracy,
    }
    try:
        return handlers[args.command](args, stream)
    except CalibrationAborted:
        # Not an error, so it does not get the "error:" prefix: stopping a sweep
        # with Escape is a supported way out, and reporting a user's own decision
        # as a fault is how a tool teaches people to distrust its output.
        print("Stopped. Nothing was saved.", file=stream)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Nothing was saved.", file=stream)
        return 1
    except GazeError as exc:
        # Every error this package raises derives from GazeError (D7), so one
        # handler covers the lot. A traceback here would be noise: these are
        # conditions with remedies, not crashes.
        print(f"error: {exc}", file=stream)
        return 1


if __name__ == "__main__":
    sys.exit(main())
