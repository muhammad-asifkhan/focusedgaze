"""The ``focusedgaze`` console entry point.

Six commands, the full set:

    download-models   fetch what may be fetched, explain what may not   (Phase 6)
    check             diagnose the environment                          (Phase 6)
    calibrate         manage calibration profiles                       (Phase 6)
    export-onnx       convert PyTorch L2CS weights to the ONNX graph    (Phase 6)
    serve             run the gaze WebSocket server                     (Phase 7)
    demo              print live readings from the camera               (Phase 2)

``demo`` and the live path of ``serve`` both needed ``GazeEstimator``, which
landed with Phase 2. ``serve --replay`` remains, because replaying recorded
readings is how the wire format is tested without a camera.

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
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, TextIO

from . import __version__
from .exceptions import GazeError
from .server import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_SEND_HZ

__all__ = ["main"]

#: Marks each check line. ASCII on purpose: this output gets pasted into issues
#: and terminals, and a Windows console in a legacy code page renders anything
#: else as replacement characters. The project has already fixed one bug of
#: exactly that shape (commit "Fix em-dash in CLI console output").
_MARK: Final[dict[str, str]] = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]"}


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


def _cmd_download_models(args: argparse.Namespace, out: TextIO) -> int:
    from .assets import ensure_all

    reports = ensure_all(allow_download=not args.no_download)
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

    results = run_checks(camera=not args.no_camera)

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
        result = robust_fit_samples(samples, name=args.name)
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

    # Interactive capture is the one path that needs the pipeline.
    print(
        "Interactive calibration needs the gaze pipeline, which is Phase 2 and is "
        "not implemented yet.\n"
        "\n"
        "What works now:\n"
        "  focusedgaze calibrate --list\n"
        "  focusedgaze calibrate --from-samples samples.json --name alice\n"
        "  focusedgaze calibrate --migrate old_calibration.pkl --name alice\n"
        "  focusedgaze calibrate --activate alice\n"
        "  focusedgaze calibrate --delete alice\n",
        file=out,
    )
    return 1


def _cmd_demo(args: argparse.Namespace, out: TextIO) -> int:
    """Open the camera and print live readings until interrupted.

    The quickest way to confirm the whole chain works on a given machine. Prints
    rather than drawing a window: a preview needs a GUI toolkit, and the thing
    worth confirming is that coordinates arrive and move.
    """
    from .capture import WebcamGazeTracker
    from .types import GazeStatus

    tracker = WebcamGazeTracker(profile=args.profile)
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

    The live source needs ``GazeEstimator``, which is Phase 2. Until it lands,
    ``--replay`` is the only source, and it is a real one: it drives
    ``gaze_test.html`` and the browser game exactly as a camera would, which is
    what makes the wire format testable today rather than after Phase 2.
    """
    import asyncio

    from .server import GazeServer, GazeSnapshot

    if not args.replay:
        print(
            "focusedgaze serve needs a gaze source.\n"
            "\n"
            "The live camera source is GazeEstimator, which is Phase 2 and is not\n"
            "implemented yet. Until it lands, replay recorded readings:\n"
            "\n"
            "    focusedgaze serve --replay readings.json\n"
            "\n"
            "where readings.json is a list of [ok, x, y] rows, or an object with a\n"
            '"readings" key holding one. That drives gaze_test.html and the game\n'
            "over the real wire format.",
            file=out,
        )
        return 1

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

    import torch  # type: ignore[import-not-found]
    from l2cs import getArch  # type: ignore[import-not-found]

    output = Path(args.output)
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
        torch.randn(1, 3, args.input_size, args.input_size),
        str(output),
        input_names=["input"],
        output_names=["pitch_bins", "yaw_bins"],
        dynamic_axes={"input": {0: "batch_size"}},
        opset_version=12,
        dynamo=False,  # the older, more stable exporter; avoids onnxscript
    )
    print(f"Exported {weights} -> {output}", file=out)
    print(
        "Note: the output tensors are named pitch_bins/yaw_bins but hold yaw "
        "first. That mislabelling is upstream and is relied upon by the decode "
        "step; do not correct it here alone.",
        file=out,
    )
    return 0


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

    check = sub.add_parser("check", help="diagnose the environment")
    check.add_argument(
        "--no-camera",
        action="store_true",
        help="skip the camera probe (headless machines and CI)",
    )
    check.add_argument("--json", action="store_true", help="machine-readable output")

    calibrate = sub.add_parser("calibrate", help="manage calibration profiles")
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

    serve = sub.add_parser("serve", help="run the gaze WebSocket server")
    serve.add_argument(
        "--host", default=DEFAULT_HOST,
        help=f"bind address (default: {DEFAULT_HOST}; keep it loopback, the stream "
             "is unauthenticated)",
    )
    serve.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port")
    serve.add_argument("--hz", type=float, default=DEFAULT_SEND_HZ, help="broadcast tick rate")
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

    export = sub.add_parser("export-onnx", help="convert PyTorch L2CS weights to ONNX")
    export.add_argument(
        "--weights", default="L2CSNet_gaze360.pkl", help="the PyTorch checkpoint to convert"
    )
    export.add_argument("--output", default="l2cs_gaze360.onnx", help="where to write the graph")
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
        "check": _cmd_check,
        "calibrate": _cmd_calibrate,
        "export-onnx": _cmd_export_onnx,
        "serve": _cmd_serve,
        "demo": _cmd_demo,
    }
    try:
        return handlers[args.command](args, stream)
    except GazeError as exc:
        # Every error this package raises derives from GazeError (D7), so one
        # handler covers the lot. A traceback here would be noise: these are
        # conditions with remedies, not crashes.
        print(f"error: {exc}", file=stream)
        return 1


if __name__ == "__main__":
    sys.exit(main())
