"""Record Tier 1 golden fixtures from the CURRENT, UNMODIFIED pipeline.

Run this BEFORE any refactoring. Everything it captures is numeric (no frames,
no face, nothing biometric), so the output is safe to commit and runs in CI.

What it captures, and why these three:

  calibration_apply   the degree-3 polynomial mapping (pitch, yaw) -> (x, y).
                      Phase 5 replaces pickle-of-sklearn-objects with raw
                      coefficients, which is the single most likely place for
                      numbers to shift silently.
  one_euro            the filter that turns a jittery signal into a steady
                      cursor. Phase 2 moves it into core/filters.py.
  positioning_gate    distance estimate and zone decision. Phase 2 moves it.

Everything upstream of these needs real images and is Tier 2 (local only).

Locating the legacy pipeline (rule 4, no hard-coded paths):
    FOCUSEDGAZE_LEGACY_DIR   points at the original gaze-detection/ folder.
    If unset, falls back to ../gaze-detection relative to this repository,
    which is the normal side-by-side checkout layout.

Usage (from the repo root, with the gaze venv):
    python tests/golden/record_tier1.py
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# The legacy pipeline lives outside this package and uses relative paths, so we
# add it to sys.path and run from its own directory. Never hard-code an absolute
# path here: a second developer checking this out has a different drive layout,
# and the harness has to run on their machine too.
LEGACY = pathlib.Path(
    os.environ.get("FOCUSEDGAZE_LEGACY_DIR") or (REPO_ROOT.parent / "gaze-detection")
)
OUT = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "tier1"


def _legacy_missing_message() -> str:
    return (
        f"Legacy pipeline not found at: {LEGACY}\n"
        "\n"
        "Set FOCUSEDGAZE_LEGACY_DIR to the original gaze-detection/ folder, e.g.\n"
        "    set FOCUSEDGAZE_LEGACY_DIR=<drive>:\\path\\to\\gaze-detection (Windows)\n"
        "    export FOCUSEDGAZE_LEGACY_DIR=/path/to/gaze-detection      (POSIX)\n"
        "\n"
        "Or check the two projects out side by side:\n"
        "    <parent>/gaze-detection/\n"
        "    <parent>/focusedgaze-sdk/     <- this repository\n"
    )


class FakeLandmark:
    """Stands in for a MediaPipe NormalizedLandmark: only .x/.y/.z are read."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z


def record_calibration_apply(cal_utils, model, model_sha256: str) -> dict:
    """Sweep a deterministic (pitch, yaw) grid through apply_calibration().

    `model_sha256` identifies the exact model these expected values came from.
    The fixture pins its own input by content rather than trusting whatever
    happens to sit at a path, which is the fix for audit section 33.
    """
    cases = []
    # Radians. The real signal sits roughly within +/-0.6 rad; the grid spans a
    # little wider so clamping at the [0,1] edges is covered too.
    for i in range(-6, 7):
        for j in range(-6, 7):
            pitch, yaw = i * 0.12, j * 0.12
            x, y = cal_utils.apply_calibration(model, pitch, yaw)
            cases.append({"pitch": pitch, "yaw": yaw,
                          "x": float(x), "y": float(y)})
    clamped = sum(1 for c in cases for v in (c["x"], c["y"]) if v in (0.0, 1.0))
    return {"description": "apply_calibration over a fixed (pitch,yaw) grid",
            "model_file": "synthetic_calibration.pkl",
            "model_sha256": model_sha256,
            "degree": model.get("degree"),
            "clamped_values": clamped,
            "n": len(cases), "cases": cases}


def record_one_euro(OneEuro) -> dict:
    """Drive the 1-euro filter with a fixed signal at fixed timestamps.

    Deliberately mixes a slow ramp, a step, and jitter: the three regimes where
    an adaptive filter behaves differently.
    """
    import math

    cfg = {"min_cutoff": 0.7, "beta": 0.6, "d_cutoff": 1.0}   # gaze_server defaults
    f = OneEuro(cfg["min_cutoff"], cfg["beta"], cfg["d_cutoff"])
    out = []
    t = 1000.0
    for i in range(120):
        t += 1 / 30
        if i < 40:
            v = 0.5 + i * 0.005                       # slow ramp
        elif i < 45:
            v = 0.9                                   # step
        else:
            v = 0.9 + math.sin(i * 1.7) * 0.02        # jitter around a hold
        out.append({"t": t, "raw": v, "filtered": float(f.filter(v, t))})

    # Non-monotonic timestamps. `filter()` only recomputes its frequency when
    # t > t_prev, and the strictly-increasing signal above never exercises the
    # other side of that guard. A stalled clock (repeated t) or a clock that
    # steps backwards is not hypothetical: it is what a paused capture thread
    # or a re-used timestamp produces. Pinning it here means a refactor cannot
    # quietly change how a stalled stream behaves.
    for dup in (0.0, 0.0, -0.5, 0.0):
        t += dup
        v = 0.88
        out.append({"t": t, "raw": v, "filtered": float(f.filter(v, t))})
    return {"description": "OneEuro filter response to ramp + step + jitter",
            "config": cfg, "n": len(out), "samples": out}


def record_positioning(gate_mod, use_focal: bool = True) -> dict:
    """Evaluate the positioning gate over synthetic iris/nose geometries.

    Args:
        use_focal: When False, the gate is constructed somewhere its
            ``models/camera_focal.json`` is unreachable, so it falls back to the
            assumed-HFOV formula.

    Why both are recorded: the legacy gate picks its focal branch by whether a
    FILE EXISTS, not by an argument. Recording only the calibrated branch would
    leave the fallback unpinned, and that fallback is the branch every user who
    has never measured a focal length actually runs. A wrong half-angle or a
    missing degrees-to-radians conversion would pass a suite that only ever
    tests the measured path.
    """
    if use_focal:
        gate = gate_mod.PositioningGate()
    else:
        # Construct from a directory with no models/ in it, so _load_focal fails
        # and the gate falls back. Only construction needs the changed cwd.
        import tempfile

        prev = pathlib.Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                gate = gate_mod.PositioningGate()
            finally:
                os.chdir(prev)
        assert gate._focal_override is None, "expected the fallback branch"

    frame_shape = (720, 1280, 3)
    cases = []

    # Degenerate geometry: both irises at the same point, so ipd_px < 1 and the
    # gate can produce no distance at all. Another branch the grid below never
    # reaches.
    degenerate = [FakeLandmark(0.5, 0.5) for _ in range(478)]
    st_deg = gate.evaluate(degenerate, frame_shape)
    cases.append({"ipd_frac": 0.0, "nose_dx": 0.0, "result_is_none": st_deg is None})

    # Vary inter-pupil separation (=> distance) and nose offset (=> centring).
    for ipd_frac in (0.045, 0.055, 0.065, 0.075, 0.09):
        for nose_dx in (0.0, 0.08, 0.18):
            lm = {}
            cx, cy = 0.5 + nose_dx, 0.5
            lm[gate_mod.LEFT_IRIS_CENTER] = FakeLandmark(cx - ipd_frac / 2, cy)
            lm[gate_mod.RIGHT_IRIS_CENTER] = FakeLandmark(cx + ipd_frac / 2, cy)
            lm[gate_mod.NOSE_TIP] = FakeLandmark(cx, cy)
            landmarks = [FakeLandmark(0.5, 0.5) for _ in range(478)]
            for idx, v in lm.items():
                landmarks[idx] = v
            st = gate.evaluate(landmarks, frame_shape)
            cases.append({
                "ipd_frac": ipd_frac, "nose_dx": nose_dx, "result_is_none": False,
                "distance_cm": float(st["distance_cm"])
                if st.get("distance_cm") is not None else None,
                "dx": float(st.get("dx", 0.0)),
                "dy": float(st.get("dy", 0.0)),
                "distance_ok": bool(st.get("distance_ok")),
                "centered": bool(st.get("centered")),
                "in_zone": bool(st.get("in_zone")),
                "zone": st.get("zone"),
            })
    return {"description": "PositioningGate.evaluate over synthetic geometries",
            "frame_shape": list(frame_shape), "n": len(cases), "cases": cases,
            # The gate loads models/camera_focal.json via a RELATIVE path, so
            # its distances depend on the process working directory. Recorded
            # here so a replay can reproduce the same context deliberately.
            "focal_override": list(gate._focal_override) if gate._focal_override else None}


def main() -> int:
    if not LEGACY.is_dir():
        print(_legacy_missing_message())
        return 1

    prev = pathlib.Path.cwd()
    os.chdir(LEGACY)                       # legacy modules use relative paths
    sys.path.insert(0, str(LEGACY))
    try:
        import calibration_utils
        import positioning_gate

        # The calibration model is the COMMITTED SYNTHETIC one, not the live
        # profile in the legacy tree. This fixture proves that the extracted
        # apply_calibration matches the legacy one, which holds for any model so
        # long as both sides use the same one, so it does not need a real
        # person's calibration.
        #
        # It used to read models/calibration_model.pkl, which ordinary
        # recalibration overwrites. That made a mutable file owned by another
        # activity into a fixture input, and it silently invalidated the fixture
        # the first time somebody recalibrated (audit section 33).
        cal_path = OUT / "synthetic_calibration.pkl"
        if not cal_path.exists():
            print(f"ERROR: no synthetic calibration model at {cal_path}")
            print("       Generate it with:")
            print("         python tests/golden/make_synthetic_calibration.py")
            return 1
        model = calibration_utils.load_calibration(str(cal_path))
        cal_digest = hashlib.sha256(cal_path.read_bytes()).hexdigest()

        # OneEuro lives inside gaze_server. Importing it is heavy (it builds the
        # ONNX session via gaze_pipeline) but it is the real implementation, and
        # copying the class here would defeat the point of a golden file.
        import gaze_server

        OUT.mkdir(parents=True, exist_ok=True)
        written = []
        for name, data in (
            ("calibration_apply.json",
             record_calibration_apply(calibration_utils, model, cal_digest)),
            ("one_euro.json", record_one_euro(gaze_server.OneEuro)),
            ("positioning_gate.json", record_positioning(positioning_gate, use_focal=True)),
            ("positioning_gate_nofocal.json",
             record_positioning(positioning_gate, use_focal=False)),
        ):
            path = OUT / name
            path.write_text(json.dumps(data, indent=1), encoding="utf-8")
            written.append((name, data["n"]))

        print("Tier 1 golden fixtures recorded from the UNMODIFIED pipeline:")
        for name, n in written:
            print(f"  {name:26s} {n} cases")
        print(f"\nwritten to {OUT}")
        return 0
    finally:
        os.chdir(prev)


if __name__ == "__main__":
    raise SystemExit(main())
