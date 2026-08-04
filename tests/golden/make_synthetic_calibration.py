"""Fit a synthetic degree-3 calibration model for the Tier 1 fixture.

Why this exists
---------------
`calibration_apply.json` proves that focusedgaze's `apply_calibration` matches
the legacy one. That equivalence holds for ANY model, provided both sides use
the same one. It does not need a real person's calibration, and using one made
the fixture depend on a mutable file that ordinary recalibration overwrites
(audit section 33).

So the fixture's input is a synthetic model, fitted here from generated pairs
and committed alongside the fixture. It is nobody's data, which keeps audit
section 9.3 satisfied, and it cannot be overwritten by using the system.

The model is fitted with the UNMODIFIED legacy fitter, so the pickle is
structurally identical to a real profile: same dict keys, same scikit-learn
objects, same degree. The point is to change whose data it is, not what it is.

Regenerating
------------
    python tests/golden/make_synthetic_calibration.py

Needs the legacy checkout, located via FOCUSEDGAZE_LEGACY_DIR. Writing a new
model invalidates `calibration_apply.json`, so re-record it in the same pass:

    python tests/golden/record_tier1.py
"""

from __future__ import annotations

import hashlib
import math
import os
import pathlib
import sys

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "tier1"
OUT = FIXTURES / "synthetic_calibration.pkl"

# Deliberately not a plain linear map. A real calibration is a mildly nonlinear
# surface, and the cubic terms have to be exercised or a degree mismatch would
# not be detectable (see the mutation check in test_golden_tier1.py).
#
# The coefficients are chosen so the surface leaves [0, 1] near the corners of
# the recorded grid, which is what keeps the clamp branch covered.
def _target(pitch: float, yaw: float) -> tuple[float, float]:
    x = 0.5 + 0.95 * yaw + 0.28 * yaw**3 + 0.10 * pitch * yaw
    y = 0.5 - 0.95 * pitch - 0.28 * pitch**3 + 0.10 * pitch * yaw
    return x, y


def build_samples() -> list[tuple[float, float, float, float]]:
    """Generated (pitch, yaw, x, y) pairs spanning the working range."""
    samples = []
    # 41x41 over +/-0.72 rad: dense enough that the robust fitter's outlier
    # pass keeps well above its min_keep floor of 60.
    steps = 41
    for i in range(steps):
        for j in range(steps):
            pitch = -0.72 + 1.44 * i / (steps - 1)
            yaw = -0.72 + 1.44 * j / (steps - 1)
            x, y = _target(pitch, yaw)
            # A little deterministic wobble, so the fit is a fit and not an
            # exact algebraic inversion. No RNG: this must be reproducible.
            x += 0.002 * math.sin(37.0 * pitch + 11.0 * yaw)
            y += 0.002 * math.cos(29.0 * pitch + 17.0 * yaw)
            samples.append((pitch, yaw, x, y))
    return samples


def main() -> int:
    legacy = os.environ.get("FOCUSEDGAZE_LEGACY_DIR")
    if not legacy:
        print("ERROR: set FOCUSEDGAZE_LEGACY_DIR to the gaze-detection checkout.")
        return 2
    legacy_dir = pathlib.Path(legacy)
    if not legacy_dir.is_dir():
        print(f"ERROR: FOCUSEDGAZE_LEGACY_DIR does not exist: {legacy_dir}")
        return 2

    prev = pathlib.Path.cwd()
    os.chdir(legacy_dir)
    sys.path.insert(0, str(legacy_dir))
    try:
        import calibration_utils as cal

        samples = build_samples()
        # robust_fit_samples is the path the live system actually uses, and it
        # is where degree=3 comes from: fit_calibration's own default is 2.
        model, n_dropped = cal.robust_fit_samples(samples)
        FIXTURES.mkdir(parents=True, exist_ok=True)
        cal.save_calibration(model, str(OUT))
    finally:
        os.chdir(prev)

    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"samples fitted : {len(samples)}")
    print(f"outliers dropped: {n_dropped}")
    print(f"degree         : {model.get('degree')}")
    print(f"written        : {OUT}  ({OUT.stat().st_size} bytes)")
    print(f"sha256         : {digest}")
    print()
    print("Now re-record the fixture:  python tests/golden/record_tier1.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
