"""The extracted One Euro filter must match the pre-refactor recording exactly.

This is the Phase 2 equivalence proof for `core/filters.py`. It compares the new
package implementation against the same Tier 1 fixture that
`test_golden_tier1.py` runs against the legacy code, so both are measured with
the same ruler.

Needs no camera, no model, no legacy checkout: pure numbers, runs in CI.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from focusedgaze.core.filters import OneEuroFilter, OneEuroFilter2D

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "tier1"
TOL = 1e-9


def _golden() -> dict:
    path = FIXTURES / "one_euro.json"
    if not path.exists():
        pytest.skip("fixture missing: run tests/golden/record_tier1.py")
    return json.loads(path.read_text(encoding="utf-8"))


def test_extracted_filter_matches_legacy_recording() -> None:
    """Byte-for-byte behavioural equivalence with the shipping filter."""
    data = _golden()
    cfg = data["config"]
    f = OneEuroFilter(cfg["min_cutoff"], cfg["beta"], cfg["d_cutoff"])

    worst = 0.0
    for s in data["samples"]:
        worst = max(worst, abs(f.filter(s["raw"], s["t"]) - s["filtered"]))
    assert worst <= TOL, f"extracted filter drifted by {worst:.3e} (tolerance {TOL:.0e})"


def test_callable_alias_matches_filter_method() -> None:
    """`f(x, t)` and `f.filter(x, t)` must be the same operation."""
    data = _golden()
    cfg = data["config"]
    a = OneEuroFilter(cfg["min_cutoff"], cfg["beta"], cfg["d_cutoff"])
    b = OneEuroFilter(cfg["min_cutoff"], cfg["beta"], cfg["d_cutoff"])
    for s in data["samples"][:30]:
        assert a.filter(s["raw"], s["t"]) == b(s["raw"], s["t"])


def test_reset_returns_to_first_sample_behaviour() -> None:
    """After reset the next sample passes through untouched, as at startup.

    This matters in practice: the server resets the filter when the face is
    lost, so the cursor does not glide in from wherever it was left.
    """
    f = OneEuroFilter(0.7, 0.6)
    f.filter(0.1, 100.0)
    f.filter(0.9, 100.0 + 1 / 30)
    f.reset()
    assert f.filter(0.42, 200.0) == 0.42


def test_two_instances_do_not_share_state() -> None:
    """Independent filters must not interfere.

    The legacy pipeline kept smoothing state in a module-level global; this test
    exists so that class of defect cannot come back through the filter.
    """
    a = OneEuroFilter(0.7, 0.6)
    b = OneEuroFilter(0.7, 0.6)
    a.filter(0.0, 100.0)
    a.filter(0.0, 100.0 + 1 / 30)
    # `b` has seen nothing, so its first sample must pass through unchanged.
    assert b.filter(0.75, 100.0) == 0.75


def test_2d_filter_is_two_independent_axes() -> None:
    """OneEuroFilter2D must equal two scalar filters run side by side."""
    data = _golden()
    cfg = data["config"]
    two = OneEuroFilter2D(cfg["min_cutoff"], cfg["beta"], cfg["d_cutoff"])
    fx = OneEuroFilter(cfg["min_cutoff"], cfg["beta"], cfg["d_cutoff"])
    fy = OneEuroFilter(cfg["min_cutoff"], cfg["beta"], cfg["d_cutoff"])

    for i, s in enumerate(data["samples"]):
        # Feed the axes different signals, so a mix-up would show up.
        gx, gy = two.filter(s["raw"], 1.0 - s["raw"], s["t"])
        assert gx == fx.filter(s["raw"], s["t"]), f"x axis diverged at sample {i}"
        assert gy == fy.filter(1.0 - s["raw"], s["t"]), f"y axis diverged at sample {i}"
