"""Dwell selection.

Pure, so a fake clock drives all of it: no camera, no screen, no person. Each
tolerance here exists because of a measured property of the tracker, and the
tests are written against the behaviour a user would notice rather than the
internal state.
"""

from __future__ import annotations

import pytest

from focusedgaze.control import DwellSelector, Target, edge_zones_for
from focusedgaze.exceptions import ConfigError

PLAY = Target("play", 0.1, 0.1, 0.4, 0.4)
STOP = Target("stop", 0.6, 0.1, 0.9, 0.4)


def _selector(**kw) -> DwellSelector:
    kw.setdefault("targets", [PLAY, STOP])
    kw.setdefault("dwell_s", 1.0)
    return DwellSelector(**kw)


def _hold(sel, x, y, *, start=0.0, until=2.0, step=0.1):
    """Look at one point over a span, returning every state."""
    states, t = [], start
    while t <= until + 1e-9:
        states.append(sel.update(x, y, t))
        t += step
    return states


# ---------------------------------------------------------------------------
# Selecting.
# ---------------------------------------------------------------------------


def test_looking_at_a_target_for_the_dwell_time_selects_it() -> None:
    sel = _selector()
    states = _hold(sel, 0.25, 0.25, until=1.5)
    assert [s.selected for s in states].count("play") == 1


def test_selection_is_reported_on_exactly_one_update() -> None:
    """So a caller can act on it without deduplicating. Reporting it every frame
    while the gaze rests there would fire the action repeatedly."""
    sel = _selector()
    states = _hold(sel, 0.25, 0.25, until=1.4)
    assert sum(1 for s in states if s.selected) == 1


def test_nothing_is_selected_before_the_dwell_elapses() -> None:
    sel = _selector()
    states = _hold(sel, 0.25, 0.25, until=0.9)
    assert all(s.selected is None for s in states)
    assert states[-1].target == "play"
    assert 0.0 < states[-1].progress < 1.0


def test_progress_climbs_toward_one() -> None:
    """The caller draws this. A dwell with no visible progress is
    indistinguishable from a hang."""
    sel = _selector()
    progress = [s.progress for s in _hold(sel, 0.25, 0.25, until=0.9)]
    assert progress == sorted(progress)
    assert progress[0] < 0.2 and progress[-1] > 0.8


def test_looking_away_before_the_dwell_completes_selects_nothing() -> None:
    sel = _selector()
    _hold(sel, 0.25, 0.25, until=0.5)
    states = _hold(sel, 0.75, 0.25, start=0.6, until=0.9)
    assert all(s.selected is None for s in states)
    assert states[-1].target == "stop"


def test_a_per_target_dwell_overrides_the_default() -> None:
    """A destructive action can be made to need a longer look."""
    slow = Target("delete", 0.1, 0.6, 0.4, 0.9, dwell_s=2.0)
    sel = DwellSelector(targets=[slow], dwell_s=0.5)
    assert all(s.selected is None for s in _hold(sel, 0.25, 0.75, until=1.5))
    assert any(s.selected == "delete" for s in _hold(sel, 0.25, 0.75, start=1.6, until=2.5))


def test_looking_at_nothing_reports_nothing() -> None:
    sel = _selector()
    state = sel.update(0.5, 0.8, 0.0)
    assert state.target is None and state.selected is None and state.progress == 0.0


# ---------------------------------------------------------------------------
# The three tolerances.
# ---------------------------------------------------------------------------


def test_jitter_just_outside_a_target_does_not_cancel_the_dwell() -> None:
    """The tracker's own error is 2-3 cm. A plain inside/outside test flickers
    and cancels a dwell the user experiences as perfectly steady."""
    sel = _selector(hysteresis=0.03)
    t, selected = 0.0, False
    while t <= 1.4:
        # Oscillate across the target's right edge by less than the margin.
        x = 0.4 + (0.02 if int(t * 10) % 2 else -0.02)
        state = sel.update(x, 0.25, t)
        selected = selected or state.selected == "play"
        t += 0.1
    assert selected, "jitter at the boundary cancelled a steady dwell"


def test_jitter_beyond_the_margin_does_cancel() -> None:
    """The control. Hysteresis must not be so wide it selects things you are
    not looking at."""
    sel = _selector(hysteresis=0.01)
    t, selected = 0.0, False
    while t <= 1.4:
        x = 0.4 + (0.2 if int(t * 10) % 2 else -0.02)
        state = sel.update(x, 0.25, t)
        selected = selected or state.selected == "play"
        t += 0.1
    assert not selected


def test_a_blink_does_not_cancel_a_dwell() -> None:
    """Blinks give NO_FACE for several frames; at 7 fps that is most of a
    second. Cancelling on every blink makes selection nearly impossible."""
    sel = _selector(blink_grace_s=0.5)
    for t in (0.0, 0.1, 0.2, 0.3):
        sel.update(0.25, 0.25, t)
    for t in (0.4, 0.5):                       # eyes shut
        state = sel.update(None, None, t)
        assert state.target == "play", "the dwell was dropped mid-blink"
        assert state.tracking is False
    selected = any(
        sel.update(0.25, 0.25, t).selected == "play"
        for t in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
    )
    assert selected


def test_a_blink_does_not_count_toward_the_dwell() -> None:
    """Otherwise a well-timed blink selects, which is the opposite of
    deliberate."""
    sel = _selector(dwell_s=1.0, blink_grace_s=5.0)
    sel.update(0.25, 0.25, 0.0)
    sel.update(None, None, 0.1)
    state = sel.update(0.25, 0.25, 1.05)        # 0.95 s of it was a blackout
    assert state.selected is None, "the blackout was counted as dwelling"


def test_gaze_lost_for_longer_than_the_grace_period_cancels() -> None:
    sel = _selector(blink_grace_s=0.3)
    sel.update(0.25, 0.25, 0.0)
    state = sel.update(None, None, 1.0)
    assert state.target is None and state.tracking is False


def test_resting_on_a_target_does_not_reselect_it_immediately() -> None:
    """The Midas touch problem: without re-arming, looking at a control is
    indistinguishable from choosing it over and over."""
    sel = _selector(dwell_s=0.5, rearm_s=1.0)
    states = _hold(sel, 0.25, 0.25, until=1.2)
    assert sum(1 for s in states if s.selected) == 1


def test_a_target_can_be_chosen_again_after_the_rearm_delay() -> None:
    sel = _selector(dwell_s=0.3, rearm_s=0.5)
    states = _hold(sel, 0.25, 0.25, until=3.0, step=0.1)
    assert sum(1 for s in states if s.selected) >= 2


def test_progress_reads_zero_while_a_target_is_blocked() -> None:
    """Held over something just chosen: show it highlighted, but do not imply
    another selection is coming."""
    sel = _selector(dwell_s=0.3, rearm_s=1.0)
    states = _hold(sel, 0.25, 0.25, until=0.8)
    after = [s for s in states if s.selected is None and s.target == "play"]
    assert after and all(s.progress == 0.0 for s in after[-3:])


# ---------------------------------------------------------------------------
# Edge zones.
# ---------------------------------------------------------------------------


def test_edge_zones_report_immediately_with_no_dwell() -> None:
    """Scrolling is continuous. Waiting a second before the page moves reads as
    a broken control."""
    sel = _selector(edges=edge_zones_for(0.09))
    state = sel.update(0.02, 0.5, 0.0)
    assert state.edge is not None and state.edge.name == "left"
    assert state.edge.held_s == 0.0


def test_an_edge_zone_reports_how_long_it_has_been_held() -> None:
    """Callers scale scroll speed by this so a glance does not fling the page."""
    sel = _selector(edges=edge_zones_for(0.09))
    sel.update(0.02, 0.5, 0.0)
    assert sel.update(0.02, 0.5, 0.8).edge.held_s == pytest.approx(0.8)


def test_leaving_an_edge_resets_its_hold() -> None:
    sel = _selector(edges=edge_zones_for(0.09))
    sel.update(0.02, 0.5, 0.0)
    sel.update(0.5, 0.5, 0.5)
    assert sel.update(0.02, 0.5, 1.0).edge.held_s == 0.0


def test_the_default_edge_band_is_the_shipping_value() -> None:
    from focusedgaze.config import RuntimeConfig

    left = edge_zones_for()[0]
    assert left.width == pytest.approx(RuntimeConfig().edge_zone)


@pytest.mark.parametrize("band", [0.0, -0.1, 0.5, 1.0])
def test_a_nonsense_edge_band_is_rejected(band) -> None:
    with pytest.raises(ConfigError, match="band"):
        edge_zones_for(band)


# ---------------------------------------------------------------------------
# Construction and lifecycle.
# ---------------------------------------------------------------------------


def test_a_target_with_no_area_is_rejected() -> None:
    with pytest.raises(ConfigError, match="no area"):
        Target("empty", 0.5, 0.5, 0.5, 0.5)


def test_duplicate_target_names_are_rejected() -> None:
    """The name is what the caller acts on, so two of them is a bug that would
    otherwise surface as the wrong action firing."""
    with pytest.raises(ConfigError, match="duplicate"):
        DwellSelector(targets=[PLAY, Target("play", 0.5, 0.5, 0.6, 0.6)])


@pytest.mark.parametrize("kw", [{"dwell_s": -1.0}, {"rearm_s": -1.0},
                                {"blink_grace_s": -1.0}, {"hysteresis": -0.1}])
def test_negative_timings_are_rejected(kw) -> None:
    with pytest.raises(ConfigError):
        DwellSelector(targets=[PLAY], **kw)


def test_reset_abandons_a_dwell_in_progress() -> None:
    """For switching screens: a dwell half-completed on a target that no longer
    exists must not complete against whatever replaced it."""
    sel = _selector()
    _hold(sel, 0.25, 0.25, until=0.8)
    sel.reset()
    assert all(s.selected is None for s in _hold(sel, 0.25, 0.25, start=0.9, until=1.5))


# ---------------------------------------------------------------------------
# Fixation averaging.
#
# A single reading carries the tracker's full per-frame noise; the median of
# many does not. Selection is a decision made over a whole dwell, so there is no
# reason to make it from one sample. Measured error on this project was 2.65 cm
# per reading, and at 30 fps a 1.05 s dwell holds 31 of them.
# ---------------------------------------------------------------------------


def test_the_aggregated_point_is_reported_for_the_caller_to_draw() -> None:
    """A cursor drawn from the raw reading can sit outside a target the selector
    considers hit. Reporting what it actually decided from prevents that."""
    sel = _selector(smoothing_s=0.3)
    state = sel.update(0.25, 0.25, 0.0)
    assert state.point == (0.25, 0.25)


def test_averaging_pulls_a_noisy_signal_toward_the_truth() -> None:
    """The whole point, as a measurement rather than an assertion of intent."""
    import random

    rng = random.Random(7)
    truth = 0.5
    sel = _selector(targets=[Target("t", 0.0, 0.0, 1.0, 1.0)], smoothing_s=1.0)

    raw_errors, smoothed_errors = [], []
    for i in range(60):
        noisy = truth + rng.gauss(0.0, 0.05)
        state = sel.update(noisy, truth, i * 0.033)      # ~30 fps
        if i >= 30:                                       # once the window is full
            raw_errors.append(abs(noisy - truth))
            smoothed_errors.append(abs(state.point[0] - truth))

    raw = sum(raw_errors) / len(raw_errors)
    smoothed = sum(smoothed_errors) / len(smoothed_errors)
    assert smoothed < raw / 2, (
        f"averaging barely helped: {raw:.4f} -> {smoothed:.4f}"
    )


def test_a_single_outlier_does_not_drag_the_point() -> None:
    """Median, not mean: eye data contains real outliers -- a saccade away and
    back, a half-blink -- and one of those moves a mean far more."""
    sel = _selector(targets=[Target("t", 0.0, 0.0, 1.0, 1.0)], smoothing_s=1.0)
    for i in range(9):
        sel.update(0.5, 0.5, i * 0.05)
    state = sel.update(0.99, 0.5, 0.5)          # one wild reading
    assert abs(state.point[0] - 0.5) < 0.05, f"outlier moved the point to {state.point}"


def test_smoothing_can_be_switched_off() -> None:
    """Lag is real: a 0.3 s window trails the eye by roughly half that. A live
    cursor may want the raw point."""
    sel = _selector(smoothing_s=0.0)
    sel.update(0.25, 0.25, 0.0)
    state = sel.update(0.35, 0.25, 0.1)
    assert state.point == (0.35, 0.25)


def test_the_window_forgets_old_readings() -> None:
    """Otherwise the point is dragged by where the eye was seconds ago."""
    sel = _selector(targets=[Target("t", 0.0, 0.0, 1.0, 1.0)], smoothing_s=0.2)
    for i in range(5):
        sel.update(0.1, 0.1, i * 0.05)
    for i in range(5, 20):
        state = sel.update(0.9, 0.9, i * 0.05)
    assert state.point[0] > 0.8, f"stale readings still in the window: {state.point}"


def test_reset_empties_the_smoothing_window() -> None:
    sel = _selector(smoothing_s=1.0)
    for i in range(5):
        sel.update(0.1, 0.1, i * 0.05)
    sel.reset()
    state = sel.update(0.9, 0.9, 1.0)
    assert state.point == (0.9, 0.9)


def test_negative_smoothing_is_rejected() -> None:
    with pytest.raises(ConfigError, match="smoothing_s"):
        DwellSelector(targets=[PLAY], smoothing_s=-0.1)


def test_overlapping_targets_resolve_to_the_first_listed() -> None:
    inner = Target("inner", 0.2, 0.2, 0.3, 0.3)
    sel = DwellSelector(targets=[inner, PLAY], dwell_s=1.0)
    assert sel.update(0.25, 0.25, 0.0).target == "inner"
