# API reference

Reference for every public name.

**Everything documented here is implemented.** This page once split the package into
"implemented" and "not implemented yet" and listed only `core.filters` and
`core.positioning` under the first heading; that stopped being true and the split has been
removed rather than left to rot. `focusedgaze.__all__` now exports the full public surface,
not just `__version__`.

This reference lags the code in coverage, not in accuracy: names it does not mention may
still exist. `focusedgaze.__all__` and the module docstrings are authoritative. Recent
additions not yet written up here include `focusedgaze.calibration.screen` (`DotRenderer`,
`screen_size_px`, `target_to_pixels`) and the `CalibrationAborted` exception.

---

## `focusedgaze.core.filters`

### `OneEuroFilter`

One Euro filter for a single scalar signal. Smooths without the lag of a fixed low-pass, by
adapting the cutoff to the signal's velocity.

```python
OneEuroFilter(min_cutoff: float = 0.7, beta: float = 0.6, d_cutoff: float = 1.0)
```

| Argument | Default | Meaning |
|---|---|---|
| `min_cutoff` | `0.7` | Cutoff at rest. Lower smooths more when still |
| `beta` | `0.6` | Velocity coefficient. Higher is more responsive when moving |
| `d_cutoff` | `1.0` | Cutoff for the internal derivative estimate |

The defaults are the deployed system's values and are pinned by a golden fixture to within
1e-9 of the pre-refactor output. Changing them changes behaviour for every existing user.

**`filter(x: float, t: float) -> float`**

Feed one sample and get the smoothed value. `t` is a timestamp **in seconds**, not a frame
index: the filter derives its sampling frequency from the difference between successive
timestamps. Passing a counter produces confidently wrong smoothing.

Non-monotonic timestamps are handled rather than rejected. The frequency is only recomputed
when time advances, so a repeated or backwards timestamp, which a paused capture thread
really does produce, will not divide by zero or invert the filter. This is covered by the
fixture in both directions.

**`reset() -> None`**

Forget all history. Call when the signal is interrupted, such as a lost face or a mode
change, so the filter does not glide from a stale value when it resumes.

### `OneEuroFilter2D`

Two independent `OneEuroFilter` instances, one per axis, matching the original system. The
axes are filtered separately because horizontal and vertical gaze error are not correlated
closely enough for a shared velocity estimate to help.

```python
OneEuroFilter2D(min_cutoff: float = 0.7, beta: float = 0.6, d_cutoff: float = 1.0)
filter(x: float, y: float, t: float) -> tuple[float, float]
reset() -> None
```

---

## `focusedgaze.core.positioning`

### `PositioningConfig`

Frozen dataclass of the tunables. Defaults preserved exactly from the original.

| Field | Default | Meaning |
|---|---|---|
| `min_distance_cm` | `45.0` | Nearest accepted distance |
| `max_distance_cm` | `65.0` | Furthest accepted distance |
| `warn_margin_cm` | `5.0` | Band inside the limits where guidance warns before rejecting |
| `center_tolerance` | `0.12` | Allowed offset from frame centre, as a fraction of frame size |
| `real_ipd_cm` | `6.3` | Assumed interpupillary distance, used to infer distance |
| `assumed_hfov_deg` | `60.0` | Fallback horizontal field of view when focal length is unmeasured |

### `FocalCalibration`

A measured focal length in pixels, tied to the frame width it was measured at.

**`for_width(frame_width: int) -> float`** scales the stored focal length to a different
frame width.

**`load(path) -> FocalCalibration` / `save(path) -> None`** read and write JSON. The path is
explicit and there is no default, which is deliberate: resolving it implicitly is what
caused the original bug where the same landmarks produced different distances depending on
the working directory.

**`to_dict()` / `from_dict(data)`** for embedding in your own config.

### `PositioningStatus`

Result of an evaluation. `as_dict()` gives a plain dictionary suitable for serialising to a
client.

### `PositioningGate`

```python
PositioningGate(config: PositioningConfig | None = None, focal: FocalCalibration | None = None)
```

**`evaluate(landmarks, frame_width, frame_height) -> PositioningStatus | None`**

Returns `None` when the irises are under one pixel apart, meaning the face is far too
distant or the landmarks are degenerate. `None` is a supported state, not an error.

`landmarks` must come from the refined **478-point** MediaPipe model, because the iris
points are required. The 468-point model will not work.

**`focal_px(frame_width: int) -> float`**

The focal length in use. Returns the measured value when one was supplied, otherwise derives
one from `assumed_hfov_deg`. The two paths give materially different answers, 117.4 cm
against 121.2 cm for identical landmarks in testing, so which one you are on matters. Both
are covered by separate fixtures.

**`measure_focal(ipd_px: float, frame_width: int, distance_cm: float) -> FocalCalibration`**

Compute a focal calibration from a known distance. Sit at a measured distance, capture, and
save the result.

---

## Everything on this page exists

This section used to be a "Not implemented yet" table listing `GazeEstimator`,
the landmark and model wrappers, the result types, the exception tree, the
capture layer, calibration, every CLI command and the WebSocket server. **All of
it shipped.** The table was left behind after the header above was rewritten to
say the split had been removed, so the page spent a release contradicting
itself — and a reader deciding whether to adopt the package would have believed
the table, because a specific list reads as more authoritative than a general
claim.

Nothing is deferred to a later phase. What is genuinely absent is recorded in
`docs/getting-started.md` under "What this cannot do", which is about the
limits of webcam gaze rather than about unfinished work.

## Added since 0.1.0

Names that exist and are not yet written up in full below. Module docstrings are
authoritative.

| Name | Module | What it is |
|---|---|---|
| `LiveGazeSource` | `focusedgaze.server` | The camera behind `GazeServer`, satisfying `GazeSource`. What `focusedgaze serve` now runs by default. |
| `STALE_AFTER_S` | `focusedgaze.server` | How old a reading may be before `latest()` reports it unusable. |
| `CalibrationProfile.backend` | `focusedgaze.calibration` | Which gaze backend the profile was fitted against. `None` on profiles written before 0.1.1. |
| `CalibrationProfile.backend_complaint()` | `focusedgaze.calibration` | `(severity, message)` when a profile and a backend disagree, else `None`. |
| `profiles_for_backend()` | `focusedgaze.calibration.profile` | Names of saved profiles matching a backend. Excludes unstamped ones. |
| `check_runtime()`, `check_openvino()` | `focusedgaze.diagnostics` | The inference-runtime check, dispatched per backend. |

Two behaviour changes worth knowing when reading older code:

- **`ModelConfig.backend` now defaults to `"intel"`**, not `"l2cs"`. See
  `docs/getting-started.md` Step 2.
- **`GazeEstimator.__init__` raises `CalibrationError`** when the profile was
  calibrated against a different backend, and logs a warning when the profile
  records no backend at all.

## Thread safety

Nothing in focusedgaze is thread-safe, and this is worth stating explicitly because the
filters carry mutable state between calls.

An `OneEuroFilter` holds the previous sample and timestamp. Calling `filter()` on one
instance from two threads will interleave those updates and produce nonsense that looks
plausible. Use one instance per stream.

The same applies to `GazeEstimator` when it exists. Its whole design point is that bounding
box smoothing becomes instance state rather than a module global, which makes two instances
in one process safe from each other, but it does not make one instance safe from two
threads.

## A defect that is preserved deliberately

The ONNX graph names its first output tensor `pitch_bins`, and that tensor contains **yaw**.
The names are wrong in the exported graph, and the original code unpacks them in the correct
order despite the labels.

focusedgaze carries that behaviour over unchanged, with the explanation in a comment and a
regression test pinning the decode, because it is exactly the kind of thing a refactor
silently reintroduces by trusting the labels.

Related: the angle is a **softmax-weighted expectation** over 90 bins, not an argmax. An
argmax implementation agrees with it only when the distribution is sharply unimodal, and
otherwise returns a plausible, smooth, wrong answer.
