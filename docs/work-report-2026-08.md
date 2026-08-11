# Work report: making focusedgaze usable, and shippable

Branch: `feat/apache-backend-and-gaze-control`, from `dev`.

The package could not be used by a person and could not be installed by anyone
else. Both are now addressed. Along the way six defects were found, four of them
latent for months and none of which failed loudly.

Everything below was measured on one machine: Windows 11, AMD Radeon integrated
graphics, Python 3.12, a 34.4 x 19.4 cm screen. Raw reports are in
[`measurements/`](measurements/).

---

## 1. The library could not produce screen coordinates

`focusedgaze calibrate` printed *"Interactive calibration needs the gaze
pipeline, which is Phase 2 and is not implemented yet"* and exited 1. That
message had outlived what it described: Phase 2 had shipped. The real blocker was
that **nothing drew a dot**. Without a profile every `process()` returned
`NOT_CALIBRATED` with `x=None, y=None`, so the headline feature was unreachable.

Added `calibration/screen.py`: a full-screen renderer using the `cv2` already
depended on, hooked to the `on_frame` seam `ui.py` had been designed around all
along. Both `calibrate` and `accuracy` use it — `accuracy` had the same gap and
said *"Look at each dot as it is named"* while naming nothing.

Two collection modes:

- **Pursuit** (default) — follow a moving dot.
- **Dwell grid** (`--grid`) — 36 static dots, hollow while you settle, solid
  while recording.

The grid exists because pursuit is frame-rate dependent in a way that is not
obvious: the dot is redrawn once per pipeline iteration, so at 7 fps it *steps*
rather than glides, and a stepping target does not elicit smooth pursuit. Worse,
slowing the sweep to collect more samples makes it *worse* — at `--seconds 120`
the dot takes 20 seconds to cross the screen, which is effectively stationary and
the eye wanders off it. Measured: 120 s collected 917 samples against 45 s's
~320, and **lost** horizontal gain (0.79 → 0.62) with held-out error rising from
0.089 to 0.163. Noise in a least-squares predictor biases the fitted gain toward
zero.

## 2. The interaction layer

A consumer with only `(x, y)` has to invent dwell timing, jitter tolerance,
blink handling and re-arming. New `focusedgaze.control`, pure and clock-injected:

```python
selector = DwellSelector(targets=[Target("play", 0.05, 0.35, 0.45, 0.65)],
                         edges=edge_zones_for(0.09))
state = selector.update(result.x, result.y, result.timestamp)
if state.selected:
    run(state.selected)
```

Three tolerances, each from a measured property:

| | Why |
|---|---|
| **Hysteresis** | The gaze jitters by more than a small target's width. A plain inside/outside test flickers and cancels a dwell the user experiences as steady. |
| **Blink grace** | Blinks give `NO_FACE` for several frames; at 7 fps that is most of a second. The dwell survives, and the clock **freezes** so a well-timed blink cannot select. |
| **Re-arm** | Without it, resting on a control fires it forever — the Midas touch problem. |

## 3. The install problem, and the Apache-2.0 backend

`pip install focusedgaze` cannot ship a working system, because the L2CS weights
derive from Gaze360, whose licence is explicit:

> §2(3): the material *"will not be used nor included in commercial applications
> in any form (such as original files, encrypted files, files containing
> extracted features, **models trained on dataset, other derivative works**,
> etc)"*
>
> *"The Licensed Material will not be copied nor distributed in any form other
> than for Your backup."*

NOTICE described this as "a conservative reading". It is not — it is the text,
and it names trained models specifically. Research use is further limited to
*"direct research colleagues who belong to the same research institution"*.

So a second backend was added: Intel's `gaze-estimation-adas-0002`, **Apache-2.0,
Copyright Intel Corporation**, per the model's own `model.yml`. It is
redistributable, auto-downloadable and digest-verified.

Measured on this machine:

| | L2CS-Net | Intel |
|---|---|---|
| Licence | Gaze360, distribution prohibited | **Apache-2.0** |
| Size | 91 MB | **7.5 MB** |
| Latency | 141.7 ms (DirectML) / 236.3 ms (CPU) | **2.0 ms (CPU)** |
| Auto-download | Never | **Yes** |

**70x faster, on CPU, with no GPU involved.** The camera becomes the bottleneck
rather than the model. That matters beyond throughput: 140 ms of lag makes dwell
interaction feel broken, and it is the direct cause of the stepping-dot problem
in §1.

`ModelConfig.backend` selects between them and **defaults to `l2cs`**, so every
existing profile, fixture and recorded measurement keeps its meaning. `--backend`
is wired through `download-models`, `setup`, `check`, `calibrate`, `accuracy`,
`demo` and `serve`. The asset registry is backend-aware, so `check` reports the
model you selected rather than failing on one you have no reason to own.

**Accuracy of the Intel backend is unmeasured.** Its published 6.95° is on
Intel's own validation set and is not comparable with L2CS's Gaze360 figure. The
eye-crop geometry in `core/eyes.py` is a chosen convention and may need tuning.

## 4. Defects found

Each of these produced plausible, wrong behaviour rather than an error.

**The positioning gate was a no-op.** `_is_usable()` checked only that a reading
carried pitch and yaw — but the estimator's gate *reports* rather than vetoes, so
`OUT_OF_RANGE` and `OFF_CENTER` results arrive complete with angles. The check
passed every state it existed to reject. Consequence, from the data: an accuracy
run collected at **69.4 cm** against a 65 cm limit.

**The gate was blind during calibration.** With no profile the estimator returns
`NOT_CALIBRATED` *before* consulting the zone, so `OUT_OF_RANGE` cannot appear on
that path at all. Distance is now checked directly against the configured bounds.

**Out-of-zone samples were being trained on.** `collect_pursuit_samples`'
docstring claimed a reading taken outside the positioning zone "simply had no
angle" and was dropped for free. It never did. Every sweep silently trained on
samples from positions the calibration was not valid at.

**MediaPipe crashed on tracking loss.** `FaceLandmarker.reset()` zeroed the frame
counter "so timestamps restart cleanly". They must not: `detect_for_video`
requires strictly increasing stamps and keeps that state itself, so rewinding
replays stamps it has seen and raises `ValueError: Input timestamp must be
monotonically increasing` — uncaught, killing the process. Trigger: lose the face
and find it again. It took out **two of three** calibration attempts.

**`export-onnx` wrote to the wrong directory.** Its default output was a relative
path — the current directory — while the runtime only reads `model_dir()`.
Following the documented instructions exactly left `check` still reporting the
model missing, with nothing connecting the two.

**`apply()` clamped, so the "raw" diagnostic was inert.** A field added to record
unclamped predictions was recording already-truncated ones. Now `apply_raw()`,
with `apply()` defined as `clamp(apply_raw())` — delegating, not duplicating, so
they cannot drift bit-wise.

Two tests were also passing for the wrong reason: one asserted the export
dependency message while relying on torch genuinely being absent (it broke the
moment the `export` extra was installed), and one began **opening the webcam and
running a live calibration inside the unit suite** once a model existed on disk.

## 5. Accuracy work

Distance was found to be the dominant remaining error. A calibration is only
valid at the distance it was collected at — screen offset for a fixed gaze angle
scales with viewing distance — so a profile taught at ~47 cm and used at 63.7 cm
under-reaches by roughly the ratio:

```
predicted gain from geometry : 0.71 - 0.78
measured gain                : x=0.67  y=0.61
```

Three changes followed:

- `CalibrationProfile.distance_cm` records where a profile was collected.
- The pre-flight shows live distance while holding you, so you can calibrate
  where you actually sit.
- `PositioningConfig.compensate_distance` (**off by default**) rescales
  predictions for the difference, via `profile.rescaled_for`.

And `GazeEstimator.recentre()`: a session offset measured from a single centre
dot. Across five runs the whole mapping shifted bodily between sessions by −0.25
to +0.34 of screen height, twice on an *identical* profile minutes apart. This is
standard practice for eye trackers and is the largest single gain available to a
consumer.

### Results

| Run | Setup | Average | x gain | y gain |
|---|---|---|---|---|
| 1 | Migrated profile from another machine | 5.07 cm | 0.79 | 0.43 |
| 2 | Pursuit, 120 s | 7.64 cm | 0.62 | 0.58 |
| d | Grid, calibrated ~47 cm, measured 63.7 | 5.19 cm | 0.67 | 0.61 |
| **e** | **Grid, calibrated 59.4 cm, measured 60.3** | **2.65 cm** | **0.87** | **0.83** |

**2.65 cm is ~2.5°**, which is at published webcam state of the art (2–3°). For
context, Vision Pro measures 0.93–1.11° using IR cameras at the eye — a sensing
gap, not a model gap.

Selection accuracy, computed from run e's per-point errors:

```
   target   hit rate   grid that fits
     5 cm      44%     6 x 3 = 18 targets
     6 cm      89%     5 x 3 = 15 targets
    12 cm     100%     2 x 1 =  2 targets
```

89% across 15 on-screen targets today. The gap to 100% is driven by a single
outlier at 5.4 cm in one corner — the region the sweep covered worst.

## 6. Also fixed

The sweep's rows landed **2-1-2** across the three screen bands `region_of`
reports, so the middle third collected half the samples of the top and bottom on
every run by every user — measured at 362 / 187 / 368. No duration fixes it;
longer sweeps preserve the ratio. `rows` now defaults to 6, giving 2-2-2, and a
control test asserts that 4 and 5 rows *are* skewed so the guard cannot pass for
the wrong reason.

Calibration now reports what it discarded and why, which previously looked
identical whether the user was absent, off centre, or three centimetres too far:

```
Discarded 171 of 232 readings:
     22  no face detected
    140  too far (median 68 cm; this needs 45-65 cm, so move closer)
      9  off centre
```

## 7. Open

- **Intel backend accuracy is unmeasured** against L2CS on a real face.
- **The eye-crop convention** in `core/eyes.py` is unvalidated. Head-pose order
  (`[yaw, pitch, roll]`, degrees), channel order and vector handedness are all
  pinned by tests but not confirmed against a person.
- **Fixation averaging** during dwell is not implemented. Independent noise falls
  as 1/√n; at 30 fps a one-second dwell gives 31 samples instead of 7.
- **Head-pose features** in the calibration polynomial. The 4x4 matrix is already
  captured and exposed via `GazeEstimator.last_observation`, but the polynomial
  still takes only `(pitch, yaw)`.
- **PyPI 0.0.0 is a placeholder** — a 21 KB wheel of stubs. Publishing a real
  release is the remaining blocker on `pip install` meaning anything.

## Verification

```
690 passed, 7 skipped, 5 deselected
ruff:  All checks passed!
mypy:  Success: no issues found in 34 source files
```

Baseline at the start of this work was 509 tests.
