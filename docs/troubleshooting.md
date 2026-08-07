# Troubleshooting

Organised by what you notice, not by what the code does.

> **Start here: `focusedgaze check`.** Most of this page is environment problems that
> produce a *working* system that is quietly worse than you expect, and none of them raise.
> One command reports all of them with the remedy attached:
>
> ```bash
> focusedgaze check              # add --no-camera on a headless machine
> ```
>
> It checks the ONNX provider (including the silent CPU fallback), the model files and
> their digests, whether a calibration exists and is selected, whether the camera opens,
> and whether the room is lit. Exit code is 0 unless something is genuinely broken; a
> warning means usable-but-worse.

> **Status.** Most of these symptoms come from the original system, which is the working
> implementation focusedgaze is being extracted from. They apply to focusedgaze as each
> phase lands. Rows about commands that do not exist yet are marked.

## The cursor is in the wrong place

| It feels like | Usually because | Do this |
|---|---|---|
| Everything is offset in one direction | Drift over the session, which is normal | Look at the centre of the screen and recentre. In the original system that is the `c` key |
| It was fine, now it is uniformly shifted | You moved, or the chair did | Recentre first. Recalibrate only if that does not fix it |
| It is wrong in a way that varies by region | The calibration does not fit you | Recalibrate. See [calibration.md](calibration.md) |
| It is worst near the bottom of the screen | Expected. Bottom-edge error is 13–14% of screen against 3–8% elsewhere | Not a fault. Design around it, see [accuracy.md](accuracy.md) |
| A different person is using it | Calibration is per person and does not transfer | They calibrate for themselves |

**Check the cheap thing first.** In the original browser integration, the single most common
cause of a wrong cursor was not being fullscreen, because the coordinate mapping assumes the
canvas fills the screen. It was described in that project's cheat sheet as the number one
thing people forget. If your integration has an equivalent assumption, check it before
blaming the tracker.

## Nothing happens at all

| It feels like | Usually because | Do this |
|---|---|---|
| No cursor, no readings | No face detected | Check the camera is not muted. A muted camera returns frames, so the failure looks like a tracking bug rather than a device problem |
| Still no face, camera is on | The room is too dark | Turn a light on. A previous recording attempt here failed at a measured brightness of 2.8 out of 255 |
| Face detected, no gaze | The gaze model is missing | focusedgaze does not download it. See [install.md](install.md) |
| Import fails on a fresh install | No ONNX provider installed | Install one of `[directml]`, `[cuda]`, `[cpu]` |
| It worked yesterday | Camera held by another process | Close whatever else opened it. A crashed process can hold a webcam |

## It is slow or jerky

| It feels like | Usually because | Do this |
|---|---|---|
| Roughly five updates a second | Running on CPU | Install `[directml]` on Windows. Inference goes from ~104 ms to ~15 ms |
| Video is smooth, gaze lags | Camera backend | On Windows, MSMF gives ~31 fps at 720p against ~10 fps for DSHOW |
| Jittery even when you hold still | Filter reset or wrong timestamps | Pass real timestamps in seconds, not frame counters. See [usage.md](usage.md) |
| Lurches after tracking resumes | Filter kept stale state | Call `reset()` when the face is lost |

## Distance and positioning

| It feels like | Usually because | Do this |
|---|---|---|
| It refuses to calibrate | You are outside 45–65 cm, or not centred | Sit in the band. The gate is protecting the fit, not being fussy |
| Distance readings look wrong | No measured focal length | Measure it once at a known distance. The assumed-field-of-view fallback gave 121.2 cm where the measured path gave 117.4 cm for identical landmarks |
| Distance changed when you moved the script | You are on the original implementation | That was a real bug: the focal file resolved through a relative path, so the answer depended on your working directory. focusedgaze makes focal length explicit configuration |
| `evaluate()` returns `None` | Irises under one pixel apart | Face is far too distant, or landmarks are degenerate. `None` is a supported state |
| Distance is always wrong by a similar factor | Using the 468-point landmark model | The refined **478-point** model is required, because distance comes from the iris points |

## Development and tests

| It feels like | Usually because | Do this |
|---|---|---|
| Tests pass but prove nothing | Comparison tests skipped | Run `pytest -q -rs` and read the skip reasons. A skip here has already concealed a failing assertion |
| Comparison tests all skip | `FOCUSEDGAZE_LEGACY_DIR` unset | Point it at the original `gaze-detection` folder |
| `ModuleNotFoundError: websockets` in a skip reason | Missing `server` extra | The original server imports it. Install `[server]` to run the full comparison |
| The calibration golden test fails | A real numeric drift | The fixture now commits its own model and verifies it by digest, so a failure here is a change in behaviour rather than a moving input. Audit sections 37 and 43 |
| Numbers drift for no reason | Environment moved under you | The reference implementation's environment is part of the measurement. See `requirements-dev.txt` |

## Things that are working as intended

Worth listing, because each one gets reported as a bug.

**`download-models` refuses to fetch the gaze weights.** Deliberate. They derive from a
dataset restricted to non-commercial research, so focusedgaze does not distribute, mirror or
fetch them. It prints instructions and stops, and **exits 0**: this is the licence policy
working, not a failure, and a command that was permanently red on a correct installation
would be one nobody reads.

**Calibration will not run outside 45–65 cm.** The face crop scales with distance, so a
model fitted at one distance and used at another is being asked about inputs it never saw.

**Bottom-edge accuracy is much worse than the average.** Measured and expected.

**Two `GazeEstimator` instances will interfere in the original code.** Bounding box
smoothing is a module-level global there. focusedgaze makes it instance state, with a test
pinning it. *(Phase 2.)*
