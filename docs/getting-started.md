# Getting started

A step-by-step guide from nothing to a working gaze-controlled application, and
how to swap the model underneath it.

Every command is copy-pasteable. Commands are shown for PowerShell; on Linux or
macOS use `export` instead of `setx` and forward slashes.

---

## Before you start

| You need | Notes |
|---|---|
| A webcam | Any. 720p is enough. Built-in laptop cameras work. |
| Python 3.12+ | 3.12, 3.13 and 3.14 are supported. |
| A GPU | **No.** The default path runs at ~2 ms per frame on a plain CPU. |
| Good lighting | The single biggest environmental factor. Face a window or a lamp, not away from one. |

Budget about **five minutes** to a working tracker, most of it the calibration.

---

## Step 1 — Install

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install "focusedgaze[intel,calibration] @ git+https://github.com/muhammad-asifkhan/focusedgaze.git@dev"
```

Two extras, and both are needed for this guide:

- `intel` — the OpenVINO runtime for the default gaze model.
- `calibration` — scikit-learn, needed to **fit** a calibration. *Applying* one
  needs only NumPy, so an application that ships a fitted profile can skip it.

> **Why the git URL?** The PyPI release `focusedgaze 0.0.0` is a placeholder — a
> 21 KB wheel of stubs that imports cleanly and does nothing. Install from git
> until a real release is published.

---

## Step 2 — Choose a backend

focusedgaze can drive two different gaze models. **Pick one now**, because a
calibration made against one does not work with the other.

| | **Intel** (recommended) | **L2CS** |
|---|---|---|
| Licence | Apache-2.0 — redistributable | Gaze360 — non-commercial research, **cannot be redistributed** |
| Download | Automatic | You fetch it by hand |
| Size | 7.5 MB | 91 MB |
| Speed (measured, AMD integrated) | **2.0 ms** | 141.7 ms |
| Needs a GPU | No | Benefits from one |

**Choose Intel unless you have a specific reason not to.** It is the only one
that can ship inside an application you distribute.

Every command below takes `--backend intel` or `--backend l2cs`. There is
currently **no config file for this** — pass the flag each time, or set it in
Python (see Step 8).

---

## Step 3 — Get the models

### Intel (automatic)

```powershell
focusedgaze download-models --backend intel
```

Fetches two files (`.xml` topology and `.bin` weights) from Intel's own storage,
verified against a SHA-256 digest. Done.

### L2CS (manual, by licence)

focusedgaze will never download these. You must obtain them yourself:

1. Download `L2CSNet_gaze360.pkl` from the
   [official L2CS-Net distribution](https://github.com/Ahmednull/L2CS-Net).
2. Install the conversion tools (one time, ~2.5 GB):
   ```powershell
   pip install "focusedgaze[export]"
   pip install git+https://github.com/Ahmednull/L2CS-Net.git
   ```
3. Convert it:
   ```powershell
   focusedgaze setup --weights .\L2CSNet_gaze360.pkl
   ```

Read `NOTICE` first. The Gaze360 licence restricts use to non-commercial
research and forbids redistribution, including of models trained on it.

---

## Step 4 — Check your machine

```powershell
focusedgaze check --backend intel
```

Every line should read `[ ok ]`. What each one catches:

| Check | Why it matters |
|---|---|
| `model:*` | Files present **and digest-verified** — catches a truncated download |
| `camera` | Confirms frames actually arrive at the requested resolution |
| `camera-brightness` | A dark room degrades tracking silently |
| `calibration` | Warns if you have no profile yet — expected on a fresh install |

Fix any `[FAIL]` before continuing. A `[warn]` about calibration is normal at
this point.

---

## Step 5 — Calibrate

**This is the step that decides your accuracy.** Read the three rules before you
run it.

```powershell
focusedgaze calibrate --backend intel --name me --grid
```

Takes about 90 seconds: 36 dots, each **hollow while you settle on it, solid
while you are being measured**.

### The three rules

1. **Sit where you will actually use it.** A profile is only valid near the
   distance it was collected at. Calibrating leaning forward and then sitting
   back measurably wrecks accuracy — 1.4 cm became 7.9 cm in our testing.
2. **Move your eyes, not your head.** The pre-flight shows your live distance;
   aim for a comfortable, repeatable posture.
3. **Actually look at each dot.** The solid phase is what is recorded.

### Reading the result

```
Collected 413 in-zone samples (reference sweep: 1728).
Coverage per screen region:
  top     left= 53  centre= 54  right= 54
  middle  left= 46  centre= 42  right= 49
  bottom  left= 44  centre= 22  right= 23

Fitted 413 samples -> ...\profiles\me.json
       degree 3, 14 outlier(s) dropped, fit error 0.1168, held-out error 0.1224
       collected at 59 cm.
```

- **Coverage** should be non-zero everywhere. A region with no samples is where
  the fit extrapolates, and it will be your worst region.
- **Held-out error** below ~0.15 is healthy.
- If it refuses to fit, it prints exactly what it discarded and why:
  ```
  Discarded 171 of 232 readings:
      140  too far (median 68 cm; this needs 45-65 cm, so move closer)
  ```

---

## Step 6 — Check it worked

```powershell
focusedgaze demo --backend intel --profile me
```

Coordinates should track your gaze. `x` and `y` are fractions of the screen,
origin **top-left**, `y` increasing **downward**.

To measure it properly:

```powershell
focusedgaze accuracy --backend intel --profile me --save result.json
```

Nine points, ~25 seconds. What to expect on a typical laptop:

| Average error | Verdict |
|---|---|
| under 2 cm | very good |
| 2–3 cm | normal, usable |
| 3–6 cm | works, but check your distance matched the calibration |
| over 6 cm | something is wrong — see Troubleshooting |

**Run it twice.** A single run is not a characterisation; run-to-run variation on
the same profile has been measured at 4×.

---

## Step 7 — Use it from Python

```python
from focusedgaze import WebcamGazeTracker

with WebcamGazeTracker(profile="me") as tracker:
    for result in tracker.stream():
        if result.ok:
            print(result.x, result.y)          # 0.0-1.0, origin top-left
```

`result.status` tells you why a frame is unusable — `NO_FACE`, `OUT_OF_RANGE`,
`OFF_CENTER`, `NOT_CALIBRATED`. None of these raise; they are normal per-frame
states.

### Selecting the backend in Python

```python
from dataclasses import replace
from focusedgaze import GazeConfig, WebcamGazeTracker
from focusedgaze.config import ModelConfig

config = GazeConfig(model=ModelConfig(backend="intel"))

with WebcamGazeTracker(profile="me", config=config) as tracker:
    ...
```

### Recentring — do this at session start

The mapping drifts between sessions as your posture changes. One dot fixes it:

```python
# show a dot at the screen centre, collect ~1 second of readings
readings = [(r.pitch, r.yaw) for r in samples if r.pitch is not None]
tracker.estimator.recentre(readings)     # everything after is corrected
```

This is the largest single accuracy gain available to an application, and it
takes one second of the user's time.

---

## Step 8 — Build a gaze-controlled application

```python
from focusedgaze import DwellSelector, Target, WebcamGazeTracker, edge_zones_for

selector = DwellSelector(
    targets=[
        Target("play", 0.05, 0.35, 0.45, 0.65),
        Target("quit", 0.55, 0.35, 0.95, 0.65, dwell_s=2.0),   # slower on purpose
    ],
    edges=edge_zones_for(0.09),
)

with WebcamGazeTracker(profile="me") as tracker:
    for result in tracker.stream():
        state = selector.update(result.x, result.y, result.timestamp)

        if state.selected:
            run(state.selected)
        if state.edge:
            scroll(state.edge.name, speed=state.edge.held_s)

        draw_cursor(state.point)                  # the aggregated point
        draw_progress_ring(state.target, state.progress)
```

It handles blinks, jitter and repeat-fire for you. Two things to get right:

- **Draw `state.point`, not `result.x/y`.** The selector decides from an
  aggregate; a cursor drawn from the raw reading can sit outside a target the
  selector considers hit.
- **Draw `state.progress`.** A dwell with no visible progress is
  indistinguishable from a frozen program.

### Size your targets for the tracker

This matters more than the model. From measured error:

| Target size | Hit rate | Fits on a 34 cm screen |
|---|---|---|
| 4 cm | 22% | 32 targets |
| **6 cm** | **89%** | **15 targets** |
| 12 cm | 100% | 2 targets |

**Aim for 6 cm or larger.** Apple's own front-camera eye tracking has similar
precision; it feels good because of large targets and dwell feedback, not
because the sensor is better.

---

## Step 9 — Changing models and weights

### Switch backend

```powershell
focusedgaze calibrate --backend l2cs --name me-l2cs --grid
focusedgaze demo      --backend l2cs --profile me-l2cs
```

**You must recalibrate.** The two models have different angle conventions; a
profile from one produces nonsense with the other.

### Where the files live

```powershell
focusedgaze check --json      # shows the model directory
```

Default: `%LOCALAPPDATA%\focusedgaze\Cache\models` on Windows.

### Install a model somebody else converted

If a colleague has already converted the L2CS weights, you do not need torch or
the download:

```powershell
focusedgaze setup --onnx .\l2cs_gaze360.onnx
```

It **loads and runs the graph before copying it**, so a wrong or truncated file
fails while it is still yours rather than after it is in the cache.

### Point at a shared directory instead

```powershell
setx FOCUSEDGAZE_MODEL_DIR "\\your-share\models\focusedgaze"
```

This variable is **exclusive**: when set, nothing else is consulted and no
network access happens for any asset. Good for locked-down or offline machines,
and for a team sharing one converted model.

> Passing the L2CS `.onnx` to someone else is redistribution. The Gaze360 licence
> permits it only among direct research colleagues at the same institution. The
> Intel model has no such restriction.

### Where profiles live

```powershell
focusedgaze calibrate --list        # * marks the active one
setx FOCUSEDGAZE_PROFILE_DIR "D:\profiles"
```

Profiles are plain JSON — they load without scikit-learn and execute no code.

---

## Step 10 — Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `NOT_CALIBRATED`, `x` is `None` | No profile | Run Step 5 |
| Cursor consistently offset | Posture drift since calibration | `recentre()`, or recalibrate |
| Cursor pulls toward screen centre | You are sitting further away than you calibrated | Sit at the calibration distance, or `--compensate-distance` |
| Accuracy fine, then bad after a break | New posture | `recentre()` at session start |
| Calibration refuses to fit | Out of the 45–65 cm zone | Read the discard breakdown it prints |
| Very low frame rate | Using L2CS without a GPU | `--backend intel` |
| Worse near screen edges | Normal — corners are the hardest region | Keep targets away from edges |
| Distance always wrong by a constant factor | The 468-point landmark model, which has no iris points | `focusedgaze check` verifies the digest |

---

## What this cannot do

- **Sub-centimetre accuracy.** That needs infrared illumination and glint
  tracking. Vision Pro measures ~0.9°, Tobii 0.3–0.6°; a webcam lands at 2–3°.
  The gap is the sensor, not the software.
- **Work without calibration.** Gaze mapping is per-person, per-machine and
  per-seating-position. There is no default profile, and shipping one would be
  worse than none.
- **Survive a moved camera.** Move the laptop or change chairs, and recalibrate.
