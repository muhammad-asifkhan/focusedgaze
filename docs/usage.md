# Using focusedgaze

Two layers. The lower one is pure computation and is the point of the design. The upper one
owns a webcam and is a convenience built on top.

> **Status.** The two modules shown under [What works today](#what-works-today) are
> implemented, tested against the original implementation, and the examples below were
> executed to produce the output shown. Everything under
> [The intended API](#the-intended-api) is a contract, not code. It is marked as such and
> none of it runs.

## What works today

Two modules are extracted: the One Euro filter and the positioning gate.

### Smoothing a jittery signal

Raw gaze coordinates jump around, partly from real micro-movements of the eye and partly
from model noise. The One Euro filter smooths them without the lag a simple moving average
introduces, by adapting how hard it smooths to how fast the signal is moving. Slow movement
gets smoothed heavily, fast movement is passed through almost untouched, so the pointer sits
still when you stare and keeps up when you flick across the screen.

```python
from focusedgaze.core.filters import OneEuroFilter2D

# Defaults match the deployed system exactly.
f = OneEuroFilter2D(min_cutoff=0.7, beta=0.6, d_cutoff=1.0)

samples = [
    (0.500, 0.500, 0.000),
    (0.550, 0.480, 0.033),
    (0.520, 0.510, 0.066),
    (0.700, 0.300, 0.100),   # a fast flick
    (0.710, 0.295, 0.133),
]

for x, y, t in samples:
    sx, sy = f.filter(x, y, t)
    print(f"raw ({x:.3f}, {y:.3f})  ->  smoothed ({sx:.4f}, {sy:.4f})")
```

Output:

```
raw (0.500, 0.500)  ->  smoothed (0.5000, 0.5000)
raw (0.550, 0.480)  ->  smoothed (0.5075, 0.4973)
raw (0.520, 0.510)  ->  smoothed (0.5094, 0.4989)
raw (0.700, 0.300)  ->  smoothed (0.5540, 0.4550)
raw (0.710, 0.295)  ->  smoothed (0.5963, 0.4129)
```

Compare the second row with the fourth. On the small move the filter passes through about
15% of the raw change, and on the flick about 25%. That is the adaptive part working: the
velocity term opens the cutoff up when the signal is genuinely moving. It is a shift in
responsiveness, not a switch, so a fast movement still arrives smoothed over a few frames
rather than instantly.

The third argument is a timestamp in seconds, not a frame number. The filter derives its
sampling rate from the gaps between timestamps, so passing a counter will give you
confidently wrong smoothing. Pass `time.perf_counter()` or the capture timestamp.

Call `reset()` when the signal is interrupted, for example when the face is lost or the user
switches modes. Without it the filter glides from a stale value when tracking resumes.

`OneEuroFilter` is the same thing for a single axis. `OneEuroFilter2D` filters the two axes
independently, exactly as the original system does.

### Checking whether the user is positioned usably

`PositioningGate` answers a question you need before the gaze reading means anything: is the
face close enough, far enough, and centred enough for the calibration to apply? It works on
MediaPipe landmarks alone and needs no gaze model.

```python
from focusedgaze.core.positioning import PositioningGate, PositioningConfig

cfg = PositioningConfig()
print(f"accepted distance: {cfg.min_distance_cm:.0f} to {cfg.max_distance_cm:.0f} cm")

gate = PositioningGate()
print("gate ready:", type(gate).__name__)
```

Output:

```
accepted distance: 45 to 65 cm
gate ready: PositioningGate
```

Distance is estimated from the pixel distance between the irises against an assumed real
interpupillary distance, which means it needs the refined 478-point landmark model rather
than the 468-point one.

**Measure your focal length if you can.** Without a measured value the gate falls back to
estimating from an assumed 60 degree horizontal field of view, and that estimate is rough.
The difference is not academic: for identical landmarks, the measured and assumed paths
produced **117.4 cm against 121.2 cm** in testing. A one-off measurement at a known distance
is worth taking, and the gate can record it.

That discrepancy was originally a bug rather than a configuration choice. The gate resolved
its focal file through a relative path, so the same landmarks gave different distances
depending on which directory you launched Python from. focusedgaze makes the focal length
explicit configuration instead of an implicit file lookup.

## The intended API

> **None of this runs.** It is the contract Phases 2 to 4 are being built against, published
> here so the shape can be argued with before it is set, rather than discovered afterwards.

### The pure layer

```python
from focusedgaze import GazeEstimator, CalibrationProfile

est = GazeEstimator(profile=CalibrationProfile.load("default"))
result = est.process(frame_bgr, timestamp=t)

if result.ok:
    print(result.x, result.y)        # screen coordinates in [0, 1]
    print(result.pitch, result.yaw)  # raw model output, radians
```

`process()` does no I/O. It takes a frame and a timestamp and returns a result. That is the
whole interface, and it is what makes the library usable from a video file, from another
capture library, from a camera shared with a hand tracker, or from a test that has to be
reproducible.

Two `GazeEstimator` instances in one process must not interfere with each other. In the
original code they would, because bounding-box smoothing lives in a module-level global.
Making that instance state is one of the two riskiest edits in Phase 2, and it has a test
pinning it.

### The convenience layer

```python
from focusedgaze import WebcamGazeTracker

with WebcamGazeTracker(profile="default") as tracker:
    for result in tracker.stream():
        if result.ok:
            print(result.x, result.y)
```

Use the context manager. It guarantees the camera is released, including on an exception,
and a webcam held by a crashed process is an annoying thing to debug.

## Running the tests

The suite compares focusedgaze against the original implementation where both exist.

```bash
set FOCUSEDGAZE_LEGACY_DIR=<path to your gaze-detection checkout>
pytest -q -rs
```

Point `FOCUSEDGAZE_LEGACY_DIR` at the original `gaze-detection` folder. Without it the
comparison tests skip rather than fail, which is correct behaviour for an environment that
cannot reach the reference implementation, but it means a clean run is not proof of much.

**Always pass `-rs`.** It prints the reason for each skip. In this project a skip has
already concealed a failing assertion for an unknown period, and a bare run reports the
count without the cause.

The calibration fixture carries its own committed model and verifies it by SHA-256, so a
recalibration elsewhere on the machine cannot invalidate it. If that digest ever fails to
match, the error names both digests and refuses to continue rather than reporting a
meaningless numeric drift.
