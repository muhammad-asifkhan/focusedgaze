# Using focusedgaze

Two layers. The lower one is pure computation and is the point of the design. The
upper one owns a webcam and is a convenience built on top.

> **Status.** Everything under [What works today](#what-works-today) is
> implemented and its examples were executed against the shipped package. Anything
> under [The intended API](#the-intended-api) is a published contract, not code,
> and none of it runs.

> **Calibration is verified.** The pure-NumPy `apply()` reproduces the recorded
> legacy output exactly across all 169 fixture cases, its polynomial term
> ordering is checked against a real scikit-learn for degrees 1 to 8, and all
> four mutation checks pass: the comparison was shown to catch transposed
> coefficients, a permuted term ordering, swapped x/y coefficient sets and a
> degree mismatch. See `MIGRATION_AUDIT.md` §43.

---

## Install

```bash
pip install -e ".[cpu,calibration,dev]"
```

Pick exactly one ONNX execution provider: `cpu`, `directml` (Windows GPU), or
`cuda`. The base install declares none on purpose, so the choice is yours and a
missing provider is reported with a named remedy rather than a bare `ImportError`.

`calibration` adds scikit-learn and scipy. They are needed to *fit* a profile and
never to *apply* one.

---

## What works today

Configuration, result types, the exception tree, the One Euro filter, the
positioning gate, the asset registry, and calibration.

### Configuration

Every tunable that used to be a module-level constant is now a frozen dataclass,
so nothing can drift underneath you at runtime and two configurations can coexist
in one process. Defaults reproduce the deployed system exactly.

```python
from focusedgaze import GazeConfig

cfg = GazeConfig()
print(cfg.filter.min_cutoff)                 # 0.7
print(cfg.filter.beta)                       # 0.6
print(cfg.camera.width, cfg.camera.height)   # 1280 720
print(cfg.positioning.min_distance_cm)       # 45.0
print(cfg.runtime.dwell_ms)                  # 1050.0
```

Override by construction, from a dict, or from a file. `GazeConfig.from_file`
reads TOML or JSON, chosen by extension.

```python
from focusedgaze import GazeConfig, FilterConfig

snappier = GazeConfig(filter=FilterConfig(min_cutoff=1.0, beta=0.9))
partial  = GazeConfig.from_dict({"filter": {"beta": 0.9}, "camera": {"index": 1}})

assert GazeConfig.from_dict(snappier.to_dict()) == snappier
```

Invalid values raise `ConfigError`, which is also a `ValueError`, so the handler
you already have around argument validation will catch it.

```python
from focusedgaze import ConfigError, FilterConfig

try:
    FilterConfig(min_cutoff=-1.0)
except ConfigError as exc:
    print("rejected:", exc)
```

### Results and status

`GazeResult` is frozen, and `ok` is derived from `status` rather than stored
beside it, so the two cannot disagree.

```python
from focusedgaze import GazeResult, GazeStatus

r = GazeResult(x=0.5, y=0.4, pitch=0.1, yaw=-0.2,
               distance_cm=55.0, status=GazeStatus.OK, timestamp=0.0)
print(r.ok)                                  # True

blind = GazeResult(x=None, y=None, pitch=None, yaw=None,
                   distance_cm=None, status=GazeStatus.NO_FACE, timestamp=0.0)
print(blind.ok)                              # False
```

The five statuses separate cases the legacy pipeline collapsed into a bare `None`,
which is why a caller could not tell a bad frame from a broken install.

| Status | Meaning |
|---|---|
| `OK` | A usable gaze point. `x` and `y` are set. |
| `NO_FACE` | No face detected in this frame. |
| `OUT_OF_RANGE` | Face found, but too close or too far. |
| `OFF_CENTER` | Face found at a usable distance, but not centred enough. |
| `NOT_CALIBRATED` | Gaze angles available, but no profile to map them to a screen. |

These are recoverable per-frame conditions, not errors. None of them raise.
Genuine faults raise from the exception tree instead.

### Smoothing a jittery signal

Raw gaze coordinates jump around, partly from real micro-movements of the eye and
partly from model noise. The One Euro filter smooths them without the lag a moving
average introduces, by adapting how hard it smooths to how fast the signal moves.
It is usable on its own, on any 2D signal.

```python
from focusedgaze.core.filters import OneEuroFilter2D

f = OneEuroFilter2D(min_cutoff=0.7, beta=0.6, d_cutoff=1.0)
for i, (x, y) in enumerate([(0.50, 0.40), (0.52, 0.41), (0.51, 0.40)]):
    sx, sy = f.filter(x, y, i / 30.0)
    print(f"{sx:.4f} {sy:.4f}")
```

Higher `beta` is snappier on fast movement and lets more jitter through. Lower
`min_cutoff` is steadier at rest and adds lag. Note that smoothing harder stops
buying steadiness once the filter can no longer keep up: `examples/filter_demo.py`
measures that rather than asserting it.

### Checking whether the user is positioned usably

```python
from focusedgaze.core.positioning import PositioningGate
from focusedgaze import PositioningConfig

gate = PositioningGate(PositioningConfig())
print(gate.config.min_distance_cm, "to", gate.config.max_distance_cm, "cm")
```

Distance is estimated from the inter-pupil distance in pixels against a measured
focal length. Without a focal calibration it falls back to an assumed horizontal
field of view, which is less accurate. Both branches are pinned by the golden
fixtures.

### Exceptions

Everything the package raises descends from `GazeError`, so one handler covers the
library without swallowing unrelated failures.

```python
from focusedgaze import GazeError, ProviderError, ModelNotFoundError

print(issubclass(ProviderError, GazeError))               # True
print(issubclass(ModelNotFoundError, FileNotFoundError))  # True
```

`ProviderError` exists so a missing ONNX provider is an expected state with a
named remedy. `ModelNotFoundError` is also a `FileNotFoundError`, and
`ConfigError` is also a `ValueError`, because those are what callers already
catch.

### Models and the cache

Weights are never shipped in the wheel. The registry knows each asset, its
expected SHA-256, and whether it may be fetched automatically.

```python
from focusedgaze.assets import REGISTRY, cache_dir

print(cache_dir())
for asset in REGISTRY.values():          # REGISTRY is a dict keyed by asset name
    print(asset.name, asset.licence, "auto-download:", asset.auto_download)
```

Each entry records its licence, the URL it came from, the `source` page that URL
was read off, and the expected digest, so a checksum in this repository can be
traced back to something citable rather than taken on trust.

The MediaPipe face landmarker may download automatically. **The gaze weights may
not.** They derive from the Gaze360 dataset, which is non-commercial research
only, so the package prints instructions and stops rather than fetching them on
your behalf. See `NOTICE`.

Set `FOCUSEDGAZE_MODEL_DIR` to a directory of local models to skip the network
entirely, which is what you want on an offline or locked-down machine.

### Calibration

Read the calibration warning at the top of this page first.

Calibration is per person, per machine, and per seating position. It is the file
the whole system depends on.

```python
from focusedgaze import CalibrationCollector, list_profiles

collector = CalibrationCollector()
collector.add_sample(pitch=0.10, yaw=-0.20, target_x=0.1, target_y=0.1)
# ... many more samples, spread across the screen ...

print(list_profiles())        # named profiles already on this machine
```

Collection is headless and scriptable, so it does not require the on-screen
routine.

Fitting needs scikit-learn; applying does not. That split is the whole point of
the format. A saved profile stores plain polynomial coefficients plus metadata,
not a pickled scikit-learn object, so loading one neither requires scikit-learn
nor breaks when scikit-learn changes version, and it is not an arbitrary-code
execution risk the way unpickling is.

`migrate_pickle` converts an existing legacy `.pkl` profile to the new format.

---

## The intended API

> **None of this runs yet.** It is the contract Phases 2 and 4 are being built
> against, published so it can be reviewed and argued with before it exists.

### The pure layer

```python
from focusedgaze import GazeEstimator, CalibrationProfile

est = GazeEstimator(profile=CalibrationProfile.load("default"))
result = est.process(frame_bgr, timestamp=t)     # -> GazeResult
```

No camera, no window, no socket. You supply frames from anywhere: a video file,
another capture library, a shared camera, or a test fixture. This is what makes
the package testable in CI without hardware, and what makes it usable to someone
who already has frames.

Two `GazeEstimator` instances in one process must not interfere. In the legacy
pipeline the bounding-box smoothing lived in a module-level global, so they did.
Making that instance state is one of the two riskiest edits in Phase 2 and has a
test waiting for it.

### The convenience layer

```python
from focusedgaze import WebcamGazeTracker

with WebcamGazeTracker(profile="default") as tracker:
    for result in tracker.stream():
        if result.ok:
            print(result.x, result.y)
```

The context manager guarantees the camera is released, including on exception.

### A note on naming

The original brief named the calibration entry point `Calibrator`. It landed as
`CalibrationCollector` plus `fit_calibration`, splitting collection from fitting.
That is a change to the published contract, so it is flagged as an open question
rather than quietly adopted.

---

## Running the tests

```bash
set FOCUSEDGAZE_LEGACY_DIR=<path to the legacy gaze-detection directory>
pytest -q -rs
```

Always pass `-rs`. A skip in this suite has already concealed a failing assertion
once, and a bare `pytest -q` reports the count without the reason.

Tests needing a webcam are marked `hardware` and deselected by default, so the
suite runs on a machine with no camera and no GPU.
