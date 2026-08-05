# focusedgaze: the complete usage guide

**What this document is.** The full picture of how `focusedgaze` is used once every
phase has landed: installation, first-run setup, both API layers, the CLI, the
WebSocket server, and precisely what goes in and what comes out of each.

**What this document is not.** A description of the package as installed today.
Large parts of it are the target, not the state. Every section carries a marker:

| Marker | Meaning |
|---|---|
| **[SHIPPED]** | Implemented and tested. Works now. |
| **[UNPROVEN]** | Implemented, but not yet verified numerically. Do not trust its numbers. |
| **[PLANNED]** | Contract only. No code behind it yet. |

That distinction is load-bearing. This project has twice shipped documentation
that described an intention as though it were behaviour, and both times the
document was believed over the code. If a marker and the code disagree, the code
is right and the marker is a bug.

---

## 1. The pipeline, end to end

```
camera frame (BGR, 1280x720)
  -> MediaPipe FaceLandmarker            478 landmarks, VIDEO mode
  -> smoothed square face crop           448x448, bbox smoothed across frames
  -> L2CS-Net gaze model (ONNX)          Gaze360 weights
  -> (pitch, yaw)                        radians, head-relative
  -> polynomial calibration              per person, per machine, per seat
  -> (x, y)                              normalised [0,1] over the whole screen
  -> One Euro filter                     adaptive smoothing
  -> GazeResult
```

Two layers sit on top of that, and the split is the point of the design:

- **`GazeEstimator`** is pure. Frames in, results out. No camera, no window, no
  socket. Usable with video files, another capture library, a shared camera, or
  a test fixture, and testable in CI on a machine with no hardware.
- **`WebcamGazeTracker`** is a convenience wrapper that owns a webcam.

---

## 2. Installation **[SHIPPED]**

```bash
pip install focusedgaze[directml]     # Windows, GPU
pip install focusedgaze[cpu]          # anywhere, slower
pip install focusedgaze[cuda]         # NVIDIA
```

**You must pick exactly one execution provider.** The base install deliberately
declares none, so the choice is explicit. Installing with no provider is a valid
state that fails later with a `ProviderError` naming the extras that fix it,
never a bare `ImportError`.

| Extra | Pulls in | Needed for |
|---|---|---|
| `cpu` / `directml` / `cuda` | an ONNX Runtime build | inference |
| `calibration` | scikit-learn, scipy | **fitting** a profile, never applying one |
| `server` | websockets | the WebSocket bridge |
| `export` | torch, torchvision, onnx | one-time PyTorch to ONNX conversion |
| `dev` | pytest, ruff, mypy, build, twine | development |

Base install is NumPy, OpenCV, MediaPipe and platformdirs. Nothing else is
imported at package import time.

**Inputs:** none. **Outputs:** an importable package. Model weights are *not*
included and the wheel stays under 5 MB.

---

## 3. First-run setup

### 3.1 `focusedgaze download-models` **[PLANNED CLI, SHIPPED registry]**

```bash
focusedgaze download-models [--dir PATH] [--force]
```

**Inputs:** optionally a target directory; `--force` re-fetches even if cached.
**Outputs:** model files in the platform cache directory, each verified by
SHA-256. Exit code 0 on success.

Two models, treated differently, and the difference is legal rather than
technical:

| Model | Size | Licence | Auto-download |
|---|---|---|---|
| MediaPipe face landmarker | ~3.8 MB | Apache-2.0 | **Yes** |
| L2CS-Net gaze weights | ~91 MB | Gaze360, non-commercial research | **No** |

The gaze weights are **never fetched for you and never redistributed**. The
command prints instructions and stops. This is a deliberate decision, not an
unimplemented feature. See `NOTICE`.

**Offline or locked-down machines:** set `FOCUSEDGAZE_MODEL_DIR` to a directory
that already contains the files. It takes precedence over the cache and skips all
network access.

### 3.2 `focusedgaze calibrate` **[PLANNED]**

```bash
focusedgaze calibrate [--profile NAME] [--camera INDEX]
```

**Inputs:** a person sitting in their normal position, following a moving dot.
**Outputs:** a named calibration profile saved to the platform config directory,
plus a printed held-out error.

Calibration is **per person, per machine, and per seating position**. It is the
file the whole system depends on. Move the laptop, change chairs, or swap users,
and it needs redoing.

### 3.3 `focusedgaze check` **[PLANNED]**

```bash
focusedgaze check
```

**Inputs:** none. **Outputs:** a diagnostic report and a non-zero exit if
anything is unusable.

```
camera        OK    index 0, 1280x720 @ 30fps, MSMF
providers     OK    DmlExecutionProvider (CPUExecutionProvider available)
models        OK    face_landmarker.task, l2cs_gaze360.onnx
profiles      OK    2 found, active: "arsalan-laptop"
port 8765     FREE
```

This command matters more than it looks. Most real-world failures here are
environment problems: a muted webcam, the CPU provider silently selected instead
of the GPU, a missing calibration, a port already in use. One command that
reports all of them turns a troubleshooting table into a tool.

### 3.4 `focusedgaze demo` **[PLANNED]**

A live preview window with a dot following your eyes. **Inputs:** a calibrated
profile and a camera. **Outputs:** a window. The quickest way to confirm the
whole chain works.

---

## 4. The pure API **[PLANNED]**

The layer that makes this a library rather than an application.

```python
from focusedgaze import GazeEstimator, CalibrationProfile, GazeConfig

est = GazeEstimator(
    profile=CalibrationProfile.load("arsalan-laptop"),
    config=GazeConfig(),
)

result = est.process(frame_bgr, timestamp=t)
```

### Inputs to `process()`

| Argument | Type | Meaning |
|---|---|---|
| `frame_bgr` | `np.ndarray`, shape `(H, W, 3)`, `uint8` | One frame in **BGR** order, as OpenCV delivers it. |
| `timestamp` | `float` | Seconds, monotonic within a stream. Drives the filter and MediaPipe's VIDEO mode. |

Three things about the input that are easy to get wrong:

1. **BGR, not RGB.** OpenCV's native order. Passing RGB gives a working pipeline
   with quietly worse detection, which is the hardest kind of bug to notice.
2. **Mirroring matters.** Calibration is recorded through a horizontally flipped
   frame. If you calibrated mirrored, you must infer mirrored, or left and right
   invert.
3. **Frames must arrive in order.** Bounding-box smoothing and the One Euro
   filter are both stateful across frames. Feeding shuffled or isolated frames
   produces plausible nonsense rather than an error.

### Outputs from `process()`

A frozen `GazeResult`. **[SHIPPED]**

| Field | Type | Meaning |
|---|---|---|
| `x`, `y` | `float \| None` | Normalised `[0,1]` over the whole physical screen. Origin **top-left**, `y` increases **downward**. `None` unless `status is OK`. |
| `pitch`, `yaw` | `float \| None` | Raw model output, **radians**. Available whenever a face was found, even if uncalibrated. |
| `distance_cm` | `float \| None` | Estimated eye-to-camera distance. |
| `status` | `GazeStatus` | Why this frame is or is not usable. |
| `timestamp` | `float` | Echoed back from the input. |
| `ok` | `bool` | **Derived from `status`**, not stored. Two sources of truth for one fact cannot disagree if there is only one. |

`GazeResult` is frozen: mutating it raises `FrozenInstanceError`.

### Status values **[SHIPPED]**

| Status | `ok` | Meaning | What to do |
|---|---|---|---|
| `OK` | `True` | Usable gaze point. | Use `x`, `y`. |
| `NO_FACE` | `False` | No face this frame. | Skip. Normal and frequent. |
| `OUT_OF_RANGE` | `False` | Too close or too far. | Prompt the user to move. |
| `OFF_CENTER` | `False` | Not centred enough. | Prompt the user to recentre. |
| `NOT_CALIBRATED` | `False` | Angles available, no profile. | Run calibration. |

**None of these raise.** They are recoverable per-frame conditions. This replaces
the legacy design where `None` meant "no face", "out of range" and "something is
broken" indiscriminately, so a caller could not tell a bad frame from a broken
install. Genuine faults raise from the exception tree instead.

### A ten-line script

```python
import cv2
from focusedgaze import GazeEstimator, CalibrationProfile

est = GazeEstimator(profile=CalibrationProfile.load("default"))
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    r = est.process(cv2.flip(frame, 1), timestamp=cv2.getTickCount() / cv2.getTickFrequency())
    if r.ok:
        print(f"{r.x:.3f} {r.y:.3f}")
```

### Working from a video file, with no camera at all

The reason the pure layer exists:

```python
from focusedgaze import GazeEstimator
from focusedgaze.capture import VideoFileSource

est = GazeEstimator(profile=...)
with VideoFileSource("session.mp4") as src:
    for frame, ts in src:
        result = est.process(frame, timestamp=ts)
```

### Thread safety

`GazeEstimator` is **not** thread-safe. It carries per-frame state, so one
instance belongs to one stream on one thread. Two instances in one process are
fully independent, which was not true of the legacy pipeline: its bounding-box
smoothing lived in a module-level global, so two consumers corrupted each other.

---

## 5. The convenience API **[PLANNED]**

```python
from focusedgaze import WebcamGazeTracker

with WebcamGazeTracker(profile="arsalan-laptop") as tracker:
    for result in tracker.stream():
        if result.ok:
            print(result.x, result.y)
```

**Inputs:** a profile name or object, an optional `GazeConfig`, an optional
camera index. **Outputs:** `GazeResult` objects.

Three consumption styles:

```python
for result in tracker.stream():   ...   # blocking iterator
tracker.on_gaze(callback)               # callback per result
latest = tracker.latest()               # non-blocking most-recent read
```

The context manager guarantees the camera is released, **including on
exception**. `close()` is idempotent. Use the `with` form: a webcam left open by
a crashed process needs the process killed to recover.

---

## 6. Calibration in detail **[UNPROVEN]**

> The code exists. Its numerics are not yet verified: the four mutation checks
> have not been run, so the pure-NumPy `apply()` reproducing scikit-learn's
> polynomial term ordering is undemonstrated. **A wrong polynomial does not
> raise.** It returns a smooth, believable surface in the wrong place. Treat
> coordinates as provisional until this marker changes.

### Headless collection

For scripted or custom calibration UIs:

```python
from focusedgaze import CalibrationCollector, fit_calibration

c = CalibrationCollector()
for target_x, target_y in grid_of_points:
    pitch, yaw = show_dot_and_read_gaze(target_x, target_y)
    c.add_sample(pitch=pitch, yaw=yaw, target_x=target_x, target_y=target_y)

profile = fit_calibration(c.samples)
print(profile.validation_error)      # held-out error, fraction of screen
profile.save("arsalan-laptop")
```

**Inputs:** `(pitch, yaw)` in radians paired with the `(target_x, target_y)` the
user was actually looking at, normalised `[0,1]`.
**Outputs:** a `CalibrationProfile` with a held-out error score.

### What a profile contains

Versioned, self-describing, and **not a pickle**:

| Field | Why it is recorded |
|---|---|
| `schema_version` | So a future format change is detected, not misread. |
| `created_at` | Profiles go stale when someone moves. |
| `screen_size`, `camera_size` | A profile is only valid for the geometry it was fitted on. |
| `validation_error` | Held-out error. The single number worth reading. |
| coefficients, `powers` | Plain arrays plus the term table that interprets them. |

The legacy format pickled live scikit-learn objects, which made loading a profile
require scikit-learn at runtime, break across scikit-learn versions, and execute
arbitrary code from a file. All three are gone. **Applying a profile is pure
NumPy.** `migrate_pickle` converts old files.

### Managing profiles

```python
from focusedgaze import list_profiles
from focusedgaze.calibration import set_active_profile, delete_profile
```

Profiles live in the platform config directory, are named, listable, and
switchable.

### Accuracy you can expect

| Condition | Error |
|---|---|
| Same session as calibration | ~2.0-2.4 cm |
| Session held out | ~3.0 cm |
| Screen corners and bottom edge | Worse |

These are measurements from the deployed system, not targets. Design around the
corner degradation: put small click targets away from the edges.

---

## 7. Configuration **[SHIPPED]**

Every tunable is a frozen dataclass. No module-level constants, no globals,
nothing that can drift underneath you at runtime.

```python
from focusedgaze import GazeConfig, FilterConfig, CameraConfig

cfg = GazeConfig()                                            # defaults
snappy = GazeConfig(filter=FilterConfig(min_cutoff=1.0, beta=0.9))
loaded = GazeConfig.from_file("gaze.toml")                    # TOML or JSON
built = GazeConfig.from_dict({"camera": {"index": 1}})
```

### Every default

| Group | Field | Default | Notes |
|---|---|---|---|
| `camera` | `index` | `0` | |
| | `width`, `height` | `1280`, `720` | |
| | `backend` | `"auto"` | Per-platform selection. |
| | `mirror` | `True` | **Must match how you calibrated.** |
| `filter` | `min_cutoff` | `0.7` | Lower is steadier at rest, more lag. |
| | `beta` | `0.6` | Higher is snappier, more jitter. |
| | `d_cutoff` | `1.0` | Rarely needs changing. |
| `positioning` | `min_distance_cm` | `45.0` | |
| | `max_distance_cm` | `65.0` | |
| | `center_tolerance` | `0.12` | |
| | `real_ipd_cm` | `6.3` | Population-average inter-pupil distance. |
| | `assumed_hfov_deg` | `60.0` | Fallback when focal length is uncalibrated. |
| `model` | `bins` | `90` | |
| | `input_size` | `448` | |
| | `providers` | DirectML, then CPU | Ordered preference, with fallback. |
| `landmarks` | `bbox_smoothing` | `0.3` | |
| | `max_faces` | `1` | |
| `runtime` | `send_hz` | `60.0` | |
| | `port` | `8765` | |
| | `dwell_ms` | `1050.0` | Dwell-to-click threshold. |
| | `edge_zone` | `0.09` | |

Invalid values raise `ConfigError`, which is also a `ValueError`.

---

## 8. Errors **[SHIPPED]**

Everything raised descends from `GazeError`, so one handler covers the library
without swallowing unrelated failures.

```
GazeError
├── ConfigError            (also ValueError)      bad configuration value
├── ModelNotFoundError     (also FileNotFoundError) weights absent, message names the fix
├── ProviderError          (also RuntimeError)    no usable ONNX provider
├── CalibrationError                             cannot fit, load, or apply
│   └── ProfileVersionError                      unsupported schema, migrate it
├── CameraError                                  cannot open, read, or configure
└── PositioningError                             gate could not be evaluated
```

The distinction that matters: **`PositioningError` is the fault case**, such as
an unreadable focal calibration. A user simply sitting too close is *not* an
error; that is an `OUT_OF_RANGE` status on a perfectly good result.

Logging goes to the `focusedgaze` logger. **The library installs no handlers**
and no public function prints. Configuring output is the application's job.

---

## 9. The WebSocket server **[PLANNED]**

```bash
focusedgaze serve [--host HOST] [--port PORT] [--profile NAME]
```

Requires the `server` extra. **Inputs:** a calibrated profile and a camera.
**Outputs:** a WebSocket endpoint at `ws://localhost:8765`.

### Messages emitted

```jsonc
{"type": "gaze",  "ok": true, "x": 0.5123, "y": 0.4210, "t": 1785910856.12}
{"type": "gaze",  "ok": false, "x": null, "y": null, "t": 1785910856.15}
{"type": "input", "mode": "gaze", "source": "gaze", "ok": true,
 "x": 0.5123, "y": 0.4210, "t": 1785910856.12}
{"type": "mode",  "mode": "gaze", "gesture_available": false,
 "error": null, "t": 1785910856.10}
```

`x` and `y` are rounded to 4 decimal places and are `null` when `ok` is false,
rather than a stale point.

### Two message types, and why both exist

- **`gaze`** is the raw device feed. One consumer reads it: the standalone test
  page. It is what makes `focusedgaze serve` verifiable without a game.
- **`input`** is the resolved pointer. This is what an application should read.

The SDK's `input` message is **gaze-only**: no gesture fields. An application
that also has gesture input replaces it through a `resolve_input` hook. The hook
**replaces** the message rather than appending, or two `input` messages race on
every tick.

### Pacing

`input` is sent every tick at `send_hz`; `gaze` only when the underlying reading
changes, which is around 15-19 Hz. Nothing is sent when no client is connected.

### Accepted from the client

```jsonc
{"cmd": "mode", "mode": "gaze"}
```

---

## 10. Putting it together: a clean machine

```bash
pip install focusedgaze[directml]
focusedgaze download-models      # landmarker auto; gaze weights print instructions
focusedgaze check                # confirm camera, provider, models
focusedgaze calibrate            # sit normally, follow the dot
focusedgaze demo                 # confirm the dot follows your eyes
```

Then ten lines of Python, as in section 4.

---

## 11. Current status summary

| Component | Status |
|---|---|
| Config, types, exceptions | **[SHIPPED]** |
| One Euro filter, positioning gate | **[SHIPPED]** |
| Asset registry and downloader | **[SHIPPED]** |
| Calibration | **[UNPROVEN]** |
| `GazeEstimator`, landmarks, ONNX model | **[PLANNED]** |
| Capture layer, `WebcamGazeTracker` | **[PLANNED]** |
| CLI | **[PLANNED]** |
| WebSocket server | **[PLANNED]**, contract fixed |

For what runs today with executed examples, see [usage.md](usage.md). That
document is the honest present; this one is the intended destination.
