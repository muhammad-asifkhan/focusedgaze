# What you need, and what you get

The short version. [complete-usage.md](complete-usage.md) has the detail; this
page is the part people actually need before deciding whether to use it.

---

## In one sentence

**You give it webcam frames. It gives you the point on your screen you are
looking at.**

---

## Part 1: What YOU need to provide

### 1. Hardware

| Thing | Requirement | Why |
|---|---|---|
| Webcam | 720p, 30fps | The pipeline is built and tuned for 1280x720. |
| Your face | 45 to 65 cm from the screen | Outside this the estimate degrades and the software refuses. |
| Seating | Roughly centred on the camera | Far off-centre is rejected. |
| Lighting | A normally lit room | Too dark and no face is found at all. |
| GPU | Optional | GPU ~15 ms per frame. CPU ~104 ms, about 7x slower but works. |

### 2. Software

| Thing | Requirement |
|---|---|
| Python | 3.12 or newer |
| OS | Windows (the only tested platform). Linux and macOS are structurally supported, untested. |
| Provider | Pick one: `directml` (Windows GPU), `cuda` (NVIDIA), or `cpu`. |

### 3. Two model files

**These are not included in the package.** The wheel stays small on purpose.

| File | Size | How you get it |
|---|---|---|
| `face_landmarker.task` | 3.8 MB | **Downloads automatically.** Apache-2.0. |
| Gaze model (ONNX) | ~91 MB | **You fetch this yourself.** The tool prints instructions and stops. |

The second one is not laziness. Those weights come from the Gaze360 dataset,
which is **non-commercial research only**. The package will not download or
redistribute them for you. If your use is commercial, resolve the licence before
going further, not after.

### 4. A calibration, per person

**This is the part people underestimate.** Calibration is:

- per **person** (your eyes are not someone else's)
- per **machine** (a different camera and screen is a different geometry)
- per **seating position** (move the laptop, redo it)

It takes about two minutes: you follow a dot around the screen. Without it you
get gaze *angles* but no screen *position*.

### 5. Per frame, at runtime

| Input | Type | Notes |
|---|---|---|
| Frame | NumPy array, `(720, 1280, 3)`, `uint8` | **BGR** order, as OpenCV gives it. |
| Timestamp | `float`, seconds | Must increase across a stream. |

Three ways to get this wrong that will **not** raise an error, and will just make
it quietly worse:

1. **Passing RGB instead of BGR.** Detection degrades. Nothing complains.
2. **Mirroring that does not match your calibration.** Left and right invert.
3. **Feeding frames out of order.** Smoothing is stateful. You get plausible
   nonsense, not an exception.

---

## Part 2: What you GET back

### Per frame, one result

```python
result = estimator.process(frame, timestamp=t)
```

| Output | Type | Meaning |
|---|---|---|
| `result.x` | `0.0` to `1.0` | How far **across** the screen. `0` is the left edge. |
| `result.y` | `0.0` to `1.0` | How far **down** the screen. `0` is the **top**. |
| `result.ok` | `True` / `False` | Whether `x` and `y` are usable this frame. |
| `result.status` | enum | *Why*, if not usable. |
| `result.pitch`, `.yaw` | radians | Raw gaze angles. Present even without calibration. |
| `result.distance_cm` | float | How far away you are. |
| `result.timestamp` | float | Echoed back. |

**Coordinates are fractions of the screen, not pixels.** To get pixels, multiply:

```python
px = result.x * screen_width
py = result.y * screen_height
```

`y` counts **downward** from the top, matching screens and image coordinates, not
graphs.

### When `ok` is False, `status` tells you why

| Status | What it means | What to do about it |
|---|---|---|
| `NO_FACE` | No face in this frame. | Nothing. Normal and frequent. Skip the frame. |
| `OUT_OF_RANGE` | Too close or too far. | Tell the user to move. |
| `OFF_CENTER` | Not centred enough. | Tell the user to recentre. |
| `NOT_CALIBRATED` | No calibration profile. | Run calibration. |

**None of these are errors and none of them raise.** They are normal states. Real
faults (no model file, no ONNX provider, unreadable profile) raise exceptions
instead, so you can always tell "this frame was no good" from "this setup is
broken".

### How good is it?

| Situation | Typical error |
|---|---|
| Same session you calibrated in | **2.0 to 2.4 cm** |
| A later session | **~3.0 cm** |
| Screen corners and bottom edge | Noticeably worse |

Design around that last row: **do not put small click targets in the corners.**

### How fast?

| Thing | Rate |
|---|---|
| Camera | ~31 fps |
| Inference | ~15 ms GPU, ~104 ms CPU |
| Gaze updates | ~15 to 19 per second |
| WebSocket output | 60 messages per second |

---

## Part 3: Is this right for you?

**Good fit**

- Hands-free pointing, dwell-to-click, accessibility input
- Attention and heat-mapping research
- Gaze-controlled games and demos
- One person, seated, at a desk, on Windows

**Poor fit**

- **Anything commercial**, until the Gaze360 licence question is settled
- Precise pointing. 2 to 3 cm is a large target, not a cursor.
- Multiple people at once (one face only)
- Moving users, phones, tablets
- Anyone who cannot calibrate first

---

## Part 4: Start to finish

```bash
pip install focusedgaze[directml]   # 1. install, pick a provider
focusedgaze download-models          # 2. get models (one is manual)
focusedgaze check                    # 3. confirm camera, provider, models
focusedgaze calibrate                # 4. ~2 min, follow the dot
focusedgaze demo                     # 5. confirm it follows your eyes
```

Then:

```python
from focusedgaze import GazeEstimator, CalibrationProfile

est = GazeEstimator(profile=CalibrationProfile.load("default"))
result = est.process(frame, timestamp=t)
if result.ok:
    print(result.x * screen_w, result.y * screen_h)
```

---

## Honest status

Not all of the above is built yet. This page describes the finished product.

| Working now | Not yet |
|---|---|
| Config, result types, errors | `GazeEstimator` itself |
| Smoothing, positioning gate | The camera layer |
| Model registry and downloader | The CLI commands |
| Calibration (**numbers not yet verified**) | The WebSocket server |

See [usage.md](usage.md) for what runs today, with examples that were actually
executed. If this page and the code disagree, **the code is right**.
