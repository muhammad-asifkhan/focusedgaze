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

The gaze model is obtained once by one person: download `L2CSNet_gaze360.pkl` from the
[official L2CS-Net distribution](https://github.com/Ahmednull/L2CS-Net) and run
`focusedgaze setup --weights <path>`. The resulting `.onnx` is **portable** — the
execution provider is chosen at load time, so the same file works on another GPU vendor
or another OS. Everyone else runs `focusedgaze setup --onnx <path>`, or points
`FOCUSEDGAZE_MODEL_DIR` at a shared copy, and needs neither torch nor the download.

The second one is not laziness. Those weights come from the Gaze360 dataset,
which is **non-commercial research only**. The package will not download or
redistribute them for you. If your use is commercial, resolve the licence before
going further, not after.

### 4. A calibration, per person

**This is the part people underestimate.** Calibration is:

- per **person** (your eyes are not someone else's)
- per **machine** (a different camera and screen is a different geometry)
- per **seating position** (move the laptop, redo it)

It takes about a minute: a positioning check, then a dot sweeps the screen for 45
seconds and you follow it with your eyes. Without it you get gaze *angles* but no
screen *position*. Afterwards you are shown how well the sweep covered each region,
and the fit is refused if it missed one entirely.

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

**Roughly 3 to 6 cm average on a 34 cm-wide screen**, and the spread matters more
than the average.

| Situation | Typical error |
|---|---|
| Screen centre, good calibration | **1.0 cm** |
| Average over the whole screen | **3.3 to 6.2 cm** |
| Worst corner | **7.8 to 12.0 cm** |

Two measured runs by the same person on the same machine, twenty minutes apart,
differed by a factor of two, and **which corner was worst swapped between them**.
That is down to how well the calibration sweep covered each part of the screen,
not a fixed property of the tracker.

Two things follow. **Do not put small click targets in the corners.** And if one
region is consistently bad, recalibrate and make sure your eyes actually follow
the dot out there, rather than assuming the tracker is weak in that area.

Full per-point numbers in [accuracy.md](accuracy.md).

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
pip install focusedgaze[directml]                 # 1. install, pick a provider
focusedgaze setup --weights L2CSNet_gaze360.pkl   # 2. models + provider, one command
focusedgaze check                                 # 3. confirm camera and lighting
focusedgaze calibrate --name you                  # 4. ~45 s, follow the dot
focusedgaze demo --profile you                    # 5. confirm it follows your eyes
```

Run `focusedgaze setup` with no arguments first: it tells you exactly what is
missing, including where to get the checkpoint it will not fetch for you.

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

All of the above is built. This page used to describe a finished product that did
not exist yet; it now describes the one in the repository.

| Working now | Caveats |
|---|---|
| Config, result types, errors | |
| Smoothing, positioning gate | |
| Model registry and downloader | The gaze weights are never fetched. That is deliberate. |
| `GazeEstimator`, the camera layer, `WebcamGazeTracker` | |
| All eight CLI commands, including an interactive `calibrate` | Needs a desktop session to draw the dot |
| Calibration and the accuracy grid | Accuracy numbers are from two runs, and they disagreed by 2x |
| The WebSocket server | |

Still true: this is **0.0.0** and the PyPI release is not a working package. Install
from the repository. See [usage.md](usage.md) for examples that were actually
executed. If this page and the code disagree, **the code is right**.
