# focusedgaze

Webcam eye-gaze tracking as a Python library. Point a laptop camera at a face and
get a screen coordinate.

> **Status: in development (0.0.0).** The name is reserved; the API below is the
> agreed contract but is not yet implemented. Do not depend on this release.

---

## What it does

```
webcam frame
  → MediaPipe face landmarks → smoothed square face crop
  → L2CS-Net gaze model (ONNX) → (pitch, yaw)
  → per-person polynomial calibration → (x, y) in [0, 1] over the screen
  → One Euro filter → steady coordinates
```

Accuracy on the reference setup: roughly 2.0–2.4 cm within a session, ~3.0 cm on a
held-out session, worst at screen corners and along the bottom edge.

## Two layers

```python
# Pure: you supply frames. No camera, no I/O, testable anywhere.
est = GazeEstimator(profile=CalibrationProfile.load("default"))
result = est.process(frame_bgr, timestamp=t)

# Convenience: it owns the webcam.
with WebcamGazeTracker(profile="default") as tracker:
    for result in tracker.stream():
        if result.ok:
            print(result.x, result.y)
```

The pure path is the point. Anyone who already has frames — a video file, another
capture library, a shared camera, a test — can use this library.

## Install

```bash
pip install focusedgaze[directml]   # Windows GPU via DirectX 12
pip install focusedgaze[cuda]       # NVIDIA
pip install focusedgaze[cpu]        # anywhere
```

The base install is provider-agnostic on purpose: you choose the ONNX execution
provider. Other extras: `[calibration]`, `[server]`, `[export]`.

## Model weights are not included

`focusedgaze download-models` fetches the MediaPipe face landmarker automatically
(Apache 2.0, from Google).

It does **not** fetch the gaze model. Those weights derive from the **Gaze360**
dataset, which is restricted to **non-commercial research use**, so this project
neither distributes nor mirrors them. You obtain them from the official L2CS-Net
distribution and convert them locally:

```bash
focusedgaze export-onnx --weights path/to/L2CSNet_gaze360.pkl
```

See [NOTICE](NOTICE) before using this for anything commercial.

## Platform support

| Platform | Status |
|---|---|
| Windows 10/11 | Tested |
| Linux | Structurally supported, untested (CI runs the non-hardware suite only) |
| macOS | Structurally supported, untested |

Camera backends and ONNX providers are abstracted, so other platforms should work
— but a classifier is a claim, and only Windows is claimed for v0.1.

Python 3.12–3.14.

## Licence

MIT for this code — see [LICENSE](LICENSE).
The gaze model weights it consumes are **not** MIT — see [NOTICE](NOTICE).

Author: Muhammad Asif Khan · <https://github.com/muhammad-asifkhan>
