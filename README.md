# focusedgaze

Webcam eye-gaze tracking as a Python library. Point a laptop camera at a face and get a
screen coordinate.

> **Status: in development (0.0.0).** Implemented and tested: configuration, result types,
> the exception tree, the One Euro filter, the positioning gate, the model registry and
> downloader, and calibration. **Not yet written**: the gaze pipeline itself, the capture
> layer, the CLI and the server. The API shown below is the agreed contract, not working
> code, and every block that describes it is marked. Do not depend on this release. See
> [MIGRATION_AUDIT.md](MIGRATION_AUDIT.md) for what has actually landed.
>
> **CI is green** on Python 3.12, 3.13 and 3.14: 214 passed, 5 skipped. It was red for five
> pushes on a platform assumption in the asset registry, since fixed. See
> `MIGRATION_AUDIT.md` §42.

### Documentation

| Start here | For |
|---|---|
| [docs/what-you-need.md](docs/what-you-need.md) | **What you must supply and what you get back.** The short version. |
| [docs/complete-usage.md](docs/complete-usage.md) | The full guide to the finished product, every section status-marked. |
| [docs/usage.md](docs/usage.md) | What runs **today**, with examples that were executed. |
| [docs/wire_format.md](docs/wire_format.md) | The WebSocket contract, read off the source. |

---

## What it does

```
webcam frame
  → MediaPipe face landmarks → smoothed square face crop
  → L2CS-Net gaze model (ONNX) → (pitch, yaw)
  → per-person polynomial calibration → (x, y) in [0, 1] over the screen
  → One Euro filter → steady coordinates
```

## What you need before any of it works

Three things, and the second and third are the ones that catch people out.

**A webcam, and light.** The face has to be detectable. A muted camera or an unlit room
produces no landmarks and therefore no gaze, and the failure looks identical to a bug. This
has cost this project two recording sessions already.

**A calibration, per person.** There is no useful uncalibrated mode. The model gives you a
gaze direction in radians, and turning that into a point on your screen depends on where
your screen is, how far away you sit, and your face. A calibration is specific to one
person, one machine, and roughly one seating position. Someone else sitting down in your
chair will get bad results until they calibrate for themselves.

**The gaze model weights, which you fetch yourself.** focusedgaze will not download them.
They derive from the Gaze360 dataset, which its authors restrict to non-commercial research
use, so this project does not distribute or mirror them. This is a deliberate refusal, not
a missing feature. See [Licence](#licence) below, and read [NOTICE](NOTICE) before you use
this commercially.

## Install

```bash
pip install focusedgaze[directml]   # Windows GPU via DirectX 12
pip install focusedgaze[cuda]       # NVIDIA
pip install focusedgaze[cpu]        # anywhere
```

Pick one. The base install is deliberately provider-agnostic: focusedgaze does not choose
an ONNX execution provider for you, because the right choice depends on hardware it cannot
see. Installing the base package with no provider extra still imports cleanly, which CI
checks on every run. Reporting a missing provider as a named error rather than a bare
`ImportError` is the agreed design, but the model loader does not exist yet, so today there
is nothing to raise it.

The provider is worth getting right. On the reference machine, an RTX 4060 running Windows,
the gaze model takes about 15 ms per frame through DirectML and about 104 ms on CPU. That
is the difference between roughly 30 frames per second end to end and roughly 5.

Other extras: `[calibration]` to fit a profile, `[server]` for the WebSocket bridge,
`[export]` to convert the PyTorch weights to ONNX.

Python 3.12–3.14. Tested on 3.12, 3.13 and 3.14 in CI.

## What works today

The filter and the positioning gate are extracted, tested against the original
implementation, and usable now:

```python
from focusedgaze.core.filters import OneEuroFilter2D

# Smooths a jittery 2D signal. Defaults match the deployed system.
f = OneEuroFilter2D(min_cutoff=0.7, beta=0.6, d_cutoff=1.0)

for x, y, t in [(0.50, 0.50, 0.00), (0.55, 0.48, 0.03), (0.52, 0.51, 0.07)]:
    sx, sy = f.filter(x, y, t)
    print(f"{sx:.4f} {sy:.4f}")
```

`focusedgaze.core.positioning` provides `PositioningGate`, which tells you whether a face is
close enough, far enough, and centred enough to give a usable reading. It works on
MediaPipe landmarks without needing the gaze model.

## The intended API

> **None of this runs yet.** It is the contract Phase 2 through Phase 4 are being built
> against, shown here so the shape can be reviewed rather than discovered late.

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

The pure path is the point of the design. Anything that already has frames can use this
library: a video file, another capture library, a camera shared with a hand tracker, or a
test that needs to be reproducible. Owning the webcam is the convenience layer, not the
foundation.

## Accuracy

The honest summary is that accuracy is uneven across the screen, and the bottom is the
worst part.

On the reference setup, held-out validation error after calibration is about **8.9% of
screen size**, and that average hides the spread: roughly **3–8% across the top and
centre**, and **13–14% along the bottom edge**. If you are placing targets, put them where
the tracker is good, and do not put anything small or important along the bottom.

Accuracy degrades when the lighting changes, when you move closer or further than you
calibrated at, and when a different person sits down. The positioning gate exists to catch
the distance case: it enforces the 45–65 cm range the calibration was collected in.

> **A figure this README used to quote is not currently supported.** Earlier versions
> claimed 2.0–2.4 cm within a session and about 3.0 cm on a held-out session. Those numbers
> come from an accuracy script in the original project whose output has never been recorded
> in this repository, so there is nothing here to back them. They have been removed rather
> than repeated. The percentage figures above are recorded, and come from the original
> system's documentation. Re-measuring in centimetres is scheduled before the milestone
> scripts are deleted.

## Platform support

| Platform | Status |
|---|---|
| Windows 10/11 | Tested |
| Linux | Structurally supported, untested. CI runs the non-hardware suite only |
| macOS | Structurally supported, untested |

Camera backends and ONNX providers are abstracted, so other platforms should work. A
classifier is a claim though, and only Windows is claimed for v0.1. CI does prove the pure
core computes identical results on Linux, which is evidence the abstractions are real
rather than aspirational, but nobody has pointed a camera at it there.

## Licence

The code is MIT. See [LICENSE](LICENSE).

The model weights are not, and this matters if you are evaluating focusedgaze for a product.
The gaze model is an ONNX export of L2CS-Net trained on the **Gaze360** dataset, whose
authors state that use of the dataset and code is for non-commercial research only. Weights
trained on it are normally treated as a derived work carrying the same restriction. So:
focusedgaze does not ship them, does not mirror them, and will not download them for you.
You obtain them from the official L2CS-Net distribution and convert them locally.

This is a conservative reading of the upstream terms and not legal advice. Full detail is in
[NOTICE](NOTICE).

Author: Muhammad Asif Khan, <https://github.com/muhammad-asifkhan>
