# focusedgaze

Webcam eye-gaze tracking as a Python library. Point a laptop camera at a face and get a
screen coordinate.

> **Status: 0.1.0, the first release that does anything.** `0.0.0` on PyPI is a
> placeholder whose modules are almost all stubs; it cannot be replaced, because PyPI
> never permits re-uploading a version. Install `0.1.0` or later.
>
> Everything is implemented and tested: `GazeEstimator`, `WebcamGazeTracker`, the capture
> layer, interactive calibration, the asset registry, the WebSocket server, a dwell-based
> control layer, and all eight CLI commands. The extraction reproduces the original
> pipeline **bit-identically** on 60 recorded frames. See
> [MIGRATION_AUDIT.md](https://github.com/muhammad-asifkhan/focusedgaze/blob/main/MIGRATION_AUDIT.md) §49.
>
> **Two gaze backends.** Intel's `gaze-estimation-adas-0002` is the default: Apache-2.0,
> downloads automatically, and measured at **2.0 ms per frame on a plain CPU** and 1.43 cm
> of error. L2CS-Net remains fully supported behind `--backend l2cs`, but its weights derive
> from Gaze360 and cannot be redistributed, so you must fetch and convert them yourself —
> which is why it is not the default. A fresh install works with no manual model step.
>
> **New here?** Read
> [docs/getting-started.md](https://github.com/muhammad-asifkhan/focusedgaze/blob/main/docs/getting-started.md).
> If something is not working, run `focusedgaze setup` then `focusedgaze check`.
>
> **Known issue in 0.1.0:** the eye-crop roll correction follows the Open Model Zoo
> reference, but this package derives head roll from MediaPipe rather than Open Model Zoo's
> head-pose network. If the two disagree in sign, accuracy degrades under head tilt rather
> than improving. To be settled in 0.1.1.

### Documentation

| Start here | For |
|---|---|
| [docs/getting-started.md](https://github.com/muhammad-asifkhan/focusedgaze/blob/main/docs/getting-started.md) | **Start here.** Install to a working gaze-controlled app, step by step, including swapping the model. |
| [docs/what-you-need.md](https://github.com/muhammad-asifkhan/focusedgaze/blob/main/docs/what-you-need.md) | **What you must supply and what you get back.** The short version. |
| [docs/complete-usage.md](https://github.com/muhammad-asifkhan/focusedgaze/blob/main/docs/complete-usage.md) | The full guide to the finished product, every section status-marked. |
| [docs/usage.md](https://github.com/muhammad-asifkhan/focusedgaze/blob/main/docs/usage.md) | What runs **today**, with examples that were executed. |
| [docs/wire_format.md](https://github.com/muhammad-asifkhan/focusedgaze/blob/main/docs/wire_format.md) | The WebSocket contract, read off the source. |

---

## What it does

```
webcam frame
  → MediaPipe face landmarks → face crop, or per-eye crops + head pose
  → gaze model (L2CS-Net via ONNX, or Intel via OpenVINO) → (pitch, yaw)
  → per-person polynomial calibration → (x, y) in [0, 1] over the screen
  → One Euro filter → steady coordinates
  → optional dwell selection → "the user chose 'play'"
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

**Model files — fetched for you on the default backend.** `focusedgaze setup` downloads
the face landmarker and the Intel gaze model, both Apache-2.0, both digest-verified. There
is no manual step and nothing to license.

**Unless you choose `--backend l2cs`, whose weights you fetch yourself.** focusedgaze will
not download those. They derive from the Gaze360 dataset, which its authors restrict to
non-commercial research use, so this project does not distribute or mirror them. This is a
deliberate refusal, not a missing feature — and it is the reason that backend is not the
default. See [Licence](#licence) below, and read [NOTICE](https://github.com/muhammad-asifkhan/focusedgaze/blob/main/NOTICE) before you use
this commercially.

## Install

```bash
pip install "focusedgaze[intel,calibration]"    # recommended: Apache-2.0 model, CPU, no GPU
```

That is the whole install. The Intel backend's weights are Apache-2.0, so they are fetched
automatically and digest-verified, and at 0.139 GFLOPs they run at ~2 ms on a plain CPU.
`calibration` adds scikit-learn, which is needed to **fit** a profile but not to apply one.

To use L2CS-Net instead, pick an ONNX execution provider and supply the weights yourself
(see [NOTICE](https://github.com/muhammad-asifkhan/focusedgaze/blob/main/NOTICE)):

```bash
pip install "focusedgaze[directml,calibration]"   # Windows GPU via DirectX 12
pip install "focusedgaze[cuda,calibration]"       # NVIDIA
pip install "focusedgaze[cpu,calibration]"        # anywhere
```

Pick one. The base install is deliberately provider-agnostic: focusedgaze does not choose
an ONNX execution provider for you, because the right choice depends on hardware it cannot
see. Installing the base package with no provider extra still imports cleanly, which CI
checks on every run. A missing provider is reported as a named `ProviderError` naming the
extras that fix it, never as a bare `ImportError`.

The provider is worth getting right. On the reference machine, an RTX 4060 running Windows,
the gaze model takes about 15 ms per frame through DirectML and about 104 ms on CPU. That
is the difference between roughly 30 frames per second end to end and roughly 5.

Other extras: `[calibration]` to fit a profile, `[server]` for the WebSocket bridge,
`[export]` to convert the PyTorch weights to ONNX.

Python 3.12–3.14. Tested on 3.12, 3.13 and 3.14 in CI.

## The API

Two layers. You still need the model weights and a calibration, as described above.

```python
from focusedgaze import GazeEstimator, WebcamGazeTracker, CalibrationProfile

# Pure: you supply frames. No camera, no network, testable anywhere.
est = GazeEstimator(profile=CalibrationProfile.load("alice"))
result = est.process(frame_bgr, timestamp=t)
if result.ok:
    print(result.x, result.y)

# Convenience: it owns the webcam and always gives it back.
with WebcamGazeTracker(profile="alice") as tracker:
    for result in tracker.stream():
        if result.ok:
            print(result.x, result.y)
```

`GazeEstimator` never reaches the network: a missing model raises with the command that
fetches it rather than downloading anything, which is what lets the whole pipeline run in
CI against a recorded fixture.

Smaller pieces are usable on their own. `focusedgaze.core.filters.OneEuroFilter2D` smooths
any jittery 2D signal, and `focusedgaze.core.positioning.PositioningGate` reports whether a
face is close enough, far enough and centred enough, working on MediaPipe landmarks without
the gaze model.

The pure path is the point of the design. Anything that already has frames can use this
library: a video file, another capture library, a camera shared with a hand tracker, or a
test that needs to be reproducible. Owning the webcam is the convenience layer, not the
foundation.

## Accuracy

The honest summary is that accuracy is uneven across the screen, and **which part is worst
varies between calibrations**.

The originating project's documentation reports held-out validation error around **8.9% of
screen size**, spread as roughly **3–8% across the top and centre** and **13–14% along the
bottom edge**. Our own two measured runs did not reproduce that pattern consistently: one was
worst at the bottom-right, the other was among its best there and worst at the top-left.

So the design advice is the durable part: keep small or important targets away from the edges
and corners, and give anything out there a generous hit area.

Accuracy degrades when the lighting changes, when you move closer or further than you
calibrated at, and when a different person sits down. The positioning gate exists to catch
the distance case: it enforces the 45–65 cm range the calibration was collected in.

> **Measured in centimetres, and it is a range on purpose.** Roughly **3 to 6 cm average**
> on a 34 cm-wide screen, best at the centre (1.0 cm in the better run), worst at whichever
> corner the calibration sweep covered least. Two runs by the same person on the same machine
> twenty minutes apart differed by a factor of two, and the failure pattern *inverted* between
> them, so a single number would mislead. Accuracy depends more on how well your calibration
> covered the screen than on anything else measured here. See
> [docs/accuracy.md](https://github.com/muhammad-asifkhan/focusedgaze/blob/main/docs/accuracy.md) and `MIGRATION_AUDIT.md` section 50.
>
> This replaces an earlier README claim of 2.0–2.4 cm, which had no source in this repository
> and was deleted rather than repeated.

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

The code is MIT. See [LICENSE](https://github.com/muhammad-asifkhan/focusedgaze/blob/main/LICENSE).

The model weights are not, and this matters if you are evaluating focusedgaze for a product.
The gaze model is an ONNX export of L2CS-Net trained on the **Gaze360** dataset, whose
authors state that use of the dataset and code is for non-commercial research only. Weights
trained on it are normally treated as a derived work carrying the same restriction. So:
focusedgaze does not ship them, does not mirror them, and will not download them for you.
You obtain them from the official L2CS-Net distribution and convert them locally.

This is a conservative reading of the upstream terms and not legal advice. Full detail is in
[NOTICE](https://github.com/muhammad-asifkhan/focusedgaze/blob/main/NOTICE).

Author: Muhammad Asif Khan, <https://github.com/muhammad-asifkhan>
