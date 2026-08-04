# Accuracy

What the numbers are, where they come from, and where focusedgaze is worse than the headline
figure suggests.

> **Read the provenance section before quoting anything here.** Some of these numbers are
> recorded measurements from the original system and some are gaps. They are labelled.

## The headline

On the reference setup, held-out validation error after calibration is around **8.9% of
screen size**.

That average is not the useful number, because the error is not evenly distributed:

| Region | Held-out error |
|---|---|
| Top and centre of screen | roughly **3–8%** of screen |
| Bottom edge | roughly **13–14%** of screen |

The bottom edge is between two and four times worse than the middle. If you are laying out
an interface driven by gaze, that is the fact to design around: keep small or important
targets out of the bottom strip, and give anything down there a generous hit area.

## Why the bottom is worse

Not fully characterised, and worth being honest about rather than inventing a mechanism. The
plausible contributors are that the eyelid occludes more of the iris when looking down,
which degrades the landmarks the crop is built from, and that a webcam mounted above the
screen sees a steeper and more foreshortened view of the eye at the bottom of its range.

Neither has been isolated by measurement in this project. The degradation is measured; the
explanation is not.

## Latency and throughput

Measured on the reference machine, an RTX 4060 running Windows, at 1280x720:

| Metric | CPU | GPU via DirectML |
|---|---|---|
| Gaze model inference | ~104 ms | ~15 ms |
| End-to-end update rate | ~5 fps | ~30 fps |

Camera throughput at 720p is about **31 fps** using the MSMF backend, against about
**10 fps** using DSHOW. The backend choice is worth as much as the provider choice on
Windows.

The practical reading: on CPU, focusedgaze is fine for processing recorded video and
unpleasant for live pointing. Five updates a second feels broken to a user even though the
accuracy is identical.

## What changes your accuracy

In rough order of how much damage each does:

**A different person.** The largest single factor. A calibration is fitted to one person's
eyes and does not transfer. Someone else using your profile gets systematically wrong
results, not slightly noisier ones.

**Distance drift.** The face crop scales with distance, so using the system well outside the
45–65 cm band it was calibrated in asks the model about inputs it never saw. This is what
the positioning gate exists to catch.

**Lighting changes.** Landmark quality depends on the face being clearly visible. Going from
daylight to a single lamp is enough to matter. Complete failure is easy to spot, because you
get no face at all; partial degradation is the dangerous case because the numbers keep
arriving and are quietly worse.

**Seating position.** Head angle relative to the camera is part of what the calibration
absorbed. A different chair changes it.

**Glasses.** Starting or stopping wearing them changes the iris appearance enough to warrant
recalibrating.

**Screen geometry.** The calibration maps to normalised screen coordinates, so changing
resolution is survivable, but changing the physical screen or moving the camera relative to
it is not.

## Measuring it yourself

Do not trust a figure measured on the data the model was fitted to. Fitting error is not
evidence of anything: a degree-3 polynomial will always describe its own training samples
well.

The original system's routine is the right shape:

1. Calibrate normally, following the moving dot.
2. Look at a set of **static points the fit has never seen**, at known screen positions.
3. For each, record where you were told to look against where the model says you looked.
4. Report per-point error, not just the mean, so the corner and edge degradation is visible.

Reporting only an average hides exactly the thing you need to know. An 8.9% mean built from
4% in the middle and 14% at the bottom describes neither region.

## Provenance, and what is missing

**Recorded, and safe to quote:** the percentage figures, the latency figures, and the camera
throughput figures. All come from the original system's own documentation, measured on the
reference machine described there.

**Not recorded, and deliberately absent:** any accuracy figure in centimetres.

An earlier version of the README claimed 2.0 to 2.4 cm within a session and about 3.0 cm on
a held-out session. Those numbers have no source in this repository. The first traces to an
accuracy script in the original project whose output has never been captured here, and the
second appears nowhere at all. They were removed rather than repeated, and this page will
not restore them until the script has been run on unmodified code and its output recorded.

That measurement is scheduled before the milestone scripts are deleted, because deleting
them removes the ability to make it at all.

**Not measured at all:** accuracy after the migration. Every figure on this page describes
the original implementation. focusedgaze is being extracted from it with a golden-file
harness pinning equivalence numerically, but the pipeline stage those figures actually
depend on is not extracted yet, and the fixture covering it does not exist. Until it does,
treat this page as describing the system focusedgaze is being built to reproduce rather than
focusedgaze itself.
