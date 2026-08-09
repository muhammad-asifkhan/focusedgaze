# Accuracy

What the numbers are, where they come from, and where focusedgaze is worse than the headline
figure suggests.

> **Read the provenance section before quoting anything here.** Some of these numbers are
> recorded measurements from the original system and some are gaps. They are labelled.

## The headline is a range, and the range is the point

**Roughly 3 to 6 cm average error on a 34 cm-wide screen.** Best at the centre, where the
better of two measured runs reached **1.0 cm**. Worst at whichever corner the calibration
sweep covered least, which was **12.0 cm** in one run and **7.8 cm** in the other, at
*opposite* corners.

There is deliberately no single headline number, because two runs of the same system, by the
same person, on the same machine, twenty minutes apart, differed by a factor of two:

| | Run 1 | Run 2 |
|---|---|---|
| Average over 9 points | **6.2 cm** | **3.3 cm** |
| Centre (50,50) | 7.4 cm | **1.0 cm** |
| Worst point | 12.0 cm at bottom-right | 7.8 cm at top-left |
| Degrades toward | right and bottom | top-left |

**The failure pattern inverts between them.** Run 2's worst point is run 1's best. That is
not a sensor limit: a sensor limit lands in the same place twice. It is an artifact of how
well the calibration sweep happened to cover each part of the screen.

So the single most useful thing on this page:

> **Accuracy depends more on how well your calibration covered the screen than on anything
> else measurable here.** If one region is bad, recalibrate and make sure your eyes actually
> follow the dot into that region, rather than assuming the tracker is weak there.

### The same runs in percent

Normalised to **screen width**, 34.4 cm, stated explicitly because "% of screen" is ambiguous
between width, height and diagonal, and the inherited figures below do not say which they
used:

| | cm | % of width |
|---|---|---|
| Run 2 average | 3.3 | **9.7%** |
| Run 2 centre | 1.0 | **2.9%** |
| Run 2 worst | 7.8 | 22.7% |
| Run 1 average | 6.2 | 18.2% |
| Run 1 worst | 12.0 | 34.9% |

Run 2 is broadly consistent with the inherited 8.9% overall, and its centre figure sits
inside the inherited 3-8% band for the top and centre. Run 1 is well outside both, which is
the variance above rather than a contradiction.

### Full per-point results

Screen 34.4 x 19.4 cm, DirectML, 32-36 samples per point, all nine points measured in both
runs. Positions are percentages of the screen.

| Point | Run 1 | Run 2 |
|---|---|---|
| (5,5) top-left | 1.1 cm | 7.8 cm |
| (50,5) top-centre | 0.6 cm | 5.2 cm |
| (95,5) top-right | 8.2 cm | 4.0 cm |
| (5,50) left | 2.0 cm | 2.1 cm |
| (50,50) centre | 7.4 cm | 1.0 cm |
| (95,50) right | 10.4 cm | 1.6 cm |
| (5,95) bottom-left | 3.3 cm | 1.9 cm |
| (50,95) bottom-centre | 11.3 cm | 3.4 cm |
| (95,95) bottom-right | 12.0 cm | 3.0 cm |

Run 2 was fitted from 1728 in-zone samples at degree 3 with 72 outliers rejected. Both
averages were recomputed from the per-point values rather than copied.

## The inherited figures, and where the bottom edge comes in

From the originating project's own documentation: held-out validation error after calibration
around **8.9% of screen size**, distributed unevenly.

| Region | Held-out error |
|---|---|
| Top and centre of screen | roughly **3-8%** of screen |
| Bottom edge | roughly **13-14%** of screen |

Those describe a bottom-edge weakness the two runs above do **not** reproduce consistently:
run 1 was worst at the bottom, run 2 was among its best there. Treat the inherited pattern as
one plausible shape rather than a law, and measure your own setup.

The design advice survives either way: keep small or important targets away from the edges
and corners, and give anything out there a generous hit area.

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

Three specific traps, all found by actually running the original tooling (sections 50.5 and
50.6):

**Never average away a point that collected no samples.** The original's held-out validation
reported 24.4% for a model the 9-point test measured at 9.7%, because two of its five
validation points collected nothing and it averaged the three survivors. A measurement whose
basis silently narrowed is not a measurement. Refuse to report a figure instead.

**Never report a single edge average.** One run's summary said "even accuracy across the
screen" while its own per-point table ranged from 0.6 cm to 12.0 cm: a good left edge
cancelled a bad right one. Report per-point and per-quadrant.

**Record which calibration produced the number.** A result that does not name its input
cannot be reproduced, and the input is the thing most likely to have changed underneath it.

## Provenance, and what is missing

**Recorded, and safe to quote:** the centimetre range and the per-point tables above, both
measured on the unmodified original pipeline and recorded in `MIGRATION_AUDIT.md` section 50.
Also the inherited percentage figures, the latency figures and the camera throughput figures,
which come from the original system's own documentation.

**Quote the range, not one of the two runs.** Either run on its own is a real measurement and
a misleading summary. An earlier version of the README claimed 2.0 to 2.4 cm from a source
that did not exist in this repository, and those were deleted for having no provenance.
Quoting "3.3 cm" now would have provenance and still be wrong, because the next run of the
same system measured 6.2 cm.

**One caveat on run 1.** Its calibration model cannot be identified. The original system
writes every recalibration to one mutable path, and run 2's fit overwrote run 1's twenty
minutes after it was measured. Run 2's model is identified by digest, `461b863c...`,
confirmed by the validation error stored inside the file rather than by its timestamp. See
section 50.4.

**Not re-measured after the migration, and it should not need to be.** Every figure on this
page was measured on the original implementation. focusedgaze reproduces that pipeline
**bit-identically** on 60 recorded frames: 60/60 identical outputs, zero crop-box differences,
drift of exactly 0.000000e+00 rad (`MIGRATION_AUDIT.md` section 49). The accuracy figures
therefore carry across as a matter of arithmetic rather than of hope.

What has *not* been done is an end-to-end accuracy run through focusedgaze with a person in
front of a camera. The equivalence evidence is stronger than such a run would be, being exact
rather than sampled, but it is evidence about the pipeline rather than about a session.
