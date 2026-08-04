# Calibration

> **Status: not implemented.** Calibration lands in Phase 5 and is the highest-risk
> numerical work in the migration. This page describes the routine as it exists in the
> original system, which is what Phase 5 has to reproduce exactly. Commands shown as
> `focusedgaze calibrate` do not exist yet.

## Why you cannot skip it

The gaze model gives you a direction: a pitch and a yaw, in radians, describing where the
eyes are pointed relative to the face. It does not know where your screen is, how big it is,
how far away you sit, or how your face is shaped. Turning a direction into a point on a
screen needs all of that, and none of it is in the model.

Calibration is the function that learns it, by watching you look at points whose position is
already known.

The consequence people find surprising: **a calibration belongs to one person, on one
machine, in roughly one seating position.** It is not a setting you configure once for the
application. Hand your laptop to someone else and the cursor will land in the wrong place
for them until they calibrate for themselves. Move from a desk chair to a sofa and it
degrades. That is not a defect being worked around, it is what the function is fitting.

## How the routine works

A smooth-pursuit collection followed by a held-out check.

**Collection.** A dot moves around the screen and you follow it with your eyes. Because the
dot's position is known at every instant, every frame yields a training pair: the model's
`(pitch, yaw)` against the screen `(x, y)` you were looking at. The original routine
gathers **over 1,500 samples** this way. Following a moving dot collects far more data than
staring at a grid of fixed points, and it collects it across the whole screen rather than at
nine places.

**Fitting.** A **degree-3 polynomial** maps `(pitch, yaw)` to `(x, y)`, fitted with robust
outlier rejection: samples whose residual exceeds **2.5 times the median absolute
deviation** are dropped and the fit is repeated, subject to keeping at least **60** samples.
Outlier rejection matters because you blink, and you occasionally look away from the dot.

**Validation.** Five static points the fit has never seen. The error on those is the number
worth trusting, because error measured on the data you fitted is not evidence of anything.

## Gating: why it refuses to start

The routine will not collect while you are outside the position it expects: between
**45 and 65 cm** from the screen, with your face reasonably centred.

This is not fussiness. The face crop the gaze model sees scales with distance, so a model
fitted at 50 cm and used at 80 cm is being asked about inputs it never saw. The gate keeps
collection and live use in the same regime. The same gate runs during live tracking, which
is why it can tell you to sit back rather than silently returning worse numbers.

The gate is implemented and usable now, ahead of the rest of calibration. See
[usage.md](usage.md).

## What a good result looks like

On the reference setup, held-out error after calibration is around **8.9% of screen size**,
and the average hides a real spread:

| Region | Held-out error |
|---|---|
| Top and centre | roughly 3–8% of screen |
| Bottom edge | roughly 13–14% of screen |

If your validation number comes out near the top of that range or worse, the usual causes
are lighting, seating, or having drifted out of the calibrated distance during collection.

> **These figures are percentages of screen size because that is what was recorded.** The
> original project also had a script reporting error in centimetres, but its output was never
> captured in this repository, so there is no centimetre figure here that can be stood
> behind. Re-measuring is scheduled before that script is deleted.

## Recalibrate, or just recentre?

Most of the time the answer is recentre, and people reach for recalibration too early.

**Recentre** when the cursor feels uniformly shifted, as though everything is offset in one
direction. Drift over a session is normal and a recentre corrects it in a second. In the
original system that is the `c` key.

**Recalibrate** only when accuracy is genuinely bad *after* recentring, and specifically
when:

- a different person is using it,
- the lighting has changed substantially, for example daylight to a lamp,
- your seating position has changed in a way you cannot undo,
- you have started or stopped wearing glasses.

Recalibration takes a couple of minutes and needs you to sit still and follow the dot all
the way through. A recentre takes a second. Try the cheap one first.

## Profile storage

The original system stores the fitted model as a **pickle** containing live scikit-learn
objects, which has two problems: loading a profile requires scikit-learn at runtime, and the
file breaks across scikit-learn versions.

Phase 5 replaces that with JSON metadata plus raw coefficient arrays, with the polynomial
evaluated in NumPy. Applying a profile then needs no scikit-learn at all, and a profile
written today keeps working after an upgrade. A one-shot migration reads existing `.pkl`
files.

This is the riskiest change in the migration, because a wrong polynomial does not crash. It
returns a smooth, believable surface in the wrong place. The equivalence fixture is
therefore mutation-checked before the swap is accepted: it must be shown to catch transposed
coefficients, a wrong feature-expansion ordering, swapped x and y coefficient sets, and a
degree mismatch. A fixture that merely passes has not been shown to catch anything.

## Your calibration profile is personal data

It is fitted to one person's eyes and is derived from a recording of their face. It is
excluded from the repository, excluded from the wheel by a build-time check that fails the
release if one is present, and it should not be committed or shared as a test fixture.
