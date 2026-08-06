# MIGRATION_AUDIT.md — Phase 0

Audit of `gaze-detection/` ahead of extraction into a distributable Python SDK.
No package code has been written. This is the Phase 0 exit artifact.

**Living document.** Every deviation from the brief gets logged in §8 for the life
of the project.

---

## 1. Method

Read all 18 `.py` files in `gaze-detection/`, plus `README.md` and
`GAZE_SYSTEM_DOCS.md`. Dependencies were derived from **actual import statements**,
not from `requirements.txt` (per B1). License findings come from the upstream
repositories, not from memory.

`GazeModuleToPipLibrary.docx` could not be opened — it is a binary Word file and no
converter is available in this environment. I worked from your §3 corrections
instead, which you describe as superseding it. **If the DOCX contains anything not
reflected in §3, tell me.**

---

## 2. Module inventory

18 files. Four buckets, as requested.

### 2.1 CORE — becomes `src/<pkg>/core/`

| File | Public surface | Problems found |
|---|---|---|
| `gaze_pipeline.py` | `get_gaze_reading`, `detect_face`, `predict_gaze_from_crop`, `get_smoothed_square_crop`, `reset_bbox_smoothing` | **Builds the ONNX session at import time** from `pathlib.Path.cwd()` (L15-33). Importing the module from the wrong directory fails outright — this is why `gaze_server.py` calls `os.chdir()` at L40. **Module-level mutable global** `_smoothed_bbox` (L35): two estimators in one process would corrupt each other. **`print()` at import** (L33). MediaPipe imported lazily inside `detect_face` (L113). Hard-codes `BINS=90`, `BBOX_SMOOTHING=0.3`, ImageNet normalisation, 448×448 input. |
| `positioning_gate.py` | `PositioningGate` (`evaluate`, `focal_px`, `calibrate_focal`), constants | Relative `FOCAL_PATH = models/camera_focal.json` (L43) — resolves against cwd. Tunables are module constants (`MIN_DIST_CM`, `MAX_DIST_CM`, `CENTER_TOL`, `REAL_IPD_CM`, `ASSUMED_HFOV_DEG`). Otherwise clean, well-documented, and already dependency-light (cv2 + numpy + math). Class-based, so it holds no global state. |
| `gaze_features.py` | `extract_features`, `FEATURE_COLUMNS`, `eye_aspect_ratio`, `iris_ratio`, `head_pose_from_matrix` | Cleanest file in the set. **`scipy` is already an optional import** (L29-30, guarded `try/except`, head rotation skipped without it) — good precedent for the extras design. Mixed purpose: iris/EAR/head-pose are genuinely core, but `FEATURE_COLUMNS` exists to serve the CSV dataset schema, which is not SDK concern. |

### 2.2 CALIBRATION — becomes `src/<pkg>/calibration/`

| File | Public surface | Problems found |
|---|---|---|
| `calibration_utils.py` | `fit_calibration`, `apply_calibration`, `robust_fit_samples`, `save_calibration`, `load_calibration`, plus a whole dataset-CSV subsystem | **Three unrelated concerns in one file**: (a) polynomial fit/apply, (b) pickle persistence, (c) a CSV dataset logger with schema versioning, migration and an auto-generated data dictionary. Only (a) and (b) belong in the SDK. **Persistence is `pickle`** (L333-340) — the model dict contains live `PolynomialFeatures` and `LinearRegression` objects, so **loading a profile requires scikit-learn at runtime and breaks across sklearn versions** (D3 confirmed, worse than the brief assumed: it is not just unsafe, it is version-fragile). `CURRENT_USER` and `SESSION_ID` are computed at **import time** (L97-98). Relative paths throughout. Default `degree=2` in `fit_calibration` but `robust_fit_samples` passes `degree=3` — the README's "degree-3" is only true via the robust path. |

### 2.3 SERVER — becomes `[server]` extra

| File | Notes |
|---|---|
| `gaze_server.py` | The WebSocket bridge. **No longer purely gaze** — it now also owns input-mode state, camera hand-over between pipelines, and imports `gesture_bridge`. See Q6; this materially affects Phase 7. |

### 2.4 SCAFFOLDING — not library code

| Files | Notes |
|---|---|
| `milestone1..9_*.py` (9 files) | Demos and manual tests. **They are entangled**: `milestone6`, `7`, `8`, `9` import `milestone4_calibration`, and `8`/`9` also import `milestone7_gaze_game`. They cannot be deleted piecemeal without breaking each other. |
| `analyze_dataset.py`, `inspect_dataset.py` | Dataset tooling built on the CSV subsystem. |
| `export_to_onnx.py` | One-time PyTorch → ONNX conversion. **The only file that imports `torch` or `l2cs`.** |

### 2.5 OUT OF SCOPE

| File | Notes |
|---|---|
| `gesture_bridge.py` | Adapter to the gesture module. Explicitly out of scope for SDK v1 — but it lives in `gaze-detection/` and `gaze_server.py` imports it. See Q6. |

---

## 3. Dependency map (from actual imports)

### 3.1 What is genuinely needed

| Bucket | Packages | Evidence |
|---|---|---|
| **Core runtime** | `opencv-python`, `mediapipe`, `numpy`, one `onnxruntime*` | imported by `gaze_pipeline`, `positioning_gate`, `gaze_features` |
| **Calibration (fit only)** | `scikit-learn` | `calibration_utils` L15-16 |
| **Calibration (optional)** | `scipy` | `gaze_features` L29, already guarded |
| **Server** | `websockets` | `gaze_server` only |
| **Export only** | `torch`, `l2cs`, (`onnx` implicitly) | `export_to_onnx.py` only |

### 3.2 Declared in `requirements.txt` but **never imported**

Verified by grepping every `.py` outside the venv:

| Package | Imports found |
|---|---|
| `joblib` | **0** — the README claims joblib saves the calibration model; the code uses `pickle` directly |
| `pandas` | **0** |
| `matplotlib` | **0** |
| `sounddevice` | **0** — the README credits it with "audio cues"; nothing imports it |
| `PyAutoGUI`, `PyGetWindow`, `PyScreeze`, `MouseInfo` | **0** — `milestone8` drives the mouse via `ctypes`, not PyAutoGUI |
| `face-detection` | **0** — present only as a transitive dependency of `l2cs` |

`requirements.txt` pins **55 packages** with `==`. A large fraction are transitive or
entirely unused. This is a snapshot of one working venv, not a dependency
specification — which is exactly D5's point, and worse than stated.

### 3.3 B2 — RESOLVED, and it resolves the easy way

The brief flagged `l2cs` and `face-detection` as git-URL dependencies that **PyPI
will reject in package metadata**, and asked me to investigate option (a) first.

**Finding: runtime inference is pure ONNX Runtime. `l2cs` is imported in exactly one
file, `export_to_onnx.py`, and `face-detection` is never imported at all.**

So both dependencies disappear from the SDK entirely. No vendoring of model-definition
code is needed — at runtime we load an ONNX graph and run it. `l2cs` is only required
to *produce* that graph from PyTorch weights, which is a one-time developer step.

**Consequence for the `[export]` extra:** it still cannot declare `l2cs`, because that
is still a git URL. The export path will be documented as a manual procedure
(`pip install git+…` by hand) rather than a declared extra, or moved to a separate
dev-only requirements file. See DEV-1 in §8.

---

## 4. LICENSE FINDINGS — **BLOCKING (B4)**

This is the item the brief said to stop on. It needs your decision.

| Component | License | Redistributable? |
|---|---|---|
| **L2CS-Net code** (`github.com/Ahmednull/L2CS-Net`) | **MIT** | Yes, with attribution. Not that we need it — see §3.3. |
| **MediaPipe `face_landmarker.task`** | Apache 2.0, Google-hosted | Yes; and we would link to Google's own URL regardless |
| **Gaze360 dataset + derived models** (`github.com/erkil1452/gaze360`) | **"The usage of the dataset and the code is for non-commercial research use only."** | **Almost certainly NOT** |

### Why this matters

`L2CSNet_gaze360.pkl` — and therefore our derived `l2cs_gaze360.onnx` — is trained on
Gaze360. A model trained on a non-commercial-research-only dataset is normally treated
as a derived work carrying the same restriction. Three consequences:

1. **We must not ship the weights in the wheel.** Already the plan (B3), now also a
   licensing requirement rather than just a size one.
2. **We must not host a mirror.** The brief listed "publish a mirror" as an option for
   B2; for the *weights* it is not available to us.
3. **We must not auto-download the weights from a non-authoritative source.** The origin
   project's setup instructions fetch `L2CSNet_gaze360.pkl` from a general-purpose model
   host rather than from the L2CS-Net or Gaze360 authors' own distribution. Regardless of
   any other consideration, a URL that is not the upstream one is unsuitable for an SDK's
   automatic downloader: its licensing provenance is not established, and it carries no
   availability guarantee. The SDK therefore points users at the official L2CS-Net
   distribution and asks them to obtain the weights themselves.

### What I did not do

I did not read Gaze360's full `LICENSE.md` (the CSAIL site returns 403 to automated
fetches). I have the headline restriction from the repository, not the complete terms.
**I am not qualified to give you a legal opinion and I am not going to guess.** See
Q1 for the decision I need.

---

## 5. Package name

`focusedgaze` — **available**. `https://pypi.org/pypi/focusedgaze/json` returns **404**.

PyPI normalises names, so `focused-gaze`, `focused_gaze` and `Focused.Gaze` all
collapse to the same record; the single check covers the collision cases the brief
raised.

Fallbacks held in reserve, unverified until needed: `gazekit`, `pygazetrack`.

**Not claimed yet.** Availability today is not a reservation — someone else can take
it. If you want it held, the cheapest move is to publish an empty `0.0.0` placeholder
to PyPI now. Say the word.

---

## 6. Response to every §3 correction

### Blockers

| # | Status | How it will be applied |
|---|---|---|
| B1 | Confirmed, worse than stated | Deps derived from imports (§3.1). Also removing 8+ declared-but-unused packages (§3.2). |
| B2 | **Resolved via option (a)** | `l2cs`/`face-detection` vanish from runtime — inference is pure ONNX (§3.3). No vendoring needed. |
| B3 | Accepted | Download-on-demand into `platformdirs.user_cache_dir`, SHA-256 verified, `download-models` CLI, env-var offline override. |
| B4 | **BLOCKED — needs your decision** | Findings in §4. Q1. |
| B5 | Confirmed | `torch`/`torchvision` are export-only. Caveat: the extra cannot declare `l2cs` (git URL) — documented procedure instead. |
| B6 | Accepted | Floor set to what I actually test. Reference env is Python 3.14; I will verify `mediapipe`/`onnxruntime-directml` wheel availability per version before claiming a range. Q4. |

### Design

| # | Status | How it will be applied |
|---|---|---|
| D1 | Accepted | Camera backend per platform, provider preference list with fallback + one log line naming the provider that loaded. Extras `[directml]`/`[cuda]`/`[cpu]`. |
| D2 | Accepted, and it is the most valuable change | Pure `GazeEstimator.process(frame, timestamp)`; `WebcamGazeTracker` layered on top. Note this also fixes the import-time-session and global-bbox defects, since both become instance state. |
| D3 | Accepted, and the problem is worse than described | Not merely unsafe — the pickle embeds sklearn estimator objects, so it is version-fragile too. New format: JSON metadata + coefficient arrays; `apply()` in pure NumPy; **sklearn becomes fit-time-only**; one-shot `.pkl` migration. |
| D4 | Accepted | Frozen dataclasses. All defaults preserved exactly (§7). |
| D5 | Accepted | `pyproject.toml` with ranges as the single source of truth; pins move to a dev constraints file. |
| D6 | Accepted | Single-sourced version. |
| D7 | Accepted | Full annotations, `py.typed`, exception tree. |
| D8 | Accepted | `[server]` extra; `core/` must import with `websockets` absent — enforced by a CI test, not just by intent. |

### §3.3 fixes

All accepted: filename typos, em-dash/en-dash command corruption, truncated
`pip install -e .`, and real identity values.

**Identity corrected after Phase 1** — the package is authored and published by
**Muhammad Asif Khan**, `github.com/muhammad-asifkhan`. See §13 and §26: the project is
attributed to a single author throughout.

---

## 7. Defaults to be preserved verbatim (rule 1)

Extraction must not change behaviour. Recording these now so drift is detectable:

| Constant | Value | Source |
|---|---|---|
| Capture | 1280×720, MSMF, ~31 fps | `gaze_server.py` |
| ONNX input | 448×448, ImageNet mean/std | `gaze_pipeline.py` |
| `BINS` | 90; angle = `Σ(softmax(bins)·i)·4 − 180` degrees — an expectation, **not** an argmax | `gaze_pipeline.py` |
| `BBOX_SMOOTHING` | 0.3 | `gaze_pipeline.py` |
| Face crop padding | 0.3 | `gaze_pipeline.py` |
| 1€ filter | `MIN_CUTOFF 0.7`, `BETA 0.6`, `D_CUTOFF 1.0` | `gaze_server.py` |
| Broadcast | 60 Hz nominal (measured ~35-44 Hz; Windows timer granularity) | `gaze_server.py` |
| Calibration | degree 3 via `robust_fit_samples`, MAD factor 2.5, `min_keep` 60 | `calibration_utils.py` |
| Positioning gate | 45–65 cm, `CENTER_TOL` 0.12, `REAL_IPD_CM` 6.3, `ASSUMED_HFOV_DEG` 60 | `positioning_gate.py` |

**Correction (section 32.6b).** The `BINS` row previously described the angle as
`argmax·4 − 180`. That was wrong, and it was wrong in the one table Phase 2's `model.py`
is meant to be written from. The code computes a **softmax-weighted expectation over the
90 bins** (`gaze_pipeline.py` L61-64) — it softmaxes the logits, dots them with the bin
indices, then scales and offsets. An argmax implementation would agree with it only when
the distribution is sharply unimodal, and would otherwise return a plausible, smooth,
wrong answer while diverging discontinuously near a bin boundary. Section 32.2 measured
the practical consequence of the distinction: because the decode is an expectation, it is
smooth, which is why it survives an execution-provider change with 188x of tolerance to
spare. Row corrected in place; recorded here rather than silently edited, because the
error was live long enough to be built on.

**The yaw/pitch defect**: `gaze_pipeline.py` L59-60 unpacks `yaw_bins, pitch_bins` even
though the ONNX graph *names* tensor[0] `pitch_bins`. `export_to_onnx.py` L~26 confirms
this is deliberate — L2CS's `forward()` returns yaw first and the export labels are
wrong strings. This will live in `core/model.py` with the comment carried over intact
and a regression test pinning the decode, because it is precisely the kind of thing a
refactor reintroduces silently.

---

## 8. Deviations log

```
DEVIATION — Phase 0
  Spec says:   Read GazeModuleToPipLibrary.docx as the draft plan.
  I did:       Could not open it (binary .docx, no converter available). Worked
               from your §3 corrections, which supersede it.
  Because:     No tooling to extract the text in this environment.
  Cost of not: Any draft-plan detail not restated in §3 is unknown to me.
  Blocking?    no — but tell me if the DOCX has anything §3 omits.
```

```
DEVIATION — DEV-1 (affects Phase 6/9)
  Spec says:   Ship an [export] extra with torch/torchvision/onnx.
  I recommend: [export] declares torch/torchvision/onnx only; `l2cs` is documented
               as a manual `pip install git+…` step, not a declared dependency.
  Because:     PyPI rejects direct-URL dependencies in ANY extra, not just the base
               dependency list. An [export] extra naming l2cs would make the wheel
               unpublishable — the same blocker as B2.
  Cost of not: Upload to PyPI fails at Phase 10.
  Blocking?    no — proceeding on this basis unless you object.
```

---

## 9. Risks not covered by the brief

1. **`gaze_server.py` is no longer a pure gaze server.** It now owns input-mode state
   and camera hand-over, and imports `gesture_bridge`. Phase 7 assumes it can be lifted
   into `<pkg>.server` unchanged. It cannot, without either dragging the gesture concern
   into the SDK or splitting the file first. → Q6.

2. **No automated tests exist today.** Rule 2's golden-file harness is the only safety
   net, and it needs a recorded session with a real face. That recording must come from
   you (Q5) — I cannot generate a face.

3. **The calibration profile is personal data.** `models/calibration_model.pkl` is
   derived from a specific person's eyes. It is already untracked from git; it must also
   never enter a wheel or a test fixture.

4. **Accuracy is unverified post-refactor.** The README's 2.0–2.4 cm figure comes from
   `milestone6`. If the milestone scripts are deleted, the ability to re-measure goes
   with them. → Q7.

---

## 10. Questions before Phase 1

Numbered for easy reply. **Q1 is blocking; the rest shape the work.**

1. **LICENSE — blocking.** Gaze360 is non-commercial research use only, and the
   weights derive from it. Which do you want?
   **(a)** Ship the SDK as a research/non-commercial tool, state the restriction
   prominently, and have the downloader fetch from the *original upstream* only —
   never our mirror.
   **(b)** Hold publication while you seek clarification or permission from the
   Gaze360/L2CS authors.
   **(c)** Retrain or swap to a permissively-licensed gaze model — clean legally, much
   larger scope.
   **(d)** Publish code-only with no downloader; users obtain weights themselves.
   I recommend **(a) or (d)** for a v0.1, but this is your call and I will not proceed
   past Phase 5 without it.

2. **Upstream weight URL.** The README's HuggingFace link is a third-party mirror
   unrelated to the model's authors. Do you have an official URL from the L2CS-Net
   authors? If not, option (d) above becomes considerably more attractive. **I will not
   invent a URL or a checksum** (rule 7).

3. **Target platforms for v0.1.** The brief requires abstracting the Windows-only
   backends (D1), which I agree with. But do you want macOS/Linux **supported and
   tested**, or **structurally allowed but untested**? I can only honestly claim what I
   can run, and I have Windows here.

4. **Python version floor.** Reference env is 3.14. Do you want the floor at 3.10 (needs
   verifying mediapipe/onnxruntime wheels across 3.10–3.14) or 3.12 (narrower, honest,
   less matrix work)?

5. **Golden-file recording.** Rule 2 needs a short recorded webcam session with your
   face, plus the current pipeline's outputs, committed as a test fixture. Are you happy
   for a video of your face to live in the repo? If not, the alternative is a synthetic
   or opaque fixture (recorded landmark arrays rather than frames), which tests less but
   ships nothing personal.

6. **`gaze_server.py` and the gesture integration.** The server now imports
   `gesture_bridge` and manages camera hand-over. For Phase 7, do you want:
   **(a)** the SDK ships a *gaze-only* server, and the game keeps a thin local wrapper
   that adds the gesture mode (my recommendation — keeps the SDK's scope clean);
   **(b)** the SDK server grows a plug-in hook for extra input sources;
   **(c)** defer the server extra to v0.2 entirely.

7. **Milestone scripts.** Which do you consider disposable? My reading:
   - `milestone1`, `2`, `3`, `5` — superseded by the SDK + `check`/`demo` CLI → deletable
   - `milestone4` — the calibration routine; its *logic* becomes `calibration/ui.py`, so
     the file goes but the behaviour must be preserved
   - **`milestone6`** — the only accuracy measurement you have; I recommend keeping it
     as an `examples/` script or porting it to a CLI command
   - `milestone7`, `8`, `9` — demos; `8` (live cursor) is the best candidate to become
     `<pkg> demo`
   Confirm, and note that 6/7/8/9 import 4 and 7, so deleting them is order-dependent.

8. **Package name.** `focusedgaze` is free. Do you want me to publish a `0.0.0`
   placeholder now to reserve it, or accept the risk?

---

---

# PHASE 1 — Skeleton and safety net

Status: **complete**, stopped at the gate.

## 1.1 What was built

Package root: `focusedgaze-sdk/` (sibling of `gaze-detection/` in this workspace; it
becomes its own repository at Phase 11).

- Full §4.1 layout: **24 stub modules**, each carrying a docstring naming the phase
  that fills it, so an unimplemented file is never mistaken for a finished one.
- `pyproject.toml` — hatchling backend, dynamic version read from `__init__.py` (D6),
  dependency **ranges** not pins (D5), all six extras (D1/D8), `py.typed` shipped (D7),
  and a build `exclude` list that keeps weights, fixtures and `.pkl` files out of any
  wheel.
- `LICENSE` (MIT, my code) and `NOTICE` — the Gaze360 non-commercial restriction stated
  plainly, along with the "we neither distribute nor mirror the weights" policy from
  your §1 decision.
- `README.md` with the platform-support table (§3: Windows tested, others structurally
  supported and explicitly untested).
- `cli.py` with a real `main()`. The console script is declared in `pyproject.toml`, so
  a stub that could not be invoked would be a dangling entry point; it currently prints
  its version and says plainly that commands arrive in Phase 6.
- `.gitignore` excluding `tests/fixtures/tier2/`, all `*.onnx` / `*.task` / `*.pkl`.

**Exit criteria met:** `pip install -e .` succeeds, `import focusedgaze` succeeds,
`__version__` resolves to `0.0.0`, and `focusedgaze --version` runs.

## 1.2 Wheel availability (§4 instruction, checked before writing classifiers)

| Package | cp312 | cp313 | cp314 |
|---|---|---|---|
| `onnxruntime` 1.28.0 | ✅ | ✅ | ✅ |
| `onnxruntime-directml` 1.24.4 | ✅ | ✅ | ✅ |
| `numpy` 2.5.1 | ✅ | ✅ | ✅ |
| `platformdirs` | pure-Python | | |
| `mediapipe` 1.0.0 | not resolvable by filename tag | | |
| `opencv-python` 5.0.0.93 | not resolvable by filename tag | | |

`requires-python = ">=3.12"` is therefore defensible. The two "not resolvable" rows are
a limitation of my filename heuristic, not evidence of missing wheels — both are
installed and working on 3.14 in the reference environment. **I have not verified them
on 3.12 or 3.13 by actually installing there.** If you want that claim tested rather
than inferred, it needs a real 3.12 environment; say so and I will add it to Phase 9.

## 1.3 Golden-file harness (rule 2)

Two tiers, per your §5.

**Tier 1 — committed, numeric, runs in CI.** Recorded from the *unmodified* pipeline by
`tests/golden/record_tier1.py`:

| Fixture | Cases | Covers |
|---|---|---|
| `calibration_apply.json` | 169 | degree-3 polynomial over a (pitch, yaw) grid incl. the [0,1] clamp edges |
| `one_euro.json` | 120 | filter response across ramp, step and jitter |
| `positioning_gate.json` | 15 | distance, centring, zone decision over synthetic geometries |

**Tier 2 — local only, gitignored, `@pytest.mark.hardware`.** `record_tier2.py` captures
frames plus the pipeline's expected `(pitch, yaw)`, writes `frames.npz` + a
`manifest.json` carrying a SHA-256. `test_golden_tier2.py` verifies the digest, then
replays the sequence *in order* from a clean bbox-smoothing state — order matters
because face detection is stateful across frames.

Regeneration by anyone is supported and documented in the recorder's docstring: a third
party records their own session and the fixture stays self-consistent, because expected
values are captured at record time. The suite is therefore not permanently tied to my
machine.

**Test selection:** `pytest tests/` → 4 passed, 2 deselected (hardware excluded by
default). `pytest -m ""` → 4 passed, 2 skipped (Tier 2 skips cleanly with an actionable
message when no fixture exists).

**Implementation selection** lives in `tests/golden/adapters.py`, not in the test files.
Tests import `get_impl()`; the adapter decides between legacy and SDK
(`FOCUSEDGAZE_GOLDEN_IMPL=legacy|sdk` forces one). This is what lets the assertions stay
frozen while the code beneath them is replaced.

## 1.4 Findings

**F1 — `PositioningGate`'s output depends on the process working directory. (New bug,
found by the harness on its first run.)**

It loads `models/camera_focal.json` through a *relative* path in `__init__`. Recording
ran with cwd inside `gaze-detection/` and found the file; the test ran from the SDK root
and silently fell back to the assumed-FOV default. Identical landmarks produced
**117.406 cm vs 121.244 cm** — a 3.8 cm difference, larger than the system's entire
claimed accuracy, decided purely by where Python was launched.

This is a latent bug in the current system, not merely a packaging inconvenience: any
script run from the wrong directory gets a quietly different distance estimate. The
harness pins the recorded focal context explicitly so comparisons are meaningful.
**Phase 2 removes the ambiguity** by making focal length explicit configuration rather
than an implicit file lookup.

**F2 — Fixture inputs must be stored at full precision. (My error, fixed.)**

I first rounded the recorded inputs to 6–10 decimal places. The 1-euro filter derives
its sampling frequency from timestamp deltas, so a timestamp rounded to 6 dp changed the
computed frequency and pushed the output past the 1e-9 tolerance. Fixtures now store
raw `float64`, which JSON round-trips exactly through `repr`. Worth recording because
the same trap applies to any future fixture.

**F3 — A4 confirmed (the degree 2 / 3 discrepancy).** `fit_calibration` defaults to
`degree=2`; `robust_fit_samples` passes `degree=3`. The live system only ever reaches
the fitter through the robust path, so effective behaviour is degree 3. The recorded
fixture carries `"degree": 3` from the actual saved model. Phase 5 makes the degree
explicit configuration; effective behaviour is preserved.

## 1.5 Deviations

```
DEVIATION — Phase 1
  Spec says:   Publish a 0.0.0 placeholder to PyPI to reserve `focusedgaze`
               before Phase 1 code lands (§8).
  I did:       Prepared everything needed for it — name, metadata, README stub,
               LICENSE, NOTICE, buildable pyproject — but did NOT publish.
  Because:     Publishing needs your PyPI credentials or a Trusted Publishing
               setup, neither of which I have. Uploading under your identity is
               also outward-facing and irreversible: a version number on PyPI
               can be yanked but never reused.
  Cost of not: The name stays unreserved and someone could take it.
  Blocking?    no — but it is the one Phase 1 item still open. See §1.6.
```

```
DEVIATION — Phase 1
  Spec says:   Phase 1 exit = "harness records and replays a session".
  I did:       Tier 1 records and replays fully (4 tests green). Tier 2's
               tooling is built and its guard verified, but no fixture was
               recorded — the camera saw no face across 45 attempts, so the
               recorder refused to write.
  Because:     A no-face fixture exercises only the null path and would give
               false confidence, so the recorder treats it as an error (exit 2)
               rather than writing a useless file. It needs you in frame.
  Cost of not: The frames -> (pitch, yaw) stage has no regression net until a
               fixture exists. Phase 2 refactors exactly that stage.
  Blocking?    Not for starting Phase 2, but I would rather have it BEFORE
               Phase 2's exit gate than after. See §1.6.
```

## 1.6 Open items carried into Phase 2

1. **PyPI placeholder not published** — needs credentials. Options: you publish it, you
   provide a token, or you set up Trusted Publishing (which Phase 10 wants anyway).
2. **Tier 2 fixture not recorded** — needs you at the camera, ~30 seconds:
   `python tests/golden/record_tier2.py --frames 60`. The camera appeared to have no
   face in view; note it also read as muted earlier in this session.
3. **`milestone6` accuracy baseline not captured.** Your §7 requires running it on the
   unmodified code and recording the numbers here *before anything is deleted*. It is an
   interactive on-screen accuracy test — it needs a person following targets, so I
   cannot produce it alone. Nothing is deleted until Phase 8, so this is not yet
   blocking, but the window closes then.

---

---

# PHASE 1 ADDENDUM — authorship and harness corrections

## 13. Authorship

The package is authored and published by:

```
Muhammad Asif Khan
https://github.com/muhammad-asifkhan
```

The project is attributed to a single author throughout: every commit, every file and
every reference. There is no second identity anywhere in the repository or its history.

### Every hit found, and its disposition

Grepped the whole tree for the superseded identity strings:

| Location | Disposition |
|---|---|
| `focusedgaze-sdk/pyproject.toml` — `authors` | Changed; `maintainers` added |
| `focusedgaze-sdk/pyproject.toml` — Homepage / Source / Issues | Changed to `muhammad-asifkhan/focusedgaze` |
| `focusedgaze-sdk/README.md` — author line | Changed |
| `focusedgaze-sdk/NOTICE` | New §0 naming the author |
| `MIGRATION_AUDIT.md` §3.3 | Corrected, pointing here |
| `focusedgaze-sdk/LICENSE` | **Already correct** — it read "Muhammad Asif Khan" before I touched it |
| **`README.md` in the originating game repository** | **NOT changed.** That is a different, already-published repository outside this project's scope. Rewriting its clone URL would break its setup instructions. |

**Git remotes:** at that point `focusedgaze-sdk/` was not yet a git repository, so there
was no SDK remote to re-point. The originating game repository's remote was left alone for
the same reason as the row above — it is a separate project.

### Downstream consequences, flagged now (not deferred to Phase 10)

1. **PyPI account.** The `0.0.0` placeholder must be published from the project owner's
   PyPI account, since that account owns the project long-term. The Phase 1 deviation
   stands: publication is not performed automatically.

2. **Trusted Publishing is owner-bound.** PyPI's OIDC publisher configuration names a
   specific GitHub **owner + repository + workflow filename**. A publisher configured
   against any other owner will **reject** a release from
   `muhammad-asifkhan/focusedgaze` — the tokenless upload simply 403s. When Phase 10
   drafts `.github/workflows/release.yml`, the setup instructions will name
   `muhammad-asifkhan/focusedgaze` explicitly and state that the PyPI project's
   publisher settings must match owner, repo and workflow filename exactly.

3. **Copyright, raised not decided.** The MIT `LICENSE` names a single copyright holder,
   which is the normal form. `NOTICE` §0 now credits contribution, which is a separate
   matter from ownership. **If the code has more than one author in substance, whether
   the copyright line itself should name more than one person is a decision for the
   authors, not a packaging detail.** → Q9 below.

## 14. Rule 4 violation in my own harness — fixed

`record_tier1.py` hard-coded `<legacy-dir>` with no override, and
`adapters.py` used the same absolute path as its default. That is a direct rule 4
violation, and it would have made the harness unrunnable for a second developer — which
matters immediately now that the project moves to another person's account.

Fixed in all three files, consistently:

- `FOCUSEDGAZE_LEGACY_DIR` is read first.
- The fallback is **relative**: `../gaze-detection` from the repository root, i.e. the
  ordinary side-by-side checkout. (It resolves correctly in the current layout.)
- A missing directory now prints the variable name and the expected layout, not an
  absolute path from someone else's machine.
- Documented in both recorder docstrings and here.
- Verified: no an absolute machine path string remains anywhere under `tests/`, and the suite
  still passes (4 passed, 2 deselected) using only the relative fallback.

**Worth recording plainly:** I found this class of bug in the shipping code (F1, the
working-directory-dependent positioning gate) and introduced the same class of bug into
the test code on the same day. The golden harness is subject to the same engineering
rules as the package, and I will treat it that way from here.

## 15. The Phase 1 wheel claim — RESOLVED, verified

**Original caveat (Phase 1):** the `Programming Language :: Python :: 3.12` and `3.13`
classifiers were **inferred, not verified**. cp312/cp313/cp314 wheels were confirmed on
PyPI for `onnxruntime`, `onnxruntime-directml` and `numpy`, but `mediapipe` and
`opencv-python` use wheel tags a filename check could not resolve, and installation had
been done on **3.14 only, on Windows only**.

**RESOLVED — CI run 2 (`14f7b36`), all three matrix rows `success` on Linux.**

| Job | Install | Conclusion |
|---|---|---|
| `test (3.12)` | yes | success |
| `test (3.13)` | yes | success |
| `test (3.14)` | yes | success |

`mediapipe` and `opencv-python` install and run on Linux across 3.12, 3.13 and 3.14.
Neither anticipated failure mode occurred: no missing wheel, and no missing system library
(`opencv-python` on a headless runner often needs `libGL`; it did not here).

**Consequences:**

- `requires-python = ">=3.12"` is **honest and tested**, not inferred.
- The `3.12` and `3.13` classifiers are **honest and tested**. They stay.
- **The Phase 9 wheel-verification task is CLOSED, ahead of schedule.** Phase 9 keeps its
  other items: clean-venv wheel install and distribution content audit.

## 15a. Three guarantees exercised for the first time

The same run was the first execution ever of three checks that had been asserted but never
run:

| Check | Result | Why it matters |
|---|---|---|
| `mypy --strict` on `src/` | **passes** on 3.12/3.13/3.14 | Phase 3's exit criterion, met early for the code that exists |
| **D8 bare-import guarantee** | **passes** | A venv with `pip install -e .` and no ONNX provider, no `websockets`, no `scikit-learn` imports `focusedgaze` cleanly. A design promise since Phase 0, never previously tested |
| Test suite on **Linux** | **19 passed** | The golden fixtures reproduce identically on a platform they were never recorded on — evidence the extracted `filters.py` and `positioning.py` are genuinely portable rather than accidentally Windows-shaped |

## 15b. Lesson — a hard-failing first step hides everything behind it

CI run 1 failed on nine ruff errors: an unsorted `__slots__`, an unsorted `__all__`,
`Sequence` imported from `typing` rather than `collections.abc`, two now-redundant quoted
annotations, and import ordering in three test files. All cosmetic, all auto-fixable.

Because `Lint` failed hard and first, **Type-check, the bare-venv import check and the
entire test suite were skipped**. Three checks that had never run once were hidden behind a
style error, and the run reported one problem while concealing whether there were others.

The dangerous part is what would have happened next. Fixing only the lint would have turned
the run green, and "CI passes" would have been reported — while `mypy --strict` had still
never been attempted on any platform. **A green run after a masked failure produces false
confidence, because it looks identical to a run where everything was genuinely checked.**

**Fix:** `Type-check`, the bare-venv check and `Test` now carry `if: not cancelled()`, so a
lint failure no longer masks them. The job still fails overall; it reports everything first.
`not cancelled()` rather than `always()`, because `always()` keeps running through a manual
cancellation and makes a cancelled run cost as much as a full one.

**General principle:** in any pipeline of independent checks, an early hard failure must not
prevent later ones from reporting. Ordering should decide what is *reported first*, never
what is *reported at all*.

## 15c. Coverage baseline (Phase 8 target: >= 80% on non-hardware paths)

Measured with the same command CI runs. The CI job-log endpoint returns HTTP 403 without
authentication, so this is a local run of `pytest --cov=focusedgaze --cov-report=term-missing`
rather than the run-2 log itself; the command and the test set are identical.

```
TOTAL   164 statements   22 missed   87%
```

**The headline number is not meaningful yet, in both directions.**

| Module | Statements | Missed | Cover |
|---|---|---|---|
| `core/positioning.py` | 103 | 7 | 93% |
| `core/filters.py` | 45 | 2 | 96% |
| `cli.py` | 13 | 13 | 0% |
| every other module | 0 | 0 | "100%" |

- **Inflated by stubs.** Eighteen modules are empty Phase 2-7 placeholders with zero
  statements, which coverage reports as 100%. They contribute nothing and flatter the total.
- **Deflated by `cli.py`.** Its 13 statements are a version banner with no test, dragging
  the total down by roughly six points.

**The honest figure for code that actually exists** — `filters.py` plus `positioning.py`,
the two modules extracted so far — is **148 statements, 9 missed, 94%**.

That is comfortably above the Phase 8 target, but the target is not yet meaningfully
tested: it will only start to bite once the stubs are filled, at which point the total will
drop sharply before recovering. **A falling coverage percentage during Phases 2-7 is
expected and is not a regression** — it means real code is replacing empty files faster
than tests are being written for it. The number to watch is per-module, not the total.

Uncovered lines in the two real modules are error and edge paths: the degenerate-landmark
return, the file-load failure branch in `FocalCalibration.load`, and a few guidance-message
branches in the positioning gate.

## 16. Additional question

9. **Copyright line.** `LICENSE` names Muhammad Asif Khan. Should any other party be
   named in the copyright line or credited separately? Not a decision to be made
   unilaterally.

---

---

# PHASE 2 PRELUDE — version control, publishing config, decisions

## 17. Repository

`focusedgaze-sdk/` is now a git repository on branch `main`. `MIGRATION_AUDIT.md`
moved here from the game-repo root: it documents this package, it is Phase 0's
deliverable, and it has to be committable with the code it describes.

**`.gitignore` was audited before the first commit**, because a file committed once
stays in history. Three gaps were closed: `gaze_env/` (this project's conventional venv
name, previously uncovered), editor/OS junk, and recorded video (`*.mp4/avi/mkv`). Also
added `*.pt`/`*.pth` alongside `*.onnx`/`*.task`.

A dry run before committing confirmed **43 files tracked, and none of**: model weights,
`.pkl` calibration profiles, `tests/fixtures/tier2/`, any virtualenv, `__pycache__`, or
build output.

Commits so far:

| Commit | Contents |
|---|---|
| 1 | Phase 0 — the migration audit |
| 2 | Phase 1 — skeleton, packaging, golden harness, `core/filters.py` |
| 3 | CI and release workflows |

**Authorship:** the repository-local git identity is set to the person at the keyboard
(the contributor's own identity), with `Co-authored-by: Muhammad Asif Khan` trailers,
so history reflects joint work rather than a single account. **If the machine changes
hands, `git config user.name/user.email` in this repo should change with it** — it is
deliberately local, not global.

Per rule 6, every phase gate from here is its own commit.

## 18. Trusted Publishing — configured for the real owner

`.github/workflows/release.yml` is written against **`muhammad-asifkhan/focusedgaze`**,
with no placeholders. The setup steps are documented inline in the workflow because PyPI
binds a trusted publisher to four values that must match exactly:

```
PyPI project    focusedgaze
Owner           muhammad-asifkhan
Repository      focusedgaze
Workflow        release.yml
Environment     pypi        (and `testpypi` on test.pypi.org)
```

A publisher configured against any other owner, repository, workflow filename or
environment would reject releases from this repository with a 403 — the failure mode
flagged earlier.

The release pipeline makes TestPyPI a **required predecessor job**, so PyPI cannot be
reached without a successful dry run. The build job also **fails the release** if any
distribution contains weights, calibration pickles, or test fixtures — Phase 9 asks for
that audit by hand; this makes it a gate instead of a habit.

`ci.yml` runs ruff, mypy and the non-hardware suite across 3.12/3.13/3.14, which is what
converts the *inferred* 3.12/3.13 support (§15) into a tested claim. It also installs
into a bare venv to prove the base package imports with no ONNX provider, no
`websockets` and no `scikit-learn` present (D8).

Both workflows validated as YAML.

## 19. PyPI placeholder — recommendation

**Recommendation: publish it now.** Reasons, in order of weight:

1. **The Trusted Publishing config binds the PyPI project name.** A rename late in the
   migration means reconfiguring the publisher as well as editing the package — two
   coupled changes at the worst possible moment.
2. **`focusedgaze` is a guessable name in an active field.** Availability at Phase 0 is
   not a reservation, and nine phases remain.
3. **A rename at Phase 9 touches everything**: `pyproject.toml`, `README`, `NOTICE`, the
   module directory, every import, the workflows, and this audit.
4. **The cost of publishing is small and bounded.** A `0.0.0` can be yanked; the version
   number is then unusable, which is irrelevant for a placeholder.

The artifacts are already built and audited, so it is one command:

```bash
python -m twine upload dist/*      # from Asif's PyPI account
```

| Artifact | Size | Contents |
|---|---|---|
| `focusedgaze-0.0.0-py3-none-any.whl` | 16 KB | 32 files: source, `py.typed`, LICENSE, NOTICE |
| `focusedgaze-0.0.0.tar.gz` | 25 KB | 36 files |

`twine check` passes on both. Audited: **no weights, no `.pkl`, no fixtures, no
`__pycache__`** in either. Well inside the <5 MB target.

**I have not published**, and will not — it needs Asif's credentials and it is an
irreversible outward-facing action under someone else's identity. The deviation from
Phase 1 stands, now with the artifact ready.

## 19a. Line endings

`.gitattributes` added (`* text=auto eol=lf`, plus `binary` for every image, archive,
model and video format). Git had warned about LF/CRLF conversion on 16 files across the
first commits and it went unaddressed; with two Windows developers and Linux CI, the
outcome would otherwise depend on each machine's `core.autocrlf` and surface as
whole-file phantom diffs.

**`git add --renormalize .` changed nothing, and no second commit was made.** Git had
already normalised these files to LF when they were staged, so the index was correct
already — the warning described the *checkout* direction, not the stored bytes.
Verified with `git ls-files --eol`: every non-empty tracked file is `i/lf`. (Two entries
report `i/none`; both are genuinely empty files, `py.typed` and `tests/golden/__init__.py`.)
The attributes file makes the guarantee explicit and durable rather than incidental.

## 19b. Correction — `Co-authored-by` was over-applied

The trailer asserts shared authorship of a *specific commit*. Phases 0 and 1 were written
by one person, so the first four commits carry a trailer they
should not.

**Intent (not yet executed at the time of writing):** the trailers are to be dropped from
the solo commits during the sanitisation history rewrite described in §19f, so that no
separate history churn is incurred for a metadata line. This paragraph is written in the
future tense deliberately and will be updated to past tense **only after the verification
gate in §19f passes** — an audit whose only value is being accurate about itself must not
claim an operation succeeded before it has run.

From this point the trailer is used only where work is genuinely joint.

Related: the repository-local `user.name` was initially a GitHub handle rather than a
real name. It is now **Muhammad Asif Khan**, matching the package metadata. Still
deliberately local to this
repository, so it should be changed if the machine changes hands.

## 19c. CI matrix — a red first run is the expected result

`ci.yml` installs `mediapipe` and `opencv-python` on **Linux** across 3.12/3.13/3.14.
Wheel availability there is **inferred, not verified**, and differs from Windows — §15
records that I only ever installed on 3.14, and only on Windows.

The first CI run is therefore expected to fail, and that failure is the deliverable, not
an obstacle. **When it fails I will report the specific `(python-version, package)` pairs
that have no wheel — I will not quietly drop rows from the matrix to make it green.**
Narrowing `requires-python` or the classifiers, if warranted, is a decision that follows
from that evidence.

Note the workflows cannot run yet: there is no GitHub remote. The first run happens when
`muhammad-asifkhan/focusedgaze` is created and this repository is pushed.

## 19d. Phase 2 gate condition

The Phase 2 gate **stays open until the Tier 2 replay passes**, regardless of how much
code is extracted.

`landmarks.py` and `model.py` are the only modules Tier 2 covers, and they carry the two
highest-risk edits in the phase: the `_smoothed_bbox` module global becomes instance
state, and the yaw/pitch decode — where the ONNX graph's output *names* are wrong — is
rewritten. Tier 1 cannot see either of them. Extracting them without the replay would be
refactoring the riskiest code in the migration with no safety net over it.

Work proceeds on `positioning.py` (Tier 1 covered) in the meantime.

## 19e. Fixture branch coverage — audited across all fixtures

The assumed-HFOV fallback was unpinned: `positioning_gate.json` was recorded with
`camera_focal.json` reachable, so only the measured branch was proven. Fixed, and the
same question asked of every other fixture.

**Positioning — two fixtures now.**

| Fixture | `focal_override` | Branch |
|---|---|---|
| `positioning_gate.json` | `[1073.4275…, 1280]` | measured focal |
| `positioning_gate_nofocal.json` | `null` | assumed-HFOV fallback |

Recorded from the legacy gate constructed inside a temporary directory, so its relative
`models/camera_focal.json` lookup genuinely failed. The two branches produce different
distances (121.24 cm vs the measured branch's value), and a guard test asserts they
differ — otherwise both parametrised runs could pass while proving one branch.

**Mutation-checked, because "it passes" is not the same as "it would catch a bug".**
Against the recorded fallback value of `121.243557`:

| Formula | Result | Caught? |
|---|---|---|
| correct `w / (2·tan(radians(60)/2))` | `121.243557` | exact match |
| degrees→radians conversion omitted | `-10.928397` | **caught** |
| angle not halved | `40.414519` | **caught** |

These are precisely the two errors that would otherwise have sailed through.

**Other unpinned branches found by asking the same question elsewhere:**

- **`one_euro.json`** — the filter only recomputes its sampling frequency when
  `t > t_prev`, and the recorded signal was strictly increasing, so the other side of
  that guard was never exercised. A stalled clock (repeated timestamp) or one that steps
  backwards is not hypothetical: a paused capture thread produces exactly that. Four
  cases appended — two repeated timestamps, one backwards step, one repeat after it.
  Fixture is now 124 samples.
- **`positioning_gate*.json`** — `evaluate()` returns `None` when the irises are under a
  pixel apart, and the geometry grid never degenerated. A coincident-landmark case is now
  recorded in both fixtures, with `result_is_none` asserted against the legacy behaviour.
- **`calibration_apply.json`** — checked, no gap. `apply_calibration(model, …)` takes the
  model as an argument rather than discovering it from a path, and its only branch is the
  `[0,1]` clamp, which the grid already spans (±0.72 rad, deliberately wider than the
  real signal range).

Principle recorded for later phases: **a branch selected by a file's presence, rather
than by an argument, is unpinned unless it was recorded both ways.** Phase 5's calibration
profile loading and Phase 6's asset cache are the next two places this will apply.

### Committed for Phase 5: mutation-check the calibration polynomial

Replacing the pickled scikit-learn pipeline with raw NumPy coefficients is the highest-risk
numerical swap in the migration, and `calibration_apply.json` currently proves equivalence
**without having been shown to catch a wrong implementation**. Passing a fixture and being
sensitive to error are different properties, as the focal-length check demonstrated.

Before that swap is accepted, the fixture must be shown to catch at least:

- **transposed coefficient order** — coefficients applied to the wrong feature
- **wrong feature-expansion ordering** — `PolynomialFeatures` emits a specific term order
  (`1, a, b, a², ab, b², …`); assuming a different one silently produces a plausible but
  wrong surface
- **x/y coefficient sets swapped** — symmetric-looking and easy to miss
- **degree mismatch** — fitting or evaluating at degree 2 where the model is degree 3
  (§F3: the legacy default and the robust path disagree, so this is a live risk)

A wrong polynomial does not crash; it returns a smooth, believable surface that is simply
in the wrong place. Only sensitivity testing distinguishes that from a correct one.

## 19f. Publication decision needed — MIGRATION_AUDIT.md

**Flagged, not decided. This needs an answer before the first push.**

This file is currently tracked in the repository that will become
`github.com/muhammad-asifkhan/focusedgaze`. It contains:

1. **Absolute paths from one machine**, in the Phase 0 inventory and findings.
2. **Personal email addresses**, in the authorship sections.
3. **A critical assessment of the origin project** — unused dependency pins, a
   working-directory bug, a version-fragile pickle, an inconsistent polynomial degree.
   Accurate and useful engineering record, but also a public critique of a named
   collaborator's code.
4. **A characterisation of a third party's conduct** regarding redistribution of
   Gaze360-derived weights — written on inference rather than investigation.

Item 4 carried the most weight. The download policy rests on the Gaze360 licence alone
and does not need it: the restriction is a documented fact, and "not the upstream source"
is sufficient grounds to reject a URL for an automatic downloader without saying anything
about whoever published it.

**DECISION: option (b), sanitise — applied to the file and to the history.** Absolute
paths became placeholders, personal emails were removed from the file body, and §4 now
states the restriction, the derivation and the resulting never-distribute policy without
characterising anyone. The technical record survives intact.

The options that were weighed:

- **(a) Public as-is** — maximum transparency about how the decisions were reached.
- **(b) Sanitise** — chosen. The engineering value is real and worth publishing, and none
  of it depends on naming a third party's conduct or on anyone's home directory layout.
- **(c) Keep it out of the published tree** — retain it locally or in a private repo, and
  ship only `NOTICE` + `CHANGELOG` publicly.

**My recommendation is (b)**, because the audit's engineering value is real and worth
publishing, while none of that value depends on naming a third party's conduct or on
anyone's home directory layout. But this is your call and the repository is not mine, so
nothing is pushed until you decide.

## 20. Q9 — answered

Answered at the time as a split between author and contributor. **Superseded by the
decision in §28:** the project is attributed to Muhammad Asif Khan alone, with no
contributor credit anywhere. `LICENSE`, `NOTICE` and the README all name him only.

---

---

# THE SANITISATION EPISODE — full record

Written under standing rule 9. Three attempts were needed; two caused damage. The failures
are the point of this section: each one was invisible until it did harm, and each was
declared safe by a check that turned out to be measuring the wrong thing.

## 21. Why the history was rewritten at all

The decision (§19f) was to sanitise `MIGRATION_AUDIT.md` for publication. Editing the file
was not enough: the original text lived in the first commit and every commit after it, and
would have been public the moment the repository was pushed. Personal email addresses were
also in commit metadata as well as file contents.

## 22. Attempt 1 — the line-wrapped literal

**What was tried.** `git filter-repo --replace-text` with a rules file of literal phrases,
including a multi-word phrase describing a third party's conduct.

**What happened.** The rewrite reported success. The verification gate — a per-commit grep
across every blob — found the phrase still present in one commit.

**Root cause.** The phrase was **line-wrapped in the source**: the words sat on two lines
with the second line's indentation between them. The rule was written with a single space,
so it never matched. The rules on either side of it *did* match, so the sentence was
partly rewritten and the target text survived inside a mangled sentence.

**Why it was dangerous.** The replacement failed *silently*, and a file-level "did anything
change?" check would have reported success. Only a per-commit content grep caught it.

**Fix.** Rules must be single-line fragments short enough that they cannot straddle a wrap —
one or two words, never a clause. Recorded as brief A2.

**Recovery.** Restored from a bundle. Nothing lost.

## 23. Attempt 2 — the rules file that rewrote every `#` in the repository

**What was tried.** The corrected rules, in a file annotated with `#` comments explaining
each rule and the deliberate `pyproject.toml` exclusion. A pre-flight check was written
first, and reported **PASS on all 10 rules**.

**What happened.** The rewrite replaced **every `#` character in the repository** with
filter-repo's default replacement marker. Every Python comment, every `.gitignore`
and `.gitattributes` entry, every
workflow comment. `.gitattributes` was mangled badly enough that git began emitting
`is not a valid attribute name` errors on ordinary commands.

**Root cause.** `git filter-repo --replace-text` **has no comment syntax**. Its format is
one rule per non-empty line; a line without `==>` means "find this, replace it with
that marker". The bare `#` separator lines in the file were therefore rules meaning
"find every `#`".

**Why the pre-flight missed it — the more important failure.** The pre-flight parsed the
rules file the way it was *intended* to be read:

```python
if not line or line.startswith("#") or "==>" not in line:
    continue          # <- skipped comments; filter-repo does not
```

It validated a **different interpretation of the input than the tool it was guarding**, so
it reported PASS on 10 rules while the tool was about to act on roughly 14 more that were
never shown to anyone. A guard that does not read its input identically to the tool is
decoration.

**Fixes, all three applied.**
1. The rules file contains **no comments**. Rationale moved to `fg-replacements.README.md`.
2. The pre-flight parses the file **exactly as filter-repo does**, reports the total rule
   count it sees, separates explicit (`==>`) from implicit rules, and **fails loudly** if
   any implicit rule exists.
3. A **dry-run into a throwaway clone** became mandatory, with `git diff --stat` against the
   pre-rewrite state. Corruption that broad is obvious in one `--stat`; that check alone
   would have caught this before it touched anything real.

**Recovery.** Restored from the fresh bundle. Verified: zero corruption markers in the
working tree, zero across every blob in every commit, 19 tests passing.

**Why the bundle discipline mattered.** The bundle in hand at that moment was 7 commits
while HEAD was 8. Restoring from it would have destroyed `STANDING_BRIEF.md`, the sanitised
audit and the `NOTICE` fix. A fresh bundle was taken first specifically because of that
gap — hence the standing rule: **a fresh `--all` bundle before every rewrite attempt.**

## 24. Attempt 3 — success, plus a composition trap found in the dry-run

**Pre-checks, all passed before execution.**

- **HEAD-only scan: zero of ten patterns present at the tip.** This closed the
  tip-versus-history question by evidence rather than argument — if the tip contains none
  of the patterns, the rewrite cannot damage tip prose. The dry-run's `git diff --stat`
  against the original tip was consequently **empty**: history changed, tip byte-identical.
- **Pre-flight: 10 rules, 10 explicit, 0 implicit, all firing.**

**The trap the dry-run caught.** `--replace-message` and `--message-callback` **do not
compose**. filter-repo implements the former as a message callback internally, so passing
both is not an error — one silently wins. Measured in the throwaway clone:

| | Result |
|---|---|
| `--replace-message` | fired (`licensing provenance` present) |
| `--message-callback` | **silently ignored** — all 4 `Co-authored-by` trailers survived |

**Fix.** Both jobs now happen in a **single** `--message-callback`, generated from the same
rules file so blobs and messages share one source of truth (`rewrite.py`). After stripping
trailers, the only remaining message content needing replacement was one phrase in one
commit — the email addresses existed *only* inside the trailers.

**A quirk worth recording.** On the real repository `filter-repo` raised an
`AssertionError` during its post-write repack, *after* the new history was written.
Assessed before proceeding: `git fsck` clean, 9 commits, hashes identical to the dry-run
(so the rewrite is deterministic), clean status. The `gc` was completed manually. Nothing
in the gate suggested the result was affected, but the tool did not exit cleanly and that
is recorded rather than buried.

**Gate result — all checks clean.** Blob scan clean, corruption canary 0, commit-message
scan clean, `Co-authored-by` count 0, 9 commits in the correct order. After
`reflog expire --expire=now --all` and `gc --prune=now`: `in-pack: 100`, `packs: 1`,
`garbage: 0`, and the four pre-rewrite commits confirmed unreachable.

## 25. What the commit-message scan added

The gate originally walked `ls-tree` blobs only. Extending it to commit messages found
what blob scanning could not:

```
'asifcalm53@gmail.com' in 4 messages   (inside Co-authored-by trailers)
<a characterisation phrase>  in 1 message   (not covered by anything)
```

`--replace-text` does not touch messages, and the trailer-stripping callback would not have
caught the second. **Messages are as public as file contents**; any future gate covers both.

## 26. Author metadata — the field the gate did not scan

The gate scanned message *bodies*. It did not scan the commit **author** and **committer**
fields, which are equally public once pushed.

```
5 commits  <full name>     <personal address>
4 commits  <GitHub handle> <same address>
```

Inconsistent in two ways: the same person under two names, and a personal mailbox in
metadata after the same address had been scrubbed from `NOTICE` and this audit.

**Decision at the time — option (c).** Neither "leave it" nor "replace the identity":

- The author **name** was to be kept and only the email replaced.
- **Only the EMAIL field is rewritten**, to a GitHub noreply address. Name and email are
  separate fields; this preserves authorship while removing the personal mailbox from
  public history.
- Asif remains owner and author of record for the package.

**Status: NOT YET EXECUTED.** The exact noreply address must be supplied — GitHub's form is
`<ID>+<username>@users.noreply.github.com`, found under Settings → Emails. **It will not be
guessed** (rule 8). The planned method is `--email-callback`, touching only the email, not
a full re-filter.

**`.mailmap` added regardless of that decision.** The name inconsistency is a *display*
problem, and a `.mailmap` fixes it in `git log`, `git shortlog` and forge contributor lists
**without rewriting a single commit**. Verified: `git shortlog -sne HEAD` collapses all nine
to one identity. When the email rewrite happens, the mailmap gains a row mapping the old
address to the canonical one so historical references still resolve.

## 27. Pre-push state — Part B findings

| Check | Result |
|---|---|
| GitHub `muhammad-asifkhan/focusedgaze` | Exists, **empty — no commits** |
| First push | **Clean.** No unrelated-histories problem, nothing to reconcile |
| Local remote | **None** — `filter-repo` removes remotes by design; must be added fresh |
| Repo name | Matches `pyproject.toml` and `release.yml` exactly |
| Local branch | `main` |
| PyPI `focusedgaze` | **404 — project does not exist** |
| TestPyPI `focusedgaze` | **404 — project does not exist** |
| GitHub environments `pypi` / `testpypi` | **Cannot check** — not publicly visible |
| Pending publisher registered? | **Cannot check** — not publicly visible |

### OPEN ITEM — confirm before the first tag

Neither PyPI project exists. The likely explanation is that a **pending publisher** was
registered on each: a pending publisher does not create a project and is not publicly
visible, which fits the observed 404s exactly. If so, everything is correct and
`release.yml` — whose inline instructions assume the pending-publisher path — needs no
change.

**This must be confirmed with Asif before the first tag.** The difference between "pending
publisher registered" and "nothing registered" surfaces only as a **403 at upload time**,
which is the worst possible moment to discover it. Both PyPI and TestPyPI need checking
**separately**; they are independent registrations.

Two caveats on these findings: the GitHub check returned a slightly self-contradictory
summary ("does not exist" alongside "This repository is empty" — the latter is the
meaningful signal, and it should be eyeballed), and the **default branch setting** on an
empty repository cannot be verified without authentication. GitHub has defaulted to `main`
since 2020 and the local branch is `main`, so they should match, but that is inference.

---

## 28. Attribution — single author, decided and applied

**Decision: the entire project is attributed to Muhammad Asif Khan
(`github.com/muhammad-asifkhan`). No contributor credit, no co-author, no second identity
anywhere — every commit, every file, every reference.**

This supersedes all earlier authorship guidance in this document, including 19b, 20 and
the option-(c) decision recorded in 26. Those sections are left in place with pointers
here rather than deleted: a record that quietly rewrites its own earlier decisions is
worth less than one that shows them changing.

### What was changed

| Target | Change |
|---|---|
| Commit **author** and **committer** — name *and* email, all commits | Rewritten unconditionally to `Muhammad Asif Khan <asifcalm53@gmail.com>` |
| `.mailmap` | **Deleted.** It existed only to map a GitHub handle to a real name. With one identity across all commits it is obsolete, and keeping it would preserve the very mapping being removed. |
| `NOTICE` | Contributors block removed; section 0 renamed "Author" and names one person. The provenance sentence about the originating game project went with it. |
| `README.md` | Contributor line removed. |
| `MIGRATION_AUDIT.md`, `STANDING_BRIEF.md` | Every passage naming a second person reworded. Engineering content — the failures, the reasoning, the gates — kept in full; that is the value of these documents. |
| `.github/workflows/release.yml` | The "for example, a different owner" comment no longer names one. |
| Commit messages | Messages naming a second person rewritten by the history pass. |

### Method

Author and committer were rewritten with **unconditional** `--name-callback` and
`--email-callback` — not a mapping from old values. A mapping leaves any identity it did
not anticipate untouched; an unconditional callback cannot. The four identity strings were
also added to `fg-replacements.txt` as literal rules, so the same pass removed them from
file contents and commit messages throughout history.

### Why the tip edits were committed *before* the rewrite

`filter-repo` rewrites what is committed, not the working tree. An uncommitted edit at the
time of the rewrite would leave the tip carrying the replacement-mangled version, with the
intended text sitting as an uncommitted diff — a trap already hit once in this migration.

### Note on removal by substitution

Deleting a block by string replacement can leave a dangling heading or a half-sentence that
still parses as Markdown but reads as nonsense. `NOTICE`, `README.md`, `STANDING_BRIEF.md`
and this file were therefore opened and read at HEAD after the rewrite, not merely grepped.

---

## 29. The attribution rewrite — third history pass

Executed under the sequence in the standing brief. Two failures surfaced in the dry-run
clone; the real repository was never touched by either.

### Failure A — the line-wrap bug, hit a second time

The rule was the superseded contributor's full personal name: a two-word literal joined by
a single space. In historical blobs that name wrapped across lines in two places — once
with the given name ending a line and the sentence continuing on the next, and once with
the given name ending a line and the family name beginning the following one, with the
paragraph's indentation sitting between them. In both cases the two words were separated
in the file by a newline plus indentation rather than by the single space the rule
required.

The rule matched neither. **This is precisely the failure documented in section 22 and in
the brief's A1** — the rule that exists to prevent it was written, read, and then walked
into anyway.

The reason is worth stating plainly: A2 says "single-line fragments only, one or two words,
never a clause". A two-word *name* felt like it satisfied that. It does not. The test is
not "is this short" but **"can this string be split by a line break?"** Any multi-word
literal can.

**Fix.** Three rules instead of one, ordered longest-first so the full name is consumed
before either half can be: the two-word full name mapping to the canonical author's full
name, then the given name alone mapping to the canonical full name, then the family name
alone mapping to the canonical family name. Longest-first ordering matters — with the
single-word rules ahead of the two-word one, the first rule would consume half the name
and leave the second rule with nothing to match.

A wrapped occurrence now yields clumsy text in historical blobs, which the standing
decision accepts: history is scrubbed, not readable prose.

**Note on how this section is written (standing rule 10).** The rules and the wrapped
source text above are *described* rather than reproduced. An earlier revision quoted both
verbatim, which reintroduced into the tip exactly the strings the pass existed to remove —
the same self-defeating-documentation pattern as Failure B below, and the fifth instance
of it in this project. Any future write-up of a replacement pass describes the shape of
its rules and never prints them.

### Failure B — the documentation defeated its own corruption canary

Section 23 quoted filter-repo's default replacement marker verbatim to explain the
corruption incident. The gate's canary flags any blob containing that marker, so it fired
on **the write-up rather than on damage** — three blobs, all this file.

Two distinct problems, both fixed:

1. **At the tip**, the marker is now described rather than reproduced, so the canary stays
   a reliable signal.
2. **In history**, a replacement rule scrubs the marker from the older blobs that still
   contained it.

The general lesson matches the one in section 25 about the brief's A1: **documentation that
quotes the exact string being removed reintroduces it.** Describe the shape of the problem,
not its literal text. This has now bitten three times — A1, the failure write-up, and the
canary — and is the single most repeated mistake in this migration.

### What was executed

| Step | Result |
|---|---|
| Fresh pre-rewrite bundle | `focusedgaze.pre-rewrite3-20260804-003242.bundle`, 12 commits |
| Tip edits committed first | NOTICE, README, `.mailmap` deleted, release.yml, audit, brief |
| Pre-flight | 7 rules, 7 explicit, 0 implicit, all firing |
| Dry-run 1 | FAILED — Failure A and B |
| Dry-run 2 | FAILED — Failure B in history |
| Dry-run 3 | PASSED |
| Real repository | Gate PASSED |
| `reflog expire` + `gc --prune=now` | `in-pack: 124`, `packs: 1`, `garbage: 0` |
| Old commits reachable | none — seven spot-checked hashes all gone |
| Identity across all 15 commits | exactly one, author and committer |
| Tests | 19 passed |
| Tip prose | read, not merely grepped; no dangling headings or half-sentences |

### Method note

Author and committer were rewritten with **unconditional** `--name-callback` and
`--email-callback` returning constants, not a mapping from old values. A mapping only
rewrites the identities it was told about; a constant cannot miss one. That is why the
verification is `git log --format="%an <%ae>|%cn <%ce>" | sort -u` returning exactly one
line — an assertion a mapping-based approach could pass while still leaving a stray
committer untouched.

---

## 30. The canary disarmed itself — caught before the next rewrite

A rule was added to scrub filter-repo's replacement marker from historical blobs (section
29, Failure B). `gate.py` detects a corrupting rewrite by searching every blob for that
same marker.

**With both in place, a genuinely corrupting rewrite would scrub its own evidence before
the canary ran.** The gate would have reported clean on a repository that had just been
destroyed — the precise failure mode the canary exists to prevent, reintroduced by the fix
for a previous failure.

Nothing was harmed: the conflict existed only between the attribution pass and its
detection, and no rewrite ran while both were present.

### Fix

- **`fg-replacements.txt` emptied.** Every rule in it had been applied and would match
  nothing anyway, which `preflight.py` treats as a failure. The file's resting state is now
  empty; rules are added for one specific pass and removed once it lands.
- **`preflight.py`** reports an empty rules file as the expected between-rewrites state
  rather than crashing on it.
- **`fg-replacements.README.md`** carries a standing prohibition: never add a rule targeting
  the marker the canary searches for. If historical blobs ever legitimately contain it
  again, scrub it in a **separate pass with the canary temporarily disabled**, and record in
  this audit that the canary was off and for which pass.
- **Standing rule 10** added to the brief, generalising it: documentation must not quote
  what it describes, and no rule may target a string a check depends on.

### Why this is the same mistake as the other three

Rule 10 exists because this project has now produced four instances of one pattern: the
artefact that describes a problem becomes indistinguishable from the problem. A quoted
phrase reintroduces the phrase; a quoted marker trips the marker detector; a rule that
removes the marker blinds the detector. In each case the mechanism was correct and the
*content* defeated it.

---

## 31. Environment separation — which venv is which

Two virtual environments exist, and **they must never be mixed**.

| Environment | Location | What it is | Install policy |
|---|---|---|---|
| **Legacy** | `<legacy-dir>/gaze_env` | The virtualenv of the ORIGINAL pipeline. It is the **reference implementation** the golden harness compares against. | **FROZEN. Nothing may be installed into it again.** |
| **SDK** | `<repo-root>/.venv` (gitignored) | The package's own development environment. Installed with `-e ".[cpu,calibration,dev]"` — the exact extras CI installs. | Normal dev environment; add what is needed. |

### Why this matters more than it looks

The Tier 1 golden fixtures were recorded by running the legacy pipeline **inside
`gaze_env`**, and every equivalence proof in this migration is a comparison against numbers
produced by that specific dependency set. A transitive `numpy`, `protobuf` or `mediapipe`
bump — the kind an unrelated `pip install` can pull in — would change the reference
implementation's output. The fixtures would then be measuring the refactor against a moved
baseline, and the failure would surface much later as a mysterious drift with no obvious
cause.

**The reference implementation is data, not just code. Its environment is part of the
measurement.**

### What had already been installed into the legacy venv

Tooling was installed there before the risk was recognised: `pytest`, `pytest-cov`,
`coverage`, `ruff`, `build`, `twine`, `pyyaml`, `git-filter-repo`, an editable
`focusedgaze` itself, plus their transitive dependencies — **35 packages** beyond
`requirements.txt`.

**Audited for damage. None found:**

| Package | Pinned | Installed | Status |
|---|---|---|---|
| `numpy` | 2.5.1 | 2.5.1 | unchanged |
| `protobuf` | 7.35.1 | 7.35.1 | unchanged |
| `opencv-python` | 5.0.0.93 | 5.0.0.93 | unchanged |
| `opencv-contrib-python` | 5.0.0.93 | 5.0.0.93 | unchanged |
| `mediapipe` | 0.10.35 | 0.10.35 | unchanged |
| `scikit-learn` | 1.9.0 | 1.9.0 | unchanged |
| `scipy` | 1.18.0 | 1.18.0 | unchanged |
| `onnxruntime-directml` | 1.24.4 | 1.24.4 | unchanged |

Every addition was purely additive; nothing was upgraded or downgraded. **The Tier 1
fixtures remain valid and do not need re-recording.** This was luck as much as judgement —
the SDK's dependency ranges (`numpy>=1.26,<3`, `opencv-python>=4.8,<6`,
`mediapipe>=0.10.9,<2`) happened to be satisfied by the pinned versions, so pip had no
reason to move them. A slightly different range would have silently invalidated the
baseline.

### Standing rule

**Nothing is installed into the legacy venv again.** If a tool is needed to run the legacy
pipeline, it goes in the SDK venv and the legacy code is imported from there via
`FOCUSEDGAZE_LEGACY_DIR`, which is exactly what the harness's adapter already does.

If the legacy venv ever must change, the Tier 1 fixtures are re-recorded in the same pass
and the change is recorded here — never one without the other.

### Consequence for local test runs

The SDK venv installs `[cpu,calibration,dev]`, matching CI, so a local run reproduces CI
rather than approximating it. It deliberately does **not** include the `server` extra, which
means `websockets` is absent — and the legacy `gaze_server` module imports it. The harness's
adapter treats that as `ImplUnavailable` and **skips** the legacy-comparison tests rather
than failing them, which is the designed behaviour for an environment that cannot reach the
reference implementation.

To exercise the full legacy comparison locally, add the `server` extra to the SDK venv. The
figure to quote for a full local run is the one where nothing skips.

---

## 12. What happens next

Phase 1 is complete and I have stopped at its gate. On your go-ahead I start **Phase 2**
(extract the pure core behind `GazeEstimator.process()`), which per A3 must include an
explicit test proving two `GazeEstimator` instances in one process do not interfere —
the current module-level `_smoothed_bbox` global makes that impossible today.

Per your closing instruction, **each phase's documentation is written as part of that
phase, immediately after its implementation** — not deferred. This section is Phase 1's.

---

---

# 32. The two environments run different libraries — measured, Phase 2 blocker

Raised before any Phase 2 code was written. The SDK venv and the legacy venv do not run
the same libraries, so a Tier 2 replay across them would measure the library upgrade and
the refactor together, and report the sum as a refactor bug.

## 32.1 The actual divergence — two packages, not the whole set

Measured by importing `importlib.metadata` in each interpreter, not read from a manifest.

| Package | Legacy `gaze_env` | SDK `.venv` | Same? |
|---|---|---|---|
| python | 3.14.6 | 3.14.6 | yes |
| **mediapipe** | **0.10.35** | **1.0.0** | **NO** |
| **onnxruntime provider** | **directml 1.24.4 (GPU)** | **onnxruntime 1.28.0 (CPU)** | **NO** |
| numpy | 2.5.1 | 2.5.1 | yes |
| protobuf | 7.35.1 | 7.35.1 | yes |
| opencv-python | 5.0.0.93 | 5.0.0.93 | yes |
| opencv-contrib-python | 5.0.0.93 | 5.0.0.93 | yes |
| scikit-learn | 1.9.0 | 1.9.0 | yes |
| scipy | 1.18.0 | 1.18.0 | yes |
| absl-py | 2.5.0 | 2.5.0 | yes |
| flatbuffers | 25.12.19 | 25.12.19 | yes |

The divergence is **exactly two packages**. Everything section 31 identified as
baseline-critical — numpy, protobuf, opencv — is identical. That narrows the remediation
considerably.

Cause: `pyproject.toml` declares ranges (D5, correctly), the SDK venv was installed with no
constraints file, and pip took the newest allowed in each range.

## 32.2 ONNX — measured, and it is not a problem

The concern was that a different provider *and* a different runtime version would exceed
the 1e-4 rad tolerance, which was chosen for GPU non-determinism within one provider.

**This one is answerable without a face**, so it was measured rather than argued. Eight
deterministic synthetic tensors (fixed seed, ImageNet-normalised, 448x448) were decoded
through the real `l2cs_gaze360.onnx` in both environments, using the pipeline's exact
decode. Inputs were verified bit-identical across environments before outputs were
compared.

| | max abs delta | in degrees |
|---|---|---|
| pitch | 5.326322e-07 rad | 3.05e-05 deg |
| yaw | 2.663161e-07 rad | 1.53e-05 deg |

**188x of headroom under the 1e-4 rad tolerance**, DirectML/1.24.4 versus CPU/1.28.0.

All observed deltas are integer multiples of roughly 1.33e-07 rad, i.e. float32 rounding in
the bin-weighted sum, not algorithmic divergence. This is expected and worth stating: the
decode is a **probability-weighted expectation over 90 bins, which is smooth** — a small
perturbation in the logits moves the output slightly. A hard argmax would have been
discontinuous and could have jumped a full 4-degree bin on a near-tie.

**Conclusion: the ONNX provider difference does not need to be eliminated for Tier 2 to be
meaningful.** Aligning it is still worthwhile for cleanliness, but it is not the blocker.

## 32.3 MediaPipe — API identical, engine rebuilt, numerics UNMEASURED

**API surface: identical.** Every element the legacy pipeline touches was compared by
reflection in both interpreters — the image wrapper and its SRGB format constant, all nine
`FaceLandmarkerOptions` fields with their defaults, the running-mode enum members, the
`FaceLandmarker` method set, the `detect_for_video` signature, `FaceLandmarkerResult`
fields, `BaseOptions` fields, and the normalised-landmark fields. **Zero differences.**
Declared dependencies and the wheel tag are also identical.

**Upstream release note (authoritative, tag v1.0.0, published 2026-07-28):** it lists **no
FaceLandmarker change and no vision-task Python change**. Its Python section covers Text
APIs only. Decisively, one of its own bullets records bumping the MediaPipe version to
0.10.36 — **1.0.0 is that build renumbered.** PyPI confirms no release exists between
0.10.35 and 1.0.0. The major version bump is a versioning milestone, not an API break.

**But the native engine was rebuilt**, and that is where the residual risk sits:

| | 0.10.35 | 1.0.0 |
|---|---|---|
| bundled task-runtime shared library | 28,736,000 bytes | 43,301,376 bytes |
| sha256 prefix | `aa8e6c1b` | `a8970c64` |

The release note also lists an Abseil bump (2023-10-18 to 2025-01-14), compiling against
OpenCV 5 features, an empty-image check added to the image-scaling calculator, and updated
manylinux build scripts. A toolchain and dependency change of that size can shift float
results in the last bits even with identical model weights and identical graph semantics.

**The landmarker `.task` asset is external and unchanged** (sha256 prefix `64184e22`,
loaded from the legacy directory by both), so the landmark model weights are certainly
identical. Any difference would come from the inference runtime, not from the model.

**What was NOT established: whether landmark outputs are numerically identical.** The A/B
needs an image containing a face; the only candidate image in the tree detects no face
under either version, and no Tier 2 fixture exists yet. **This is recorded as unmeasured
rather than assumed.** Bit-identical output is plausible but unproven, and the crop
geometry is a min/max over the full landmark set, which is more sensitive to a single moved
outlier landmark than an average would be.

## 32.4 What CI actually proved about mediapipe — nothing

Resolved with pip's own resolver against the real index, for the Linux platform CI uses:

```
Linux / Python 3.12 -> Would install mediapipe-1.0.0
Linux / Python 3.13 -> Would install mediapipe-1.0.0
Linux / Python 3.14 -> Would install mediapipe-1.0.0
```

So the green CI run installed **1.0.0**, not 0.10.35. But a grep over `src/` and `tests/`
shows `mediapipe` is imported in exactly two places — `tests/golden/adapters.py` and
`tests/golden/record_tier2.py` — and in `adapters.py` it sits inside `make_landmarker()`, a
lazy closure only Tier 2 calls. On Linux the legacy directory does not exist, so the legacy
adapter raises `ImplUnavailable` and those tests skip.

**The 19 passing tests never execute a line of mediapipe.** CI proved a wheel installs on
three Pythons. It did not, and could not, prove anything about landmark equivalence.
Section 15's wheel-availability conclusion stands unchanged; no equivalence conclusion may
be drawn from it.

## 32.5 The published mediapipe range cannot be supported — and its floor is unreachable

The declared range is not merely untested at the edges. Its floor **cannot be installed on
any Python this package supports**:

```
Linux/3.12  mediapipe==0.10.9 -> No matching distribution
            available: 0.10.20, 0.10.21, 0.10.30, 0.10.31, 0.10.32, 0.10.33, 0.10.35, 1.0.0
Linux/3.14  mediapipe==0.10.9 -> No matching distribution
            available: 0.10.30, 0.10.31, 0.10.32, 0.10.33, 0.10.35, 1.0.0
```

With `requires-python = ">=3.12"`, the reachable floor is 0.10.20 on 3.12 and 0.10.30 on
3.14. The declared floor describes versions no user can resolve. It is inert rather than
harmful, but it is a published claim with nothing behind it. The recommendation is in the
turn report; **`pyproject.toml` is unchanged pending confirmation**, because it affects
every downstream install.

## 32.6 Collateral findings

**(a) The legacy pipeline is not at the documented path.** `CONTEXT_HANDOFF.md` sections
2, 6 and 7 point at a location that does not exist on this machine. The pipeline, its
frozen venv, the ONNX model and the landmarker asset are all present and intact at a
different drive and directory. Every documented command that references the old path
fails, including the Tier 2 recorder invocation. Not corrected in this turn — the path
appears in several sections and a partial fix would leave the document self-contradicting.
Proposed as a single consistent pass.

**(b) Section 7 of this audit misdescribes the decode.** It records the angle as an argmax
scaled by 4 and offset by 180. The code computes a **softmax-weighted expectation** over
the 90 bins (`gaze_pipeline.py` L61-64), which is a different function with different
numerical behaviour. Phase 2's `model.py` is to be written from that table, and an
implementation following it literally would produce a plausible, smooth, wrong result — the
exact failure class standing rule 2 exists to catch. Flagged; the table is not edited in
this turn because section 7 is described as the frozen record of preserved defaults.

**(c) `pyproject.toml` refers to a constraints file that does not exist.** Its comment
states that exact pins live in a dev requirements file for CI. There is no such file, and
`ci.yml` installs the editable package with extras and no constraints. This is the
mechanism that let the two environments drift apart, so the comment describes the control
that would have prevented the problem while the control itself is absent.

**(d) Standing rule 10 is violated at the tip of this file.** Section 29 reproduces
verbatim both the wrapped source text that defeated a replacement rule and the three
replacement rules themselves, including the superseded personal name on both sides of the
arrow. That is the same self-defeating-documentation pattern rule 10 was written for,
occurring in the section that documents it, and it is a fifth instance rather than a
fourth. Flagged only — rewriting it is a tip edit on published history and is the reader's
call.

## 32.7 The one authorised change made this turn

`CONTEXT_HANDOFF.md` section 1 identified the game repository by a GitHub handle belonging
to the superseded identity, two rows above the line asserting that no second identity
appears anywhere. Replaced with the phrase "the originating game repository" and no URL,
matching the wording already used in section 13.

Verified: the working tree of `CONTEXT_HANDOFF.md` now matches zero of the four superseded
identity patterns. **Tip only. History is published and was not touched.** One further
occurrence remains in `STANDING_BRIEF.md` Part D (Phase 11), which names the downstream
repository by its bare repo name; reported, not changed.

---

---

# 33. The Tier 1 calibration fixture depends on a mutable file — found by removing a skip

## 33.1 How it surfaced

Section 31 recommended adding the `server` extra to the SDK venv so the legacy-comparison
tests stop skipping, on the principle that **a run where nothing skips is the only figure
worth quoting**. That was done. The result was not a clean 19.

```
before (no websockets):  15 passed, 4 skipped, 2 deselected
after  (websockets):      1 FAILED, 18 passed, 2 deselected

E  AssertionError: calibration drifted by 1.000e+00 (tolerance 1e-09)
```

The four skips were not four passing tests waiting for a dependency. **Three were, and one
was a failure that the skip had been concealing.** The skip reason named a missing
`websockets` import, which reads as an environment gap rather than as a masked assertion,
so nothing about the earlier output invited suspicion.

This is the same lesson as section 15b, in a new place: there, a hard failure early in CI
hid three later checks; here, a missing optional dependency hid a failing fixture. **A
skipped test and a passing test are not interchangeable, and a suite reported as "N passed,
M skipped" is not evidence until M is zero or every skip is individually justified.**

## 33.2 Root cause — proven, not inferred

The drift is exactly 1.0, which is the width of the `[0,1]` clamp in `apply_calibration`:
a different polynomial pushes at least one grid corner onto the opposite rail.

The legacy `calibration_model.pkl` had been re-fitted. Rather than assume that, every
calibration model in the legacy models directory was replayed through the fixture grid:

| model file | worst drift |
|---|---|
| `calibration_model.backup-20260803.pkl` | **0.000000e+00 — exact match** |
| `calibration_model.pkl` (live) | 1.0 |
| `calibration_model.run5-16pct.pkl` | 1.0 |
| `calibration_model.run7-10.7pct.pkl` | 1.0 |
| four further same-day backups | 1.0 |

**The fixture is not wrong and not stale in its numbers.** It is an exact record of one
specific calibration model. That model still exists on disk under a backup name, and the
file the harness actually loads has since been replaced by a newer fit.

## 33.3 The real defect — a fixture input discovered from a mutable path

`adapters.py` loads the calibration model from the default path inside the legacy models
directory. That path is **mutable external state owned by another activity entirely** —
re-running calibration overwrites it as a matter of normal use. So the fixture pins its
expected outputs while leaving one of its inputs free to change underneath it.

This is the third instance of one root cause in this project:

1. **F1 (section 1.4)** — the positioning gate resolved its focal file by relative path, so
   the answer depended on the process working directory.
2. **Section 19e** — a branch selected by a file's presence was unpinned until recorded
   both ways.
3. **This** — a fixture's input model is discovered from a path rather than pinned by
   identity.

Each time, a value the test depends on was located rather than declared. The generalisation
worth carrying into Phases 5 and 6: **anything a fixture depends on must be identified by
content, not by location.** Phase 5 replaces this pickle with JSON plus raw coefficients,
which is the natural moment to fix it properly.

## 33.4 Recommended fix — not applied, needs a decision

Preferred: **pin the model by identity, not by path.** Record the model's SHA-256 in
`calibration_apply.json` at record time, have the adapter load the exact recorded file, and
fail with a message naming the mismatch ("fixture recorded against model <digest>, found
<digest>") instead of surfacing as an opaque numeric drift. The matching model still exists,
so this is recoverable without re-recording anything.

Explicitly rejected: **re-recording the fixture against the current model.** That would turn
green immediately and would be the wrong move — it discards the pinned baseline in favour of
whichever calibration happened to be on disk today, and it treats a harness defect as if it
were a data refresh. The fixture's value is that it does not move.

Also rejected: restoring the live `calibration_model.pkl` from the backup. It is a real
person's current calibration and is not this migration's to overwrite.

## 33.5 Status of the environment change

`websockets==16.1.1` was installed into the SDK venv only, via the `server` extra's declared
range. A `pip freeze` diff taken before and after shows **exactly one added line and nothing
upgraded or downgraded** — mediapipe, numpy, protobuf, opencv, onnxruntime, scikit-learn and
scipy are all unchanged, so the section 31 baseline discipline holds and the section 32
divergence table is still accurate.

The legacy venv was not touched.

---

---

# 34. Publisher confirmation, and the corrections batch before the first tag

## 34.1 Trusted Publishing — CONFIRMED, open item closed

Confirmed by the project owner on 2026-08-04:

| Check | Result |
|---|---|
| PyPI pending publisher | registered |
| TestPyPI pending publisher | registered (separate registration) |
| GitHub environment `pypi` | exists |
| GitHub environment `testpypi` | exists |
| Values verified against `release.yml` | yes |

**The pending-publisher reading of the two 404s was correct.** Section 27 recorded that
neither project existed on PyPI or TestPyPI and noted this was *consistent with* a pending
publisher, which is not publicly visible — but that the alternative, nothing registered at
all, produces an identical observation and surfaces only as a 403 at upload. That ambiguity
is now resolved in favour of the first reading. Neither project exists, so **the first
upload creates it**, which is exactly what a pending publisher is for.

**`release.yml` needs no change.** Its inline setup instructions already document the
pending-publisher path (use the account-level publishing page when the project does not yet
exist, the project settings page when it does), and they name the four bound values that
must match. Nothing in the confirmed configuration contradicts them.

`CONTEXT_HANDOFF.md` section 9 item 2 is closed and moved to that section's Closed list.
Section 27's open item is superseded by this section rather than edited, so the reasoning
that resolved it stays legible.

## 34.2 The decode correction in section 7

Section 7's `BINS` row described the angle as an argmax scaled and offset. The code
computes a softmax-weighted expectation over the 90 bins. Corrected in place, with the
correction stated in section 7 itself rather than applied silently, because that table is
the frozen record Phase 2's `model.py` is to be written from and the wrong description was
live long enough to have been built on.

The distinction is not cosmetic. An argmax agrees with an expectation only when the bin
distribution is sharply unimodal, and diverges discontinuously near a bin boundary — a
4-degree jump on a near-tie. It is also the reason section 32.2's cross-provider
measurement came out as well as it did: an expectation is smooth, so float32 differences in
the logits move the output by fractions of a bin instead of jumping one whole bin.

## 34.3 Standing rule 10, applied to section 29

Section 29 reproduced verbatim both the wrapped source text that defeated a replacement
rule and the three replacement rules themselves, superseded personal name included on both
sides of the arrow. Rewritten to describe the shape of each — a two-word name split by a
newline and indentation, and a longest-first rule ordering — without printing either.

This was the fifth instance in this project of documentation reintroducing what it
describes, and the second to survive at the tip after the rule against it was written. A
note now sits inside section 29 stating the constraint, so the next person to write up a
replacement pass meets the rule at the point of temptation rather than in a list of
standing rules elsewhere.

Verified after the edit: the whole working tree matches zero of the superseded identity
patterns.

## 34.4 The path pass

`CONTEXT_HANDOFF.md` sections 1, 2, 6 and 7 pointed at a legacy-pipeline location that does
not exist. The pipeline, its frozen venv, the ONNX model and the landmarker asset are all
intact on a different drive. Every documented command referencing the old path failed as
written, including the Tier 2 recorder invocation that is currently the Phase 2 blocker —
so the one instruction most needed by the next session was the one most certainly broken.

All four sections corrected in a single pass rather than piecemeal, because a partial fix
leaves the document contradicting itself, which is harder to diagnose than a uniformly
stale one. Section 2 now says plainly that the location moved and that
`FOCUSEDGAZE_LEGACY_DIR` is what makes the harness independent of it.

Two stale claims in the same sections were corrected at the same time: the suite is not
"19 passed" (section 33), and the SDK venv is no longer without `websockets`.

Section 7 now also instructs running the suite with `-rs`. A bare run reports a skip count
without a reason, and in this project a skip has already concealed a failing assertion.

## 34.5 The mediapipe range

`pyproject.toml` now declares `mediapipe>=0.10.30,<1.1`, replacing a range whose floor no
user could resolve (section 32.5).

- **Floor 0.10.30** — the lowest release with an installable distribution under
  `requires-python = ">=3.12"`. The previous floor described versions that do not exist for
  any supported interpreter.
- **Ceiling `<1.1`** — 1.0.0 is upstream's 0.10.36 renumbered, by its own release note, and
  is the only 1.x this project has run. `<2` claimed a whole major series that has never
  been executed here.

The rationale is written into `pyproject.toml` as a comment, not left only in this audit,
because the next person to widen the bound will be reading the dependency list rather than
section 32. **Widening means replaying the golden fixtures against each candidate version
and recording the result — not relaxing the bound and hoping.**

## 34.6 `requirements-dev.txt` now exists

Section 32.6c recorded that `pyproject.toml` referred to a dev pin file that had never been
created, and that its absence is the mechanism by which the two environments drifted onto
different mediapipe majors unnoticed. The file now exists, pinning the SDK venv's actual
resolved versions.

It is deliberately **not** installed by CI. CI resolves the declared ranges freshly, so an
incompatible new release surfaces as a red build instead of being masked by a pin — which
is the entire reason section 32's divergence was findable at all. The file is for
reproducing a specific result: recording a fixture, chasing a numeric difference, bisecting.

It documents the legacy venv's two differing versions and how to match them exactly, so the
choice between "approximate the reference" and "reproduce the reference" is explicit rather
than accidental.

**One pin in the first draft was wrong.** `platformdirs` was written from memory as 4.5.0;
the venv has 4.11.0. Caught by diffing every declared pin against `pip freeze` before
committing, which is the only reason it is a footnote rather than a defect. Rule 8 exists
for exactly this: a version number not read from the environment is invented, however
plausible it looks.

---

---

# 35. First release run — build green, TestPyPI rejected the publisher

`v0.0.0` tagged at `e1de9bf` and pushed on 2026-08-04. This was the first execution of
`release.yml` in the project's history: nothing in it had ever run.

## 35.1 Result

| Job | Conclusion |
|---|---|
| **Build sdist and wheel** | **success** |
| **Publish to TestPyPI** | **failure** |
| **Publish to PyPI** | **skipped** |

Every build step passed, including the two that had never been exercised: `twine check` on
both artifacts, and the forbidden-content audit that fails a release carrying weights,
calibration pickles or test fixtures. **The Phase 9 distribution audit is therefore proven
as a gate rather than as an intention** — it ran, on real artifacts, and reported clean.

The `pypi` job was **skipped, not failed**, which is the designed behaviour: it declares
`needs: testpypi`, so a TestPyPI failure prevents PyPI from being reached at all. The
Phase 10 requirement that TestPyPI be a required predecessor is confirmed working on its
first real test. **Nothing was uploaded to either index.**

## 35.2 The exact failure

From the check-run annotation on the failing job (the job-log endpoint returns 403 without
repository admin rights, so annotations were used instead — see 35.4):

```
Trusted publishing exchange failure:
Token request failed: the server refused the request for the following reasons:

* `invalid-publisher`: valid token, but no corresponding publisher
  (Publisher with matching claims was not found)
```

The token itself was valid. TestPyPI simply had no publisher whose configuration matched
the claims GitHub presented. Those claims were:

| Claim | Value presented |
|---|---|
| `repository` | `muhammad-asifkhan/focusedgaze` |
| `repository_owner` | `muhammad-asifkhan` |
| `workflow_ref` | `muhammad-asifkhan/focusedgaze/.github/workflows/release.yml@refs/tags/v0.0.0` |
| `environment` | `testpypi` |
| `ref` | `refs/tags/v0.0.0` |

**All four bound values match what `release.yml` documents.** Owner, repository, workflow
filename and environment are exactly as specified in its inline setup block. The repository
side is therefore correct, and the mismatch is on the TestPyPI registration.

This is the failure mode section 27 predicted and section 34.1 believed closed. The
confirmation recorded in 34.1 was given in good faith and covered PyPI and TestPyPI as a
pair; the evidence now says the two registrations are not in the same state. **Section 34.1
is not retracted — it is corrected here: the PyPI registration remains unverified by
execution, and the TestPyPI one is demonstrably not matching.**

## 35.3 What this does and does not tell us about PyPI

**It says nothing about the PyPI registration.** The `pypi` job never ran, so its publisher
has still never been exercised. A green TestPyPI job would not have proven PyPI either —
they are independent registrations on independent services, which is precisely why
`release.yml` documents them as separate steps. The next run tests exactly one of them at a
time, in order.

## 35.4 Version 0.0.0 is NOT consumed

Checked directly after the run:

```
test.pypi.org/pypi/focusedgaze/0.0.0/json -> 404
pypi.org/pypi/focusedgaze/0.0.0/json      -> 404
test.pypi.org/pypi/focusedgaze/json       -> 404
pypi.org/pypi/focusedgaze/json            -> 404
```

Neither project exists and neither version number has been used. The expectation going into
this run was that tagging would permanently consume `0.0.0`, since a version can be yanked
but never reused. **That did not happen, because the upload never occurred** — the publisher
exchange failed before any artifact was transferred. `0.0.0` remains available on both
indexes and can be used by the retry once the registration is fixed.

Worth recording as a general point: **an irreversible step is only irreversible once it
actually executes.** The cost of this failed run was zero, not one burned version number.

## 35.5 Two observations, neither acted on

**(a) The job-log endpoint requires admin.** `GET /actions/jobs/{id}/logs` returns
403 `Must have admin rights to Repository` without authentication, as section 15c recorded
for the CI logs. The failure detail was recovered from the **check-run annotations**
endpoint instead, which is publicly readable and carried the full error including the
claim set. Recorded because it is the working route for any future unauthenticated
diagnosis of a failed run in this repository.

**(b) A deprecation warning, non-blocking.** The run emitted
`Node.js 20 is deprecated ... actions/download-artifact@v4 ... forced to run on Node.js 24`.
It did not cause the failure and is not being changed as part of diagnosing one — but it
will need addressing before the deprecation becomes an error.

## 35.6 Deliberately not done

Per instruction, and per standing rule 1 (a failed gate is a result, not an error to work
around):

- **The publish was not retried.** A retry against an unchanged registration produces an
  identical failure.
- **`release.yml` was not modified.** The workflow is not wrong. Every value it presents is
  the value it documents, and editing it to accommodate a mismatched registration would
  encode the mismatch into the repository rather than fix it.
- **No fallback to token-based upload was introduced.** That would convert a configuration
  problem into a permanent credential in the repository.

The fix is on the TestPyPI side. Until it lands, the release pipeline's status is: build
verified end to end, publishing unverified on both indexes.

---

---

# 36. First successful release, and the documentation rules

## 36.1 Correction to section 34.1: the confirmation was read, not exercised

Section 34.1 recorded Trusted Publishing as confirmed on both indexes. Section 35 then
recorded TestPyPI rejecting the publisher with `invalid-publisher`. Both are now explained.

The TestPyPI pending publisher had its **environment field set to `pypi`** rather than
`testpypi`. Owner, repository and workflow filename all matched, and the environment did
not. That is exactly the shape of the observed error: the token was valid, three of four
bound values agreed, and no publisher matched the complete claim set. The pypi.org
registration was correct throughout and was never touched.

**Section 34.1 was not wrong about what was checked. It was wrong about what checking
proves.** The values were confirmed by reading them back, and reading a configuration back
tells you what it says, not whether it works. Only using it tells you that.

This is the mutation-check principle from standing rule 2, applied to configuration instead
of code. Rule 2 exists because a fixture that passes has not been shown to catch anything,
so the fixture is deliberately broken to prove it is sensitive. A configuration read back
is the same category of non-evidence: it agrees with itself. The equivalent of a mutation
check is an actual publish, and it cost a release run to learn.

Recorded as a general rule for the remaining phases: **verifying a configuration by reading
it back is not the same as verifying it by using it, and only the second kind counts.**

## 36.2 The release that landed

Re-run via `workflow_dispatch` on `main` rather than a new tag, since section 35 established
that `0.0.0` had never been consumed.

| Job | Conclusion |
|---|---|
| Build sdist and wheel | success |
| Publish to TestPyPI | success |
| Publish to PyPI | success |

**Both publishers were exercised for the first time, in sequence, and both passed.** The
PyPI registration had never been reached before this run, so its correctness was unverified
until now for the same reason TestPyPI's was.

Published and confirmed by fetching each index's JSON API:

```
https://pypi.org/project/focusedgaze/          0.0.0   wheel + sdist
https://test.pypi.org/project/focusedgaze/     0.0.0   wheel + sdist
```

Both records carry `Requires-Python: >=3.12` and the corrected `mediapipe<1.1,>=0.10.30`
bound from section 34.5, so the range narrowing is now the published metadata rather than a
local edit.

**`0.0.0` is now consumed on both indexes, permanently.** A version number can be yanked but
never reused. This was intended: it is the placeholder that reserves the name, which section
19 recommended and which had been open since Phase 1. That open item is closed.

**A predicted difference that turned out not to matter.** A dispatch run presents
`ref: refs/heads/main` rather than the tag ref a tag-triggered run presents. Neither
publisher restricts by ref, so the claim set still matched. Worth recording because the
prediction was made before the run and held: had it mattered, the error would have changed
shape rather than disappearing.

## 36.3 The no-em-dash rule, and a deliberate backlog

A house style rule now applies to every document, commit message, docstring and code
comment: no em-dashes. The rationale is that it is a strong tell of generated text, and in
one case it was worse than that (see 36.4).

Sweeps completed, each rewriting sentences by what the dash was doing rather than
substituting a character:

| Scope | Count | State |
|---|---|---|
| `README.md` | written clean | done |
| `NOTICE`, `CONTEXT_HANDOFF.md` | 51 | done |
| `src/` and `tests/` | 35 | done |
| `CHANGELOG.md` | 9 | done |
| `requirements-dev.txt` | 3 | done |
| `CONTEXT_HANDOFF.md` inside code fences | 6 | **deliberately left** |
| `.github/workflows/`, `.gitignore` | 4 | **deliberately left** |
| `MIGRATION_AUDIT.md`, `STANDING_BRIEF.md` | 269 | **deliberate backlog** |

**The three exclusions are decisions, not omissions.**

Code fences, workflows and `.gitignore` are left because the rule exempts anything
executable, and this project has been broken twice by dash characters in that context. A
typographic preference is not worth re-running that risk.

`MIGRATION_AUDIT.md` and `STANDING_BRIEF.md` are internal documents, long, and under
constant edit. Rewriting 269 sentences across them would produce a large diff over text
nobody outside this project reads, and would collide with every future edit. **The rule
applies to new writing in both files from now on. The existing backlog is left
unconverted on purpose, and this paragraph is the record of that decision** so a future
reader does not mistake it for an incomplete sweep.

## 36.4 An em-dash was a real defect, not only a tell

`cli.py` had an em-dash in its argparse description and in its status line. On a default
Windows terminal that renders as a replacement glyph, so the first output a new user saw
from the tool was mojibake in its own help text.

The same class of problem was found in `tests/golden/record_tier2.py`, whose on-screen
prompt to the person being recorded contained one, and in four `pytest.skip` messages and
one assertion message. All were strings printed to a console at the moment someone was
trying to read an instruction.

Fixed as a bug rather than as style. The package was also swept for other non-ASCII in
printed strings: there is none. Every non-ASCII character in `src/` was an em-dash, with no
smart quotes, ellipsis characters or non-breaking spaces to worry about.

## 36.5 Standing rule 8 applied to an instruction

The documentation brief offered "2.0 to 2.4 cm within a session, around 3.0 cm on a
held-out session" as its worked example of preferring specifics to adjectives. Neither
number has a source in this repository. The first traces to `milestone6`, whose output has
never been captured here and which section 9 has listed as an open item since Phase 0. The
second appears nowhere at all.

The README now carries the figures that are sourced, from the originating project's own
documentation: held-out error around **8.9% of screen size**, roughly **3–8% across the top
and centre**, and **13–14% along the bottom edge**. The unsupported centimetre figures were
removed and the gap is named in the README itself rather than quietly dropped.

`DOCUMENTATION_PLAN.md` was added to the repository with that example corrected in place,
carrying a note explaining why. A brief that demands sourced numbers cannot itself contain
an unsourced one: the next writer copies the example straight into a public page, which is
how the original claim reached the README in the first place.

**Rule 8 is usually applied to model URLs, checksums and platform claims. It applies to
instructions with the same force.** An instruction is not evidence, whoever wrote it.

---

---

# 37. Section 33.4 implemented: the fixture now pins its own input

The calibration fixture read its model from a path that ordinary recalibration overwrites,
so it began failing the first time somebody recalibrated and reported it as a numeric drift
of exactly 1.0 with no hint of the cause. Fixed here. **The suite is back to 19 passed.**

## 37.1 The approach, and why it beats the two obvious ones

The fixture proves that the extracted `apply_calibration` matches the legacy one. That
equivalence holds for **any** model, provided both sides use the same one. It never needed a
real person's calibration.

So the model is now **synthetic**: fitted from generated pairs by
`tests/golden/make_synthetic_calibration.py` and committed as
`tests/fixtures/tier1/synthetic_calibration.pkl` (1,190 bytes).

Two alternatives were considered and are worse:

- **Load the real backup by name from the legacy tree.** Keeps the dependency on an external
  tree that has now moved three times in one session, and keeps a real person's calibration
  in the loop for no benefit.
- **Commit the real backup.** Fixes mutability but puts personal data in a public repository,
  against section 9.3.

The synthetic model has neither problem. It is fitted with the **unmodified legacy fitter**,
via `robust_fit_samples`, so the pickle is structurally identical to a real profile: same
dict keys, same scikit-learn objects, same degree. What changed is whose data it is, not
what it is.

## 37.2 This is not the re-recording that was rejected

Section 33.4 rejected re-recording, and this re-records. The distinction matters.

The rejected version would have re-recorded against **whatever model happened to be on
disk**, trading a pinned baseline for an arbitrary one and turning the suite green by
lowering the bar. This replaces a **mutable input with an immutable committed one**. The
baseline is not weakened, it is finally nailed down: the fixture and its input now travel
together in the same commit and cannot drift apart.

## 37.3 Both properties preserved, proven rather than assumed

| Property | Before | After |
|---|---|---|
| Polynomial degree | 3 | **3** |
| Grid values on the `[0,1]` clamp | 154 | **120** |

Degree is read from the fitted model, not asserted. The clamp count went **down**, which is
not a regression: fewer saturated cases means more of the grid exercises the polynomial
itself. Both rails are still hit, 60 values at 0.0 and 60 at 1.0, so neither side of the
clamp is unpinned. The count is now recorded in the fixture as `clamped_values`, so losing
it later is visible in a diff rather than silent.

## 37.4 Mutation-checked

A fixture that passes has not been shown to catch anything. Against the re-recorded fixture,
tolerance 1e-9:

| Mutation | Worst drift | Caught |
|---|---|---|
| (control) unmutated | 0.000e+00 | passes, as required |
| swapped x/y coefficient sets | 1.000e+00 | yes |
| reversed coefficient order | 8.318e-01 | yes |
| degree 2 instead of degree 3 | 1.997e-02 | yes |
| single coefficient +1e-6 | 7.200e-07 | yes |

The last row matters most: sensitivity extends to a perturbation four orders of magnitude
below the ones a broken refactor would produce. The degree row is the live risk from F3,
where `fit_calibration` defaults to 2 while the robust path passes 3.

## 37.5 The guard fails loudly, and that was tested too

`FixtureModelMismatch` names both digests and the file, and refuses to continue.

It is deliberately **not** an `ImplUnavailable`. That class is caught by the session fixture
and converted into a skip, and a skip is how the original defect stayed invisible. Writing
the check without that distinction would have reproduced the bug inside its own fix.

That was a real trap, not a hypothetical: the first version of this change raised
`FixtureModelMismatch` inside `_load_legacy`, whose broad `except Exception` converts
anything into `ImplUnavailable`. It would have skipped. Caught before commit by re-reading
the exception path rather than by the tests, which would have reported a plausible skip.

Verified by tampering with the recorded digest and re-running: **1 failed, 4 errors, zero
skips.**

## 37.6 The biometric guard rail was strengthened, not relaxed

`test_fixtures_contain_nothing_biometric` asserted that only `.json` lives in the Tier 1
fixture directory, and committing a `.pkl` tripped it. The guard is doing real work, so it
was not weakened to accommodate the new file.

It now permits exactly one non-JSON file, **identified by digest rather than by name**. A
real calibration profile dropped in under the filename `synthetic_calibration.pkl` still
fails. That is the case worth catching, because it is the one that looks harmless in a diff.

`.gitignore` keeps its blanket `*.pkl` exclusion with a single negation for this one path,
so every other pickle stays excluded by default.

## 37.7 The real backup is untouched, and its digest is recorded here

`calibration_model.backup-20260803.pkl` remains where it is in the legacy tree. It was not
committed and not deleted.

```
sha256  b430c1037aaa4a80d06c8f13447e9f599489c1a2cd3541a806a99acb65fa25a4
```

**This is the artifact that verified the original fixture**, at 0.0 drift across all 169
cases. It is recorded so the original equivalence can be re-derived if it is ever
questioned. The current fixture no longer depends on it.

The synthetic model's own digest, recorded in `calibration_apply.json`:

```
sha256  66fdd379bc1d702c7d0fc232510ac736be21a60406844323c113c36555a18948
```

## 37.8 The archive-recovery route worked, and is worth knowing

Between the previous section and this one, the legacy tree appeared to have been deleted:
the directory was gone and a 774 MB `game_integration.zip` sat in its place. The backup file
was **not** in that archive, because the archive predated it.

The archive's plain `models/calibration_model.pkl` was extracted to a scratch directory and
replayed through the fixture grid using the archived `calibration_utils.py`. Result: **0.0
drift across 169 cases**, and a digest matching the loose backup exactly. The fixture model
was recovered and independently verified from a zip before the loose file turned up again at
a third path.

Two things worth carrying forward:

1. **A presumed-lost artifact is worth ten minutes of checking before it is treated as
   lost.** The recovery needed four files out of 60,849 entries and produced a proof rather
   than a plausible-looking candidate.
2. **Verify a recovered artifact by replay, not by filename or timestamp.** The zip entry's
   mtime matched the backup's, which was suggestive but not evidence. Replaying it through
   the fixture was.

## 37.9 The legacy path now has one home

The path moved three times in one session, ending at a location on `C:` with an underscore.
Every document that repeated the literal value went stale each time.

`CONTEXT_HANDOFF.md` section 2 is now the **only** place in the repository holding the
literal value. Every other reference, in documents and in code, goes through
`FOCUSEDGAZE_LEGACY_DIR`. The harness already worked this way, which is why nothing in
`src/` or `tests/` needed changing: only the prose was stale. Section 7's command now says
to take the value from section 2 rather than repeating it, because repeating it is exactly
how the previous three staleness incidents happened.

## 37.10 A demonstration that confirmed what it was built to confirm

Recorded under rule 9 because it is the same principle as the mutation check, in a place it
was not expected.

`examples/filter_demo.py` was written to show that the shipped filter settings beat
over-smoothing, with an over-smoothed configuration included as the losing comparison. The
closing text asserted that over-smoothing wins on jitter and loses on lag, which is the
textbook tradeoff and is what the example was built to illustrate.

Running it disproved that. The over-smoothed filter's jitter was **worse**, 0.00414 against
the tuned 0.00214, because it never reaches the target at all: what registered as jitter was
the filter still creeping toward a position it had left 25 frames earlier. The honest
conclusion is more interesting than the intended one. Smoothing harder stops buying
steadiness once the filter can no longer keep up.

**A demonstration that confirms what you expected has not been tested against anything.** It
is the same failure mode as a fixture that passes without having been shown to catch a bug:
agreement with a prior belief is not evidence, whether the belief is about code or about a
tradeoff. The example now states what its own numbers show.

---

# Section 38 - The environment loss of 2026-08-05

## 38.1 Both virtualenvs were dead, including the frozen reference

Found at the start of the session, before any work: **neither virtualenv could execute a
single statement.** Two independent causes had combined.

1. The entire tree had been moved to a new root. This is the **fourth** move; the path
   record in `CONTEXT_HANDOFF.md` section 2 was stale in every entry.
2. Separately, the base CPython 3.14 installation both venvs were built against had been
   removed from the machine.

Cause 2 is the one that killed them. A Windows venv keeps a *copy* of `python.exe`, which
makes it look self-contained, but it still resolves its standard library through the `home`
key in `pyvenv.cfg`. When that directory is gone the copied interpreter aborts before
running any user code, and **the error names the missing base executable, saying nothing
about virtualenvs at all.** Read quickly, it looks like a corrupt Python install rather than
a broken venv.

## 38.2 Why this was worse than a broken dev environment

The SDK venv is disposable. The **legacy** venv is not: section 6 of the handoff records
that its exact package set *is* the measurement the Tier 1 golden fixtures were recorded
against. Every equivalence claim in this migration rests on being able to run it.

The rule protecting it said **never install anything into it again**. That rule was
honoured, and it did not help. It guarded against *modification* and was silent about the
directory being *moved* and its base interpreter being *uninstalled*. Both happened.

## 38.3 It survived because site-packages is self-contained

Only the interpreter bootstrap was broken. `Lib\site-packages` was intact at 132 entries,
with the frozen set present and matching what section 6 records: mediapipe 0.10.35, numpy
2.5.1, onnxruntime-directml 1.24.4, scikit-learn 1.9.0, scipy 1.18.0, opencv 5.0.0.93,
protobuf 7.35.1, websockets 16.1.1.

So the fix was a **repair, not a rebuild**: `pyvenv.cfg` alone was rewritten to point at a
3.14 interpreter that exists, with the original preserved beside it as
`pyvenv.cfg.bak-before-repair`. **`site-packages` was not touched.** Verified by importing
the full stack and confirming the DirectML execution provider still enumerates. A rebuild
would have silently re-resolved every dependency to current versions and destroyed the
baseline while appearing to succeed.

## 38.4 The caveat this leaves open

The interpreter now under the legacy venv is **3.14.3**. The venv was built on **3.14.4**,
which is no longer on this machine. The frozen environment is running one CPython patch
level below the one that recorded the fixtures.

The compiled extensions are `cp314` and therefore ABI-compatible across 3.14.x, and a patch
release is very unlikely to move floating-point results. But per rule 1, unlikely is not
measured, and section 6 is explicit that the environment is part of the measurement.

### Measured, not assumed: Tier 1 reproduces byte-for-byte

Rather than leave this as a reasoned-about risk, it was tested. `record_tier1.py` was run
**with the repaired legacy venv** and its output compared against the committed fixtures.

The recorder writes to a path fixed relative to its own location and takes no output
override, so running it in place would have **overwritten the committed baseline** - the
exact trap section 33.4 rejected and section 37 fixed. It was therefore run against a copy
of `tests/golden/` and `tests/fixtures/` in a scratch directory, leaving the real fixtures
untouched.

Result: **all four fixtures byte-for-byte identical.** `calibration_apply.json` (169
cases), `one_euro.json` (124 cases), `positioning_gate.json` (16 cases) and
`positioning_gate_nofocal.json` (16 cases), across both focal branches.

So for every pure-numeric path the golden harness covers, the patch-level interpreter
difference changes nothing. **The caveat is closed for Tier 1.**

### What this does NOT prove

Two limits, stated because a green result is the easiest place to overclaim:

1. **Tier 1 does not exercise the ONNX model.** The recorder prints a line naming the
   DirectML provider, which proves the session builds, but the fixtures cover the
   calibration polynomial, the One Euro filter and the positioning gate. Those are numpy
   and scikit-learn. No inference output is pinned by Tier 1.
2. **Tier 2 is still unrecorded**, and it covers exactly the parts most exposed to an
   environment difference: MediaPipe landmarks and ONNX inference. The interpreter caveat
   remains formally open there until Tier 2 exists and is replayed.

The remedy, if Tier 2 ever disagrees, is unchanged: install 3.14.4 and repoint `home`.

### A related finding: the test suite never runs the legacy venv

Worth recording because it corrects a natural misreading. `tests/golden/adapters.py`
inserts the legacy directory onto `sys.path` of the **current** interpreter. It does not
invoke the legacy venv's `python.exe`. So a normal `pytest` run executes legacy *source*
against the **SDK venv's** packages, never the frozen set.

The frozen venv therefore matters for **recording** fixtures, not for replaying them. That
narrows what the freeze rule is protecting, and it means a green suite says nothing about
the health of the legacy environment. This is also why the byte-for-byte check above had to
be run explicitly with the legacy interpreter: the suite passing would not have told us.

It also compounds the section 32 warning that the two venvs run different mediapipe and
different ONNX providers. Any legacy-versus-SDK comparison run through the suite is holding
the packages constant and varying only the source, which is the right experiment for a
refactor but is not a check on the environment.

---

# Section 39 - Phase 7 reconnaissance: the exit criterion was not satisfiable

Full contract in `docs/wire_format.md`. This section records the finding and the decisions.

## 39.1 The game never reads the gaze message

The Phase 7 exit criterion was "the existing browser game connects to `focusedgaze serve`
and plays end to end, unmodified". **That criterion could not be met by a gaze-only server**,
and the reason had been sitting in the client the whole time.

There is exactly one WebSocket client in the game, `input-manager.js`. Its handler returns
early on any message that is not `type: "input"`. The `type: "gaze"` message the SDK was
being designed around is **discarded by the game**, with a source comment saying so
outright: the gaze feed is kept for `gaze_test.html`, and the game consumes the resolved
`input` message instead.

The failure mode this would have produced is the expensive kind. A gaze-only server would
have left the game connecting successfully, reporting itself connected, and never moving
the cursor, **with no console error and nothing in the server log**. It would have looked
like a calibration or tracking problem, not a wire-format problem.

**Verified independently rather than taken on report.** A repository-wide search for
WebSocket clients returns three files; two are comments, and one of those states explicitly
that the socket lives in `input-manager.js`, "the only WebSocket client". The early return
was read directly in the source.

The general lesson, which is a variant of rule 2: **a wire format is defined by what the
consumer reads, not by what the producer sends.** Every earlier description of this
interface in the briefing documents described the producer. The one fact that determined
the design was on the consumer side, and nobody had looked.

## 39.2 Decisions taken

Recorded here because Q6, the question this supersedes, had been posed in at least two
documents and **answered in none** - grepping the audit found only the places asking it.
That is precisely the gap rule 9 exists to close, so these do not repeat it.

**Q7-1: the SDK emits a minimal gaze-only `input` message, plus hooks.** `type`, `mode`,
`source`, `ok`, `x`, `y`, `t`, with `mode` and `source` the constant `"gaze"`. **No gesture
fields.** The game repo's wrapper replaces the message through a `resolve_input` hook to add
gesture. So bare `focusedgaze serve` drives the game, while the SDK's wire format stays
gaze-only and no gesture-shaped field name enters it.

The hook must **replace** the SDK's message rather than append to it, or two `input`
messages race on every tick.

**This rests on an inference that has not been executed.** With the gesture fields absent
the client's counter sync compares `undefined > undefined`, which is `false` in JavaScript,
so no click fires spuriously, and dwell-to-click remains enabled because `api.mode` stays
`"gaze"`. Reading the client line by line says the game is fully playable. **Running it does
not exist yet.** Confirm against a stub server before relying on it, and treat the
sequence-counter behaviour as the specific thing to watch. Q7-2, whether to send explicit
zeros for those counters instead of relying on `undefined`, stays open and is contingent on
that test.

**Q7-5: `gaze_test.html` stays a supported consumer**, so the `type: "gaze"` message remains
in the SDK's wire format as the raw device feed. It is the only thing that makes bare
`focusedgaze serve` testable without the game, which is worth keeping for `focusedgaze
check` and for SDK users who are not this game.

## 39.3 The frame-sharing coupling is already gone

`read_shared_frame()` is defined and never called: `set_mode` passes `own_camera=True`, so
the shared-frame tracker is never constructed. The gaze and gesture pipelines do **not**
share frames in the live path. The only surviving coupling is a two-event camera lease.

This matters because the camera lease is neither a gaze concept nor a gesture concept, it
is an operating-system fact about a device that admits one owner. It belongs in the SDK as
`pause()`/`resume()`, carrying the measured 4.0 s wait and both 0.3 s driver sleeps, which
are empirical and must not be tidied into round numbers.

## 39.4 Two latent bugs found, neither fixed

Found by reading, not by running. Both left in place under rule 4: move first, improve
second, in separate commits.

1. **A guaranteed `NameError` at shutdown.** `gaze_loop`'s `finally` releases a capture
   handle that is never bound in that function. **Confirmed independently**: the function
   begins at line 310, the release is at line 380, and the only two bindings of that name
   in the file are at lines 136 and 239, both in other scopes. It has stayed invisible
   because it only fires when the loop exits, on a daemon thread, during shutdown.
2. **Valid JSON that is not an object crashes the connection handler.** The attribute lookup
   is not guarded, and the resulting error is not caught by the inner handler, so the
   connection closes.

A third, milder issue: the input message is built from mode and gesture state read without
holding the lock that guards it.

Also worth recording: the server's docstring describes an out-of-zone case, and no zone
check exists in that file - it never imports the positioning gate. The documentation
described an intention rather than the code, which is the same class of defect as section
37.10's example that confirmed what it was built to confirm.

---

# Section 40 - Parallel implementation run, 2026-08-05, and what it left unproven

Four phases were worked in parallel by separate agents on disjoint file sets, with git
writes and the shared documents reserved centrally so concurrent work could not race on the
index or on this file. Phase 7 completed (section 39). **Phases 3, 5 and 6 were terminated
mid-run by a session limit.**

## 40.1 State at the stop

Suite: **194 passed, 2 deselected, zero skips.** Up from 19. `exceptions.py` landed first
and centrally, because every other subsystem imports it and leaving it a stub would have
serialised work that has no reason to be sequential.

| Phase | Landed | Tests | Honest status |
|---|---|---|---|
| 3 config, types | yes | `test_config.py` | Defaults not yet checked against legacy source |
| 5 calibration | yes | **none** | **Written and unproven** |
| 6 assets | yes | 2 files, network-free | Closest to done |
| 7 server | doc only | n/a | Contract established, decisions taken |

## 40.2 The gap that matters: Phase 5 has no tests

1653 lines of the **highest-risk numerical work in the migration**, with no dedicated test
file, because the agent was killed before writing one. **None of the four mutation checks
that Part D of the standing brief makes non-negotiable have been run**: transposed
coefficients, wrong `PolynomialFeatures` term ordering, swapped x and y coefficient sets,
and a degree mismatch.

So the pure-NumPy `apply()` reproducing sklearn's term ordering is **unproven**. This is
precisely the failure the brief warns about, and the reason it warns: a wrong polynomial
does not crash. It returns a smooth, believable surface in the wrong place.

**A green suite here means "nothing imports wrongly", not "the arithmetic matches."** The
work was committed rather than discarded so it is not lost, with the gap stated in its
commit message so it is visible in history rather than in someone's memory. Phase 5 is
**written-and-unproven**, not nearly-done.

## 40.3 A function shadowing a submodule, and why the tests could not have passed

`assets/__init__.py` re-exports a function named `download` from the submodule
`assets.download`. The function wins, so **both** obvious spellings bind the wrong object:

    from focusedgaze.assets import download as m   -> the function
    import focusedgaze.assets.download as m        -> also the function

The second surprises people. `import a.b as x` resolves `getattr(a, "b")` first and only
falls back to `sys.modules` when that lookup **fails**; here it succeeds and hands back a
callable. Monkeypatching then raises an `AttributeError` naming a function, which reads like
a missing symbol rather than a shadowing problem, and sends the reader looking in the wrong
file.

The tests now go through `sys.modules`, the one spelling that cannot be shadowed, with the
reason recorded at the import so a later tidy-up does not "fix" it back.

**The shadowing itself was left in place.** `download` is in the declared public `__all__`,
so renaming it is an API decision rather than a test fix, and the API shape is on the list
of things to ask about. Open item.

The general point, which is section 39's lesson in another costume: **these two tests could
never have passed as written.** They were committed unrun. Code that has not been executed
has not been written yet, whoever or whatever produced it.

## 40.4 On parallelising this work

What worked: disjoint file ownership, one shared dependency resolved centrally up front, and
reserving git and the shared documents so four writers could not collide.

What it cost: three of four agents stopped mid-task, and partial work from an agent that
cannot report its own gaps is worth less than it looks. Every landed file needed independent
verification anyway, and two of the failures found were found by running the suite, not by
reading the reports. **Treat an agent's completion claim as a hypothesis.** The suite is the
evidence, and for Phase 5 the suite does not yet cover the thing most likely to be wrong.

---

# Section 41 - Tier 2 recorded, and the diagnosis that was wrong for two attempts

## 41.1 The recorded fixture

**Tier 2 exists as of 2026-08-05.** 60 frames, 59 with a face (98.3%), pitch from -0.72 to
-0.08 rad and yaw from -0.41 to 0.43 rad, so the subject genuinely looked around rather
than staring at one point. Captured from the **unmodified** legacy pipeline through the
legacy venv, ONNX on DirectML. About 102 MB, gitignored; only its SHA-256 travels in the
manifest.

## 41.2 Both previous failures were misdiagnosed

The two earlier attempts were recorded as a muted camera and an unlit room. Probing the
hardware before touching anything showed **the camera opened on all three backends and
returned frames.** Neither diagnosis held up.

What was actually happening: the recorder slept a flat **1.0 s** for auto-exposure. Measured
on this webcam, mean brightness sits around **52/255 for the first 3.5 s** and only climbs
to about **100/255 by 6.5 s**. So recording began at roughly half the settled brightness,
and the near-black image that produced was read as a dark room.

**A fixed sleep was never waiting for the thing it existed to wait for.** It waited for a
duration and hoped that stood in for a state. The failure it caused pointed at the
environment, which is why it survived two attempts: the misdiagnosis was plausible, and
nobody measured the camera because the explanation already felt sufficient.

## 41.3 The first fix was also wrong, in an instructive way

Replacing the sleep with a poll was right; the first poll was not. It compared each frame
with the **previous** one and stopped after 10 steady frames, which it reached at **1.4 s
and 50.9/255** - part-way up the ramp, and barely better than the sleep it replaced.

At roughly 33 ms apart, a slow climb between consecutive frames is a fraction of a percent,
so "N consecutive steady frames" is satisfied long before anything has settled. It now
compares against a reading **1.5 s earlier**, a span comparable to the ramp itself.

Generalising, and worth remembering next time something waits for a signal to stabilise:
**a stability test whose window is shorter than the transient it is watching will always
report stable.** It does not fail loudly. It returns early with a confident answer, which is
the same shape of defect as the fixture in section 37 that matched a mutable input, and as
the demonstration in 37.10 that confirmed what it was built to confirm.

## 41.4 Two guards, both refusing rather than warning

- **Minimum brightness.** A fixture captured in the dark would pass while pinning degraded
  behaviour, which is worse than not having one, because it converts a missing test into a
  misleading one.
- **Minimum face rate, 50%.** Zero faces was already refused, but a handful is barely
  better: the fixture would pin mostly the null path while presenting as coverage. Verified
  by the guard actually firing on a 1/60 run before the real recording succeeded.

Both are flags, so a deliberate exception remains possible and remains visible in the
command that made it. The manifest now records settled brightness, settle time and face
rate, so the conditions are data rather than recollection.

Also removed `--settle-frames`, which the rewrite left doing nothing. An option that does
nothing is a lie in the help text.

## 38.5 The durable fix, partly applied

**A virtualenv is not a durable artifact.** What saved the baseline was luck of layout, not
design: `site-packages` happened to be self-contained and happened to move with the tree.

`tests/golden/legacy_environment.txt` now records the frozen set as committed data, with
the base interpreter version and the 3.14.3/3.14.4 caveat in its header, so the environment
can be **reconstructed** rather than merely repaired. That is the half that is done.

The half that is not: nothing verifies the live venv still matches that file. A drift check
comparing `pip freeze` against the committed capture would turn a silent baseline change
into a failing test, which is the same principle as pinning the calibration fixture's input
in section 37. Until that exists, the golden baseline is still one directory deletion away
from being unreproducible, and the recorded package list is the only thing standing behind
it.

---

# Section 42 - The gate that reported instead of gating

## 42.1 What was true, and for how long

CI has been **red on every push since `49a8f3d`**. Retrieved from the REST API rather than
the run pages, because the HTML view shows the newest run and this needed the sequence:

| Run created | Commit | Conclusion |
|---|---|---|
| 2026-08-04T17:30:49Z | `49a8f3d` | success |
| 2026-08-05T05:58:43Z | `5df7ac1` | **failure** |
| 2026-08-05T06:12:12Z | `d0b804f` | **failure** |
| 2026-08-05T07:16:09Z | `3ad8db1` | **failure** |
| 2026-08-05T07:24:28Z | `8006a0c` | **failure** |
| 2026-08-05T07:33:37Z | `8006a0c` | **failure** |

Five consecutive red runs across roughly ninety minutes, and **no document in this
repository said so**. `CHANGELOG.md`'s Verified section still claimed the suite passes on
Linux throughout. That claim was recorded when it was true, at `49a8f3d`, and was never
revisited when the Phase 3/5/6 batch landed on top of it.

Per-step, identical on all three matrix jobs (`3.12`, `3.13`, `3.14`) in run `30985498642`:

    Install                                         success
    Lint                                            success
    Type-check                                      success
    Core imports with optional dependencies absent  success
    Test                                            FAILURE

So `ruff`, `mypy --strict` and the D8 bare-import guarantee are all still green on Linux.
The failure is confined to the suite itself.

## 42.2 Why this is a rule 1 failure and not just a red build

Standing rule 1 says verification is a **gate**, not a report. A gate that goes red and is
neither read nor recorded has silently become a report, and a report nobody reads is
decoration. The batch that turned it red is the same parallel run section 40 already
criticised for landing two tests that could never have passed as written. The lesson there
was to treat an agent's completion claim as a hypothesis, because the suite is the evidence.
That lesson was learned about the local suite and never applied to the remote one.

The specific mechanism worth naming: **the local suite and CI disagree, and only the local
one was being consulted.** On this machine the suite is green - 194 passed at `8006a0c`,
206 after section 43 - both with and without `FOCUSEDGAZE_LEGACY_DIR` set. Whatever fails on
Linux does not fail on Windows, so every local run reinforced a conclusion CI was
contradicting.

## 42.3 What is not yet known

**The failing assertion has not been identified.** The Actions log endpoint returns HTTP 403
without authentication, `gh` is not installed on this machine, and the public check-run
annotations carry only an exit-code-1 message. That annotation is attributed to `.github`
line 63, which is the `mypy` step; the per-step API reports Type-check as success, so the
line attribution is wrong and the step list is authoritative. Recorded here because a future
reader will otherwise re-derive that same false lead.

Reproduction attempts on this machine, all green, so none of them is the cause:

- full suite with the legacy tree present: 194 passed, 2 deselected, zero skips;
- full suite with `FOCUSEDGAZE_LEGACY_DIR` unset, which is the CI condition: 189 passed,
  5 skipped, 2 deselected;
- `ruff check src tests` and `mypy --strict`: both clean.

The difference is therefore Linux, the CI install, or a version resolved differently there.
Candidates worth checking first, given that the batch which broke it was Phases 3/5/6:
filesystem case sensitivity, path separators in the assets cache tests, or `platformdirs`
returning a differently-shaped path.

**This section will be incomplete until somebody reads the log.** It records that the gate
failed and that the cause is unknown; it does not diagnose it, and it must not be read as
though it had.

## 42.4 The durable fix

Nothing in this project noticed a red CI. The audit records what was run locally, and the
local run is the one that cannot see the failure. Options considered:

- a required status check on `main`, so a red run blocks the push rather than annotating it;
- a status line in the audit or CHANGELOG updated per phase gate from the API, not from
  memory.

**The first is being applied by the repository owner**, who is adding a required status
check on `main` so that a red run blocks rather than annotates. Recorded here so this
section does not sit open: the remaining gap is now owned, not merely noted.

Until it lands, a claim that the suite passes means it passes **on Windows**, and every
such claim in this repository should be read that way.

## 42.5 Closed: one test, one parameter, on every version

The failing assertion, once the log was actually read:

    FAILED tests/test_assets_registry.py::test_filename_must_be_a_bare_name[a\b.bin]
      >       with pytest.raises(ConfigError, match="bare name"):
      E       Failed: DID NOT RAISE ConfigError

    1 failed, 188 passed, 5 skipped, 2 deselected

One test, one of its three parameters, identical on 3.12, 3.13 and 3.14. Not a flake, not a
resolver difference, not an install problem.

`ModelAsset.__post_init__` validated a filename with `Path(self.filename).name != self.filename`.
`Path` is `WindowsPath` on Windows and `PosixPath` on Linux, and the two disagree about
whether a backslash is a separator:

| input | `WindowsPath(x).name` | rejected | `PosixPath(x).name` | rejected |
|---|---|---|---|---|
| `../escape.bin` | `escape.bin` | yes | `escape.bin` | yes |
| `sub/dir.bin` | `dir.bin` | yes | `dir.bin` | yes |
| `a\b.bin` | `b.bin` | yes | `a\b.bin` | **no** |

On Linux a backslash is an ordinary filename character, so `a\b.bin` genuinely *is* a bare
name and the guard correctly declined to reject it. The test asserted rejection
unconditionally, so it could only ever pass on Windows.

**The cause landed exactly where 42.3 guessed**: the Phase 6 assets code, path separators.
That is worth noting without over-reading it. The guess was cheap and it was checked, not
assumed; a guess recorded and then confirmed by evidence is a different thing from a guess
believed because it sounded right, which is the failure mode section 41 spent three
attempts on.

## 42.6 The gate worked. The reading of it failed.

This is the part to keep. `ci.yml` says in its own header that it runs on Linux
deliberately, because "proving the pure core imports and computes correctly on Linux is
what keeps the platform abstractions (D1) honest". That is precisely what it did. It found
a Windows assumption baked into a validator, on the first push after that validator landed,
and it said so five times.

Nothing was wrong with the mechanism. What failed was that nobody read it, and the reason
nobody read it is worth naming: **the local suite was green, and the local suite is the one
that cannot see this class of defect at all.** Every local run agreed with itself and
disagreed with CI, and the local one was the one being consulted. A test suite cannot
detect a platform assumption on the platform that shares it.

## 42.7 The 403 that was accepted too early

The diagnosis needed the authenticated log, and an earlier turn had reported the log
unavailable after a single unauthenticated 403 and moved on. That was a premature stop: the
endpoint was not refusing, it was asking for credentials that were sitting in the machine's
own credential helper the whole time.

The working method, recorded because it will be needed again:

    TOKEN=$(printf "protocol=https\nhost=github.com\n\n" | git credential fill \
            | grep "^password=" | cut -d= -f2-)
    curl -H "Authorization: Bearer $TOKEN" -L \
         https://api.github.com/repos/OWNER/REPO/actions/jobs/JOB_ID/logs

The stored credential is the repo owner's and carries the scope the endpoint needs. `gh` is
not installed on this machine, which is what the earlier turn had treated as the end of the
road.

The general lesson matches rule 1 from the other direction. A gate that reports "denied"
is a result, but **"I could not check" is not the same as "it cannot be checked"**, and the
difference is usually one attempt.

## 42.8 The sweep found two more holes in the same validator

The backslash divergence was caught only because a test happened to parametrize a backslash.
So the rest of `src/` was swept for anything whose meaning changes between `PosixPath` and
`WindowsPath`. Two further defects turned up in the same six lines, neither of which any
test exercised:

**A bare `..` was accepted on both platforms.** `Path('..').name` is `'..'`, so the check
passed it, and `cache/models/..` is the parent of the cache directory. This was never a
platform divergence at all: it was a traversal hole everywhere, and it survived because no
case in the parametrize list was a bare `..`. The test that found the backslash bug had the
right idea and an incomplete list.

**`C:foo.bin` was accepted on Linux.** On Windows it is drive-relative:

    PureWindowsPath('D:/cache/models') / 'C:foo.bin'   ->   C:foo.bin

which escapes the cache directory and the drive together. A colon is also how an NTFS
alternate data stream is addressed. On POSIX it is an ordinary character, so a registry
entry written or generated on Linux passes validation and becomes an escape when the same
registry is read on Windows.

Everything else in `src/` came back clean, and the reasons are worth recording so the sweep
does not have to be repeated:

| Site | Verdict |
|---|---|
| `profile.py` `_NAME_RE` | Safe by construction. `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$` is a whitelist with no separator in it, and requiring an alphanumeric first character rejects `..` as a side effect. Profile names become filenames and are joined in five places; all are covered by this one regex. |
| `registry.py` `model_dir_override` | `Path(raw).expanduser()` on a user-supplied environment variable. Platform-dependent interpretation is **correct** here: the user typed a path for their own machine. |
| `download.py` joins | `directory / asset.filename` and the `.part` sibling. These are the joins the validator protects; once the validator is platform-independent they are safe everywhere. |
| `config.py` suffix dispatch | `p.suffix.lower()` against `.toml`/`.json`. Case-folded explicitly, so it behaves identically on a case-sensitive filesystem. |
| `profile.py` `list_profiles` | `p.stem` over a real `glob` result. No string is being classified. |

## 42.9 The fix, and why it is a behaviour change

The rule is now a property of the string rather than of the running platform, expressed as a
character class instead of through `pathlib`:

    _NOT_A_BARE_NAME = re.compile(r"[/\\:]")

plus an explicit rejection of `.` and `..`. Both separators, the drive/stream colon and both
traversal spellings are rejected on every platform, so one registry entry means one thing
everywhere.

This is a **behaviour change**, not a test fix, and it is committed separately under rule 4
with that stated in the message. Filenames that a POSIX install previously accepted are now
refused. Nothing in the shipped registry is affected: `face_landmarker.task`,
`l2cs_gaze360.onnx` and `L2CSNet_gaze360.pkl` all pass unchanged, and a control test now
pins that ordinary filenames are still accepted, because a validator that rejected
everything would satisfy every negative test above.

The new guarantee is pinned two ways. The parametrized list grew from 3 cases to 11 and now
includes the ones that were silently passing. And `test_the_bare_name_rule_does_not_consult_pathlib`
asserts the property **directly against `PurePosixPath` and `PureWindowsPath`** rather than
through the ambient `Path`, because a test written in terms of `Path` agrees with whichever
defect the local platform happens to have. That test would have caught the original bug on a
Windows-only developer machine, which is the whole point.

---

# Section 43 - Phase 5 verified: the calibration polynomial has teeth

## 43.1 What section 40.2 said, and what closed it

Section 40.2 recorded the highest-risk numerical work in the migration with no dedicated
test file, and **none of the four mutation checks run**. `tests/test_calibration_profile.py`
now exists: 12 tests, of which 4 need no scikit-learn.

The headline number: **`apply()` reproduces the legacy recording exactly.**

    cases compared : 169
    worst drift    : 0.000e+00   (tolerance 1e-9)

Not merely within tolerance - bit for bit, across every recorded case. The pure-NumPy path
evaluates the identical surface the pickled scikit-learn pipeline did. The assertion is
still written against `1e-9` rather than zero, because BLAS may legitimately pick a
different kernel elsewhere and a last-bit difference is not the defect being hunted.

## 43.2 The four mutations, measured

Against the committed fixture, tolerance 1e-9:

| Mutation | Worst drift | Caught |
|---|---|---|
| (control) unmutated | 0.000e+00 | passes, as required |
| transposed coefficient order | 8.318e-01 | yes |
| wrong term ordering (rows 1 and 2 exchanged) | 1.000e+00 | yes |
| swapped x/y coefficient sets | 1.000e+00 | yes |
| degree 2 instead of degree 3 | 3.138e-02 | yes |

The transposed-order row reproduces section 37.4's 8.318e-01 exactly, which is the expected
result: same mutation, same fixture, now applied to the new implementation rather than the
legacy one. Two independent paths agreeing on a number neither was tuned to produce is worth
more than either alone.

Term ordering was also checked directly against a real `PolynomialFeatures` for **degrees 1
through 8**, not only the 3 the system uses. The claim being made is about the construction,
so it is tested as a construction.

## 43.3 The degree mutation is only detectable on a cubic surface

The weakest row above is the degree mismatch at 3.138e-02, and it is weak for a reason worth
writing down: **it is a property of the fixture's data, not of the code.** If the synthetic
target in `make_synthetic_calibration.py` were ever flattened, evaluating the model at
degree 2 would drift by nearly nothing, that row would stop failing, and the suite would go
on reporting that it catches a degree mismatch when it no longer did.

A comment would not survive a tidy-up. So the property is **asserted**:
`test_the_recorded_surface_is_genuinely_cubic` requires the largest cubic coefficient to be
at least 5% of the largest coefficient overall. Measured when written: 0.2805 against
0.9501, i.e. **29.5%**, so there is a wide margin, and a future regeneration that flattens
the surface fails with a message naming the file to fix.

This is the section 37 lesson in a third costume: a test whose input is free to change
underneath it is pinned to nothing.

## 43.4 A guard that had never executed

`_check_powers_match_sklearn` runs on every path that reads coefficients out of scikit-learn
and hard-fails if a future release reorders its feature expansion. Coverage showed **its
body had never been reached by any test**. It was a plausible-looking block with no evidence
it did anything.

Four tests now cover it, all scikit-learn-free because the check is pure NumPy: it accepts
the order this package generates (the control - a guard that rejected everything would pass
the negative test too), rejects a permuted table, rejects a table of the wrong size, and
names both orders in its message.

## 43.5 The tests were shown to fail

Standing rule 2 applied to the new file itself, since a test file that only ever passes is
the thing this project keeps getting caught by. Each defect was introduced and the relevant
test confirmed to fail:

| Defect introduced | Result |
|---|---|
| `polynomial_powers` returns a reversed table | caught at degree 1 |
| `apply()` nudged by 1e-8, four orders below a real bug | caught, both equivalence tests |
| fixture surface flattened in its cubic terms | caught by the cubic-weight assertion |
| the term-order guard replaced with a no-op | caught: the expected error was not raised |
| a model whose digest disagrees with the recording | caught by the fixture's digest assertion |

The 1e-8 row is the important one: sensitivity extends far below the magnitude any of the
four real mutations produce.

## 43.6 Both dependency branches recorded

Rule 3: a branch selected by a package being present is unpinned unless recorded both ways.

    with scikit-learn      12 passed
    without scikit-learn    4 passed, 8 skipped

The skips name their reason and are actionable. The four that still run are the term-order
guard tests, which is the right split: they are the ones that must hold in a base install,
where `apply()` runs and no fitter ever will.

One trap found while recording the second branch, worth keeping. The first attempt blocked
scikit-learn by raising a plain `ImportError`, and the tests **errored instead of skipping**.
`pytest.importorskip` only skips on `ModuleNotFoundError`; a bare `ImportError` means
installed-but-broken, which is a genuinely different condition that must not be skipped
past. The simulation was wrong, not the tests. **A negative test that does not reproduce the
real absence is testing a condition that never occurs.**

## 43.7 Coverage

| Module | Before | After |
|---|---|---|
| `calibration/profile.py` | 23% | **56%** |
| `calibration/fitter.py` | 27% | **59%** |
| Total | 66% | **78%** |

Still short of the Phase 8 target of 80%, and the remainder is concentrated in
`collector.py` (48%) and the profile's on-disk paths, neither of which this file set out to
cover. What is now covered is the arithmetic, which is the part that fails silently.

## 43.8 Still open

- `fit_calibration` defaults to `degree=2` while `robust_fit_samples`, the shipping path,
  uses 3. Section 37.4 flagged this as the live risk from F3 and it is still live. The
  degree mutation now proves the fixture would catch a mismatch; it does not remove the
  asymmetry that makes one plausible.
- `calibration/ui.py` is still a stub.
- `migrate_pickle` needs scikit-learn, because the legacy pickle holds live estimator
  objects. Inherent to the format and documented, but it means the migration path, unlike
  the apply path, is not available in a base install.
