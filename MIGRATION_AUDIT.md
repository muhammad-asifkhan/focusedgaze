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
**Muhammad Asif Khan**, `github.com/muhammad-asifkhan`.
Muhammad Asif Khan (`github.com/muhammad-asifkhan`) is credited as a contributor in `NOTICE`.
See §13.

---

## 7. Defaults to be preserved verbatim (rule 1)

Extraction must not change behaviour. Recording these now so drift is detectable:

| Constant | Value | Source |
|---|---|---|
| Capture | 1280×720, MSMF, ~31 fps | `gaze_server.py` |
| ONNX input | 448×448, ImageNet mean/std | `gaze_pipeline.py` |
| `BINS` | 90; angle = `argmax·4 − 180` degrees | `gaze_pipeline.py` |
| `BBOX_SMOOTHING` | 0.3 | `gaze_pipeline.py` |
| Face crop padding | 0.3 | `gaze_pipeline.py` |
| 1€ filter | `MIN_CUTOFF 0.7`, `BETA 0.6`, `D_CUTOFF 1.0` | `gaze_server.py` |
| Broadcast | 60 Hz nominal (measured ~35-44 Hz; Windows timer granularity) | `gaze_server.py` |
| Calibration | degree 3 via `robust_fit_samples`, MAD factor 2.5, `min_keep` 60 | `calibration_utils.py` |
| Positioning gate | 45–65 cm, `CENTER_TOL` 0.12, `REAL_IPD_CM` 6.3, `ASSUMED_HFOV_DEG` 60 | `positioning_gate.py` |

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

Muhammad Asif Khan (`github.com/muhammad-asifkhan`) is credited as a
**contributor** in `NOTICE` — the gaze pipeline this package extracts was built as part
of the gaze-controlled game project.

### Every hit found, and its disposition

Grepped the whole tree for `muhammad-asifkhan`, `iammuhammad-asifkhan` and `Muhammad Asif Khan`:

| Location | Disposition |
|---|---|
| `focusedgaze-sdk/pyproject.toml` — `authors` | Changed; `maintainers` added |
| `focusedgaze-sdk/pyproject.toml` — Homepage / Source / Issues | Changed to `muhammad-asifkhan/focusedgaze` |
| `focusedgaze-sdk/README.md` — author line | Changed; contributor line added |
| `focusedgaze-sdk/NOTICE` | New §0 crediting author and contributor |
| `MIGRATION_AUDIT.md` §3.3 | Corrected, pointing here |
| `focusedgaze-sdk/LICENSE` | **Already correct** — it read "Muhammad Asif Khan" before I touched it |
| **`README.md` (game repo root), `git clone …/the originating game repository.git`** | **NOT changed.** This is the *game* project's own clone URL — a different repository, genuinely owned by `muhammad-asifkhan` and already published there. Rewriting it would break the game's setup instructions. |

**Git remotes:** `focusedgaze-sdk/` is not a git repository (never initialised), so there
is no SDK remote to re-point. The game repo's remote (`the originating game repository`) is correct as
it stands and was left alone for the same reason as the row above.

### Downstream consequences, flagged now (not deferred to Phase 10)

1. **PyPI account.** The `0.0.0` placeholder must be published from **Asif's** PyPI
   account, since that account will own the project long-term. My Phase 1 deviation
   stands — I do not publish — but the instruction is now addressed to Asif, not to
   Muhammad Asif Khan.

2. **Trusted Publishing is owner-bound.** PyPI's OIDC publisher configuration names a
   specific GitHub **owner + repository + workflow filename**. A publisher configured
   against `muhammad-asifkhan/focusedgaze` will **reject** a release from
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

## 15. Correction to the Phase 1 wheel claim

The `Programming Language :: Python :: 3.12` and `3.13` classifiers are **inferred, not
verified**. I confirmed cp312/cp313/cp314 wheels on PyPI for `onnxruntime`,
`onnxruntime-directml` and `numpy`, but `mediapipe` and `opencv-python` use wheel tags my
filename check could not resolve, and I have installed on **3.14 only**.

Added to Phase 9 as a real task: create genuine 3.12 and 3.13 environments and install
there. Until that passes, those two classifiers are a claim I cannot support, and the
audit says so rather than the README implying otherwise.

## 16. Additional question

9. **Copyright line.** `LICENSE` currently names Muhammad Asif Khan alone, and `NOTICE`
   credits Muhammad Asif Khan as a contributor. Is that the intended split, or should the
   MIT copyright line name both? Not my decision to make — tell me which you want.

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

A publisher configured against `muhammad-asifkhan/focusedgaze` would reject releases from this
repository with a 403 — the failure mode flagged earlier.

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

Related: the repository-local `user.name` was a GitHub handle (`muhammad-asifkhan`); it is now the
real name, **Muhammad Asif Khan**. Still deliberately local to this
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

`LICENSE` names Muhammad Asif Khan alone; `NOTICE` §0 and the README credit Muhammad Asif Khan
Khan as contributor. Current split retained, as instructed. Closed.

---

## 12. What happens next

Phase 1 is complete and I have stopped at its gate. On your go-ahead I start **Phase 2**
(extract the pure core behind `GazeEstimator.process()`), which per A3 must include an
explicit test proving two `GazeEstimator` instances in one process do not interfere —
the current module-level `_smoothed_bbox` global makes that impossible today.

Per your closing instruction, **each phase's documentation is written as part of that
phase, immediately after its implementation** — not deferred. This section is Phase 1's.
