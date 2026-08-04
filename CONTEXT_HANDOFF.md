# Context Handoff — focusedgaze migration

Written to let a new session pick this up cold. Read this, then `STANDING_BRIEF.md`
(the binding instructions), then `MIGRATION_AUDIT.md` (the full record).

Last updated: 2026-08-04, immediately after the environment split.

---

## 1. What this project is

Two separate things, in two separate places:

| | Path | What it is | Git |
|---|---|---|---|
| **The game** | `D:\Projects\game_integration` | A gaze- and gesture-controlled browser game (Three.js). Working, playable. | the originating game repository |
| **The SDK** | `C:\projects\focusedgaze` | The gaze pipeline extracted from that game into a pip-installable Python package. **This is the active work.** | `github.com/muhammad-asifkhan/focusedgaze` |

The SDK is being extracted **from** `D:\Projects\game_integration\gaze-detection\` (the
"legacy pipeline"), which stays working throughout. Phase 11 will make the game depend on the
published package.

**Attribution:** the SDK is attributed entirely to **Muhammad Asif Khan
<asifcalm53@gmail.com>**, GitHub `muhammad-asifkhan`. Every commit, file and reference.
No second identity anywhere — this was enforced by a history rewrite.

---

## 2. Where everything lives

The legacy pipeline **moved off `C:` and is no longer where earlier revisions of this file
said it was.** Paths below are current as of 2026-08-04 and verified. Treat them as this
machine's layout, not as a guarantee: `FOCUSEDGAZE_LEGACY_DIR` is the mechanism that makes
the harness independent of them (section 7).

```
D:\Projects\game_integration\         the game repo (separate project)
  gaze-detection\                     THE LEGACY PIPELINE — the reference implementation
    gaze_env\                         its virtualenv — FROZEN, see section 6
    gaze_pipeline.py                  landmarks + ONNX inference
    positioning_gate.py               distance + centring
    calibration_utils.py              polynomial fit/apply (pickle-based)
    gaze_server.py                    WebSocket bridge (also has gesture integration)
    milestone1..9_*.py                scaffolding scripts
  game\workingGameTemplate\           the browser game

C:\projects\
  focusedgaze\                        THE SDK (git repo, pushed)
    .venv\                            SDK dev environment (gitignored)
    src\focusedgaze\                  the package
    tests\                            golden harness + unit tests
    STANDING_BRIEF.md                 binding instructions — READ THIS
    MIGRATION_AUDIT.md                full record, ~30 sections
    CONTEXT_HANDOFF.md                this file
  BUNDLES.md                          recovery bundles, labelled
  *.bundle                            git recovery bundles (6)
  preflight.py  rewrite.py  gate.py   history-rewrite tooling
  fg-replacements.txt                 filter-repo rules — EMPTY is the resting state
  fg-replacements.README.md           why, and the hard prohibitions
```

---

## 3. Phase status

| Phase | State |
|---|---|
| 0 — Audit | **Done.** `MIGRATION_AUDIT.md` |
| 1 — Skeleton + golden harness | **Done.** Package installs, imports; Tier 1 fixtures recorded |
| **2 — Pure core** | **OPEN.** `filters.py` and `positioning.py` done and proven equivalent. `landmarks.py`, `model.py`, `estimator.py` remain |
| 3 — Config/types/exceptions | Not started (`mypy --strict` already passing on existing code) |
| 4 — Capture layer | Not started |
| 5 — Calibration | Not started. **Highest-risk numerical work** |
| 6 — Assets + CLI | Not started |
| 7 — Server extra | Not started |
| 8 — Tests/docs | Not started |
| 9 — Packaging | Wheel-availability task **closed early** by CI; clean-venv install remains |
| 10 — Release | Not started |
| 11 — Close the loop | Not started |

### Phase 2 gate condition — THIS IS THE CURRENT BLOCKER

`landmarks.py` and `model.py` **must not land until the Tier 2 fixture exists and
replays green against the UNMODIFIED legacy pipeline.** Those two modules are the only
code Tier 2 covers, and they carry the two riskiest edits in the phase:

- `landmarks.py` kills a module-level `_smoothed_bbox` global (bbox smoothing becomes
  instance state)
- `model.py` carries the **ONNX tensor-order defect**: the graph *names* tensor[0]
  `pitch_bins` but it actually contains **yaw**. The legacy code handles this correctly.
  Carry the comment over intact and pin the decode with a test.

---

## 4. The golden harness (the safety net)

Two tiers. This is the only protection against silently changing behaviour.

**Tier 1 — committed, numeric, runs in CI.** Recorded from the unmodified legacy
pipeline. Covers the calibration polynomial, the One Euro filter, the positioning gate,
including both focal branches, non-monotonic timestamps and degenerate geometry.

**Tier 2 — gitignored, needs a real face. NOT YET RECORDED.** Covers frames →
(pitch, yaw). Record with:

```bash
cd C:\projects\focusedgaze
python tests/golden/record_tier2.py --frames 60
pytest -m hardware
```

Requires the webcam working **and lit** — two previous attempts failed, once because the
camera was muted (grey padlock image) and once because the room lights were off
(brightness 2.8/255). The recorder refuses to write a fixture with no face in it.

**Implementation selection** lives in `tests/golden/adapters.py`, not in the tests, so
assertions stay frozen while the code beneath them is replaced.

---

## 5. Key decisions already made (do not relitigate)

- **Model weights are never distributed.** The gaze model derives from Gaze360, which is
  non-commercial research only. `download-models` fetches the MediaPipe landmarker
  automatically but **prints instructions and stops** for the gaze weights. See `NOTICE`.
- **`l2cs` and `face-detection` are NOT dependencies.** Runtime inference is pure ONNX
  Runtime; `l2cs` is imported only by the one-time export script.
- **Windows is the only tested platform** for v0.1. Linux/macOS structurally supported,
  untested. CI proves the pure core runs on Linux.
- **`requires-python = ">=3.12"`**, verified by CI on 3.12/3.13/3.14.
- **No hybrid gaze+gesture mode** in the game — sharing one camera cost ~20% of the hand
  detection rate.

---

## 6. Environments — DO NOT MIX

| Env | Path | Rule |
|---|---|---|
| **Legacy** | `D:\Projects\game_integration\gaze-detection\gaze_env` | **FROZEN. Never install anything into it again.** |
| **SDK** | `C:\projects\focusedgaze\.venv` | Normal dev env. `-e ".[cpu,calibration,dev]"` plus `websockets` |

The Tier 1 fixtures were recorded using the legacy venv's exact dependency set. A
transitive `numpy`/`protobuf`/`mediapipe` bump there would move the baseline and surface
later as mysterious drift. **The reference implementation is data, not just code — its
environment is part of the measurement.**

Audited: 35 packages were added to `gaze_env` before this rule existed, but **nothing was
upgraded or downgraded** — fixtures remain valid. See audit §31.

**The two venvs do not run the same libraries.** `mediapipe` is 0.10.35 in legacy and
1.0.0 in the SDK venv; the ONNX provider is DirectML/GPU in legacy and CPU in the SDK
venv. Everything else — numpy, protobuf, opencv, sklearn, scipy — is identical. The ONNX
difference was measured and is harmless (188x under tolerance); the MediaPipe landmark
difference is **still unmeasured** because it needs a face. See audit §32 before trusting
any cross-venv comparison.

**`websockets` is now installed** in the SDK venv (audit §33.5), because the legacy
`gaze_server` imports it and without it the legacy-comparison tests **skip**. That skip
was concealing a genuine failure — see section 9 item 2 and audit §33. A `pip freeze`
diff confirmed nothing else moved when it was added.

---

## 7. Running things

```bash
# Tests. Currently 1 FAILED, 18 passed, 2 deselected — the failure is real and
# is a harness defect, not a refactor bug. See section 9 item 2 / audit §33.
cd C:\projects\focusedgaze
set FOCUSEDGAZE_LEGACY_DIR=D:\Projects\game_integration\gaze-detection
pytest -q -rs

# Hardware tests (needs the Tier 2 fixture — does not exist yet)
pytest -m hardware

# The game
cd D:\Projects\game_integration\gaze-detection && gaze_env\Scripts\python.exe gaze_server.py
cd D:\Projects\game_integration\game\workingGameTemplate && python -m http.server 8000
# then http://localhost:8000/forest.html
```

`FOCUSEDGAZE_LEGACY_DIR` points the harness at the legacy pipeline and is the reason the
suite does not care where that pipeline lives. Set it. The fallback is `../gaze-detection`
relative to the repository root, which does not resolve in the current layout — the SDK is
on `C:` and the legacy pipeline is on `D:`.

**Always run with `-rs`.** A skip in this suite has already hidden a failing assertion
once; a bare `pytest -q` reports the skip count without the reason.

---

## 8. Current CI state

**Green.** Run 2 (`14f7b36`) — all three matrix rows success on Linux:

- `mediapipe` and `opencv-python` install on 3.12/3.13/3.14 (this closed a Phase 1 open
  question)
- `mypy --strict` passes — first ever execution
- D8 bare-import guarantee holds (no ONNX provider, no websockets, no sklearn)
- 19 tests pass on Linux
- Coverage 87% overall / **94% for code that actually exists** (stubs inflate, untested
  CLI banner deflates)

Steps after `Lint` carry `if: !cancelled()` — a lint failure previously masked three
checks that had never run.

---

## 9. Open items / waiting on a human

1. **Tier 2 fixture** — blocks the Phase 2 gate. Needs a lit room and a working webcam.
2. **Tier 1 calibration fixture is FAILING** — and the failure is a harness defect, not a
   refactor bug. The fixture pins its expected outputs but loads its input model from a
   mutable path, and the live `calibration_model.pkl` has since been re-fitted. Proven by
   replay: the fixture matches one backup exactly (0.0 drift) and every other model by
   1.0, the width of the `[0,1]` clamp. **Do not "fix" this by re-recording against the
   current model** — that goes green while discarding the baseline. Fix is audit §33.4:
   pin the model by SHA-256. Needs a decision.
3. **`milestone6` accuracy baseline** — must be run on unmodified code before any
   milestone script is deleted (Phase 8). It is the source of the "2.0–2.4 cm" claim.
   **Run it before recalibrating again**: a baseline measured against a calibration that
   is later replaced has exactly the problem item 2 describes.
4. **MediaPipe landmark equivalence — unmeasured.** The two venvs run 0.10.35 and 1.0.0.
   The A/B needs an image containing a face, so it is blocked behind item 1. Audit §32.3.
5. **The mediapipe range decision** — `pyproject.toml` now declares `>=0.10.30,<1.1`. The
   old floor was unreachable on every supported Python. Widening needs a replay per
   candidate version. Audit §32.5.

### Closed

- **PyPI / TestPyPI pending publisher — CONFIRMED, 2026-08-04.** Pending publishers are
  registered on both, the `pypi` and `testpypi` GitHub environments exist, and the values
  were verified against `release.yml`. The pending-publisher path was the correct reading
  of the two 404s: neither project existed, so the first upload creates it. **`release.yml`
  needs no change** — its inline instructions already assume exactly this path.
- **SDK venv install** — completed and verified.

---

## 10. Hard-won lessons (all of these cost real damage)

Recorded in full in `MIGRATION_AUDIT.md` §21–31. The short version:

1. **A text replacement that silently no-ops is worse than one that errors.** A literal
   whose words were line-wrapped never matched — twice.
2. **`filter-repo --replace-text` has no comment syntax.** `#` lines become search terms.
   A rules file with comments replaced every `#` in the repository.
3. **A guard must parse its input exactly as the tool does.** The pre-flight skipped `#`
   lines; filter-repo did not. It reported PASS on 10 rules while the tool acted on ~14.
4. **Documentation must not quote what it describes** (standing rule 10). Writing up a
   removed string reintroduces it. This has bitten four times.
5. **A hard-failing first step hides every check behind it**, and fixing only that step
   produces a green run indistinguishable from a genuinely verified one.
6. **Verification is a gate, not a report.** Run everything, paste raw output, stop on the
   first failure.
7. **Prove the test has teeth.** Mutation-check: introduce the plausible error and confirm
   the test fails.
8. **A branch selected by a file's presence is unpinned** unless recorded both ways.

---

## 11. What to do next

1. Confirm the SDK venv install finished; run the 19 tests from it.
2. Record the Tier 2 fixture (lights on), replay it against the **unmodified** pipeline.
3. Only then: `landmarks.py`, `model.py`, `estimator.py`, plus the test proving two
   `GazeEstimator` instances in one process do not interfere.
4. Close the Phase 2 gate.

Before writing any code, read `STANDING_BRIEF.md` Part C — the ten standing rules are
binding, and most of them exist because something broke.
