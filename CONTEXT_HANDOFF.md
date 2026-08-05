# Context Handoff: focusedgaze migration

Written to let a new session pick this up cold. Read this, then `STANDING_BRIEF.md`
(the binding instructions), then `MIGRATION_AUDIT.md` (the full record).

Last updated: 2026-08-04, immediately after the environment split.

---

## 1. What this project is

Two separate things, in two separate places:

| | Path | What it is | Git |
|---|---|---|---|
| **The game** | see section 2 | A gaze- and gesture-controlled browser game (Three.js). Working, playable. | the originating game repository |
| **The SDK** | `C:\projects\focusedgaze` | The gaze pipeline extracted from that game into a pip-installable Python package. **This is the active work.** | `github.com/muhammad-asifkhan/focusedgaze` |

The SDK is being extracted **from** the legacy pipeline in the originating game
repository, which stays working throughout. Phase 11 will make the game depend on the
published package.

**Attribution:** the SDK is attributed entirely to **Muhammad Asif Khan
<asifcalm53@gmail.com>**, GitHub `muhammad-asifkhan`. Every commit, file and reference.
No second identity anywhere. This was enforced by a history rewrite.

---

## 2. Where everything lives

**This table is the single source of truth for the legacy path.** It is the only place in
the repository that records the literal value. Everywhere else, in documents and in code,
refers to `FOCUSEDGAZE_LEGACY_DIR`.

That rule exists because the path moved **three times in one session**, and every document
that repeated the literal value went stale on the next move. Verified as of 2026-08-04, but
treat it as this machine's layout rather than a guarantee: if it moves again, change it here
and nothing else needs touching.

Everything now lives under one root: `C:\Users\basim\Desktop\foucusedazed\`.
The tree moved wholesale on 2026-08-05 (fourth move; see section 12).

```
C:\Users\basim\Desktop\foucusedazed\
  game_integration\game_integration\  the game repo (separate project; note the
                                      doubled directory name, it is not a typo)
    gaze-detection\                   THE LEGACY PIPELINE - the reference implementation
      gaze_env\                       its virtualenv - FROZEN, see section 6
      gaze_pipeline.py                landmarks + ONNX inference
      positioning_gate.py             distance + centring
      calibration_utils.py            polynomial fit/apply (pickle-based)
      gaze_server.py                  WebSocket bridge (also has gesture integration)
      milestone1..9_*.py              scaffolding scripts
      models\                         weights + calibration .pkl backups (IRREPLACEABLE)
    game\workingGameTemplate\         the browser game

  projects\
    focusedgaze\                      THE SDK (git repo, pushed)
      .venv\                          SDK dev environment (gitignored)
      src\focusedgaze\                the package
      tests\                          golden harness + unit tests
      STANDING_BRIEF.md               binding instructions - READ THIS
      MIGRATION_AUDIT.md              full record
      CONTEXT_HANDOFF.md              this file
    dryrun\                           a partial earlier copy, NOT the live SDK
    BUNDLES.md                        recovery bundles, labelled
    *.bundle                          git recovery bundles (6)
    preflight.py  rewrite.py  gate.py  history-rewrite tooling
    fg-replacements.txt               filter-repo rules - EMPTY is the resting state
    fg-replacements.README.md         why, and the hard prohibitions
```

`projects\dryrun\` is a leftover partial copy containing only docs and tests. It is
**not** the live SDK and must not be edited. The live package is `projects\focusedgaze\`.

---

## 3. Phase status

| Phase | State |
|---|---|
| 0. Audit | **Done.** `MIGRATION_AUDIT.md` |
| 1. Skeleton + golden harness | **Done.** Package installs, imports; Tier 1 fixtures recorded |
| **2. Pure core** | **OPEN.** `filters.py` and `positioning.py` done and proven equivalent. `landmarks.py`, `model.py`, `estimator.py` remain |
| 3. Config/types/exceptions | **Landed, unreviewed.** `exceptions.py` done centrally. Config defaults not yet checked against legacy source |
| 4. Capture layer | Not started |
| 5. Calibration | **Landed, UNPROVEN. No tests, no mutation checks.** See audit §40.2 |
| 6. Assets + CLI | **Assets landed with tests. CLI not started** |
| 7. Server extra | **Contract established, decisions taken** (§39). Code not started |
| 8. Tests/docs | Not started |
| 9. Packaging | Wheel-availability task **closed early** by CI; clean-venv install remains |
| 10. Release | Not started |
| 11. Close the loop | Not started |

### Phase 2 gate condition: THIS IS THE CURRENT BLOCKER

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

**Tier 1: committed, numeric, runs in CI.** Recorded from the unmodified legacy
pipeline. Covers the calibration polynomial, the One Euro filter, the positioning gate,
including both focal branches, non-monotonic timestamps and degenerate geometry.

**Tier 2: gitignored, needs a real face. NOT YET RECORDED.** Covers frames →
(pitch, yaw). Record with:

```bash
cd C:\projects\focusedgaze
python tests/golden/record_tier2.py --frames 60
pytest -m hardware
```

Requires the webcam working **and lit**. Two previous attempts failed, once because the
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
- **No hybrid gaze+gesture mode** in the game: sharing one camera cost ~20% of the hand
  detection rate.
- **Phase 7 wire format, decided 2026-08-05** (audit §39, full contract in
  `docs/wire_format.md`). The game reads **only** `type:"input"` and discards
  `type:"gaze"`, so a gaze-only server would have left it connected and motionless with no
  error. Decisions: the SDK emits a **minimal gaze-only `input` message** (`type`, `mode`,
  `source`, `ok`, `x`, `y`, `t`; `mode` and `source` constant `"gaze"`; **no gesture
  fields**) plus hooks, and the game repo's wrapper **replaces** it via `resolve_input` to
  add gesture. `gaze_test.html` **stays supported**, so `type:"gaze"` remains as the raw
  device feed. This supersedes Q6, which had been asked in two documents and answered in
  none. **The claim that the game plays against the minimal message is inferred from
  reading the client, not executed** - confirm against a stub server first.

---

## 6. Environments: DO NOT MIX

| Env | Path | Rule |
|---|---|---|
| **Legacy** | `$FOCUSEDGAZE_LEGACY_DIR\gaze_env` (see section 2) | **FROZEN. Never install anything into it again.** |
| **SDK** | `<SDK>\.venv` (see section 2) | Normal dev env. `-e ".[cpu,calibration,dev]"` plus `websockets` |

**2026-08-05: both venvs were found dead and the legacy one was repaired. Read section 12
before running anything.**

The Tier 1 fixtures were recorded using the legacy venv's exact dependency set. A
transitive `numpy`/`protobuf`/`mediapipe` bump there would move the baseline and surface
later as mysterious drift. **The reference implementation is data, not just code: its
environment is part of the measurement.**

Audited: 35 packages were added to `gaze_env` before this rule existed, but **nothing was
upgraded or downgraded**. Fixtures remain valid. See audit §31.

**The two venvs do not run the same libraries.** `mediapipe` is 0.10.35 in legacy and
1.0.0 in the SDK venv; the ONNX provider is DirectML/GPU in legacy and CPU in the SDK
venv. Everything else (numpy, protobuf, opencv, sklearn, scipy) is identical. The ONNX
difference was measured and is harmless (188x under tolerance); the MediaPipe landmark
difference is **still unmeasured** because it needs a face. See audit §32 before trusting
any cross-venv comparison.

**`websockets` is now installed** in the SDK venv (audit §33.5), because the legacy
`gaze_server` imports it and without it the legacy-comparison tests **skip**. That skip
was concealing a genuine failure. See section 9 item 2 and audit §33. A `pip freeze`
diff confirmed nothing else moved when it was added.

---

## 7. Running things

```bash
# Tests. Currently 19 passed, 2 deselected.
cd C:\projects\focusedgaze
set FOCUSEDGAZE_LEGACY_DIR=<the gaze-detection path from section 2>
pytest -q -rs

# Hardware tests (needs the Tier 2 fixture — does not exist yet)
pytest -m hardware

# The game
cd %FOCUSEDGAZE_LEGACY_DIR% && gaze_env\Scripts\python.exe gaze_server.py
cd %FOCUSEDGAZE_LEGACY_DIR%\..\game\workingGameTemplate && python -m http.server 8000
# then http://localhost:8000/forest.html
```

`FOCUSEDGAZE_LEGACY_DIR` points the harness at the legacy pipeline and is the reason the
suite does not care where that pipeline lives. Set it. The fallback is `../gaze-detection`
relative to the repository root, which does not resolve in the current layout.

**The literal value lives in exactly one place: the table in section 2.** It has moved
three times in one session, and every document that repeated it went stale on the next
move. Everywhere else, including this section, refers to the variable.

**Always run with `-rs`.** A skip in this suite has already hidden a failing assertion
once; a bare `pytest -q` reports the skip count without the reason.

---

## 8. Current CI state

**Green.** Run 2 (`14f7b36`), all three matrix rows success on Linux:

- `mediapipe` and `opencv-python` install on 3.12/3.13/3.14 (this closed a Phase 1 open
  question)
- `mypy --strict` passes, first ever execution
- D8 bare-import guarantee holds (no ONNX provider, no websockets, no sklearn)
- 19 tests pass on Linux
- Coverage 87% overall / **94% for code that actually exists** (stubs inflate, untested
  CLI banner deflates)

Steps after `Lint` carry `if: !cancelled()`. A lint failure previously masked three
checks that had never run.

---

## 9. Open items / waiting on a human

1. **Tier 2 fixture.** Blocks the Phase 2 gate. Needs a lit room and a working webcam.
2. **`milestone6` accuracy baseline** must be run on unmodified code before any
   milestone script is deleted (Phase 8). It is the source of the "2.0-2.4 cm" claim.
   **Run it before recalibrating again**: a baseline measured against a calibration that
   is later replaced has exactly the problem the old item 2 described, now closed below.
3. **MediaPipe landmark equivalence: unmeasured.** The two venvs run 0.10.35 and a 1.x.
   The A/B needs an image containing a face, so it is blocked behind item 1. Audit §32.3.
4. **The mediapipe range decision.** `pyproject.toml` now declares `>=0.10.30,<1.1`. The
   old floor was unreachable on every supported Python. Widening needs a replay per
   candidate version. Audit §32.5.
5. **The legacy interpreter is a patch level below what recorded the fixtures.** See
   section 12. **Closed for Tier 1** by a byte-for-byte re-record; still open for Tier 2,
   so it is blocked behind item 1. No decision needed unless Tier 2 disagrees.

### Closed

- **PyPI / TestPyPI pending publisher: CONFIRMED, 2026-08-04.** Pending publishers are
  registered on both, the `pypi` and `testpypi` GitHub environments exist, and the values
  were verified against `release.yml`. The pending-publisher path was the correct reading
  of the two 404s: neither project existed, so the first upload creates it. **`release.yml`
  needs no change**. Its inline instructions already assume exactly this path.
- **Tier 1 calibration fixture failure: CLOSED, 2026-08-04**, by commit `49a8f3d`
  implementing audit §33.4. The fixture read its model from a path that ordinary
  recalibration overwrites, so it failed with a drift of exactly 1.0, the width of the
  `[0,1]` clamp, and no indication of the real cause. The model is now **synthetic**,
  fitted by `tests/golden/make_synthetic_calibration.py` using the unmodified legacy
  fitter and committed beside the fixture, so input and expectation travel in one commit
  and cannot drift apart. This is **not** the re-recording §33.4 rejected: it replaced a
  mutable input with an immutable one rather than lowering the bar. Mutation-checked at
  1e-9. Also removed the last real person's calibration from the test tree.
- **SDK venv install:** completed 2026-08-04, **but the venv has since been destroyed by
  the move. Rebuilt 2026-08-05. See section 12.**

---

## 10. Hard-won lessons (all of these cost real damage)

Recorded in full in `MIGRATION_AUDIT.md` §21–31. The short version:

1. **A text replacement that silently no-ops is worse than one that errors.** A literal
   whose words were line-wrapped never matched. This happened twice.
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

Updated 2026-08-05. The suite is at **194 passed, 2 deselected, zero skips**.

1. **Write the Phase 5 tests and run the four mutation checks.** Highest priority, ahead of
   any new code. 1653 lines of the riskiest arithmetic in the project are currently
   unproven, and a wrong polynomial will not announce itself. Audit §40.2.
2. **Check the Phase 3 config defaults against the legacy source**, file:line by file:line.
   The citations are in the code; nobody has confirmed them. A changed default is a
   behaviour change wearing a refactor's clothes.
3. Record the Tier 2 fixture (lights on), replay it against the **unmodified** pipeline.
4. Only then: `landmarks.py`, `model.py`, `estimator.py`, plus the test proving two
   `GazeEstimator` instances in one process do not interfere.
5. Close the Phase 2 gate.

Decide separately: whether `download` should keep shadowing the `assets.download`
submodule (§40.3), and Q7-2, whether the SDK sends explicit zero sequence counters.

Before writing any code, read `STANDING_BRIEF.md` Part C: the ten standing rules are
binding, and most of them exist because something broke.

---

## 12. The 2026-08-05 environment loss, and the repair

**Both virtualenvs were dead on arrival this session. Neither could execute a single
statement.** The whole tree had been moved to a new root (section 2), and separately the
base CPython 3.14 installation they were built against had been removed from the machine.

A Windows venv keeps a *copy* of `python.exe` but still resolves its standard library
through the `home` key in `pyvenv.cfg`. When that directory disappears, the copied
interpreter aborts before running anything, with a message naming the missing executable
rather than anything about virtualenvs. Both `pyvenv.cfg` files pointed at a base install
that no longer exists, and neither pointed at a location inside this project.

### Why this mattered more than a normal broken venv

The legacy venv is **not** a convenience. Its exact package set *is* the measurement that
the Tier 1 golden fixtures were recorded against (section 6). Losing it would have meant
losing the ability to re-derive or extend the baseline, and every later equivalence claim
rests on it.

**It was recoverable because only the interpreter bootstrap was broken, not the packages.**
`Lib\site-packages` was intact: 132 entries, with the frozen set present and unchanged at
mediapipe 0.10.35, numpy 2.5.1, onnxruntime-directml 1.24.4, scikit-learn 1.9.0, scipy
1.18.0, opencv 5.0.0.93, protobuf 7.35.1, websockets 16.1.1. That matches what section 6
records, so the measurement survived.

### What was done

- **Legacy venv: repaired, not rebuilt.** Only `pyvenv.cfg` was rewritten, to point `home`
  and `executable` at the 3.14 interpreter that does exist on this machine. The original
  file is preserved beside it as `pyvenv.cfg.bak-before-repair`. **`site-packages` was not
  touched**, so the frozen dependency set is bit-for-bit what recorded the fixtures.
  Verified by importing the whole stack and confirming the DirectML provider still
  enumerates.
- **SDK venv: rebuilt from scratch.** It is gitignored, disposable, and holds no
  measurement. Rebuilt as `-e ".[cpu,calibration,dev]"` plus `websockets`, on 3.14.3.
  Verified: 19 passed, 2 deselected, zero skips, matching the count section 7 records.

  **One trap worth knowing.** Installing `websockets` bare took 17.0.1, but the `server`
  extra declares `>=12,<17`, so the dev environment silently violated the package's own
  declared range - the state most likely to let a bound be wrong without anyone noticing.
  Pinned to **16.1.1**, which is inside the range and matches the legacy venv, so the
  legacy-comparison tests exercise the same major version the reference does. If you
  rebuild this venv, install the extra rather than the bare name.

### The one caveat, and it is a real one

The interpreter now under the legacy venv is **3.14.3**. The venv was originally built on
**3.14.4**, which is no longer on this machine. So the frozen environment is running one
CPython patch level below the one that recorded the fixtures.

The compiled extensions are `cp314`, so they are ABI-compatible across 3.14.x, and a patch
release is very unlikely to move floating-point results in numpy or onnxruntime. But
"unlikely" is not "measured", and section 6's rule is explicit that the environment is
part of the measurement.

**So it was measured rather than argued about.** `record_tier1.py` was re-run with the
repaired legacy venv, against a **scratch copy** of the harness so the committed fixtures
were never at risk. That precaution was necessary: the recorder has a hard-coded output
path and no override, so running it in place would have overwritten the baseline, which is
exactly the trap audit §33.4 rejected.

Result: all four Tier 1 fixtures **byte-for-byte identical**. 169 calibration cases, 124
filter cases, 16 + 16 positioning cases across both focal branches.

The caveat is therefore **closed for Tier 1** and **open for Tier 2**, which is not
recorded yet and covers the two things most exposed to an environment difference: MediaPipe
landmarks and ONNX inference. Tier 1 pins no inference output at all - it prints a line
naming the DirectML provider, which only proves the session builds. If Tier 2 ever
disagrees, install 3.14.4 and repoint `home` before suspecting the refactor.

**Related, and easy to misread:** `tests/golden/adapters.py` puts the legacy directory on
`sys.path` of the *current* interpreter. It never invokes the legacy venv's `python.exe`.
A normal `pytest` run therefore executes legacy *source* under the **SDK venv's** packages.
The frozen venv matters for **recording** fixtures, not for replaying them, and a green
suite tells you nothing about whether that venv still works. Audit §38.

### The lesson

**A virtualenv is not a durable artifact, and "frozen" did not mean "safe".** The rule in
section 6 protected the environment against being *modified* and said nothing about it
being *moved* or about its base interpreter being *uninstalled*. Both happened, and the
protection did not apply to either.

What actually saved the baseline was that `site-packages` is self-contained. What would
make it durable is recording the frozen set as data rather than relying on the directory:
a `pip freeze` committed to this repository, and the base interpreter version pinned
alongside it, so the environment can be reconstructed rather than merely repaired. That
work is not done. Until it is, this venv is a single directory deletion away from taking
the golden baseline with it.
