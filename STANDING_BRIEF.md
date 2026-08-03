# focusedgaze — Standing Brief for the Remainder of the Migration

This supersedes the per-turn instructions. It is the reference for everything from here to
publication. Future turns refer to it by section.

---

## Part A — Finish the history rewrite

The rewrite failed verification and was correctly rolled back. Do not retry it as-is.

### A1. Why it failed, and the general lesson

A rule targeted a multi-word phrase that was **line-wrapped in the source** — the words sat
on two lines with the indentation of the second between them, like this:

```
    ... a multi-word phrase that
    continues on the next line ...
```

The rule was written as a literal with a single space between those words, so it never
matched. The rules on either side of it *did* match, so the sentence was partly rewritten
and the target text survived inside it — in a history that reported success.

Two properties made this dangerous rather than merely wrong: the replacement failed
**silently**, and a file-level "did anything change?" check would have said yes.

**Rule that follows:** a text replacement that silently no-ops is worse than one that errors.
Never author a replacement without proving it fires.

### A2. Fix the replacements

1. **Match on single-line fragments only.** Use the shortest distinctive string that cannot
   straddle a line break — one or two words, not a clause. Prefer `widget-identifier` over
   `the widget identifier that we removed`. Do not write multi-word phrases that approach
   the file's wrap width. If a multi-line block genuinely must be replaced, use a regex with
   `\s+` between the words rather than a literal space.

2. **Pre-flight assertion — this is the actual fix.** Before running the rewrite, for every
   entry in the replacements file, grep every commit's blobs and assert the pattern matches
   **at least once somewhere**. A pattern that matches nowhere is a bug in the replacements
   file and must fail loudly at authoring time. Print a table: pattern → commits affected →
   files affected. Show the table before executing.

### A3. NOTICE collateral — decided

The rewrite redacted emails from `NOTICE`, where they were legitimate contact details. That
was collateral, not intent.

**Decision: (b) — drop the email column from `NOTICE` entirely.** GitHub handles are
sufficient contact for a public credits file. Applied to the working tree before rewriting,
so the current file and the historical replacement agree.

`pyproject.toml`'s author email **stays exactly as it is**. It is required package metadata
and its presence in every commit is correct, not a leak. Excluded explicitly.

### A4. Re-run, with the same gate

Verification is a gate, not a report: per-commit grep for every pattern, `Co-authored-by`
count must be zero, seven commits in the same order, then `reflog expire` + `gc --prune=now`
+ `count-objects -v`. Paste raw output. On any failure, restore from the bundle and report
what did not match — do not iterate silently.

---

## Part B — First push

Only after A4 passes. `git-filter-repo` removes the remote by design, so it is added fresh.

Before pushing, check and report — do not assume, and do not act on what is found:

- Is the remote empty? If the repo was initialised with a README, LICENSE or `.gitignore`, it
  has a commit and the histories are unrelated. **Report what is there and stop.**
- Default branch on GitHub — must be `main` to match local.
- Repo name exactly `focusedgaze` under `muhammad-asifkhan`.
- Environments `pypi` and `testpypi` exist under Settings → Environments.
- On PyPI **and separately on TestPyPI**: was a *pending publisher* registered, or does the
  project exist because something was uploaded? They need different configuration paths, and
  `release.yml`'s inline instructions currently assume pending. If a version was already
  uploaded, report which — a version number can be yanked but never reused.

Expect the first CI run to be red. Linux wheel availability for `mediapipe` and
`opencv-python` across 3.12/3.13/3.14 is inferred, not verified. **Report the failing
(python-version, package) pairs — that result is the Phase 9 verification deliverable. Do not
drop matrix rows to force green.**

---

## Part C — Standing rules

1. **Verification is a gate.** Run all requested checks, paste raw output, stop on the first
   failure. A failed gate is a result, not an error to work around.
2. **Prove the test has teeth.** A fixture that passes has not been shown to catch anything.
   Mutation-check: introduce the specific plausible error and confirm the test fails.
3. **Branch coverage in fixtures.** A branch selected by a file's presence, an environment
   variable, or a missing argument is unpinned unless it was recorded both ways.
4. **No behaviour changes during extraction.** Move first, improve second, separate commits.
   Where behaviour *must* change, say so explicitly in the commit message and add a test that
   pins the new guarantee.
5. **No hard-coded paths**, in package code or test code.
6. **Report honest no-ops.** Never manufacture a commit to match an expected shape.
7. **The audit must be accurate about itself.** No past-tense claims about operations that
   have not run.
8. **Ask before inventing.** A model URL, a checksum, a licence term, a platform claim — if
   it is not in the repo or in these instructions, ask.

---

## Part D — Remaining work

### Phase 2 (open) — pure core
`positioning.py` and `filters.py` are done. Remaining:

- `landmarks.py` — MediaPipe wrapper and face crop. **Kills the module-level
  `_smoothed_bbox` global**; bbox smoothing becomes instance state.
- `model.py` — ONNX session, provider preference list with fallback and one log line naming
  the provider that loaded, and the pitch/yaw tensor-order decode. Carry the upstream-defect
  comment over intact and pin the decode with a test.
- `estimator.py` — `GazeEstimator.process(frame, timestamp) -> GazeResult`. No I/O.
- The A3 isolation test: two `GazeEstimator` instances in one process must not interfere.

**Gate condition:** `landmarks.py` and `model.py` do not close without the Tier 2 fixture.
Replay it against the *unmodified* pipeline first — if it does not pass there, the harness is
wrong and that must be known before it starts reporting on the refactor.

### Phase 3 — config, types, exceptions
Frozen dataclasses replacing every module constant, defaults preserved exactly. `GazeResult`,
`GazeStatus`, the exception tree. `mypy --strict` clean.

### Phase 4 — capture
`FrameSource` protocol, `WebcamSource` with per-platform backend selection, `VideoFileSource`
for CI, `WebcamGazeTracker` with guaranteed camera release.

### Phase 5 — calibration
The highest-risk numerical work: replacing the pickled scikit-learn pipeline with raw
coefficients in NumPy, so `sklearn` becomes fit-time only.

**Mutation-check before claiming equivalence.** A wrong polynomial does not crash — it
returns a smooth, believable surface in the wrong place. Test that the fixture catches:
transposed coefficients, wrong `PolynomialFeatures` term ordering, swapped x/y coefficient
sets, and a degree mismatch. Plus the `.pkl` migration path.

### Phase 6 — assets and CLI
Registry, cache download with SHA-256 verification, and the split policy: MediaPipe landmarker
auto-downloads; the gaze weights print instructions and stop. All six CLI commands.

### Phase 7 — server extra
Gaze-only WebSocket server, wire format byte-identical. The game's gesture handling stays in
the game repo as a thin wrapper.

### Phase 8 — tests, docs, examples
≥80% coverage on non-hardware paths. **Before deleting any milestone script, run
`milestone6_test_accuracy.py` on unmodified code and record the numbers in the audit.** Port
it to `focusedgaze accuracy` rather than losing it.

### Phase 9 — packaging verification
Clean-venv install of the wheel, not the dev venv. Wheel content audit (automated in
`release.yml`). Resolve the 3.12/3.13 wheel question with real runs.

### Phase 10 — release
TestPyPI first, install from it into a clean venv, then tag and publish.

### Phase 11 — close the loop
`game-gaze` depends on the published package; delete the duplicated modules.

---

## Part E — Waiting on the human

Do not close the phases these block. Remind once per phase gate, not every turn.

1. **Tier 2 fixture** — blocks the Phase 2 gate.
2. **`milestone6` accuracy baseline** — window closes at Phase 8.
3. **NOTICE email decision (A3)** — resolved: option (b).
4. **PyPI / TestPyPI project state (Part B)** — blocks the release workflow being correct.
