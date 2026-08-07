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

9. **Document everything, as you do it.** Every phase, decision, failure and fix gets
   written down in the same turn it happens — not batched, not deferred to the end.

   - `MIGRATION_AUDIT.md` gains a section for every phase gate, every deviation, every bug
     found, and every decision made. Include what was tried, what failed, why, and what the
     fix was. **Failures are the most valuable entries; never quietly drop one that got
     resolved.**
   - Every commit message explains **why**, not just what. Keep the standard of the
     existing ones.
   - Every new module gets a docstring explaining its purpose and anything non-obvious
     carried over from the legacy code.
   - Every non-obvious workaround gets an inline comment naming the reason — e.g. the ONNX
     tensor-order defect, or the `filter-repo` comment-syntax trap.
   - `CHANGELOG.md` gets an entry per phase.
   - This brief gets updated when a decision changes the plan.
   - Anything a future reader would need in order to not repeat a mistake goes in writing
     **before the turn ends**.

   Assume the reader is someone joining this project in six months with no context —
   including the people who wrote it.

10. **Documentation must not quote what it describes.** Writing up a removed string, a
    corruption marker, or a scrubbed phrase **reintroduces it** and defeats the check that
    looks for it. Describe the shape of the problem; never reproduce its literal text.

    This has caused three separate failures in this project: a brief section quoting the
    phrase whose line-wrapping broke a rewrite, a write-up reintroducing strings the same
    rewrite had just removed, and a corruption canary firing on the very section that
    explained the corruption.

    The same reasoning forbids a *rule* that targets a marker a *check* searches for — the
    detector would scrub its own evidence. See `fg-replacements.README.md`.

11. **Validate by whitelist, not by library behaviour.** Anything that becomes a filename, a
    path component or a key is checked against an explicit character class of what is
    allowed, never by asking a library whether it looks acceptable. `pathlib`'s answer
    depends on the platform, so a check written through it inherits whichever defect the
    local platform has.

    The contrast is already in this repo: the profile name regex is a whitelist and was safe
    by construction, including against traversal it was never written to consider. The
    registry check delegated to `pathlib` and had three holes, only one of which any test
    found. See MIGRATION_AUDIT.md section 42.

12. **No direct commits to `main`.** All work happens on `dev`. `main` changes only through
    a merge, and only after the full gate has passed **on `dev`**.

    This changes nothing about rules 1 to 11. It changes *where* they are enforced: before
    code is visible to the team rather than after.

    **The gate, before any merge:**

    - `pytest -q -rs`, run **both** with and without `FOCUSEDGAZE_LEGACY_DIR` set. Never a
      bare run: the skip reasons are part of the result, and a skip has concealed a failing
      assertion here before.
    - `ruff check src tests`
    - `mypy --strict`
    - Push `dev` and **observe CI green on `dev` itself**. Not inferred from a local run.
      Section 42 is the whole reason this clause is written that way: the local suite was
      green for five consecutive red CI runs, and a test suite cannot detect a
      platform assumption on the platform that shares it.

    **The merge:**

    ```
    git checkout main && git pull
    git merge dev --no-ff
    git push origin main
    ```

    `--no-ff` is required. A fast-forward folds `dev`'s history into `main` with no merge
    commit, and rule 9 depends on being able to see what arrived together and when.

    **After merging, confirm CI green on `main` separately.** A green `dev` does not
    guarantee a green `main`: `main` may have moved, and the merge result is a commit that
    has never been tested until it exists.

    `ci.yml` therefore triggers on pushes to **both** branches. With `main` alone a push to
    `dev` ran nothing and this gate could not be satisfied at all.

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

### Phase 7 - server extra
Gaze-only WebSocket server, wire format byte-identical. The game's gesture handling stays in
the game repo as a thin wrapper.

**Revised 2026-08-05 after reconnaissance (audit §39, contract in `docs/wire_format.md`).**
The exit criterion as originally written was **not satisfiable**: the game reads only
`type:"input"` and discards `type:"gaze"`, so a strictly gaze-only server leaves it
connected and motionless with no error anywhere. Decided:

- The SDK emits a **minimal gaze-only `input` message**: `type`, `mode`, `source`, `ok`,
  `x`, `y`, `t`, with `mode` and `source` the constant `"gaze"`. **No gesture fields.**
- The wrapper's `resolve_input` hook **replaces** that message rather than appending, or two
  `input` messages race every tick.
- `type:"gaze"` **stays** in the wire format; `gaze_test.html` remains supported and is what
  makes bare `focusedgaze serve` testable without the game.
- Preserve the two pacing rules: `input` every tick, `gaze` only when the reading changes.
- The camera lease moves into the SDK as `pause()`/`resume()`, carrying the measured 4.0 s
  wait and both 0.3 s driver sleeps. **Those are empirical; do not round them.**
- Delete the `os.chdir()`; do not delegate it. Four cwd-relative lookups ride on it.
- Two latent bugs are recorded in §39.4 and deliberately **not** fixed during extraction.
  Fix them after, in their own commits, per rule 4.

**Still unproven:** that the game plays against the minimal message. It is inferred from
reading the client, not executed. Confirm against a stub server before relying on it, and
watch the sequence counters specifically (Q7-2 is contingent on that result).

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
