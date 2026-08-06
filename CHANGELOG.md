# Changelog

All notable changes to focusedgaze are recorded here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Engineering decisions and the reasoning behind them live in `MIGRATION_AUDIT.md`;
this file records what changed, per phase.

## [Unreleased]

### Added
- **Phase 1:** package skeleton, `pyproject.toml` (dynamic version, dependency
  ranges, six extras, `py.typed`), MIT `LICENSE`, and a `NOTICE` documenting the
  Gaze360 non-commercial restriction and the never-distribute policy for model
  weights.
- **Phase 1:** two-tier golden-file regression harness recorded against the
  *unmodified* legacy pipeline. Tier 1 is numeric and committed (calibration
  polynomial, One Euro filter, positioning gate); Tier 2 covers frames →
  (pitch, yaw), is gitignored, and is regenerable by anyone from their own
  recording.
- **Phase 2 (in progress):** `core/filters.py` (`OneEuroFilter`,
  `OneEuroFilter2D`) and `core/positioning.py` (`PositioningGate`,
  `FocalCalibration`, `PositioningConfig`, `PositioningStatus`), both proven
  equivalent to the legacy implementation within 1e-9.
- **Phase 3:** frozen configuration dataclasses (`CameraConfig`, `FilterConfig`,
  `LandmarkConfig`, `ModelConfig`, `PositioningConfig`, `RuntimeConfig` and the
  `GazeConfig` composite), the result types (`GazeResult`, `GazeStatus`) and the
  exception tree rooted at `GazeError`. Every default is pinned by
  `test_config.py` against the legacy value it replaces, each carrying the
  `file:line` it was harvested from, and written out literally rather than read
  back from the module so the test cannot agree with itself. Sections load from
  TOML or JSON, reject unknown keys rather than ignoring them, and are frozen
  deeply enough that tuple fields cannot be mutated in place. `mypy --strict`
  clean.
- **Phase 5:** calibration. `CalibrationProfile` replaces the legacy pickle with
  a versioned JSON format holding an explicit exponent table and raw
  coefficients, so applying a profile is **pure NumPy**: no scikit-learn at
  runtime, no version-fragile estimator objects, and no arbitrary-code execution
  on load. `fit_calibration` and `robust_fit_samples` do the fitting,
  `CalibrationCollector` gathers samples, and `migrate_pickle` converts an old
  profile once.
- **Phase 6:** asset registry and cache downloader. SHA-256 verified, resumable
  after a truncated transfer, and enforcing the licence split: the MediaPipe
  landmarker auto-downloads, the gaze weights print instructions and stop. Both
  test files are network-free.
- **Phase 7:** the WebSocket wire format documented from the source before
  extraction (`docs/wire_format.md`), plus the design decisions the
  reconnaissance forced. **No server code has landed**; `server/websocket.py`
  is still a stub.
- CI and release workflows with PyPI Trusted Publishing, configured for the real
  owner and repository. TestPyPI is a required predecessor job, and the build
  fails if any distribution contains model weights, calibration profiles or test
  fixtures.
- `STANDING_BRIEF.md`, the reference for the remainder of the migration.
- `.mailmap`, which normalises the author name across commits without rewriting
  history.
- `.gitattributes`, line endings normalised to LF in the repository.

### Changed
- **Behaviour change, deliberate:** `ModelAsset` now judges a filename by the
  same rule on every platform. It was `Path(filename).name != filename`, and
  `Path` means `WindowsPath` on Windows and `PosixPath` on Linux, so one
  registry entry meant two different things depending on who read it: `a\b.bin`
  was rejected on Windows and accepted on Linux. That divergence is what turned
  CI red for five pushes. Both separators, a drive-letter or NTFS-stream colon,
  and both spellings of `..` are now rejected everywhere. The sweep that
  followed found two further holes in the same check, neither previously
  exercised by any test: a bare `..` was accepted on **both** platforms, and
  `C:foo.bin` was accepted on Linux while escaping to another drive when joined
  on Windows. No shipped registry entry is affected. `MIGRATION_AUDIT.md` §42.5
  to §42.9.
- **Behaviour change, deliberate:** the positioning gate no longer reads its
  focal length from a relative path at construction. It is now an explicit
  `FocalCalibration` argument. The legacy behaviour meant identical landmarks
  produced **117.4 cm or 121.2 cm depending on the process working directory**.
  a 3.8 cm swing, larger than the system's entire accuracy budget. Pinned by
  `test_result_does_not_depend_on_working_directory`.

### Fixed
- Golden fixtures now pin both focal branches (measured and assumed-HFOV
  fallback), the non-monotonic-timestamp path in the filter, and the degenerate
  geometry case. Previously only the branch reachable when a config file
  happened to exist was covered.
- Nine lint errors (import ordering, unsorted `__slots__` and `__all__`,
  `Sequence` from `collections.abc`, redundant quoted annotations). All
  behaviour-neutral; the golden tests were re-run afterwards rather than
  assumed.
- **CI ordering:** `Type-check`, the bare-venv import check and `Test` now run
  even when `Lint` fails. A hard-failing first step was hiding all three, so a
  single unsorted `__slots__` concealed every check that mattered, and fixing
  only the lint would have produced a green run while `mypy --strict` had still
  never executed.

### Verified
- **Linux wheel availability resolved.** `mediapipe` and `opencv-python`
  install and run on Linux across Python **3.12, 3.13 and 3.14** (CI run 2).
  `requires-python = ">=3.12"` and the 3.12/3.13 classifiers are now tested
  rather than inferred, closing the Phase 9 wheel task early.
- **Phase 5 calibration is numerically verified.** `tests/test_calibration_profile.py`
  replays all 169 recorded cases through the pure-NumPy `apply()`: worst drift
  **0.000e+00**, bit-for-bit with the legacy scikit-learn pipeline rather than
  merely within the 1e-9 tolerance. Term ordering is checked against a real
  `PolynomialFeatures` for degrees 1 to 8. All four required mutations are
  caught — transposed coefficients (8.318e-01), wrong term ordering (1.000e+00),
  swapped x/y coefficient sets (1.000e+00), degree mismatch (3.138e-02) — and
  the tests were themselves shown to fail against a deliberately broken
  implementation, including an `apply()` perturbed by only 1e-8. The
  `_check_powers_match_sklearn` guard, whose body had never executed under any
  test, is now covered. `MIGRATION_AUDIT.md` §43.
- **`mypy --strict` passes** on all three versions, the first execution.
- **D8 bare-import guarantee holds:** a venv with no ONNX provider, no
  `websockets` and no `scikit-learn` imports the package cleanly.
- **Test suite passed on Linux at `49a8f3d`**, a platform the golden fixtures
  were never recorded on. **This no longer holds.** The `Test` step has failed
  on all three Python versions on every push since, starting with `5df7ac1`,
  which is where the Phase 3/5/6 batch landed. `Lint`, `Type-check` and the
  bare-import check are still green on Linux; the failure is confined to the
  suite. **Diagnosed, fixed and verified green:** a single parametrized case,
  `test_filename_must_be_a_bare_name[a\b.bin]`, failing on a platform assumption
  in the asset registry's filename validator. CI now passes on 3.12, 3.13 and
  3.14 — 214 passed, 5 skipped, 78% coverage, matching Windows exactly. See the
  behaviour-change entry above and `MIGRATION_AUDIT.md` §42.
- Coverage baseline **94%** across the two extracted modules (`filters.py`,
  `positioning.py`); 87% reported overall, inflated by empty stub modules and
  deflated by an untested CLI banner. Now **78%** overall across a much larger
  package; see the Phase 5 entry.

### Security / privacy
- History rewritten to remove absolute machine paths, personal email addresses
  and a characterisation of a third party's conduct from every commit. Verified
  by a gate covering every blob and every commit message. See
  `MIGRATION_AUDIT.md` §21–26 for the full record, including the two failed
  attempts and what each one taught.

## [0.0.0]
- Name reservation placeholder. No functionality.
