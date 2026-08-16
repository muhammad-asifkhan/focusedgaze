# Changelog

All notable changes to focusedgaze are recorded here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Engineering decisions and the reasoning behind them live in `MIGRATION_AUDIT.md`;
this file records what changed, per phase.

## [0.1.1]

### Fixed
- **Every documentation link on the PyPI project page.** All eleven links in the
  0.1.0 description are repository-relative (`docs/getting-started.md`,
  `NOTICE`, `LICENSE`, …). PyPI serves the description on `pypi.org`, where
  those resolve against `pypi.org` and 404 — they only ever worked when read on
  GitHub. They were rewritten to absolute URLs in `8a079bf`, but that commit
  landed *after* the `v0.1.0` tag, and PyPI does not permit re-uploading a
  version, so the fix could not reach 0.1.0 and this release is what carries it.
  A test now asserts that no link in `README.md` is relative, because the
  failure is invisible from inside the repository: on GitHub the broken form
  renders perfectly.
- **`Documentation` and `Changelog` added to `[project.urls]`**, which is the
  PyPI sidebar. Those links are the only ones on the page that do not depend on
  the description rendering, so they are worth having independently.

### Changed
- **The default gaze backend is now `intel`, not `l2cs`.** This is a bug fix
  wearing a default's clothes. The L2CS weights derive from Gaze360 and this
  package may not distribute, mirror or fetch them, by a licence decision that is
  settled and correct. Defaulting to them meant a fresh install had no working
  first run at all: `demo`, `serve`, `calibrate`, `accuracy` and `check` all
  stopped at a licence notice instructing the user to obtain a 91 MB checkpoint
  and convert it. Every example in `docs/getting-started.md` passed
  `--backend intel` explicitly, which is why the dead end was invisible to
  anyone already using the project. The Apache-2.0 backend downloads unprompted
  and measured better besides — 1.43 cm against 1.96 cm, 2.0 ms against 141.7 ms.
  **L2CS is unchanged and fully supported behind `--backend l2cs`;** existing
  users who pass the flag, or who set `ModelConfig.backend`, see no difference.

### Added
- **`focusedgaze serve` now serves the camera.** It previously refused to run
  without `--replay`, printing that the live source "is Phase 2 and is not
  implemented yet" — a message that had been stale since Phase 2 landed. Every
  piece existed (`WebcamGazeTracker` streams results, `GazeSource` is three
  methods); nothing joined them, so a browser or Electron client could only ever
  be driven by a recorded JSON file. New `focusedgaze.server.LiveGazeSource` is
  that join, and `serve` uses it by default. `--replay` is unchanged and still
  needs no profile.

  Capture runs on its own thread publishing into a one-deep slot, because
  `latest()` is called on the broadcaster tick and must not block — reading a
  frame waits ~33 ms for the camera, which would couple the send rate to the
  capture rate and stall the event loop for every client at once. A reading
  older than `STALE_AFTER_S` (0.1 s) is reported `ok=False` rather than served
  again: a slot that keeps its last good value turns a dead capture thread into
  a frozen cursor that nothing reports as broken. `pause()`/`resume()` close and
  rebuild the tracker, so giving up the camera actually releases the device.

  `serve` gained `--profile`, and requires it for the camera: without a
  calibration the pipeline has only raw angles, and the wire format has no field
  for those, so every message would carry `ok: false`. Declining to start beats
  serving a feed that can never report a position.
- **Profiles record the backend they were calibrated against**, and using one
  with the other backend is now refused instead of silently producing wrong
  coordinates. The two models do not report the same angles for the same eye, so
  a profile applied across them evaluates cleanly and lands somewhere else — the
  class of failure this project keeps encountering. Checked in
  `GazeEstimator.__init__`, where a profile and a model first meet, so the
  library gets the same protection as the CLI. Profiles written before this
  release carry no stamp; those **warn** rather than fail, because they are
  valid for whichever backend made them and nothing can tell which.
- **`FOCUSEDGAZE_BACKEND` sets the default backend for the CLI**, so choosing
  the non-default one no longer costs a `--backend` on every invocation.
  Precedence is flag, then variable, then the declared default; an unrecognised
  value is refused by name rather than ignored, because a bad value there is
  invisible on the command line. Read by `cli.py` and nowhere else: `config.py`
  reads no environment, so `GazeConfig()` keeps meaning exactly what it says
  rather than depending on the shell that launched the process.
- **A refused profile now names one that would work.** When the active or
  requested profile belongs to the other backend and a profile for the selected
  backend already exists, `check`, `demo` and `accuracy` name it and print the
  command to use it. Profiles with no recorded backend are never offered this
  way — they might match and might not, and naming one would be a guess dressed
  as an answer.
- **`profiles_for_backend()`** in `calibration.profile`, which is how that
  lookup is done.
- **`focusedgaze check` verifies the inference runtime for the selected
  backend.** It previously checked `onnxruntime` unconditionally, which meant an
  install with the `intel` extra and no ONNX provider was reported as broken
  when it was fine, and an install with the Intel *models* and no OpenVINO was
  reported as healthy when it would raise on the first frame. `focusedgaze
  setup` gained the same check on its Intel path, which previously skipped it
  entirely.

## [0.1.0]

The first release that does anything. `0.0.0` on PyPI is a placeholder — a 21 KB
wheel in which almost every module is a stub — and it cannot be replaced, because
PyPI never permits re-uploading a version.

### Added
- **Interactive calibration.** `calibration/screen.py` draws the dot that nothing
  was drawing. `focusedgaze calibrate` previously refused to run, blaming a phase
  that had already shipped, so no profile could be produced and the library could
  not emit screen coordinates at all. Two collection modes: a smooth-pursuit
  sweep, and a dwell grid (`--grid`) for slow machines, where a dot redrawn once
  per pipeline iteration steps rather than glides.
- **`focusedgaze.control`.** Dwell selection with hysteresis, blink grace,
  re-arming and fixation averaging — the layer between gaze coordinates and an
  application. Pure and clock-injected, so it is tested without hardware.
- **A second, redistributable gaze backend.** Intel's `gaze-estimation-adas-0002`
  (Apache-2.0), selected with `--backend intel`. 7.5 MB against 91 MB, and
  measured at **2.0 ms against 141.7 ms** for L2CS on the same machine. It is
  fetched automatically and digest-verified; the L2CS weights cannot be, because
  the Gaze360 licence names models trained on the dataset as covered derivative
  works and forbids distribution. L2CS remains the default so no existing
  profile, fixture or measurement changes meaning. *(Superseded in
  [Unreleased]: that default left fresh installs with no working first run.)*
- **`GazeEstimator.recentre()`** — a session offset measured from one centre dot.
  Across five runs the whole mapping shifted between sessions by −0.25 to +0.34 of
  screen height, twice on an identical profile minutes apart.
- **Distance awareness.** Profiles record the distance they were collected at,
  the pre-flight shows it live, and `compensate_distance` (off by default)
  rescales for the difference.
- **`focusedgaze setup`**, including `--onnx` to install a graph somebody else
  converted, validated by loading and running it before it is copied.
- [`docs/getting-started.md`](docs/getting-started.md).

### Fixed
- **The positioning gate accepted every state it existed to reject.**
  `OUT_OF_RANGE` and `OFF_CENTER` readings carry gaze angles, so a check for "has
  an angle" passed them. An accuracy run was collected at 69.4 cm against a 65 cm
  limit. The gate was also blind while calibrating, where it matters most.
- **Out-of-zone samples were being trained on**, contradicting the docstring that
  claimed they were dropped for free.
- **A crash on tracking loss.** `FaceLandmarker.reset()` rewound the frame
  counter, so regaining a lost face replayed a timestamp MediaPipe had already
  seen and raised `ValueError: Input timestamp must be monotonically increasing`,
  uncaught. It killed two of three calibration attempts.
- **Eye crops ignored head roll** on the Intel backend. Measured across 12.6° of
  tilt: horizontal gain 0.98 → 0.48, error 1.43 cm → 7.90 cm.
- **`export-onnx` wrote outside the directory the runtime reads**, so following
  the documented instructions left `check` still reporting the model missing.
- **`apply()` clamped**, making the unclamped diagnostic inert. Now `apply_raw()`.
- **The sweep's rows landed 2-1-2** across the three screen bands, starving the
  middle third on every run by every user. Now 2-2-2.

### Known issues
- The roll correction's **sign is unverified**. It follows the Open Model Zoo
  reference, but this package derives roll from MediaPipe's transformation matrix
  rather than Open Model Zoo's head-pose network. If the two disagree, head tilt
  gets worse rather than better. To be settled in 0.1.1.
- Intel-backend accuracy is measured on one machine and one face.

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
- **Phase 2:** the gaze pipeline, complete. `core/filters.py` and
  `core/positioning.py` landed first, both proven equivalent to the legacy
  implementation within 1e-9. `core/landmarks.py` (MediaPipe wrapper and the
  smoothed square crop), `core/model.py` (ONNX session, provider selection and
  the angle decode) and `core/estimator.py` (`GazeEstimator.process`) complete
  it. The module-level `_smoothed_bbox` global is **gone**: crop smoothing is
  instance state, so two estimators in one process no longer corrupt each
  other's crops.
- **Phase 2:** `WebcamGazeTracker` and the `demo` CLI command, both of which
  were waiting on `GazeEstimator`. The CLI command set is now complete at six.
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
- **Phase 4:** the capture layer. `FrameSource` protocol (structural, so an
  application that already owns a camera can pass its own object in),
  `WebcamSource` with threaded capture and per-platform backend selection
  (MSMF on Windows, AVFoundation on macOS, V4L2 on Linux, overridable),
  `VideoFileSource`, and `FrameSequenceSource` for replaying frames already in
  memory with no codec involved. The legacy server's dedicated capture thread
  is preserved, including its deliberate frame dropping: the slot holds one
  frame and a slow consumer jumps to the freshest rather than working through a
  backlog. `WebcamGazeTracker` was gated on Phase 2 and followed with it.
- **Phase 6:** the CLI. `download-models`, `check`, `calibrate` and
  `export-onnx`; `serve` followed in Phase 7 and `demo` with Phase 2, completing
  the set at six. `check` turns most of `docs/troubleshooting.md` into one
  command: it reports a CPU-only ONNX provider, a missing or wrong model file,
  a missing or unselected calibration, a camera that will not open, and a room
  too dark for face detection, each with its remedy. Diagnosis lives in the new
  `focusedgaze.diagnostics` module so it is testable without a camera.
- **Phase 7:** the gaze-only WebSocket server, `focusedgaze.server.websocket`,
  plus the `serve` CLI command and a `ServerError` in the exception tree. Emits
  the `gaze` message with its shape preserved exactly, and a **minimal** seven-
  field `input` message (`type`, `mode`, `source`, `ok`, `x`, `y`, `t`) whose
  `mode` and `source` are the constant `"gaze"`. No gesture vocabulary enters the
  SDK. Both legacy pacing rules are preserved: `input` every tick, `gaze` only
  when the reading changes. Three hooks (`on_connect`, `on_command`,
  `resolve_input`) let the game repo's wrapper **replace** the input message with
  its full twelve-field one. The camera lease arrives as `pause()`/`resume()`,
  carrying the measured 4.0 s wait and both 0.3 s driver sleeps. The legacy
  `os.chdir` is **deleted**, not delegated, and all four relative lookups that
  rode on it now resolve explicitly; see `MIGRATION_AUDIT.md` §47.2.
- **Phase 8:** `focusedgaze accuracy`, the port of the milestone accuracy
  script, plus the `focusedgaze.accuracy` module behind it. Carries the three
  requirements audit §50 derived from *running* the original rather than reading
  it: it **refuses to print an average when any point collected no samples** and
  names which failed, reports **per-point and grouped by row and column** rather
  than a single edge average, and records the calibration profile's **digest**
  alongside screen size, drift offset and per-point sample counts. Errors are
  reported in centimetres and as a percentage of screen **width**, with the
  denominator named in the type, the JSON and the rendering.
- **Phase 8:** `focusedgaze.calibration.ui`, the smooth-pursuit routine and the
  last stub outside Phase 11. Carries the shipping numbers (1728 reference
  samples, degree 3, MAD 2.5, min_keep 60) with a test that they still agree with
  the fitter's own defaults. The sweep never lets the dot jump, because samples
  collected while the user reacquires it are labelled with a target they were not
  yet looking at, and reports **per-region coverage** against the same 3x3 grid
  the accuracy test measures on.
- CI and release workflows with PyPI Trusted Publishing, configured for the real
  owner and repository. TestPyPI is a required predecessor job, and the build
  fails if any distribution contains model weights, calibration profiles or test
  fixtures.
- `STANDING_BRIEF.md`, the reference for the remainder of the migration.
- `.mailmap`, which normalises the author name across commits without rewriting
  history.
- `.gitattributes`, line endings normalised to LF in the repository.

### Changed
- **Behaviour change, deliberate:** `migrate_pickle` now carries the legacy
  `validation_error` across instead of discarding it. Its docstring asserted the
  legacy pickle "never stored one ... so there is nothing to migrate and putting
  a number there would be a fabrication". That was false: **all nine** legacy
  calibrations in the reference tree carry the key. The claim was a belief about
  the data that the data contradicts, and the code implemented the belief.
  It mattered concretely — §50.4 identified one accuracy run's model *by* its
  stored validation error matching what the run reported, and migrating that
  profile destroyed the only field that made the identification possible. Absent
  still becomes `None` rather than a fabricated `0.0`. `MIGRATION_AUDIT.md` §51.3.
- **Behaviour change, deliberate:** valid JSON that is not an object no longer
  closes the WebSocket connection (R-11). The legacy handler called the mapping
  accessor on whatever `json.loads` returned, so `"hi"` raised `AttributeError`,
  the error escaped the read loop and the socket closed; the browser reconnected
  1.2 s later and the command was lost. Only JSON objects are dispatched now, and
  a hook that raises no longer drops the client either. `MIGRATION_AUDIT.md` §47.11.
- **Behaviour change, deliberate:** an abrupt client disconnect is logged at debug
  rather than as a traceback. Both clients reconnect forever, so a closed browser
  tab produced roughly one stack trace per second. Genuine errors still get the
  full trace. `MIGRATION_AUDIT.md` §47.5.
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
- **`focusedgaze accuracy` reproduces the recorded run's arithmetic exactly**, and
  the calibration that produced it loads and evaluates through the SDK to within
  **4.441e-16 rad** over 400 probes — 2 ULP, passing the golden tolerance by a
  factor of 2.25 million. All nine per-point figures reproduce to 2.2e-15 cm, and
  every derived statistic matches: average 3.333 cm, 9.69% of width, worst 7.8 cm
  at (5,5), centre 1.0 cm. **The sensing half is not reproducible** and is
  reported as such: the recorded run captured error magnitudes, not the
  predictions they came from, so there is nothing to replay without a person.
  `MIGRATION_AUDIT.md` §51.2.
- **The accuracy baseline is measured, and published as a RANGE.** Two runs of
  the unmodified original pipeline, same person, same machine, twenty minutes
  apart: **6.2 cm** and **3.3 cm** average over nine screen points on a 34.4 cm
  screen. The failure pattern **inverts** between them — run 1 degrades to the
  right and bottom, run 2 at the top-left, and run 2's worst point is run 1's
  best. That is a calibration-coverage artifact, not a sensor limit, so no single
  headline figure is published: quoting "3.3 cm" would have provenance and still
  be wrong. In percent of screen width, run 2 is 9.7% average and 2.9% at centre,
  broadly consistent with the inherited 8.9%. Closes an item open since Phase 0.
  `MIGRATION_AUDIT.md` §50, `docs/accuracy.md`.
- **Two defects in the legacy tooling, found by running it** and deliberately
  left unfixed because the scripts are being retired. Its held-out validation
  reported 24.4% for a model the 9-point test measured at 9.7%, because **two of
  five validation points collected no samples** and it averaged the survivors.
  And one run's summary claimed "even accuracy across the screen" while its own
  table ranged 0.6 cm to 12.0 cm, because it averages all four edges and a good
  left edge cancelled a bad right one. Both become requirements on
  `focusedgaze accuracy`: refuse to report a figure when any point collected
  nothing, and report per-point and per-quadrant rather than an edge average.
- **The extraction reproduces the legacy pipeline BIT-IDENTICALLY.** The Tier 2
  fixture replayed through both implementations in one process, on the same 60
  real frames: 60/60 bit-identical, **0/60** crop bounding boxes differing, worst
  pitch and yaw drift **0.000000e+00 rad**. Not "within the 1e-4 tolerance" — the
  94x of headroom §48 measured was not consumed at all. `MIGRATION_AUDIT.md` §49.
- **mediapipe 0.10.35 vs 1.0.0 measured on real frames**, which §32.3 recorded as
  unmeasured and explicitly declined to assume. Same 60 frames replayed under
  both interpreters: **0/60** crop boxes differ, worst drift 1.065e-06 rad, 94x
  headroom. The bbox result is the one that answers §32.3, which named the
  min/max crop geometry as the place a landmark difference would show first.
  §32.3's guess that outputs would be bit-identical was **wrong** (2/60), and is
  recorded as wrong. `MIGRATION_AUDIT.md` §48.
- **The A3 isolation test exists and has teeth.** Two estimators in one process
  do not interfere, driven hard enough to show the defect: twenty frames into one
  estimator, one frame into another, first crop compared against a clean
  reference. Plus a second test that `reset()` is local, because
  `reset_bbox_smoothing()` was global in both directions.
- **The golden harness now runs the SDK.** `_load_sdk` raised unconditionally
  from Phase 1 until now, so every golden test measured the legacy code. Opening
  it caught two harness defects that only worked because one implementation ran:
  a private-attribute poke and an unnormalised return type. `MIGRATION_AUDIT.md` §49.2.
- **The browser game plays against the minimal message, executed rather than
  inferred.** Audit §39 recorded that claim as read off the client and never run.
  The real, unmodified `input-manager.js` now runs under Node against a real
  server: 66 messages (1 `gaze`, 65 `input`), cursor driven to exactly the
  coordinates sent, `handOk` false, `gesture` empty, `pinching` false, mode and
  source `"gaze"`, and **no activation event fired at all**. That answers Q7-2:
  the SDK does not need to send explicit zero counters. `MIGRATION_AUDIT.md` §47.4.
- **Linux wheel availability resolved.** `mediapipe` and `opencv-python`
  install and run on Linux across Python **3.12, 3.13 and 3.14** (CI run 2).
  `requires-python = ">=3.12"` and the 3.12/3.13 classifiers are now tested
  rather than inferred, closing the Phase 9 wheel task early.
- **Coverage is 88%**, past the Phase 8 target of 80%. `cli.py` 94%,
  `diagnostics.py` 93%, the capture modules 93-98%.
- **The capture layer is mutation-checked.** Four defects introduced and each
  caught: a `read()` that returns the slot without waiting, a `release()` that
  does not join the capture thread, a capture loop that queues instead of
  dropping, and an `__exit__` that does not release.
- **Per-platform backend selection is tested for every platform from one
  machine.** `resolve_backend` takes the platform as an argument for that
  reason; §42 is what it cost to learn that platform behaviour checked on one
  platform is not checked.
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
