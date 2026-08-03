# Changelog

All notable changes to focusedgaze are recorded here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Engineering decisions and the reasoning behind them live in `MIGRATION_AUDIT.md`;
this file records what changed, per phase.

## [Unreleased]

### Added
- **Phase 1** — package skeleton, `pyproject.toml` (dynamic version, dependency
  ranges, six extras, `py.typed`), MIT `LICENSE`, and a `NOTICE` documenting the
  Gaze360 non-commercial restriction and the never-distribute policy for model
  weights.
- **Phase 1** — two-tier golden-file regression harness recorded against the
  *unmodified* legacy pipeline. Tier 1 is numeric and committed (calibration
  polynomial, One Euro filter, positioning gate); Tier 2 covers frames →
  (pitch, yaw), is gitignored, and is regenerable by anyone from their own
  recording.
- **Phase 2 (in progress)** — `core/filters.py` (`OneEuroFilter`,
  `OneEuroFilter2D`) and `core/positioning.py` (`PositioningGate`,
  `FocalCalibration`, `PositioningConfig`, `PositioningStatus`), both proven
  equivalent to the legacy implementation within 1e-9.
- CI and release workflows with PyPI Trusted Publishing, configured for the real
  owner and repository. TestPyPI is a required predecessor job, and the build
  fails if any distribution contains model weights, calibration profiles or test
  fixtures.
- `STANDING_BRIEF.md` — the reference for the remainder of the migration.
- `.mailmap` — normalises the author name across commits without rewriting
  history.
- `.gitattributes` — line endings normalised to LF in the repository.

### Changed
- **Behaviour change, deliberate:** the positioning gate no longer reads its
  focal length from a relative path at construction. It is now an explicit
  `FocalCalibration` argument. The legacy behaviour meant identical landmarks
  produced **117.4 cm or 121.2 cm depending on the process working directory** —
  a 3.8 cm swing, larger than the system's entire accuracy budget. Pinned by
  `test_result_does_not_depend_on_working_directory`.

### Fixed
- Golden fixtures now pin both focal branches (measured and assumed-HFOV
  fallback), the non-monotonic-timestamp path in the filter, and the degenerate
  geometry case. Previously only the branch reachable when a config file
  happened to exist was covered.

### Security / privacy
- History rewritten to remove absolute machine paths, personal email addresses
  and a characterisation of a third party's conduct from every commit. Verified
  by a gate covering every blob and every commit message. See
  `MIGRATION_AUDIT.md` §21–26 for the full record, including the two failed
  attempts and what each one taught.

## [0.0.0]
- Name reservation placeholder. No functionality.
