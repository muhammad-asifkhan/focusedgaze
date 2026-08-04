# Examples

The plan calls for four runnable scripts: `minimal.py`, `video_file.py`, `callback.py` and
`headless_calibration.py`. Every one of them needs `GazeEstimator`, `WebcamGazeTracker` or
the calibration profile, and none of those exist yet. They arrive with Phases 4 and 6.

They are not written as stubs on purpose. An example that fails on paste is worse than no
example, because it teaches the reader that the documentation cannot be trusted, and that
lesson sticks.

What is here is what runs.

| Script | Needs | Status |
|---|---|---|
| `filter_demo.py` | Nothing beyond the base install | Runs today |
| `minimal.py` | `WebcamGazeTracker` | Phase 4 |
| `video_file.py` | `VideoFileSource`, `GazeEstimator` | Phase 4 |
| `callback.py` | The callback API | Phase 4 |
| `headless_calibration.py` | `CalibrationProfile` and the fitter | Phase 5 and 6 |

Run the one that exists:

```bash
python examples/filter_demo.py
```
