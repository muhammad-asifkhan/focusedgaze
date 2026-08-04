# Installing focusedgaze

> **Status: in development (0.0.0).** The published `0.0.0` is a name placeholder with no
> working pipeline in it. The commands below are correct for when the package publishes
> properly, but they have not been verified end to end, because there is nothing to install
> yet. What currently works is described in [What actually installs today](#what-actually-installs-today).

## Pick an ONNX provider first

This is the step people get wrong, so it comes first.

focusedgaze does not choose an ONNX execution provider for you. The right choice depends on
your hardware, and a library that guesses will pick wrong on somebody's machine and be
hard to override. So the base package deliberately ships without one, and you add the extra
that matches your machine:

```bash
pip install focusedgaze[directml]   # Windows, any DirectX 12 GPU (including integrated)
pip install focusedgaze[cuda]       # NVIDIA, via onnxruntime-gpu
pip install focusedgaze[cpu]        # anywhere, no GPU needed
```

If you are on Windows and unsure, use `[directml]`. It works on integrated graphics as well
as discrete cards, and it does not need a CUDA toolkit.

The choice is worth getting right. On the reference machine, an RTX 4060 running Windows,
the gaze model takes about **15 ms** per frame through DirectML and about **104 ms** on CPU.
End to end that is roughly **30 frames per second** against roughly **5**. CPU is fine for
processing a recorded video and frustrating for live interaction.

### If you install no provider at all

The base package still imports. That is checked on every CI run, so it will not break
quietly:

```bash
pip install focusedgaze
python -c "import focusedgaze; print(focusedgaze.__version__)"
```

Loading a gaze model without a provider is intended to raise a named focusedgaze exception
telling you which extra to install, rather than a bare `ImportError` from somewhere in the
dependency tree. That is the agreed design. It is not implemented yet, because the model
loader is still a stub, so today there is nothing to raise it.

## The other extras

| Extra | Brings | You need it when |
|---|---|---|
| `calibration` | scikit-learn, scipy | Fitting a calibration profile. Applying one will not need it |
| `server` | websockets | Running the WebSocket bridge that feeds a browser |
| `export` | torch, torchvision, onnx | Converting the PyTorch gaze weights to ONNX, once |

Combine them the usual way:

```bash
pip install "focusedgaze[directml,calibration]"
```

`calibration` is split deliberately. Fitting a profile is a scikit-learn job, but applying
one is arithmetic on stored coefficients, so an application that ships a profile and only
reads it should not have to install scikit-learn. That split is a Phase 5 design commitment
and is not in place yet.

`export` does not declare `l2cs`, and cannot. `l2cs` installs only from a git URL, and PyPI
rejects direct-URL dependencies in any dependency list, including extras. Declaring it would
make the wheel unpublishable. The export command prints the manual install line instead.

## Python and platform

Python **3.12–3.14**. Tested on all three in CI, on Linux.

| Platform | Status |
|---|---|
| Windows 10/11 | Tested |
| Linux | Structurally supported, untested with a camera |
| macOS | Structurally supported, untested |

CI proves the pure computation runs and produces identical numbers on Linux, which is real
evidence the platform abstractions are not imaginary. Nobody has pointed a camera at a
non-Windows machine, so only Windows is claimed.

## Model weights

Two model files, treated completely differently, and the difference is legal rather than
technical.

**The MediaPipe face landmarker downloads automatically.** It is Apache 2.0, published by
Google, and fetched from Google's own hosting with SHA-256 verification.

```bash
focusedgaze download-models
```

**The gaze model does not, and will not.** That command deliberately stops and prints
instructions instead. The weights are an L2CS-Net model trained on the **Gaze360** dataset,
whose authors restrict use of the dataset and code to non-commercial research. Weights
trained on it are normally treated as a derived work carrying the same restriction, so
focusedgaze does not distribute them, does not mirror them, and will not fetch them on your
behalf.

You obtain them yourself from the official L2CS-Net distribution, then convert locally:

```bash
focusedgaze export-onnx --weights path/to/L2CSNet_gaze360.pkl
```

There is a second reason beyond licensing. The commonly circulated download for these
weights is a third-party mirror unconnected to the model's authors, which is unsuitable for
an automatic downloader regardless of licence: its provenance is not established and it
carries no availability guarantee.

Read [NOTICE](../NOTICE) before using focusedgaze commercially. This is a conservative
reading of the upstream terms, not legal advice.

## What actually installs today

`0.0.0` is a placeholder. `pip install focusedgaze` will fetch it and it will import, but
the pipeline is not in it. Neither `download-models` nor `export-onnx` exists yet. Both
arrive in Phase 6.

To work with what is implemented, install from a checkout:

```bash
git clone https://github.com/muhammad-asifkhan/focusedgaze
cd focusedgaze
pip install -e ".[cpu,calibration,dev]"
```

That gives you the One Euro filter and the positioning gate, both extracted and pinned
against the original implementation, plus the test suite. See [usage.md](usage.md).

For reproducing a specific numeric result rather than just running the code, use the pinned
set in [`requirements-dev.txt`](../requirements-dev.txt). It records exactly which versions
produced the current fixtures, and explains where the reference environment differs.
