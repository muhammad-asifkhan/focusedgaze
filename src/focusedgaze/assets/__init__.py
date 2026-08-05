"""Model asset registry and download.

Model weights are not shipped inside the wheel: the distribution targets under
5 MB and the gaze model alone is roughly 91 MB, and the licence position in
NOTICE forbids redistributing that one at any size. This subpackage is how the
files get to the machine that needs them.

    from focusedgaze.assets import ensure
    landmarker = ensure("face_landmarker")     # downloads, digest-verified
    gaze = ensure("gaze_model")                # raises with instructions

Two assets, opposite policies, for a licence reason set out in
:mod:`focusedgaze.assets.registry` and in NOTICE. The MediaPipe landmarker is
Apache-2.0 and downloads automatically; the Gaze360-derived gaze model is never
fetched and never mirrored, and asking for it raises ``ModelNotFoundError``
carrying the instructions. Nothing here prints: the CLI renders
:class:`~focusedgaze.assets.download.AssetReport` values from
:func:`~focusedgaze.assets.download.ensure_all`.

Set ``FOCUSEDGAZE_MODEL_DIR`` to a directory of model files to bypass the cache
entirely and guarantee no network access.
"""

from __future__ import annotations

from .download import (
    AssetReport,
    RemoteStream,
    Transport,
    build_request,
    download,
    ensure,
    ensure_all,
    urllib_transport,
)
from .registry import (
    FACE_LANDMARKER,
    GAZE_MODEL,
    GAZE_WEIGHTS,
    MODEL_DIR_ENV,
    REGISTRY,
    ModelAsset,
    asset_path,
    cache_dir,
    get_asset,
    model_dir,
    model_dir_override,
    runtime_assets,
    sha256_file,
)

__all__ = [
    "FACE_LANDMARKER",
    "GAZE_MODEL",
    "GAZE_WEIGHTS",
    "MODEL_DIR_ENV",
    "REGISTRY",
    "AssetReport",
    "ModelAsset",
    "RemoteStream",
    "Transport",
    "asset_path",
    "build_request",
    "cache_dir",
    "download",
    "ensure",
    "ensure_all",
    "get_asset",
    "model_dir",
    "model_dir_override",
    "runtime_assets",
    "sha256_file",
    "urllib_transport",
]
