"""What model files exist, where they live, and which of them we may fetch.

Weights are never shipped inside the wheel. The gaze model alone is roughly
91 MB against a 5 MB target for the distribution, and the licence position in
NOTICE forbids redistributing it at any size. So the package carries a
*description* of each file instead: its name, its digest, its licence, and
whether this package is permitted to go and get it.

THE POLICY SPLIT, AND WHY IT IS NOT A DETAIL
--------------------------------------------
Two runtime assets, treated in opposite ways, for a legal reason rather than a
technical one:

* ``face_landmarker.task`` (3.8 MB) is Apache-2.0, published by Google on
  Google's own hosting. It downloads automatically, digest-verified.
* ``l2cs_gaze360.onnx`` (91 MB) derives from L2CS-Net trained on the Gaze360
  dataset, whose authors restrict the dataset and code to non-commercial
  research use. Weights trained on it are normally treated as a derived work
  carrying the same restriction. This package therefore does not distribute,
  mirror, or fetch them. The asset carries no URL at all, and
  :class:`ModelAsset` refuses to be constructed with ``auto_download`` set and
  no URL, so the policy cannot be flipped by editing a single boolean.

That decision is recorded in NOTICE and in MIGRATION_AUDIT.md section 4. It is
settled; this module implements it rather than revisiting it.

PYTORCH CHECKPOINT VERSUS ONNX GRAPH
------------------------------------
These are two different files and only one of them is a runtime asset. The
runtime loads ``l2cs_gaze360.onnx`` and nothing else: inference is pure ONNX
Runtime, and ``torch`` is not a runtime dependency. ``L2CSNet_gaze360.pkl`` is
the upstream PyTorch checkpoint, and it exists in this registry only because
``focusedgaze export-onnx`` consumes it to produce the ONNX graph. It is marked
``required_at_runtime=False`` so the download command ignores it.

WHY THE ONNX ASSET PINS NO DIGEST
---------------------------------
The ONNX graph is produced on the user's machine by their own torch and onnx
versions, so its bytes are not reproducible between installs. An enforced
digest would reject a legitimately exported model. The digest measured on the
development machine is recorded as ``reference_sha256``: informational, never
enforced. The landmarker is the opposite case, a fixed published artifact, so
its digest is enforced on every load.

PATHS ARE RESOLVED, NEVER GUESSED
---------------------------------
``FOCUSEDGAZE_MODEL_DIR`` points at a directory of model files and takes
precedence over the managed cache; when it is set, no network access happens
for any asset. It is deliberately *exclusive* rather than the first entry in a
search path: if it is set and the file is not there, the error says so instead
of silently loading a different copy from the cache. This project has now been
bitten three times by a value that was located rather than declared (audit
sections 1.4, 19e and 33.3), and a two-entry search path is the same shape of
trap.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import platformdirs

from ..exceptions import ConfigError

__all__ = [
    "CACHE_APP_NAME",
    "FACE_LANDMARKER",
    "GAZE_MODEL",
    "GAZE_WEIGHTS",
    "MODEL_DIR_ENV",
    "REGISTRY",
    "ModelAsset",
    "asset_path",
    "cache_dir",
    "get_asset",
    "model_dir",
    "model_dir_override",
    "runtime_assets",
    "sha256_file",
]

#: Offline override. Points at a directory that already holds the model files.
MODEL_DIR_ENV = "FOCUSEDGAZE_MODEL_DIR"

#: Application name handed to platformdirs. Kept as a constant because changing
#: it silently orphans every user's existing cache.
CACHE_APP_NAME = "focusedgaze"

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")

#: Characters that stop a filename being a single name SOMEWHERE.
#:
#: Deliberately not expressed through ``pathlib``. ``Path`` is ``WindowsPath``
#: on Windows and ``PosixPath`` on Linux, and the two disagree about what these
#: characters mean, so a check written as ``Path(f).name != f`` gives different
#: answers on different machines. This registry is ONE source read on every
#: platform and its filenames are joined onto a cache directory, so an entry
#: must not mean two things depending on who reads it.
#:
#: * ``\\`` is a separator on Windows and an ordinary character on POSIX.
#: * ``:`` makes a name drive-relative on Windows, where joining ``C:x`` onto a
#:   cache directory on another drive escapes it completely, and also opens an
#:   NTFS alternate data stream. On POSIX it is ordinary.
#: * ``/`` is a separator everywhere and was already rejected.
_NOT_A_BARE_NAME = re.compile(r"[/\\:]")

# 1 MiB. Large enough that hashing 91 MB takes a couple of hundred
# milliseconds, small enough that the file never has to fit in memory.
_HASH_CHUNK = 1 << 20


@dataclass(frozen=True)
class ModelAsset:
    """One model file: what it is, where it came from, and whether we may fetch it.

    Args:
        name: Logical key used by the API and the CLI, e.g. ``"face_landmarker"``.
        filename: Name on disk. Must be a bare filename, judged identically on
            every platform: it is joined onto the cache directory, so anything
            that is a separator, a drive prefix or a traversal *anywhere* is
            rejected *everywhere*. See :data:`_NOT_A_BARE_NAME`.
        licence: Short licence note, for display.
        licence_url: Where that licence is stated.
        auto_download: Whether this package may fetch the file unprompted.
        instructions: What a user must do when the file is absent and this
            package is not allowed to fetch it. Carried in the raised error, so
            the remedy travels with the failure instead of living only in a doc.
        url: Direct download. ``None`` for anything we refuse to fetch.
        source: Where ``url`` and ``licence`` were verified from. Provenance is
            part of the record: an unattributed URL in a downloader is exactly
            what this project declined to accept for the gaze weights.
        sha256: Enforced digest. ``None`` when no digest can honestly be
            enforced, i.e. when the artifact is produced locally.
        size_bytes: Enforced size. ``None`` under the same condition.
        reference_sha256: Digest measured on the development machine.
            Informational only, never enforced. It exists so that a user who
            wants to know whether they have the same bytes can find out.
        required_at_runtime: Whether the pipeline needs this file to run.
            ``False`` marks a build-time input such as the PyTorch checkpoint.

    Invariants are checked at construction, not at download time. An asset that
    could be fetched without a digest to check it against is a bug in this file,
    and it should fail on import rather than on somebody's first run.
    """

    name: str
    filename: str
    licence: str
    licence_url: str
    auto_download: bool
    instructions: str
    url: str | None = None
    source: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    reference_sha256: str | None = None
    required_at_runtime: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.filename:
            raise ConfigError("a model asset needs both a name and a filename")
        if self.filename in (".", "..") or _NOT_A_BARE_NAME.search(self.filename):
            raise ConfigError(
                f"{self.name}: filename {self.filename!r} must be a bare name, "
                "not a path: it is joined onto the cache directory"
            )
        for label, digest in (("sha256", self.sha256),
                              ("reference_sha256", self.reference_sha256)):
            if digest is not None and not _HEX64.match(digest):
                raise ConfigError(
                    f"{self.name}: {label} must be 64 lowercase hex characters, got {digest!r}"
                )
        if self.size_bytes is not None and self.size_bytes <= 0:
            raise ConfigError(f"{self.name}: size_bytes must be positive")
        if self.auto_download:
            # Nothing is fetched that cannot be checked afterwards. This guards
            # the licence policy as much as it guards against corruption.
            missing = [f for f, v in (("url", self.url),
                                      ("sha256", self.sha256),
                                      ("size_bytes", self.size_bytes)) if v is None]
            if missing:
                raise ConfigError(
                    f"{self.name}: auto_download requires {', '.join(missing)}; "
                    "an asset this package fetches must be verifiable"
                )
        elif self.url is not None:
            raise ConfigError(
                f"{self.name}: an asset that is not auto-downloaded must carry no URL, "
                "so that no code path can be made to fetch it by mistake"
            )

    def path_in(self, directory: str | Path) -> Path:
        """Where this asset lives inside ``directory``."""
        return Path(directory) / self.filename

    def not_found_message(self, searched: Path, *, reason: str = "") -> str:
        """The text of the failure, including what to do about it.

        Args:
            searched: The path that was looked at and found wanting.
            reason: Why it was rejected, when it existed but was not usable.

        The message names a remedy because a missing gaze model is the expected
        first-run state, not an exceptional one. ``ModelNotFoundError`` carries
        this text, so the instructions reach the user through the exception
        rather than through a ``print`` in library code.
        """
        head = (f"{self.filename} was rejected at {searched}: {reason}"
                if reason else f"{self.filename} was not found at {searched}")
        lines = [head, "", self.instructions.strip(), ""]
        if self.url is not None:
            lines += [f"Direct download: {self.url}", ""]
        lines += [
            (f"Alternatively set {MODEL_DIR_ENV} to a directory that already "
             f"contains {self.filename}."),
            f"Licence: {self.licence} ({self.licence_url})",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The registry itself.
# ---------------------------------------------------------------------------

FACE_LANDMARKER = ModelAsset(
    name="face_landmarker",
    filename="face_landmarker.task",
    licence="Apache-2.0",
    licence_url="https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker",
    auto_download=True,
    # Version-pinned at /1/ rather than /latest/. Google publishes both and they
    # currently serve identical bytes, but `latest` is a mutable alias: pinning
    # a digest against it would turn an upstream refresh into a download failure
    # for every user at once.
    url=("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
         "face_landmarker/float16/1/face_landmarker.task"),
    source=("https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker"
            " (model bundle table)"),
    sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    size_bytes=3758596,
    reference_sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    instructions=(
        "The MediaPipe face landmarker is Apache-2.0 and is fetched automatically\n"
        "from Google's own hosting. If the download failed you are offline, behind\n"
        "a proxy, or the URL has moved. Fetch the file by hand and drop it in the\n"
        "directory shown above; the digest is checked either way."
    ),
)

GAZE_MODEL = ModelAsset(
    name="gaze_model",
    filename="l2cs_gaze360.onnx",
    licence="Gaze360-derived: NON-COMMERCIAL RESEARCH USE ONLY",
    licence_url="https://github.com/erkil1452/gaze360",
    auto_download=False,
    # No URL. Deliberately. See the module docstring and NOTICE section 1.
    # No enforced digest either: this file is exported on the user's own machine,
    # so its bytes depend on their torch and onnx versions.
    reference_sha256="dfc208a38f15fa372ffb42aff160fcf20ebee6aa117fe1ce0b703c387b838465",
    instructions=(
        "focusedgaze does NOT download the gaze model, and will not.\n"
        "\n"
        "It is an L2CS-Net model trained on the Gaze360 dataset, whose authors\n"
        "restrict the dataset and the code to non-commercial research use. Weights\n"
        "trained on it are normally treated as a derived work carrying the same\n"
        "restriction, so this package does not distribute, mirror or fetch them.\n"
        "You are responsible for your own compliance with those terms.\n"
        "\n"
        "To obtain it:\n"
        "  1. Get L2CSNet_gaze360.pkl from the official L2CS-Net distribution\n"
        "     linked from https://github.com/Ahmednull/L2CS-Net\n"
        "  2. Convert it locally, once:\n"
        "       focusedgaze export-onnx --weights <path to L2CSNet_gaze360.pkl>\n"
        "\n"
        "Roughly 91 MB once exported. Read NOTICE before using it commercially."
    ),
)

GAZE_WEIGHTS = ModelAsset(
    name="gaze_weights",
    filename="L2CSNet_gaze360.pkl",
    licence="Gaze360-derived: NON-COMMERCIAL RESEARCH USE ONLY",
    licence_url="https://github.com/erkil1452/gaze360",
    auto_download=False,
    # Input to `export-onnx`, not to inference: the runtime loads the ONNX graph
    # and never imports torch. Recorded here so the export command has one place
    # to read the filename and the licence from.
    #
    # No enforced digest. The copy measured here came from the origin project's
    # setup instructions, which point at a third-party mirror rather than at the
    # L2CS-Net authors' own distribution, so this digest describes one file on
    # one machine and is not an upstream-published checksum. Presenting it as
    # authoritative would be a claim nobody has verified.
    reference_sha256="8a7f3480d868dd48261e1d59f915b0ef0bb33ea12ea00938fb2168f212080665",
    required_at_runtime=False,
    instructions=(
        "The PyTorch checkpoint is not distributed with focusedgaze and is not\n"
        "downloaded. Obtain it from the official L2CS-Net distribution linked from\n"
        "https://github.com/Ahmednull/L2CS-Net, then convert it once:\n"
        "  focusedgaze export-onnx --weights <path to L2CSNet_gaze360.pkl>\n"
        "\n"
        "Only the resulting l2cs_gaze360.onnx is needed at runtime."
    ),
)

#: Every asset, keyed by logical name. Insertion order is the display order.
REGISTRY: Mapping[str, ModelAsset] = {
    a.name: a for a in (FACE_LANDMARKER, GAZE_MODEL, GAZE_WEIGHTS)
}


def get_asset(name: str) -> ModelAsset:
    """Look an asset up by logical name.

    Raises:
        ConfigError: if the name is not registered. The message lists the names
            that are, because this is reachable from a CLI argument.
    """
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise ConfigError(f"unknown model asset {name!r}; known assets: {known}") from None


def runtime_assets() -> tuple[ModelAsset, ...]:
    """The assets the pipeline actually needs in order to run."""
    return tuple(a for a in REGISTRY.values() if a.required_at_runtime)


# ---------------------------------------------------------------------------
# Where files live.
# ---------------------------------------------------------------------------


def model_dir_override(env: Mapping[str, str] | None = None) -> Path | None:
    """The directory named by ``FOCUSEDGAZE_MODEL_DIR``, or ``None``.

    Args:
        env: Environment to read. Defaults to ``os.environ``; passing one
            explicitly keeps a caller (and a test) free of global state.

    An unset variable and one holding only whitespace are the same thing. A
    variable set to the empty string is what an unset shell variable expands to
    in a script, and reading that as "the models are in the current directory"
    would resurrect the working-directory bug this package exists to avoid.
    """
    raw = (env if env is not None else os.environ).get(MODEL_DIR_ENV, "")
    stripped = raw.strip()
    return Path(stripped).expanduser() if stripped else None


def cache_dir() -> Path:
    """The managed cache directory. Not created as a side effect of asking."""
    base = platformdirs.user_cache_dir(CACHE_APP_NAME, appauthor=False)
    return Path(base) / "models"


def model_dir(env: Mapping[str, str] | None = None) -> Path:
    """Where model files are read from: the override if set, else the cache."""
    override = model_dir_override(env)
    return override if override is not None else cache_dir()


def asset_path(asset: ModelAsset, env: Mapping[str, str] | None = None) -> Path:
    """Where ``asset`` is expected to be found right now."""
    return asset.path_in(model_dir(env))


def sha256_file(path: str | Path, *, chunk_size: int = _HASH_CHUNK) -> str:
    """SHA-256 of a file, as lowercase hex, read in chunks.

    Streamed rather than slurped: the gaze model is around 91 MB and this runs
    on whatever laptop the user has.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
