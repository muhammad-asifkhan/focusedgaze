"""The registry's data, its invariants, and where it says files live.

The point of most of these is not that the current values are right. It is that
the *policy* cannot be relaxed by accident: the licence split lives in a
dataclass whose constructor refuses the combinations that would break it, and
these tests fail if someone gives the gaze model a URL or gives the landmarker
one without a digest to check it against.

Nothing here touches the network. An autouse fixture makes that structural
rather than aspirational.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
from pathlib import PurePosixPath, PureWindowsPath

import platformdirs
import pytest

from focusedgaze.assets import registry
from focusedgaze.assets.registry import (
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
from focusedgaze.exceptions import ConfigError

LEGACY_DIR_ENV = "FOCUSEDGAZE_LEGACY_DIR"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing in this module may open a socket."""
    import urllib.request

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a test attempted a network call")

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)


def _minimal(**overrides: object) -> dict[str, object]:
    """The smallest valid manual asset, for mutating one field at a time."""
    base: dict[str, object] = {
        "name": "example",
        "filename": "example.bin",
        "licence": "MIT",
        "licence_url": "https://example.invalid/licence",
        "auto_download": False,
        "instructions": "get it yourself",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# The policy. These are the tests that must never be "fixed" by relaxing them.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("asset", [GAZE_MODEL, GAZE_WEIGHTS], ids=lambda a: a.name)
def test_gaze_assets_are_never_auto_downloaded_and_carry_no_url(asset: ModelAsset) -> None:
    """The Gaze360 licence position, expressed as data.

    Both files derive from a dataset restricted to non-commercial research. The
    package does not distribute, mirror or fetch them, so the registry entry
    holds no URL at all: there is nothing for a code path to accidentally use.
    """
    assert asset.auto_download is False
    assert asset.url is None
    assert "non-commercial" in asset.licence.lower()
    assert asset.licence_url == "https://github.com/erkil1452/gaze360"


def test_landmarker_is_auto_downloadable_and_fully_verifiable() -> None:
    """The permissive half of the split, with everything needed to check it."""
    assert FACE_LANDMARKER.auto_download is True
    assert FACE_LANDMARKER.url is not None
    assert FACE_LANDMARKER.url.startswith("https://storage.googleapis.com/mediapipe-models/")
    assert FACE_LANDMARKER.licence == "Apache-2.0"
    assert FACE_LANDMARKER.sha256 is not None
    assert FACE_LANDMARKER.size_bytes == 3758596
    assert FACE_LANDMARKER.source, "a URL in a downloader needs recorded provenance"


def test_the_landmarker_url_is_version_pinned_not_latest() -> None:
    """A digest pinned against a mutable alias breaks for everyone at once.

    Google publishes the bundle under both `/1/` and `/latest/`. They currently
    serve identical bytes, but `latest` is by definition free to change, and the
    day it does every user's download would fail its digest check simultaneously.
    """
    assert FACE_LANDMARKER.url is not None
    assert "/float16/1/" in FACE_LANDMARKER.url
    assert "/latest/" not in FACE_LANDMARKER.url


def test_an_asset_that_is_not_auto_downloaded_may_not_carry_a_url() -> None:
    """The guard rail against relaxing the policy one field at a time.

    Giving the gaze model a URL is the first half of the change that would make
    it downloadable, and this refuses it on its own.
    """
    with pytest.raises(ConfigError, match="must carry no URL"):
        ModelAsset(**_minimal(url="https://example.invalid/weights.onnx"))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "omitted", ["url", "sha256", "size_bytes"],
)
def test_auto_download_requires_everything_needed_to_verify(omitted: str) -> None:
    """Nothing is fetched that cannot be checked once it arrives."""
    fields: dict[str, object] = {
        "auto_download": True,
        "url": "https://example.invalid/a.bin",
        "sha256": "0" * 64,
        "size_bytes": 10,
    }
    del fields[omitted]
    with pytest.raises(ConfigError, match=omitted):
        ModelAsset(**_minimal(**fields))  # type: ignore[arg-type]


def test_the_gaze_model_records_a_reference_digest_without_enforcing_one() -> None:
    """The ONNX graph is exported locally, so its bytes are not reproducible.

    torch and onnx versions change the serialisation, so an enforced digest
    would reject a model the user exported correctly. The measured digest is
    still recorded, because "do I have the same bytes as the reference?" is a
    question worth being able to answer.
    """
    assert GAZE_MODEL.sha256 is None
    assert GAZE_MODEL.reference_sha256 is not None
    assert len(GAZE_MODEL.reference_sha256) == 64


def test_the_pytorch_checkpoint_is_not_a_runtime_asset() -> None:
    """Inference is pure ONNX; the .pkl is only an input to `export-onnx`."""
    assert GAZE_WEIGHTS.required_at_runtime is False
    assert GAZE_WEIGHTS.filename.endswith(".pkl")
    assert GAZE_MODEL.filename.endswith(".onnx")
    names = [a.name for a in runtime_assets()]
    assert names == ["face_landmarker", "gaze_model"]


# ---------------------------------------------------------------------------
# Construction invariants.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "../escape.bin",       # traversal, a separator on every platform
        "sub/dir.bin",         # forward slash, a separator on every platform
        "a\\b.bin",            # backslash: separator on Windows, inert on POSIX
        "..\\escape.bin",      # traversal spelled the Windows way
        "..",                  # bare traversal, accepted by pathlib on BOTH platforms
        ".",                   # the cache directory itself
        "C:foo.bin",           # drive-relative on Windows, inert on POSIX
        "C:/foo.bin",          # absolute on Windows
        "/abs.bin",            # absolute on POSIX
        "trailing/",           # a directory, not a file
        "stream.bin:ads",      # NTFS alternate data stream
    ],
)
def test_filename_must_be_a_bare_name(bad: str) -> None:
    """The filename is joined onto a cache directory, so it must not escape it.

    Every case here is rejected on EVERY platform, which is the guarantee that
    matters and the one this did not previously make. The registry is a single
    source read on Windows, Linux and macOS alike, and its filenames are joined
    onto a cache directory, so an entry that is a bare name for one reader and a
    path for another means two different things from one line of source.

    The original check was ``Path(filename).name != filename``, and `Path` is
    `WindowsPath` here and `PosixPath` there. It therefore accepted ``a\\b.bin``
    on Linux and rejected it on Windows, which is what turned CI red for five
    pushes (audit section 42), and it accepted ``C:foo.bin`` on Linux while that
    value escapes to another drive entirely when joined on Windows.

    It also accepted a bare ``..`` on *both* platforms, because ``Path('..').name``
    is ``'..'``. That one was never a divergence: it was a traversal hole
    everywhere, and it survived because no case in this list exercised it.
    """
    with pytest.raises(ConfigError, match="bare name"):
        ModelAsset(**_minimal(filename=bad))  # type: ignore[arg-type]


def test_the_bare_name_rule_does_not_consult_pathlib() -> None:
    """The rule must be a property of the string, not of the running platform.

    Asserted directly against `PurePosixPath` and `PureWindowsPath` rather than
    through the ambient `Path`, because a test that used `Path` would agree with
    whichever defect the platform happens to have. This is the check that would
    have caught the original bug on a Windows-only developer machine.
    """
    # Only values some pathlib calls "bare". `sub/dir.bin` and `.` are rejected
    # by both, so they prove nothing here and live in the list above instead.
    bare_to_some_pathlib = ["a\\b.bin", "C:foo.bin", ".."]
    for bad in bare_to_some_pathlib:
        posix_thinks_bare = PurePosixPath(bad).name == bad
        windows_thinks_bare = PureWindowsPath(bad).name == bad
        # At least one platform's pathlib is happy with each of these, which is
        # exactly why the rule cannot be delegated to pathlib.
        assert posix_thinks_bare or windows_thinks_bare, (
            f"{bad!r} no longer demonstrates the divergence this test is about"
        )
        with pytest.raises(ConfigError, match="bare name"):
            ModelAsset(**_minimal(filename=bad))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "good", ["face_landmarker.task", "l2cs_gaze360.onnx", "a.b.c.bin", "no-extension"]
)
def test_ordinary_filenames_are_still_accepted(good: str) -> None:
    """The control. A validator that rejected everything would pass the tests above."""
    assert ModelAsset(**_minimal(filename=good)).filename == good  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["ABC" + "0" * 61, "0" * 63, "z" * 64, ""])
def test_digests_must_be_64_lowercase_hex_characters(bad: str) -> None:
    """An uppercase or short digest would never match and would look fine."""
    with pytest.raises(ConfigError, match="hex"):
        ModelAsset(**_minimal(reference_sha256=bad))  # type: ignore[arg-type]


def test_size_must_be_positive() -> None:
    with pytest.raises(ConfigError, match="positive"):
        ModelAsset(**_minimal(size_bytes=0))  # type: ignore[arg-type]


def test_name_and_filename_are_required() -> None:
    with pytest.raises(ConfigError, match="name and a filename"):
        ModelAsset(**_minimal(name=""))  # type: ignore[arg-type]


def test_registry_keys_match_asset_names() -> None:
    assert all(key == asset.name for key, asset in REGISTRY.items())


def test_get_asset_lists_the_alternatives_when_the_name_is_wrong() -> None:
    """Reachable from a CLI argument, so the error has to be useful."""
    assert get_asset("gaze_model") is GAZE_MODEL
    with pytest.raises(ConfigError, match="known assets: face_landmarker, gaze_model"):
        get_asset("gaze-model")


# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------


def test_cache_dir_is_absolute_and_asking_does_not_create_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Reading a path must not have a side effect on the filesystem."""
    monkeypatch.setattr(platformdirs, "user_cache_dir", lambda *a, **k: str(tmp_path / "c"))
    got = cache_dir()
    assert got.is_absolute()
    assert got == tmp_path / "c" / "models"
    assert not got.exists()


def test_cache_dir_is_named_for_this_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """The appname is what keeps one user's caches apart; pin that it is passed."""
    seen: dict[str, object] = {}

    def _spy(appname: str | None = None, appauthor: object = None, **kw: object) -> str:
        seen["appname"] = appname
        seen["appauthor"] = appauthor
        return "/anywhere"

    monkeypatch.setattr(platformdirs, "user_cache_dir", _spy)
    cache_dir()
    assert seen["appname"] == "focusedgaze"
    # appauthor=False, not None: on Windows the default nests the name twice.
    assert seen["appauthor"] is False


def test_override_is_read_from_the_real_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The os.environ branch, exercised through os.environ rather than a stub."""
    monkeypatch.setenv(MODEL_DIR_ENV, str(tmp_path))
    assert model_dir_override() == tmp_path
    assert model_dir() == tmp_path
    assert asset_path(GAZE_MODEL) == tmp_path / "l2cs_gaze360.onnx"


def test_override_can_be_supplied_explicitly_instead_of_through_the_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The explicit-mapping branch. Both ways are recorded, per standing rule 3."""
    monkeypatch.delenv(MODEL_DIR_ENV, raising=False)
    assert model_dir_override({MODEL_DIR_ENV: str(tmp_path)}) == tmp_path
    # An explicit mapping wins over the real environment being unset, and an
    # explicit empty mapping wins over it being set.
    monkeypatch.setenv(MODEL_DIR_ENV, str(tmp_path / "other"))
    assert model_dir_override({}) is None


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_a_blank_override_is_the_same_as_an_unset_one(blank: str) -> None:
    """An unset shell variable expands to the empty string.

    Reading that as "the models are in the current directory" would put back
    exactly the working-directory dependence this package removed in Phase 2.
    """
    assert model_dir_override({MODEL_DIR_ENV: blank}) is None


def test_model_dir_falls_back_to_the_cache_when_the_override_is_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.delenv(MODEL_DIR_ENV, raising=False)
    monkeypatch.setattr(platformdirs, "user_cache_dir", lambda *a, **k: str(tmp_path))
    assert model_dir() == tmp_path / "models"


def test_override_expands_a_user_relative_path() -> None:
    """`~/models` is what people actually type into a shell profile."""
    got = model_dir_override({MODEL_DIR_ENV: os.path.join("~", "gazemodels")})
    assert got is not None
    assert "~" not in str(got)


# ---------------------------------------------------------------------------
# Hashing and messages.
# ---------------------------------------------------------------------------


def test_sha256_file_matches_hashlib_across_chunk_boundaries(tmp_path: pathlib.Path) -> None:
    """Streamed hashing must agree with the one-shot answer, chunk size aside."""
    body = bytes(range(256)) * 40
    path = tmp_path / "blob.bin"
    path.write_bytes(body)
    expected = hashlib.sha256(body).hexdigest()
    assert sha256_file(path) == expected
    assert sha256_file(path, chunk_size=7) == expected


def test_sha256_of_an_empty_file_is_the_empty_digest(tmp_path: pathlib.Path) -> None:
    """The loop must terminate on a zero-length read rather than spin."""
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert sha256_file(path) == hashlib.sha256(b"").hexdigest()


def test_the_gaze_message_carries_the_whole_remedy(tmp_path: pathlib.Path) -> None:
    """No print() in library code, so the instructions travel in the message."""
    message = GAZE_MODEL.not_found_message(tmp_path / "l2cs_gaze360.onnx")
    assert "does NOT download" in message
    assert "non-commercial research" in message
    assert "export-onnx" in message
    assert "L2CS-Net" in message
    assert MODEL_DIR_ENV in message
    assert "https://github.com/erkil1452/gaze360" in message
    assert "l2cs_gaze360.onnx" in message


def test_the_landmarker_message_names_the_url_it_would_have_used(
    tmp_path: pathlib.Path,
) -> None:
    """A user behind a proxy needs the URL to fetch it by hand."""
    message = FACE_LANDMARKER.not_found_message(tmp_path / "x", reason="download failed: boom")
    assert "download failed: boom" in message
    assert FACE_LANDMARKER.url is not None
    assert FACE_LANDMARKER.url in message


def test_no_registry_message_offers_a_gaze_download() -> None:
    """A stray URL in prose is as good as one in the data, to a hurried reader."""
    for asset in (GAZE_MODEL, GAZE_WEIGHTS):
        assert "http://" not in asset.instructions
        for line in asset.instructions.splitlines():
            if "https://" in line:
                assert "github.com" in line, f"unexpected download link: {line}"


# ---------------------------------------------------------------------------
# Reality check. Skips cleanly, and says why.
# ---------------------------------------------------------------------------


def test_recorded_landmarker_digest_matches_the_real_file() -> None:
    """The recorded digest against an actual copy of the file on disk.

    Everything else here checks the registry against itself. This is the only
    test that checks it against reality, so it is worth having even though it
    can only run where a copy exists. It uses the same environment variable the
    golden harness uses and skips with a reason when that is unset.
    """
    root = os.environ.get(LEGACY_DIR_ENV)
    if not root:
        pytest.skip(f"{LEGACY_DIR_ENV} unset: no local copy of face_landmarker.task to check")
    path = pathlib.Path(root) / FACE_LANDMARKER.filename
    if not path.exists():
        pytest.skip(f"{path} does not exist")
    assert path.stat().st_size == FACE_LANDMARKER.size_bytes
    assert sha256_file(path) == FACE_LANDMARKER.sha256


def test_registry_module_exports_what_it_documents() -> None:
    """__all__ drifting from the module is a silent break for `from ... import *`."""
    for name in registry.__all__:
        assert hasattr(registry, name), name
