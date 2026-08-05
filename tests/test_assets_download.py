"""Fetching, verifying, resuming, and refusing.

Every test here runs with no network. The transport is an injected callable and
the default one is additionally poisoned by an autouse fixture, so a regression
that reintroduced a direct ``urlopen`` would fail rather than quietly reach out
to Google from CI.

The cases that matter are the ones where something goes wrong in a way that
still looks fine:

* a body of the right length and the wrong bytes,
* a server that ignores ``Range`` and re-sends the whole file, which appends a
  duplicate prefix to a resume,
* a partial file whose first half is already corrupt, which would otherwise be
  resumed and rejected forever.
"""

from __future__ import annotations

import hashlib
import io
import pathlib
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator

import platformdirs
import pytest

# The package re-exports a FUNCTION named `download`, which shadows the submodule of the
# same name. So BOTH obvious spellings bind the function, not the module:
#   from focusedgaze.assets import download as m   -> the function
#   import focusedgaze.assets.download as m        -> also the function, because
#       `import a.b as x` resolves getattr(a, "b") first and only falls back to
#       sys.modules when that fails. Here it succeeds and returns the wrong object.
# Monkeypatching then fails with an AttributeError that names a function and reads like
# a missing symbol rather than a shadowing problem. Going through sys.modules is the one
# spelling that cannot be shadowed. See MIGRATION_AUDIT.md section 40.
import focusedgaze.assets.download  # noqa: F401  (ensures the submodule is imported)

download_module = sys.modules["focusedgaze.assets.download"]
from focusedgaze.assets.download import (
    AssetReport,
    RemoteStream,
    build_request,
    download,
    ensure,
    ensure_all,
    urllib_transport,
)
from focusedgaze.assets.registry import (
    FACE_LANDMARKER,
    GAZE_MODEL,
    MODEL_DIR_ENV,
    ModelAsset,
)
from focusedgaze.exceptions import ConfigError, ModelNotFoundError

#: 2 KiB of non-repeating-enough bytes. Small enough to be instant, large enough
#: to be split across several chunks and resumed halfway.
BODY = bytes(range(256)) * 8
HALF = len(BODY) // 2
URL = "https://example.invalid/test_asset.bin"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default transport must never run unless a test asks for it."""

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a test attempted a network call")

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)


@pytest.fixture
def cache(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Point the managed cache at a temporary directory, with no override set.

    Patches ``platformdirs`` rather than ``registry.cache_dir`` so the real path
    composition still runs; only the platform's answer is substituted.
    """
    root = tmp_path / "cache"
    monkeypatch.setattr(platformdirs, "user_cache_dir", lambda *a, **k: str(root))
    monkeypatch.delenv(MODEL_DIR_ENV, raising=False)
    return root / "models"


def make_asset(body: bytes = BODY, **overrides: object) -> ModelAsset:
    """A downloadable asset describing ``body``, for mutating one field at a time."""
    fields: dict[str, object] = {
        "name": "test_asset",
        "filename": "test_asset.bin",
        "licence": "MIT",
        "licence_url": "https://example.invalid/licence",
        "auto_download": True,
        "instructions": "fetch it by hand",
        "url": URL,
        "sha256": hashlib.sha256(body).hexdigest(),
        "size_bytes": len(body),
    }
    fields.update(overrides)
    return ModelAsset(**fields)  # type: ignore[arg-type]


class FakeServer:
    """A range-capable byte server with no socket.

    Args:
        body: The complete object.
        honour_range: When ``False``, answers a range request with 200 and the
            whole body, which is what a plain file server or a caching proxy
            does.
        stop_after: Stop yielding after this many bytes, with no error. A short
            body is not an exception; it is what a dropped connection looks like
            from the reading end.
        fail_after: Raise ``URLError`` after this many bytes.
    """

    def __init__(self, body: bytes, *, honour_range: bool = True,
                 stop_after: int | None = None, fail_after: int | None = None) -> None:
        self.body = body
        self.honour_range = honour_range
        self.stop_after = stop_after
        self.fail_after = fail_after
        self.calls: list[tuple[str, int]] = []

    def __call__(self, url: str, start_byte: int, timeout: float) -> RemoteStream:
        self.calls.append((url, start_byte))
        if start_byte > 0 and self.honour_range:
            return RemoteStream(206, self._chunks(self.body[start_byte:]), len(self.body))
        return RemoteStream(200, self._chunks(self.body), len(self.body))

    def _chunks(self, payload: bytes) -> Iterator[bytes]:
        sent = 0
        for offset in range(0, len(payload), 8):
            if self.fail_after is not None and sent >= self.fail_after:
                raise urllib.error.URLError("connection reset by peer")
            if self.stop_after is not None and sent >= self.stop_after:
                return
            chunk = payload[offset:offset + 8]
            yield chunk
            sent += len(chunk)


def part_of(directory: pathlib.Path, asset: ModelAsset) -> pathlib.Path:
    return directory / f"{asset.filename}.part"


# ---------------------------------------------------------------------------
# download(): the happy path and the four ways it goes wrong.
# ---------------------------------------------------------------------------


def test_download_writes_a_verified_file_and_cleans_up(tmp_path: pathlib.Path) -> None:
    asset = make_asset()
    server = FakeServer(BODY)

    path = download(asset, dest_dir=tmp_path, transport=server)

    assert path == tmp_path / asset.filename
    assert path.read_bytes() == BODY
    assert not part_of(tmp_path, asset).exists()
    assert server.calls == [(URL, 0)]


def test_download_creates_the_destination_directory(tmp_path: pathlib.Path) -> None:
    """First run has no cache directory yet; that is not an error."""
    nested = tmp_path / "a" / "b"
    path = download(make_asset(), dest_dir=nested, transport=FakeServer(BODY))
    assert path.read_bytes() == BODY


def test_a_wrong_digest_is_rejected_and_leaves_nothing_behind(
    tmp_path: pathlib.Path,
) -> None:
    """The right number of wrong bytes. Size alone would have accepted this."""
    asset = make_asset()
    same_length_wrong_content = bytes(len(BODY))
    server = FakeServer(same_length_wrong_content)

    with pytest.raises(ModelNotFoundError, match="sha256 mismatch"):
        download(asset, dest_dir=tmp_path, transport=server)

    assert not (tmp_path / asset.filename).exists(), "a bad file was moved into place"
    assert not part_of(tmp_path, asset).exists(), "a bad partial was kept to be resumed"


def test_a_truncated_transfer_leaves_a_resume_point(tmp_path: pathlib.Path) -> None:
    """Incomplete is not the same as corrupt, and is not treated the same way."""
    asset = make_asset()
    server = FakeServer(BODY, stop_after=HALF)

    with pytest.raises(ModelNotFoundError, match="resume"):
        download(asset, dest_dir=tmp_path, transport=server)

    assert part_of(tmp_path, asset).stat().st_size == HALF
    assert not (tmp_path / asset.filename).exists()


def test_the_next_attempt_resumes_from_the_partial_file(tmp_path: pathlib.Path) -> None:
    asset = make_asset()
    with pytest.raises(ModelNotFoundError):
        download(asset, dest_dir=tmp_path, transport=FakeServer(BODY, stop_after=HALF))

    second = FakeServer(BODY)
    path = download(asset, dest_dir=tmp_path, transport=second)

    assert second.calls == [(URL, HALF)], "the second attempt refetched from the start"
    assert path.read_bytes() == BODY
    assert not part_of(tmp_path, asset).exists()


def test_a_server_that_ignores_the_range_header_restarts_instead_of_appending(
    tmp_path: pathlib.Path,
) -> None:
    """The nastiest of the failure modes, because it produces a plausible file.

    A 200 answer to a range request carries the whole body. Appending it to the
    prefix already on disk yields a file of exactly the wrong length with a
    duplicated head, and the only thing that would catch it is the digest.
    """
    asset = make_asset()
    part_of(tmp_path, asset).write_bytes(BODY[:HALF])
    server = FakeServer(BODY, honour_range=False)

    path = download(asset, dest_dir=tmp_path, transport=server)

    assert server.calls == [(URL, HALF)], "the resume was never attempted"
    assert path.read_bytes() == BODY
    assert path.stat().st_size == len(BODY)


def test_a_corrupt_partial_is_deleted_rather_than_resumed_forever(
    tmp_path: pathlib.Path,
) -> None:
    """Otherwise every future attempt resumes the same poisoned prefix."""
    asset = make_asset()
    part = part_of(tmp_path, asset)
    part.write_bytes(bytes(HALF))

    with pytest.raises(ModelNotFoundError, match="sha256 mismatch"):
        download(asset, dest_dir=tmp_path, transport=FakeServer(BODY))
    assert not part.exists()

    fresh = FakeServer(BODY)
    assert download(asset, dest_dir=tmp_path, transport=fresh).read_bytes() == BODY
    assert fresh.calls == [(URL, 0)], "the deleted partial was still treated as a resume point"


def test_a_network_failure_keeps_what_already_arrived(tmp_path: pathlib.Path) -> None:
    """A dropped connection is the case resume exists for; do not throw it away."""
    asset = make_asset()
    server = FakeServer(BODY, fail_after=512)

    with pytest.raises(ModelNotFoundError, match="download failed"):
        download(asset, dest_dir=tmp_path, transport=server)

    assert part_of(tmp_path, asset).stat().st_size == 512


def test_a_partial_at_or_past_the_expected_size_is_discarded(
    tmp_path: pathlib.Path,
) -> None:
    """Ranging past the end of the object earns a 416, not a download."""
    asset = make_asset()
    part_of(tmp_path, asset).write_bytes(BODY + b"leftovers")
    server = FakeServer(BODY)

    path = download(asset, dest_dir=tmp_path, transport=server)

    assert server.calls == [(URL, 0)]
    assert path.read_bytes() == BODY


def test_progress_is_reported_monotonically_up_to_the_total(
    tmp_path: pathlib.Path,
) -> None:
    """The CLI renders this; the library itself never prints."""
    seen: list[tuple[int, int | None]] = []
    download(make_asset(), dest_dir=tmp_path, transport=FakeServer(BODY),
             progress=lambda done, total: seen.append((done, total)))

    done = [d for d, _ in seen]
    assert done == sorted(done)
    assert done[-1] == len(BODY)
    assert {t for _, t in seen} == {len(BODY)}


def test_download_refuses_an_asset_it_is_not_allowed_to_fetch(
    tmp_path: pathlib.Path,
) -> None:
    """Called directly rather than through ensure(), the policy still holds."""
    server = FakeServer(BODY)
    with pytest.raises(ModelNotFoundError, match="never downloaded automatically"):
        download(GAZE_MODEL, dest_dir=tmp_path, transport=server)
    assert server.calls == []


# ---------------------------------------------------------------------------
# The request, and the default transport, without a socket.
# ---------------------------------------------------------------------------


def test_build_request_sets_a_range_header_only_when_resuming() -> None:
    assert build_request(URL, 0).get_header("Range") is None
    assert build_request(URL).get_header("Range") is None
    assert build_request(URL, 1024).get_header("Range") == "bytes=1024-"


@pytest.mark.parametrize(
    "url", ["http://example.invalid/a.bin", "file:///etc/passwd", "ftp://x/y"],
)
def test_build_request_refuses_anything_but_https(url: str) -> None:
    """urlopen speaks file:// too. A downloader that will read local paths on
    request is a liability even when every URL it holds today is hard-coded."""
    with pytest.raises(ConfigError, match="non-https"):
        build_request(url)


class FakeHTTPResponse:
    """Just enough of ``http.client.HTTPResponse`` for the default transport."""

    def __init__(self, body: bytes, status: int = 200,
                 headers: dict[str, str] | None = None) -> None:
        self._buffer = io.BytesIO(body)
        self.status = status
        self.headers = headers if headers is not None else {"Content-Length": str(len(body))}
        self.closed_by_us = False

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def close(self) -> None:
        self.closed_by_us = True


def test_the_default_transport_streams_and_closes_the_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = FakeHTTPResponse(BODY)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: response)

    stream = urllib_transport(URL, 0, 1.0)
    assert stream.status == 200
    assert stream.total_bytes == len(BODY)
    assert b"".join(stream.chunks) == BODY
    assert response.closed_by_us, "the response was left open"


def test_the_default_transport_adds_the_prefix_back_to_a_partial_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a 206 the Content-Length describes the tail, not the whole file."""
    tail = BODY[HALF:]
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda *a, **k: FakeHTTPResponse(tail, status=206,
                                         headers={"Content-Length": str(len(tail))}),
    )
    assert urllib_transport(URL, HALF, 1.0).total_bytes == len(BODY)


@pytest.mark.parametrize("headers", [{}, {"Content-Length": "unknown"}])
def test_the_default_transport_copes_with_an_unusable_length(
    monkeypatch: pytest.MonkeyPatch, headers: dict[str, str],
) -> None:
    """A chunked response has no Content-Length at all; that is not an error."""
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: FakeHTTPResponse(BODY, headers=headers)
    )
    assert urllib_transport(URL, 0, 1.0).total_bytes is None


# ---------------------------------------------------------------------------
# ensure(): cache, override, and the refusal.
# ---------------------------------------------------------------------------


def test_a_verified_cache_hit_never_reaches_the_transport(cache: pathlib.Path) -> None:
    asset = make_asset()
    cache.mkdir(parents=True)
    (cache / asset.filename).write_bytes(BODY)
    server = FakeServer(BODY)

    assert ensure(asset, transport=server) == cache / asset.filename
    assert server.calls == []


def test_a_cache_miss_downloads_into_the_cache(cache: pathlib.Path) -> None:
    asset = make_asset()
    got = ensure(asset, transport=FakeServer(BODY))
    assert got == cache / asset.filename
    assert got.read_bytes() == BODY


def test_a_corrupt_cached_file_is_replaced_rather_than_used(cache: pathlib.Path) -> None:
    """A cache entry is verified on the way out, not trusted because it is there."""
    asset = make_asset()
    cache.mkdir(parents=True)
    (cache / asset.filename).write_bytes(bytes(len(BODY)))
    server = FakeServer(BODY)

    assert ensure(asset, transport=server).read_bytes() == BODY
    assert server.calls == [(URL, 0)]


def test_a_bad_file_we_cannot_replace_is_reported_not_deleted(
    cache: pathlib.Path,
) -> None:
    """We can only re-fetch what we are allowed to fetch.

    Deleting a manual file we cannot replace would destroy the user's copy and
    tell them less than the mismatch message does.
    """
    manual = make_asset(auto_download=False, url=None)
    cache.mkdir(parents=True)
    victim = cache / manual.filename
    victim.write_bytes(bytes(len(BODY)))

    with pytest.raises(ModelNotFoundError, match="sha256 mismatch"):
        ensure(manual, transport=FakeServer(BODY))
    assert victim.exists()


def test_the_override_directory_wins_and_no_transport_is_used(
    cache: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The offline path: FOCUSEDGAZE_MODEL_DIR means no network, for any asset."""
    asset = make_asset()
    offline = tmp_path / "offline"
    offline.mkdir()
    (offline / asset.filename).write_bytes(BODY)
    monkeypatch.setenv(MODEL_DIR_ENV, str(offline))
    server = FakeServer(BODY)

    assert ensure(asset, transport=server) == offline / asset.filename
    assert server.calls == []


def test_the_override_can_be_passed_explicitly_instead_of_through_the_process(
    cache: pathlib.Path, tmp_path: pathlib.Path,
) -> None:
    """Same branch, reached without mutating global state. Rule 3: both ways."""
    asset = make_asset()
    offline = tmp_path / "explicit"
    offline.mkdir()
    (offline / asset.filename).write_bytes(BODY)

    got = ensure(asset, env={MODEL_DIR_ENV: str(offline)}, transport=FakeServer(BODY))
    assert got == offline / asset.filename


def test_the_override_does_not_fall_back_to_the_cache(
    cache: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Precedence is exclusive, deliberately.

    A perfectly good copy sits in the cache. The override says the models are
    somewhere else, so the failure names that somewhere else instead of quietly
    loading a different file. Three separate bugs in this project came from a
    value being located rather than declared.
    """
    asset = make_asset()
    cache.mkdir(parents=True)
    (cache / asset.filename).write_bytes(BODY)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv(MODEL_DIR_ENV, str(empty))
    server = FakeServer(BODY)

    with pytest.raises(ModelNotFoundError) as excinfo:
        ensure(asset, transport=server)

    assert str(empty) in str(excinfo.value)
    assert server.calls == []


def test_a_file_in_the_override_directory_is_still_verified(
    cache: pathlib.Path, tmp_path: pathlib.Path,
) -> None:
    """Offline does not mean unchecked."""
    asset = make_asset()
    offline = tmp_path / "offline"
    offline.mkdir()
    (offline / asset.filename).write_bytes(bytes(len(BODY)))

    with pytest.raises(ModelNotFoundError, match="sha256 mismatch"):
        ensure(asset, env={MODEL_DIR_ENV: str(offline)}, transport=FakeServer(BODY))


def test_the_gaze_model_refuses_to_download_and_says_what_to_do(
    cache: pathlib.Path,
) -> None:
    """The policy, at the surface a caller actually uses.

    This is the test to read if you are wondering why the gaze model does not
    download. It is not an oversight and it is not a TODO.
    """
    server = FakeServer(BODY)

    with pytest.raises(ModelNotFoundError) as excinfo:
        ensure("gaze_model", transport=server)

    message = str(excinfo.value)
    assert "does NOT download" in message
    assert "non-commercial research" in message
    assert "export-onnx" in message
    assert server.calls == [], "the gaze model was fetched from somewhere"


def test_disabling_downloads_reports_instead_of_fetching(cache: pathlib.Path) -> None:
    """The offline flag a CLI would pass, distinct from the env override."""
    server = FakeServer(BODY)
    with pytest.raises(ModelNotFoundError, match="downloads are disabled"):
        ensure(make_asset(), allow_download=False, transport=server)
    assert server.calls == []


def test_ensure_accepts_a_registered_name_and_rejects_an_unknown_one(
    cache: pathlib.Path,
) -> None:
    with pytest.raises(ConfigError, match="unknown model asset"):
        ensure("face-landmarker", transport=FakeServer(BODY))


# ---------------------------------------------------------------------------
# ensure_all(): the CLI surface.
# ---------------------------------------------------------------------------


def test_ensure_all_separates_a_refusal_from_a_failure(cache: pathlib.Path) -> None:
    """Both assets are missing, for entirely different reasons.

    ``download-models`` has to be able to say "this one needs your attention"
    without calling the deliberate refusal a fault, and it has to report both
    rather than stopping at the first.
    """

    def offline(url: str, start_byte: int, timeout: float) -> RemoteStream:
        raise urllib.error.URLError("no route to host")

    reports = ensure_all(transport=offline)
    by_name = {r.asset.name: r for r in reports}

    assert list(by_name) == ["face_landmarker", "gaze_model"]
    assert by_name["face_landmarker"].state == "failed"
    assert by_name["gaze_model"].state == "manual"
    assert not any(r.ok for r in reports)
    assert "export-onnx" in by_name["gaze_model"].detail
    assert all(r.path is None for r in reports)


def test_ensure_all_reports_downloaded_then_present(
    cache: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second run must not re-download, and must say so.

    Uses a substitute asset because the real landmarker's bytes cannot be
    fabricated: its digest is a real published artifact's.
    """
    asset = make_asset()
    monkeypatch.setattr(download_module, "runtime_assets", lambda: (asset, GAZE_MODEL))
    server = FakeServer(BODY)

    first = ensure_all(transport=server)
    second = ensure_all(transport=server)

    assert [r.state for r in first] == ["downloaded", "manual"]
    assert [r.state for r in second] == ["present", "manual"]
    assert len(server.calls) == 1
    assert first[0].ok and second[0].ok
    assert not first[1].ok


def test_ensure_all_never_raises_when_everything_is_broken(
    cache: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A report the CLI can print beats an exception it has to interpret."""
    monkeypatch.setattr(download_module, "runtime_assets",
                        lambda: (make_asset(), FACE_LANDMARKER, GAZE_MODEL))

    def broken(url: str, start_byte: int, timeout: float) -> RemoteStream:
        raise urllib.error.URLError("nope")

    reports = ensure_all(transport=broken)
    assert len(reports) == 3
    assert all(isinstance(r, AssetReport) for r in reports)
    assert [r.state for r in reports] == ["failed", "failed", "manual"]
