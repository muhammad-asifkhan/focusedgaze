"""Fetch a model asset, verify it, and resume an interrupted transfer.

Stdlib only. ``requests`` is not a declared dependency and is not going to
become one for the sake of four HTTP calls a year, so this is ``urllib.request``
with a ``Range`` header.

THE ONE INVARIANT
-----------------
A file at its final path has had its digest checked. Nothing else in the package
re-checks it before loading, so this module is where that guarantee is made:
bytes land in ``<name>.part``, the digest and the size are checked there, and
only then does :func:`os.replace` move it into place. A truncated or corrupted
transfer therefore cannot be left sitting where a later run would mistake it for
a good file. ``os.replace`` is atomic within a directory, so a crash between the
check and the move leaves the old state, never a half-written file.

RESUME, AND THE TWO WAYS IT GOES WRONG
--------------------------------------
A partial ``.part`` is resumed with ``Range: bytes=N-``. Two failure modes are
handled explicitly rather than hoped away:

* **The server ignores the range.** It answers 200 with the whole body instead
  of 206 with the tail. Appending that to the existing prefix produces a file of
  the right shape and the wrong contents, which is the worst possible outcome.
  A 200 in response to a range request restarts the write from zero.
* **The partial bytes are themselves corrupt.** Resuming only ever adds to
  them, so the digest fails at the end and the ``.part`` is deleted. Without
  that, every subsequent attempt would resume the same poisoned prefix and fail
  the same way forever.

A network error is different in kind and the ``.part`` survives it: an
interrupted transfer is exactly what resume exists for.

THE TRANSPORT IS A SEAM
-----------------------
:func:`download` takes a ``transport`` callable. The default one is the only
code in the package that opens a socket, and the tests substitute their own, so
the whole suite runs with no network. The seam hands back an iterator of chunks
rather than an HTTP response object, so a test fake is a few lines rather than a
mock of ``http.client``.

NO OUTPUT
---------
Library code does not print. Progress goes to an optional callback and the
per-asset outcome comes back as :class:`AssetReport` for the CLI to render.
"""

from __future__ import annotations

import os
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..exceptions import ConfigError, ModelNotFoundError
from .registry import ModelAsset, cache_dir, get_asset, model_dir_override, runtime_assets, sha256_file

__all__ = [
    "CHUNK_SIZE",
    "DEFAULT_TIMEOUT",
    "AssetReport",
    "RemoteStream",
    "Transport",
    "build_request",
    "download",
    "ensure",
    "ensure_all",
    "urllib_transport",
]

#: Seconds before a stalled connection is given up on.
DEFAULT_TIMEOUT = 30.0

#: 64 KiB. Small enough that a progress callback feels live on a slow link.
CHUNK_SIZE = 1 << 16

_HTTP_PARTIAL_CONTENT = 206

#: Called with (bytes written so far, total expected or ``None``).
ProgressCallback = Callable[[int, int | None], None]

AssetState = Literal["present", "downloaded", "manual", "failed"]


@dataclass(frozen=True)
class RemoteStream:
    """What a transport returns: a status, a byte stream, and a size if known.

    Args:
        status: HTTP status. 206 means the server honoured the range request
            and the chunks are the tail; anything else is treated as a full body.
        chunks: The response body. Consumed once.
        total_bytes: Expected total size of the complete file when the server
            said, or ``None``.
    """

    status: int
    chunks: Iterator[bytes]
    total_bytes: int | None = None


#: ``(url, start_byte, timeout) -> RemoteStream``. Substituted in tests.
Transport = Callable[[str, int, float], RemoteStream]


@dataclass(frozen=True)
class AssetReport:
    """The outcome for one asset, for a CLI to display.

    ``state`` distinguishes the two ways of not having a file, because they are
    not the same event:

    * ``"manual"``  the package is not permitted to fetch it. Expected, not a
      failure; ``detail`` holds the instructions to show the user.
    * ``"failed"``  the package was permitted and could not. A real problem.
    """

    asset: ModelAsset
    state: AssetState
    path: Path | None
    detail: str = ""

    @property
    def ok(self) -> bool:
        """Whether the file is present and verified."""
        return self.state in ("present", "downloaded")


def build_request(url: str, start_byte: int = 0) -> urllib.request.Request:
    """Build the HTTP request, with a ``Range`` header when resuming.

    Split out from the transport so the resume header can be tested without a
    socket. Rejects any scheme but https: ``urlopen`` also speaks ``file://``,
    and a downloader that will read local paths on request is a liability even
    when every URL it currently holds is hard-coded.

    Raises:
        ConfigError: if the URL is not https.
    """
    if not url.lower().startswith("https://"):
        raise ConfigError(f"refusing to fetch a non-https URL: {url!r}")
    request = urllib.request.Request(url, method="GET")
    if start_byte > 0:
        request.add_header("Range", f"bytes={start_byte}-")
    return request


def urllib_transport(url: str, start_byte: int, timeout: float) -> RemoteStream:
    """The default transport. The only place in the package that opens a socket."""
    request = build_request(url, start_byte)
    # Scheme is checked in build_request, so urlopen cannot be steered at a
    # local file here.
    response: Any = urllib.request.urlopen(request, timeout=timeout)  # noqa: S310
    status = int(response.status)
    return RemoteStream(
        status=status,
        chunks=_iter_chunks(response),
        total_bytes=_expected_total(response, status, start_byte),
    )


def _iter_chunks(response: Any) -> Iterator[bytes]:
    """Yield the body in chunks, closing the response even on an early exit."""
    try:
        while True:
            chunk: bytes = response.read(CHUNK_SIZE)
            if not chunk:
                return
            yield chunk
    finally:
        response.close()


def _expected_total(response: Any, status: int, start_byte: int) -> int | None:
    """Total size of the complete file, from ``Content-Length``, or ``None``.

    On a 206 the header describes the tail being sent, not the whole file, so
    the already-downloaded prefix has to be added back.
    """
    raw = response.headers.get("Content-Length")
    if raw is None or not str(raw).strip().isdigit():
        return None
    length = int(str(raw).strip())
    return length + start_byte if status == _HTTP_PARTIAL_CONTENT else length


def _mismatch(path: Path, asset: ModelAsset) -> str | None:
    """Why ``path`` is not an acceptable copy of ``asset``, or ``None`` if it is.

    Size is checked first because it is free and catches the common truncation
    case without hashing 91 MB to learn what ``stat`` already knew.
    """
    actual_size = path.stat().st_size
    if asset.size_bytes is not None and actual_size != asset.size_bytes:
        return f"expected {asset.size_bytes} bytes, found {actual_size}"
    if asset.sha256 is not None:
        actual = sha256_file(path)
        if actual != asset.sha256:
            return f"sha256 mismatch: expected {asset.sha256}, found {actual}"
    return None


def download(
    asset: ModelAsset,
    *,
    dest_dir: str | Path | None = None,
    transport: Transport = urllib_transport,
    timeout: float = DEFAULT_TIMEOUT,
    progress: ProgressCallback | None = None,
) -> Path:
    """Fetch one asset into ``dest_dir``, resuming and verifying.

    Args:
        asset: What to fetch. Must have ``auto_download`` set.
        dest_dir: Where it lands. Defaults to the managed cache.
        transport: Injectable HTTP seam; see the module docstring.
        timeout: Seconds of inactivity before giving up.
        progress: Called with ``(bytes_done, total_or_none)`` per chunk.

    Returns:
        The path of the verified file.

    Raises:
        ModelNotFoundError: if the asset may not be fetched, if the transfer
            failed, or if what arrived did not verify. The message names the
            remedy in every case.
    """
    if not asset.auto_download or asset.url is None:
        # Reached only by calling this directly on a manual asset. `ensure`
        # routes those away before they get here.
        raise ModelNotFoundError(
            asset.not_found_message(
                Path(dest_dir or cache_dir()) / asset.filename,
                reason="this asset is never downloaded automatically",
            )
        )

    directory = Path(dest_dir) if dest_dir is not None else cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / asset.filename
    part = directory / f"{asset.filename}.part"

    start = part.stat().st_size if part.exists() else 0
    if asset.size_bytes is not None and start >= asset.size_bytes:
        # A .part at or past the full size is junk, not a resume point: the
        # bytes are wrong or the size is. Start again rather than range past
        # the end of the object and get a 416 for it.
        part.unlink()
        start = 0

    try:
        stream = transport(asset.url, start, timeout)
        resuming = start > 0 and stream.status == _HTTP_PARTIAL_CONTENT
        if start > 0 and not resuming:
            # The server sent the whole body despite the Range header. Appending
            # would splice a duplicate prefix onto the file.
            start = 0
        written = start
        with part.open("ab" if resuming else "wb") as handle:
            for chunk in stream.chunks:
                handle.write(chunk)
                written += len(chunk)
                if progress is not None:
                    progress(written, asset.size_bytes or stream.total_bytes)
    except OSError as exc:
        # Covers URLError, socket timeouts and disk errors alike. The .part is
        # deliberately left where it is: that is the resume point.
        raise ModelNotFoundError(
            asset.not_found_message(final, reason=f"download failed: {exc}")
        ) from exc

    # Incomplete and complete-but-wrong are different situations and the .part
    # is treated differently for each.
    if asset.size_bytes is not None and written < asset.size_bytes:
        # Short body, no error: the connection or a proxy gave up early. What
        # arrived is a usable prefix, so it is KEPT and the next attempt resumes
        # from it. Nothing has been moved into place, and the digest still has
        # to pass before anything is, so keeping it costs no safety and saves
        # the whole transfer on a flaky link.
        raise ModelNotFoundError(
            asset.not_found_message(
                final,
                reason=(f"transfer ended after {written} of {asset.size_bytes} bytes; "
                        "run the command again to resume from where it stopped"),
            )
        )

    reason = _mismatch(part, asset)
    if reason is not None:
        # Full size and still wrong, so the bytes themselves are bad. Delete it:
        # resuming a poisoned prefix would re-derive the same bad digest on
        # every future attempt, and a user watching only the file size would see
        # something that looks finished.
        part.unlink(missing_ok=True)
        raise ModelNotFoundError(asset.not_found_message(final, reason=reason))

    os.replace(part, final)
    return final


def ensure(
    asset: ModelAsset | str,
    *,
    allow_download: bool = True,
    env: Mapping[str, str] | None = None,
    transport: Transport = urllib_transport,
    timeout: float = DEFAULT_TIMEOUT,
    progress: ProgressCallback | None = None,
) -> Path:
    """Return a verified local path for ``asset``, fetching it if permitted.

    Resolution order:

    1. ``FOCUSEDGAZE_MODEL_DIR`` if set. Exclusive: nothing else is consulted
       and no network access happens, for any asset.
    2. The managed cache. A file already there is verified before it is handed
       back; if it fails and we are allowed to re-fetch, it is replaced.
    3. A download, if the asset allows one and ``allow_download`` is set.

    Args:
        asset: An asset or its registered name.
        allow_download: Set ``False`` for an offline run. Turns a would-be
            download into an error naming what is missing.
        env: Environment to read the override from. Defaults to ``os.environ``.
        transport: Injectable HTTP seam.
        timeout: Seconds of inactivity before giving up.
        progress: Called with ``(bytes_done, total_or_none)`` per chunk.

    Raises:
        ModelNotFoundError: with instructions, whenever a verified file cannot
            be produced. For the gaze model this is the expected first run.
        ConfigError: if ``asset`` is a name that is not registered.
    """
    resolved = get_asset(asset) if isinstance(asset, str) else asset

    override = model_dir_override(env)
    if override is not None:
        path = resolved.path_in(override)
        if not path.exists():
            raise ModelNotFoundError(resolved.not_found_message(path))
        reason = _mismatch(path, resolved)
        if reason is not None:
            raise ModelNotFoundError(resolved.not_found_message(path, reason=reason))
        return path

    directory = cache_dir()
    path = resolved.path_in(directory)
    if path.exists():
        reason = _mismatch(path, resolved)
        if reason is None:
            return path
        if not (resolved.auto_download and allow_download):
            # Not ours to delete: we cannot replace it, so removing it would
            # only destroy the user's copy and tell them less.
            raise ModelNotFoundError(resolved.not_found_message(path, reason=reason))
        path.unlink()

    if not resolved.auto_download:
        raise ModelNotFoundError(resolved.not_found_message(path))
    if not allow_download:
        raise ModelNotFoundError(
            resolved.not_found_message(path, reason="downloads are disabled for this run")
        )
    return download(
        resolved, dest_dir=directory, transport=transport, timeout=timeout, progress=progress
    )


def ensure_all(
    *,
    allow_download: bool = True,
    env: Mapping[str, str] | None = None,
    transport: Transport = urllib_transport,
    timeout: float = DEFAULT_TIMEOUT,
    progress: ProgressCallback | None = None,
) -> tuple[AssetReport, ...]:
    """Resolve every runtime asset and report on each, raising for none of them.

    This is the surface behind ``focusedgaze download-models``. It never raises,
    because a gaze model that has to be fetched by hand is the designed
    behaviour rather than a fault, and a command that aborted on it could not go
    on to report that the landmarker is fine.

    Returns:
        One :class:`AssetReport` per runtime asset, in registry order. A caller
        deciding an exit code should look at the reports whose asset has
        ``auto_download`` set: those are the ones this package promised to
        handle. A ``"manual"`` report is instructions to print, not a failure.
    """
    override = model_dir_override(env)
    where = override if override is not None else cache_dir()
    reports: list[AssetReport] = []
    for asset in runtime_assets():
        # Sampled before the call so "present" and "downloaded" can be told
        # apart afterwards. `ensure` deliberately does not report which it did.
        existed = asset.path_in(where).exists()
        try:
            path = ensure(
                asset,
                allow_download=allow_download,
                env=env,
                transport=transport,
                timeout=timeout,
                progress=progress,
            )
        except ModelNotFoundError as exc:
            state: AssetState = "manual" if not asset.auto_download else "failed"
            reports.append(AssetReport(asset, state, None, str(exc)))
        else:
            reports.append(
                AssetReport(asset, "present" if existed else "downloaded", path)
            )
    return tuple(reports)
