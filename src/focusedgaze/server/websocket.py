"""Gaze-only WebSocket server, wire-compatible with ``gaze_server.py``.

THE CONTRACT THIS IMPLEMENTS IS THE REVISED ONE
------------------------------------------------
The original Phase 7 exit criterion was "the browser game connects to
``focusedgaze serve`` and plays end to end, unmodified". Reconnaissance
(MIGRATION_AUDIT.md section 39, `docs/wire_format.md`) established that it is
**not satisfiable**: `input-manager.js:135` is ``if (m.type !== "input") return;``,
so the game discards every ``type: "gaze"`` message. A strictly gaze-only server
leaves the game connected, reporting "socket: connected", with the cursor at its
initial (0.5, 0.5) forever and **no error anywhere**. That is risk R-1 and it is
invisible to any test that only compares the gaze message against a fixture.

So this module emits both:

* ``type: "gaze"`` - the raw feed, shape preserved exactly. Only
  ``gaze_test.html`` reads it, which is what makes bare ``focusedgaze serve``
  testable without the game.
* ``type: "input"`` - a **minimal, gaze-only** resolved pointer carrying
  ``type``, ``mode``, ``source``, ``ok``, ``x``, ``y``, ``t``, where ``mode`` and
  ``source`` are the constant ``"gaze"``. No gesture fields enter the SDK.

The game repo's wrapper replaces that message through :attr:`GazeServer.resolve_input`
with the full twelve-field legacy one.

WHY ``resolve_input`` REPLACES RATHER THAN APPENDS
---------------------------------------------------
A hook that appended would put two ``input`` messages on the wire per tick with
different ``x``/``y``, and the client would take whichever arrived last. The
cursor would then be driven by scheduling order. That is the entire reason this
is not a generic ``extra_messages`` list.

TWO CADENCES ON ONE TICK
-------------------------
The part a rewrite is most likely to get wrong (R-4):

* ``input`` goes out **every tick**, because in the wrapper gesture state can
  change between gaze readings.
* ``gaze`` goes out **only when the reading changes**, keyed on its timestamp,
  which is roughly 15-19 Hz against a 60 Hz tick.

``t`` is read by no client, on any message type, but it is load-bearing
*internally*: it is the dedup key. Reduce its precision and two readings landing
in the same millisecond collapse into one broadcast (R-5).

NO POSITIONING GATE
--------------------
The legacy server never imported one, despite its own docstring claiming an
"out of zone" case. Wiring the positioning gate into the ``ok`` decision would
look like a natural improvement and would be a behaviour change: the cursor would
vanish whenever the user leaned outside 45-65 cm.

NO ``os.chdir``
----------------
The legacy server chdirs to its own directory at startup and four cwd-relative
lookups ride on it (R-2). A library must not chdir: it is process-global and
would break any host application. All four now resolve explicitly, and each one
is named in MIGRATION_AUDIT.md section 47.2.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

from ..exceptions import ConfigError, ServerError

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_SEND_HZ",
    "GazeServer",
    "GazeSnapshot",
    "GazeSource",
    "gaze_message",
    "minimal_input_message",
    "validate_host",
    "validate_port",
]

_log = logging.getLogger(__name__)

#: Loopback only. Changing this default publishes an unauthenticated,
#: webcam-derived stream to the LAN (R-14).
DEFAULT_HOST: Final = "localhost"
DEFAULT_PORT: Final = 8765

#: Nominal tick. `gaze_server.py:93`. Audit section 7 measured 35-44 Hz in
#: practice, attributed to Windows timer granularity; the nominal value is what
#: the code asks for and is preserved as such.
DEFAULT_SEND_HZ: Final = 60.0

#: Seconds :meth:`GazeServer.pause` waits for the camera to actually be released.
#: MEASURED, from `gaze_server.py:148` ``release_gaze_camera(timeout=4.0)``.
#: Reopening a webcam before the previous owner has let go fails on Windows, so
#: this is a driver fact rather than a tunable. Do not round it.
DEFAULT_PAUSE_TIMEOUT_S: Final = 4.0

#: Seconds to let the driver settle before reopening, in :meth:`GazeServer.resume`.
#: MEASURED, from `gaze_server.py:216` ``time.sleep(0.3)`` between stopping the
#: other owner and reopening for gaze. The matching 0.3 s on the *release* side
#: lives in :mod:`focusedgaze.capture.webcam`, where the camera is actually
#: closed. Two sleeps, two places, both empirical. Do not merge them into one
#: tidier-looking constant: they guard different transitions.
DEFAULT_RESUME_SETTLE_S: Final = 0.3

#: Rule 11: an explicit character class of what a host may contain, never a
#: library's opinion of whether a string "looks like" a host. Covers hostnames,
#: dotted IPv4, and bracketed IPv6. A host reaches ``socket.bind`` and a
#: permissive check here is how a typo becomes a wildcard bind.
_HOST_RE: Final = re.compile(r"\A(?:[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?|\[[0-9A-Fa-f:.]+\])\Z")

#: Addresses that keep the stream on this machine. Anything else is reachable
#: from the network and gets a loud warning (R-14).
_LOOPBACK: Final = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})

#: Constant for both ``mode`` and ``source`` in the SDK's own input message. The
#: SDK has no mode concept; the field exists because the client reads it.
_GAZE: Final = "gaze"


def validate_host(host: str) -> str:
    """Check a bind address against an explicit character class (rule 11).

    Raises:
        ConfigError: The value is empty, over-long, or contains anything outside
            the permitted set.
    """
    if not isinstance(host, str) or not host:
        raise ConfigError(f"host must be a non-empty string, got {host!r}")
    if len(host) > 255:
        raise ConfigError(f"host is implausibly long ({len(host)} characters)")
    if not _HOST_RE.match(host):
        raise ConfigError(
            f"host {host!r} contains characters that are not permitted in a host "
            "name or address. Allowed: letters, digits, dot, hyphen, underscore, "
            "or a bracketed IPv6 address."
        )
    if host not in _LOOPBACK:
        # Not an error: binding wider is a legitimate choice a caller may make
        # deliberately. It is not something to do by accident, so it is said out
        # loud rather than validated away.
        _log.warning(
            "binding to %r, which is reachable from the network. This stream is "
            "derived from a webcam and is unauthenticated: there is no origin "
            "check, no token, and no TLS. Prefer %r.",
            host,
            DEFAULT_HOST,
        )
    return host


def validate_port(port: int) -> int:
    """Check a TCP port is in range (rule 11).

    ``0`` is permitted and means "let the OS pick a free one"; the chosen port
    then appears as :attr:`GazeServer.bound_port`. That is how a test binds
    without racing another process for a fixed number, and hard-coding 8765 in a
    test suite is exactly the kind of shared mutable resource that produces a
    failure nobody can reproduce.

    Raises:
        ConfigError: Not an integer, or outside 0-65535.
    """
    # bool is an int subclass, and serve(port=True) binding port 1 is not a
    # diagnosis anyone wants to make twice.
    if isinstance(port, bool) or not isinstance(port, int):
        raise ConfigError(f"port must be an integer, got {port!r}")
    if not (0 <= port <= 65535):
        raise ConfigError(f"port must be between 0 and 65535, got {port}")
    return port


@dataclass(frozen=True, slots=True)
class GazeSnapshot:
    """The newest gaze reading, in wire units.

    Args:
        ok: Whether this reading is usable. ``False`` covers no-face and the
            paused-camera case alike, which is the legacy behaviour: the server
            never distinguished them on the wire.
        x: Horizontal position, normalised ``[0, 1]``, origin left. ``None``
            whenever ``ok`` is ``False``.
        y: Vertical position, normalised ``[0, 1]``, origin top, **increasing
            downward**. ``None`` whenever ``ok`` is ``False``.
        t: Unix epoch seconds, full precision.

    ``x`` and ``y`` must be ``None`` rather than a stale point when ``ok`` is
    ``False`` (R-6). The client's ``if (m.ok)`` guard is what preserves its last
    good position; emitting ``0`` instead is harmless against today's client and
    lethal against any future one that drops the guard.
    """

    ok: bool = False
    x: float | None = None
    y: float | None = None
    t: float = 0.0


@runtime_checkable
class GazeSource(Protocol):
    """Where the server gets readings, and how the camera lease is worked.

    Structural, so the Phase 2 pipeline, a replay of a recording and a test stub
    all satisfy it without inheriting anything.
    """

    def latest(self) -> GazeSnapshot:
        """The newest reading. Called on the broadcaster tick; must not block."""
        ...

    def pause(self, timeout: float = DEFAULT_PAUSE_TIMEOUT_S) -> bool:
        """Give up the camera. ``True`` if it was actually released in time."""
        ...

    def resume(self) -> bool:
        """Take the camera back. ``True`` if it opened."""
        ...


def gaze_message(snapshot: GazeSnapshot) -> dict[str, Any]:
    """The ``type: "gaze"`` payload.

    Key order is the wire order: Python dicts preserve insertion order and
    ``json.dumps`` writes in that order. Constructed in one place here against
    four in the legacy file, all of which agreed.

    ``x`` and ``y`` are rounded to 4 places, matching ``round(sx, 4)`` at
    `gaze_server.py:378`. ``t`` is deliberately **not** rounded (R-5).
    """
    return {
        "type": "gaze",
        "ok": bool(snapshot.ok),
        "x": None if snapshot.x is None else round(float(snapshot.x), 4),
        "y": None if snapshot.y is None else round(float(snapshot.y), 4),
        "t": snapshot.t,
    }


def minimal_input_message(snapshot: GazeSnapshot) -> dict[str, Any]:
    """The SDK's own ``type: "input"`` payload: gaze only, no gesture fields.

    Seven fields. ``mode`` and ``source`` are the constant ``"gaze"`` because the
    client reads both and falls back to ``api.mode`` when ``source`` is falsy.

    The six gesture fields are **absent**, not zeroed. Against the real client
    that means ``handOk`` becomes ``false``, ``gesture`` becomes ``""``,
    ``pinching`` stays ``false``, and the counter sync stores ``undefined`` then
    compares ``undefined > undefined``, which is ``false``, so no click ever
    fires spuriously. That reading was **inferred** in section 39 and is now
    **executed** against the real ``input-manager.js`` under Node; see
    MIGRATION_AUDIT.md section 47.4 and ``tests/test_server_game_client.py``.
    """
    return {
        "type": "input",
        "mode": _GAZE,
        "source": _GAZE,
        "ok": bool(snapshot.ok),
        "x": None if snapshot.x is None else round(float(snapshot.x), 4),
        "y": None if snapshot.y is None else round(float(snapshot.y), 4),
        "t": time.time(),
    }


#: ``() -> dict | None``. Sent to one client on connect, before it sees traffic.
OnConnect = Callable[[], "dict[str, Any] | None"]
#: ``(dict) -> dict | None``. Handles a client command; the result, if any, is
#: broadcast to EVERY client, not just the sender. That is the legacy behaviour
#: (`gaze_server.py:452`) and it is what keeps two open tabs in agreement.
OnCommand = Callable[["dict[str, Any]"], "dict[str, Any] | None"]
#: ``(GazeSnapshot) -> dict | None``. REPLACES the SDK's input message.
#: Returning ``None`` suppresses it entirely.
ResolveInput = Callable[[GazeSnapshot], "dict[str, Any] | None"]


class GazeServer:
    """The transport, the pacing, and the two messages. Nothing else.

    Args:
        source: Where readings come from.
        host: Bind address. Validated; keep it loopback (R-14).
        port: TCP port. Validated.
        send_hz: Broadcaster tick rate.
        on_connect: Returns a message sent to each client as it connects, before
            it joins the broadcast traffic. The wrapper returns its mode status.
        on_command: Handles an inbound command object. The wrapper handles
            ``cmd == "mode"``.
        resolve_input: **Replaces** the input message.

    Raises:
        ConfigError: A host, port or rate is out of range.
        ServerError: ``websockets`` is not installed.
    """

    __slots__ = (
        "_clients",
        "_host",
        "_last_gaze_t",
        "_on_command",
        "_on_connect",
        "_period",
        "_port",
        "_resolve_input",
        "_running",
        "_source",
        "bound_port",
    )

    def __init__(
        self,
        source: GazeSource,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        send_hz: float = DEFAULT_SEND_HZ,
        on_connect: OnConnect | None = None,
        on_command: OnCommand | None = None,
        resolve_input: ResolveInput | None = None,
    ) -> None:
        self._host = validate_host(host)
        self._port = validate_port(port)
        if not (0 < float(send_hz) <= 1000):
            raise ConfigError(f"send_hz must be in (0, 1000], got {send_hz!r}")
        self._period = 1.0 / float(send_hz)
        self._source = source
        self._on_connect = on_connect
        self._on_command = on_command
        self._resolve_input = resolve_input
        self._clients: set[Any] = set()
        self._last_gaze_t: float | None = None
        self._running = False
        #: The port actually bound. Differs from ``port`` only when 0 was asked
        #: for, which is how a test gets a free one without racing.
        self.bound_port: int | None = None

    # -- the camera lease --------------------------------------------------

    def pause(self, timeout: float = DEFAULT_PAUSE_TIMEOUT_S) -> bool:
        """Release the camera so another pipeline can take it.

        The lease is genuinely shared: it is not a gaze concept and not a gesture
        concept, it is an operating-system fact about exclusive webcam ownership
        (section 6.5). It lives here because the SDK opens the camera, which
        means the SDK ships an API whose only current caller is a gesture feature
        it knows nothing about.

        Returns:
            ``True`` if the camera was released within ``timeout``.
        """
        released = bool(self._source.pause(timeout))
        if not released:
            _log.warning(
                "camera was not released within %.1fs; whatever takes it next "
                "will probably fail to open it",
                timeout,
            )
        return released

    def resume(self, settle: float = DEFAULT_RESUME_SETTLE_S) -> bool:
        """Take the camera back, after letting the driver settle.

        The sleep is empirical and is not decoration: reopening immediately after
        the other owner closed it fails on Windows.
        """
        time.sleep(settle)
        return bool(self._source.resume())

    # -- messages ----------------------------------------------------------

    def _input_message(self, snapshot: GazeSnapshot) -> dict[str, Any] | None:
        if self._resolve_input is None:
            return minimal_input_message(snapshot)
        # REPLACES. Never appended to the default, for the reason in the module
        # docstring: two input messages per tick and the client takes whichever
        # arrived last.
        return self._resolve_input(snapshot)

    # -- the loop ----------------------------------------------------------

    def _broadcast(self, payload: dict[str, Any]) -> None:
        from websockets.asyncio.server import broadcast as ws_broadcast

        if not self._clients:
            return
        # No backpressure, deliberately (R-9): a full write buffer is ignored and
        # a slow client is reaped by the keepalive ping instead. Disabling
        # ping_interval, the common reflex when connections drop, removes the
        # reaper and turns a slow client into a memory leak.
        ws_broadcast(self._clients, json.dumps(payload))

    async def _tick(self) -> None:
        """One broadcaster pass. Both cadences, in the legacy order."""
        if not self._clients:
            # Nothing is sent with no clients, but the loop still ticks. Matching
            # `gaze_server.py:476`.
            return
        snapshot = self._source.latest()
        # gaze BEFORE input, within a tick. Legacy ordering, `:480-483`.
        if snapshot.t != self._last_gaze_t:
            self._last_gaze_t = snapshot.t
            self._broadcast(gaze_message(snapshot))
        message = self._input_message(snapshot)
        if message is not None:
            self._broadcast(message)

    async def _broadcaster(self) -> None:
        while self._running:
            try:
                await self._tick()
            except Exception:  # noqa: BLE001 - one bad tick must not kill the feed
                _log.exception("broadcaster tick failed; continuing")
            await asyncio.sleep(self._period)

    async def _handler(self, websocket: Any) -> None:
        """One coroutine per connected client."""
        self._clients.add(websocket)
        _log.info("client connected (clients: %d)", len(self._clients))
        try:
            if self._on_connect is not None:
                hello = self._on_connect()
                if hello is not None:
                    # Per-client send, before this socket joins broadcast
                    # traffic, so the HUD is right immediately.
                    await websocket.send(json.dumps(hello))
            async for raw in websocket:
                await self._on_message(raw)
        except _connection_closed_errors() as exc:
            # DELIBERATE DEVIATION, stated under rule 4. The legacy handler
            # caught bare `Exception` and printed a traceback, so a browser tab
            # closing or a process being killed produced a stack trace. Both
            # clients reconnect forever, at 1200 ms and 1000 ms (R-8), so that
            # turns an ordinary disconnect into a log flood at roughly one
            # traceback per second while the page is shut.
            #
            # An abrupt close is not a failure. The legacy comment says the
            # traceback exists so failures are not silent, not so that normal
            # disconnects are noisy, and the distinction is preserved: genuine
            # errors below still get the full trace.
            _log.debug("client went away: %s", exc)
        except Exception as exc:  # noqa: BLE001
            # Never swallowed: a failure here used to close the socket with no
            # trace, which looked exactly like a hang from the browser's side.
            _log.exception("connection handler failed: %s", exc)
        finally:
            self._clients.discard(websocket)
            _log.info("client disconnected (clients: %d)", len(self._clients))

    async def _on_message(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return
        if self._on_command is None:
            return
        # R-11, fixed. `json.loads` returns any JSON value, so a client sending
        # `"hi"`, `3` or `null` produces a str, an int or None. The legacy server
        # called `.get` on that directly, the AttributeError escaped the read
        # loop, and the connection closed; the browser then reconnected 1.2 s
        # later and the command was simply lost.
        #
        # Only objects are dispatched. A non-object payload is as meaningless as
        # a malformed one and is ignored the same way.
        if not isinstance(msg, dict):
            _log.debug("ignoring a %s payload: commands must be JSON objects", type(msg).__name__)
            return
        try:
            result = self._on_command(msg)
        except Exception:  # noqa: BLE001
            # A hook belongs to the caller and may raise anything. Letting that
            # drop the socket would make one bad command look like a network
            # fault, and with both clients reconnecting forever it would loop.
            _log.exception("on_command failed; the connection is kept open")
            return
        if result is not None:
            # To EVERY client, not just the sender: two open tabs must agree.
            self._broadcast(result)

    async def serve_forever(self) -> None:
        """Run until cancelled. Opens the port only after the source is ready.

        Ordering matters (R-10): the launcher treats "port is listening" as
        "ready" and waits up to 90 s for it. Opening the port before the gaze
        source can produce readings would report success for a server with no
        gaze feed.
        """
        try:
            from websockets.asyncio.server import serve as ws_serve
        except ImportError as exc:
            raise ServerError(
                "the WebSocket server needs the `websockets` package, which the "
                "base install does not include. Install it with:\n"
                "    pip install 'focusedgaze[server]'"
            ) from exc

        self._running = True
        async with ws_serve(self._handler, self._host, self._port) as server:
            self.bound_port = _bound_port(server, self._port)
            _log.info("gaze server ready at ws://%s:%s", self._host, self.bound_port)
            try:
                await self._broadcaster()
            finally:
                self._running = False

    def stop(self) -> None:
        """Ask the broadcaster to finish after its current tick."""
        self._running = False

    @property
    def clients(self) -> int:
        """How many sockets are currently connected."""
        return len(self._clients)


def _connection_closed_errors() -> tuple[type[BaseException], ...]:
    """The exceptions that mean "the client went away", not "something broke".

    Resolved lazily so this module still imports with `websockets` absent, which
    is the whole point of the deferred import elsewhere in this file.
    """
    try:
        from websockets.exceptions import ConnectionClosed
    except ImportError:  # pragma: no cover - only in a base install
        return (ConnectionResetError,)
    return (ConnectionClosed, ConnectionResetError)


def _bound_port(server: Any, requested: int) -> int:
    """The port actually listening, which differs from 0 when 0 was asked for."""
    sockets: Iterable[Any] = getattr(server, "sockets", None) or ()
    for sock in sockets:
        try:
            return int(sock.getsockname()[1])
        except (OSError, IndexError, TypeError):  # pragma: no cover - platform edge
            continue
    return requested
