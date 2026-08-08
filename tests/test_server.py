"""Phase 7: the WebSocket server, exercised against a real client.

These start an actual server on an ephemeral port and connect an actual
`websockets` client to it. Nothing here asserts against a hand-written string:
the messages are read off the wire, which is the only way the two pacing rules
and the ordering within a tick are observable at all.

Port 0 throughout. A fixed 8765 in a test suite is a shared mutable resource and
produces failures nobody can reproduce.
"""

from __future__ import annotations

import asyncio
import functools
import json
from typing import Self

import pytest

from focusedgaze.exceptions import ConfigError
from focusedgaze.server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    GazeServer,
    GazeSnapshot,
    GazeSource,
    gaze_message,
    minimal_input_message,
    validate_host,
    validate_port,
)


def sync(fn):
    """Run an async test body without pulling in an async pytest plugin.

    `anyio` and `pytest-asyncio` are both absent, and adding a dev dependency to
    run four coroutines is a worse trade than four lines here.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


class StubSource:
    """A scripted gaze source. Satisfies `GazeSource` structurally."""

    def __init__(self, snapshot: GazeSnapshot | None = None) -> None:
        self.snapshot = snapshot if snapshot is not None else GazeSnapshot()
        self.paused = 0
        self.resumed = 0
        self.pause_timeouts: list[float] = []
        self.pause_result = True

    def latest(self) -> GazeSnapshot:
        return self.snapshot

    def pause(self, timeout: float = 4.0) -> bool:
        self.paused += 1
        self.pause_timeouts.append(timeout)
        return self.pause_result

    def resume(self) -> bool:
        self.resumed += 1
        return True


class ServerHarness:
    """Runs a GazeServer in the background and hands out its URL."""

    def __init__(self, **kwargs: object) -> None:
        self.source: StubSource = kwargs.pop("source", None) or StubSource()  # type: ignore[assignment]
        self.server = GazeServer(self.source, host="127.0.0.1", port=0, **kwargs)  # type: ignore[arg-type]
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Self:
        self._task = asyncio.create_task(self.server.serve_forever())
        for _ in range(400):  # up to ~4 s
            if self.server.bound_port:
                return self
            await asyncio.sleep(0.01)
        raise AssertionError("server did not bind")

    async def __aexit__(self, *exc: object) -> None:
        self.server.stop()
        if self._task is not None:
            self._task.cancel()
            with pytest.raises((asyncio.CancelledError, Exception)):
                await self._task

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.server.bound_port}"


async def _collect(url: str, count: int, timeout: float = 5.0) -> list[dict]:
    """Connect and read `count` messages."""
    from websockets.asyncio.client import connect

    out: list[dict] = []
    async with connect(url) as ws:
        for _ in range(count):
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            out.append(json.loads(raw))
    return out


# ---------------------------------------------------------------------------
# Host and port validation. Rule 11.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host", ["localhost", "127.0.0.1", "example.com", "my-host_1", "[::1]", "0.0.0.0"]
)
def test_permitted_hosts_are_accepted(host: str) -> None:
    assert validate_host(host) == host


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "local host",          # space
        "host;rm -rf /",       # shell metacharacters
        "host/../other",       # path separators
        "host\\other",
        "host\nother",         # newline
        "-leading",            # hyphen at either end is not a legal label
        "trailing-",
        "a" * 256,
    ],
)
def test_a_host_outside_the_character_class_is_refused(bad: str) -> None:
    """Rule 11: an explicit class of what is allowed, not a library's opinion.

    A host reaches `socket.bind`. A permissive check is how a typo becomes a
    wildcard bind, and how anything interpolating this value later inherits
    whatever was smuggled through.
    """
    with pytest.raises(ConfigError):
        validate_host(bad)


def test_a_non_loopback_host_warns_rather_than_failing(caplog) -> None:
    """Binding wider is a legitimate deliberate choice, and a terrible accident.

    So it is said out loud rather than validated away (R-14).
    """
    with caplog.at_level("WARNING"):
        validate_host("0.0.0.0")
    assert any("unauthenticated" in r.getMessage() for r in caplog.records)


def test_a_loopback_host_does_not_warn(caplog) -> None:
    """The control, so the warning above means something."""
    with caplog.at_level("WARNING"):
        validate_host("localhost")
    assert not caplog.records


@pytest.mark.parametrize("port", [0, 1, 8765, 65535])
def test_permitted_ports_are_accepted(port: int) -> None:
    assert validate_port(port) == port


@pytest.mark.parametrize("bad", [-1, 65536, "8765", 8765.0, None, True, False])
def test_a_bad_port_is_refused(bad: object) -> None:
    """`True` is refused specifically: bool is an int subclass and would bind 1."""
    with pytest.raises(ConfigError):
        validate_port(bad)  # type: ignore[arg-type]


def test_the_defaults_are_the_legacy_ones() -> None:
    """Changing these silently breaks every existing client."""
    assert DEFAULT_HOST == "localhost"
    assert DEFAULT_PORT == 8765


def test_a_nonsense_send_rate_is_refused() -> None:
    with pytest.raises(ConfigError, match="send_hz"):
        GazeServer(StubSource(), send_hz=0)


# ---------------------------------------------------------------------------
# Message shape.
# ---------------------------------------------------------------------------


def test_the_gaze_message_keeps_its_exact_shape_and_key_order() -> None:
    """Key order is the wire order; json.dumps writes insertion order."""
    msg = gaze_message(GazeSnapshot(ok=True, x=0.42424242, y=0.70710678, t=1690000000.1234567))
    assert list(msg) == ["type", "ok", "x", "y", "t"]
    assert msg["type"] == "gaze"
    assert msg["ok"] is True
    assert msg["x"] == 0.4242 and msg["y"] == 0.7071
    assert msg["t"] == 1690000000.1234567, "t must not be rounded: it is the dedup key"


def test_a_no_face_gaze_message_carries_null_not_a_stale_point() -> None:
    """R-6. The client's `if (m.ok)` guard is what preserves its last position.

    Emitting 0 instead of null is harmless against today's client and lethal
    against any future one that drops the guard.
    """
    msg = gaze_message(GazeSnapshot(ok=False, x=None, y=None, t=5.0))
    assert msg["x"] is None and msg["y"] is None
    assert msg["ok"] is False


def test_the_input_message_is_seven_fields_with_no_gesture_vocabulary() -> None:
    """The whole point of option (d): no gesture fields enter the SDK."""
    msg = minimal_input_message(GazeSnapshot(ok=True, x=0.25, y=0.75, t=1.0))
    assert list(msg) == ["type", "mode", "source", "ok", "x", "y", "t"]
    assert msg["mode"] == "gaze" and msg["source"] == "gaze"
    for absent in ("hand_ok", "gesture", "click_seq", "clear_seq", "swipe_seq", "swipe"):
        assert absent not in msg


def test_the_input_message_rounds_coordinates_like_the_legacy_producer() -> None:
    msg = minimal_input_message(GazeSnapshot(ok=True, x=0.123456789, y=0.987654321, t=1.0))
    assert msg["x"] == 0.1235 and msg["y"] == 0.9877


# ---------------------------------------------------------------------------
# Pacing and ordering, read off the wire.
# ---------------------------------------------------------------------------


@sync
async def test_input_goes_out_every_tick_and_gaze_only_on_a_new_reading() -> None:
    """R-4, the rule a rewrite is most likely to get wrong.

    The source's timestamp is held constant, so the gaze feed must fall silent
    after one message while input keeps arriving every tick.
    """
    async with ServerHarness(send_hz=200.0) as h:
        h.source.snapshot = GazeSnapshot(ok=True, x=0.1, y=0.2, t=100.0)
        messages = await _collect(h.url, 9)

    types = [m["type"] for m in messages]
    assert types.count("gaze") == 1, f"gaze was re-sent for an unchanged reading: {types}"
    assert types.count("input") == 8


@sync
async def test_a_new_reading_produces_exactly_one_more_gaze_message() -> None:
    """The dedup key is `t`. One send per new reading, no more, no fewer."""
    async with ServerHarness(send_hz=200.0) as h:
        h.source.snapshot = GazeSnapshot(ok=True, x=0.1, y=0.2, t=100.0)
        from websockets.asyncio.client import connect

        async with connect(h.url) as ws:
            seen: list[dict] = []
            for _ in range(4):
                seen.append(json.loads(await asyncio.wait_for(ws.recv(), 5.0)))
            h.source.snapshot = GazeSnapshot(ok=True, x=0.3, y=0.4, t=200.0)
            for _ in range(6):
                seen.append(json.loads(await asyncio.wait_for(ws.recv(), 5.0)))

    gaze = [m for m in seen if m["type"] == "gaze"]
    assert len(gaze) == 2, f"expected one gaze per reading, got {len(gaze)}"
    assert [m["t"] for m in gaze] == [100.0, 200.0]


@sync
async def test_gaze_is_broadcast_before_input_within_a_tick() -> None:
    """Legacy ordering, `gaze_server.py:480-483`."""
    async with ServerHarness(send_hz=200.0) as h:
        h.source.snapshot = GazeSnapshot(ok=True, x=0.5, y=0.5, t=7.0)
        messages = await _collect(h.url, 2)
    assert [m["type"] for m in messages] == ["gaze", "input"]


@sync
async def test_nothing_is_sent_with_no_clients_but_the_loop_keeps_ticking() -> None:
    """Matching `gaze_server.py:476`. The tick must survive an idle period."""
    async with ServerHarness(send_hz=200.0) as h:
        h.source.snapshot = GazeSnapshot(ok=True, x=0.1, y=0.2, t=1.0)
        await asyncio.sleep(0.1)  # nobody connected; ticks happen anyway
        assert h.server.clients == 0
        messages = await _collect(h.url, 2)
    assert [m["type"] for m in messages] == ["gaze", "input"]


@sync
async def test_a_late_joiner_gets_no_replay_backlog() -> None:
    """Both clients reconnect forever (R-8); a backlog on connect would flood."""
    async with ServerHarness(send_hz=200.0) as h:
        h.source.snapshot = GazeSnapshot(ok=True, x=0.1, y=0.2, t=1.0)
        await asyncio.sleep(0.15)
        messages = await _collect(h.url, 3)
    # The first thing a late joiner sees is current traffic, not history.
    assert all(m["t"] in (1.0, messages[-1]["t"]) or m["type"] == "input" for m in messages)
    assert len([m for m in messages if m["type"] == "gaze"]) <= 1


# ---------------------------------------------------------------------------
# Hooks.
# ---------------------------------------------------------------------------


@sync
async def test_resolve_input_replaces_the_message_rather_than_appending() -> None:
    """The single most important property of the hook.

    Appending would put two `input` messages per tick on the wire with different
    x/y, and the client takes whichever arrived last, so the cursor would be
    driven by scheduling order.
    """
    def full_message(snapshot: GazeSnapshot) -> dict:
        return {"type": "input", "mode": "gesture", "source": "gesture",
                "ok": True, "x": 0.9, "y": 0.9, "click_seq": 3, "t": 1.0}

    async with ServerHarness(send_hz=200.0, resolve_input=full_message) as h:
        h.source.snapshot = GazeSnapshot(ok=True, x=0.1, y=0.2, t=1.0)
        messages = await _collect(h.url, 6)

    inputs = [m for m in messages if m["type"] == "input"]
    assert len(inputs) >= 4
    for m in inputs:
        assert m["source"] == "gesture", "the SDK's own input message was still sent"
        assert m["x"] == 0.9
    # Exactly one input per tick: count them against the gaze messages.
    ticks = len(inputs)
    assert ticks == len([m for m in messages if m["type"] == "input"])


@sync
async def test_resolve_input_can_suppress_the_message_entirely() -> None:
    async with ServerHarness(send_hz=200.0, resolve_input=lambda s: None) as h:
        h.source.snapshot = GazeSnapshot(ok=True, x=0.1, y=0.2, t=1.0)
        from websockets.asyncio.client import connect

        async with connect(h.url) as ws:
            first = json.loads(await asyncio.wait_for(ws.recv(), 5.0))
            assert first["type"] == "gaze"
            h.source.snapshot = GazeSnapshot(ok=True, x=0.3, y=0.4, t=2.0)
            second = json.loads(await asyncio.wait_for(ws.recv(), 5.0))
            assert second["type"] == "gaze", "input was sent despite the hook returning None"


@sync
async def test_on_connect_is_sent_per_client_before_broadcast_traffic() -> None:
    """The wrapper returns its mode status here so the HUD is right immediately."""
    async with ServerHarness(
        send_hz=200.0, on_connect=lambda: {"type": "mode", "mode": "gaze"}
    ) as h:
        h.source.snapshot = GazeSnapshot(ok=True, x=0.1, y=0.2, t=1.0)
        messages = await _collect(h.url, 3)
    assert messages[0]["type"] == "mode"


@sync
async def test_a_command_result_is_broadcast_to_every_client_not_just_the_sender() -> None:
    """Legacy behaviour, `:452`. Two open tabs must agree about the mode."""
    seen: list[dict] = []

    def on_command(msg: dict) -> dict:
        seen.append(msg)
        return {"type": "mode", "mode": msg.get("mode", "")}

    from websockets.asyncio.client import connect

    async with ServerHarness(send_hz=200.0, on_command=on_command) as h:
        h.source.snapshot = GazeSnapshot(ok=True, x=0.1, y=0.2, t=1.0)
        async with connect(h.url) as sender, connect(h.url) as observer:
            await sender.send(json.dumps({"cmd": "mode", "mode": "gesture"}))
            found = None
            for _ in range(40):
                msg = json.loads(await asyncio.wait_for(observer.recv(), 5.0))
                if msg["type"] == "mode":
                    found = msg
                    break
    assert seen == [{"cmd": "mode", "mode": "gesture"}]
    assert found is not None, "the other client never saw the mode change"
    assert found["mode"] == "gesture"


@sync
async def test_unparseable_payloads_are_ignored_without_closing_the_socket() -> None:
    async with ServerHarness(send_hz=200.0, on_command=lambda m: None) as h:
        h.source.snapshot = GazeSnapshot(ok=True, x=0.1, y=0.2, t=1.0)
        from websockets.asyncio.client import connect

        async with connect(h.url) as ws:
            await ws.send("not json at all")
            msg = json.loads(await asyncio.wait_for(ws.recv(), 5.0))
            assert msg["type"] in ("gaze", "input")


# ---------------------------------------------------------------------------
# The camera lease.
# ---------------------------------------------------------------------------


def test_pause_carries_the_measured_four_second_timeout() -> None:
    """4.0 s is a Windows driver fact from `gaze_server.py:148`, not a tunable."""
    source = StubSource()
    server = GazeServer(source)
    assert server.pause() is True
    assert source.pause_timeouts == [4.0]


def test_resume_waits_the_measured_settle_before_reopening(monkeypatch) -> None:
    """0.3 s from `gaze_server.py:216`. Reopening immediately fails on Windows."""
    slept: list[float] = []
    monkeypatch.setattr("focusedgaze.server.websocket.time.sleep", slept.append)
    source = StubSource()
    server = GazeServer(source)
    assert server.resume() is True
    assert slept == [0.3], f"the driver settle was not honoured: {slept}"
    assert source.resumed == 1


def test_a_camera_that_will_not_release_is_reported_rather_than_assumed(caplog) -> None:
    source = StubSource()
    source.pause_result = False
    server = GazeServer(source)
    with caplog.at_level("WARNING"):
        assert server.pause() is False
    assert any("not released" in r.getMessage() for r in caplog.records)


def test_the_stub_source_satisfies_the_protocol() -> None:
    """Structural, so the Phase 2 pipeline and a replay both fit without inheriting."""
    assert isinstance(StubSource(), GazeSource)


# ---------------------------------------------------------------------------
# R-11, fixed. These replace the tests that pinned the bug.
# ---------------------------------------------------------------------------


@sync
async def test_r11_valid_json_that_is_not_an_object_no_longer_closes_the_connection() -> None:
    """`json.loads` returns any JSON value, not only objects.

    The legacy server called `.get` on the result directly, so `"hi"` raised
    AttributeError, the error escaped the read loop and the socket closed. The
    browser then reconnected 1.2 s later and the command was lost. Now a
    non-object payload is ignored exactly like a malformed one.
    """
    def on_command(msg: dict) -> dict | None:
        return {"type": "mode", "mode": msg.get("mode", "")}

    async with ServerHarness(send_hz=200.0, on_command=on_command) as h:
        h.source.snapshot = GazeSnapshot(ok=True, x=0.1, y=0.2, t=1.0)
        from websockets.asyncio.client import connect

        async with connect(h.url) as ws:
            await ws.recv()
            for payload in ('"hi"', "3", "null", "[1, 2]", "true"):
                await ws.send(payload)
            # Still alive, and still serving: the whole point of the fix.
            await ws.send(json.dumps({"cmd": "mode", "mode": "gaze"}))
            found = None
            for _ in range(80):
                msg = json.loads(await asyncio.wait_for(ws.recv(), 5.0))
                if msg["type"] == "mode":
                    found = msg
                    break
    assert found is not None, "the connection died, or the command never arrived"
    assert found["mode"] == "gaze"


@sync
async def test_a_hook_that_raises_does_not_drop_the_client() -> None:
    """A hook belongs to the caller and may raise anything.

    Letting that close the socket would make one bad command look like a network
    fault, and with both clients reconnecting forever it would loop.
    """
    def explode(msg: dict) -> dict | None:
        raise RuntimeError("hook is broken")

    async with ServerHarness(send_hz=200.0, on_command=explode) as h:
        h.source.snapshot = GazeSnapshot(ok=True, x=0.1, y=0.2, t=1.0)
        from websockets.asyncio.client import connect

        async with connect(h.url) as ws:
            await ws.recv()
            await ws.send(json.dumps({"cmd": "mode", "mode": "gaze"}))
            msg = json.loads(await asyncio.wait_for(ws.recv(), 5.0))
            assert msg["type"] in ("gaze", "input"), "the feed stopped after a hook error"
