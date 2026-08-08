"""The WebSocket server extra. [extra: server]

    from focusedgaze.server import GazeServer

    server = GazeServer(source)
    asyncio.run(server.serve_forever())

**Importing this package does not import `websockets`.** The dependency is
deferred to the moment the server is actually run, so `focusedgaze.core` and
everything else stay importable in a base install that has no `server` extra.
Running without it raises :class:`~focusedgaze.exceptions.ServerError` carrying
the install command, never a bare ``ImportError`` (D8).

What this ships, and what it deliberately does not, is set out in
`docs/wire_format.md` section 6.4. In short: the transport, both pacing rules,
the ``gaze`` message, a **minimal gaze-only** ``input`` message, the camera lease
as :meth:`~focusedgaze.server.websocket.GazeServer.pause` /
:meth:`~focusedgaze.server.websocket.GazeServer.resume`, and three hooks. No
gesture vocabulary: no mode concept, no ``hand_ok``, no ``click_seq``, no pinch.
Those stay in the game repo's wrapper, which replaces the input message through
the ``resolve_input`` hook.
"""

from __future__ import annotations

from .websocket import (
    DEFAULT_HOST,
    DEFAULT_PAUSE_TIMEOUT_S,
    DEFAULT_PORT,
    DEFAULT_RESUME_SETTLE_S,
    DEFAULT_SEND_HZ,
    GazeServer,
    GazeSnapshot,
    GazeSource,
    gaze_message,
    minimal_input_message,
    validate_host,
    validate_port,
)

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PAUSE_TIMEOUT_S",
    "DEFAULT_PORT",
    "DEFAULT_RESUME_SETTLE_S",
    "DEFAULT_SEND_HZ",
    "GazeServer",
    "GazeSnapshot",
    "GazeSource",
    "gaze_message",
    "minimal_input_message",
    "validate_host",
    "validate_port",
]
