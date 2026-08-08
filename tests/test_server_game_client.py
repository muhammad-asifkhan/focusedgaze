"""Execute the inference audit section 39 recorded and never ran.

Section 39 states, in bold, that the game playing against the minimal gaze-only
`input` message was **inferred from reading the client, not executed**. It named
that as the same defect class as the server docstring describing a positioning
check the file never performed, and as the example that asserted a tradeoff its
own numbers contradicted.

So this runs the real `input-manager.js` under Node, against a real
`focusedgaze` server, over a real WebSocket, and asserts on the client's own
public API afterwards. Nothing here is simulated except the gaze readings.

The client is loaded **unmodified** apart from its hard-coded `ws://localhost:8765`,
which is repointed at an ephemeral port so the test cannot collide with a real
server. The harness asserts that substitution fired rather than assuming it, per
standing brief A2.

Needs the game checkout, located by `FOCUSEDGAZE_GAME_DIR`, and Node. Skips with
an actionable reason without them, which is the same arrangement the Tier 1
golden tests use for the legacy tree.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import pathlib
import shutil
import subprocess

import pytest

from focusedgaze.server import GazeServer, GazeSnapshot

HARNESS = pathlib.Path(__file__).parent / "js" / "run_input_manager.mjs"


def _game_dir() -> pathlib.Path:
    raw = os.environ.get("FOCUSEDGAZE_GAME_DIR", "")
    if not raw:
        pytest.skip(
            "FOCUSEDGAZE_GAME_DIR unset: point it at the directory holding "
            "input-manager.js to run the browser client against the server."
        )
    path = pathlib.Path(raw)
    if not (path / "input-manager.js").is_file():
        pytest.skip(f"no input-manager.js under {path}")
    return path


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the real browser client cannot be run")
    return node


class ScriptedSource:
    """A gaze source that walks through a fixed list of readings."""

    def __init__(self, readings: list[GazeSnapshot]) -> None:
        self._readings = readings
        self._index = 0

    def latest(self) -> GazeSnapshot:
        snapshot = self._readings[min(self._index, len(self._readings) - 1)]
        self._index += 1
        return snapshot

    def pause(self, timeout: float = 4.0) -> bool:
        return True

    def resume(self) -> bool:
        return True


def _run_client(readings: list[GazeSnapshot], run_ms: int = 2000) -> dict:
    """Serve `readings` and drive the real client against it. Returns its state."""
    game_dir = _game_dir()
    node = _node()

    async def go() -> dict:
        server = GazeServer(ScriptedSource(readings), host="127.0.0.1", port=0, send_hz=60.0)
        task = asyncio.create_task(server.serve_forever())
        for _ in range(400):
            if server.bound_port:
                break
            await asyncio.sleep(0.01)
        assert server.bound_port, "server did not bind"

        url = f"ws://127.0.0.1:{server.bound_port}"
        proc = await asyncio.create_subprocess_exec(
            node, str(HARNESS), url, str(run_ms), str(game_dir),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=60)
        server.stop()
        task.cancel()
        # The task is cancelled, so CancelledError is the expected outcome and
        # anything else is teardown noise that must not mask the assertion below.
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert proc.returncode == 0, (
            f"the client harness failed (exit {proc.returncode}):\n{err.decode(errors='replace')}"
        )
        return json.loads(out.decode())

    return asyncio.run(go())


@functools.lru_cache(maxsize=1)
def _steady_run() -> dict:
    """One run with a good reading held steady. Cached: it costs two seconds."""
    return _run_client([GazeSnapshot(ok=True, x=0.25, y=0.75, t=1000.0)])


# ---------------------------------------------------------------------------
# The thing section 39 could not claim.
# ---------------------------------------------------------------------------


def test_the_real_client_receives_and_accepts_the_minimal_input_message() -> None:
    """The headline: the game's cursor moves, driven by a seven-field message."""
    state = _steady_run()
    assert state["connected"] is True
    assert state["messages_seen"] > 0, "the client received nothing at all"
    assert state["by_type"]["input"] > 0, "no input message reached the client"
    assert state["ok"] is True, "the client did not accept the reading"
    assert state["x"] == 0.25 and state["y"] == 0.75, (
        "the cursor did not take the coordinates the server sent; this is R-1, "
        "the failure mode where the game connects and never moves"
    )


def test_the_absent_gesture_fields_leave_the_client_in_its_safe_defaults() -> None:
    """`hand_ok`, `gesture` and the rest are absent, not zeroed.

    Section 39 predicted `handOk` false, `gesture` "" and `pinching` false from
    the client's `!!m.hand_ok` and `m.gesture || ""` coercions. Executed here.
    """
    state = _steady_run()
    assert state["handOk"] is False
    assert state["gesture"] == ""
    assert state["pinching"] is False
    assert state["gestureAvailable"] is False


def test_dwell_to_click_stays_enabled_because_mode_remains_gaze() -> None:
    """`api.mode` staying "gaze" is what keeps the game playable without gestures."""
    state = _steady_run()
    assert state["mode"] == "gaze"
    assert state["source"] == "gaze"


def test_no_spurious_activation_fires_from_the_missing_sequence_counters() -> None:
    """Q7-2, and the specific thing the brief asked to watch.

    The client syncs `lastClick/lastSwipe/lastClear` from the first `input`
    message, then fires on `m.click_seq > lastClick`. With the counters absent
    that stores `undefined` and compares `undefined > undefined`, which is
    `false` in JavaScript, so nothing should ever fire.

    Predicted in section 39 from the comparison rules. Executed here across
    hundreds of messages: a single spurious click would activate whatever the
    cursor is resting on, which in the game is a purchase or a scene change.
    """
    state = _steady_run()
    fired = [e["event"] for e in state["events"]]
    assert "click" not in fired, f"a click fired with no counters present: {fired}"
    assert "clear" not in fired
    assert "swipe" not in fired
    assert "pinchStart" not in fired
    assert state["by_type"]["input"] >= 20, (
        "too few input messages to make the no-click claim meaningful; "
        f"saw {state['by_type']['input']}"
    )


def test_the_client_discards_the_gaze_message_as_section_39_found() -> None:
    """R-1 restated as a positive check: gaze arrives and changes nothing.

    The server sends both, so if the client were driven by the gaze feed rather
    than by `input` this would not distinguish them. It is asserted because the
    whole revised contract rests on it being true.
    """
    state = _run_client(
        [GazeSnapshot(ok=True, x=0.9, y=0.9, t=float(i)) for i in range(1, 200)]
    )
    assert state["by_type"]["gaze"] > 0, "no gaze message was sent at all"
    assert state["by_type"]["input"] > 0


def test_a_no_face_reading_hides_the_cursor_without_moving_it() -> None:
    """R-6 through the real client: `ok` false must preserve the last position.

    The client's `if (m.ok)` guard is the mechanism. Sending `null` rather than a
    stale point is what makes it work.
    """
    readings = [GazeSnapshot(ok=True, x=0.3, y=0.6, t=1.0)] * 30
    readings += [GazeSnapshot(ok=False, x=None, y=None, t=2.0)] * 90
    state = _run_client(readings)
    assert state["ok"] is False, "the client still believes it has a reading"
    assert state["x"] == 0.3 and state["y"] == 0.6, (
        "the cursor moved on a no-face reading; x/y must be null so the client "
        "keeps its last good position while hiding the cursor"
    )
