"""The base install must stay importable with `websockets` absent (D8, rule 3).

`websockets` is the `server` extra and the base install deliberately does not
carry it. Three things therefore have to hold, and all three are checked here
rather than assumed from the fact that nothing obviously imports it:

1. `focusedgaze` and `focusedgaze.core` import cleanly.
2. `focusedgaze.server` imports cleanly too, so a caller can reference the
   classes and read the docs without the extra.
3. Running the server raises `ServerError` carrying the install command, never a
   bare `ImportError`.

The absence is simulated with a meta-path finder raising `ModuleNotFoundError`,
which is what Python raises when a package is genuinely missing. Audit 43.6
records why the distinction matters: a plain `ImportError` means
installed-but-broken, a different condition, and simulating with the wrong one
tests something that never happens.
"""

from __future__ import annotations

import asyncio
import importlib
import sys

import pytest

from focusedgaze.exceptions import GazeError, ServerError


class _BlockWebsockets:
    """Make `websockets` unimportable for the duration of a test."""

    def find_spec(self, name, path=None, target=None):
        if name == "websockets" or name.startswith("websockets."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)


@pytest.fixture
def no_websockets(monkeypatch: pytest.MonkeyPatch):
    """Remove `websockets` from the process, and restore focusedgaze after.

    The restore matters more than it looks. These tests re-import `focusedgaze`
    from scratch, which creates a **new** `ServerError` class object; anything
    still holding the old one would then fail an `isinstance` check against an
    identically named class. Leaving those fresh modules in `sys.modules` would
    hand that confusion to every test that ran afterwards.
    """
    saved = {name: mod for name, mod in sys.modules.items() if name.startswith("focusedgaze")}
    for module in [m for m in sys.modules if m == "websockets" or m.startswith("websockets.")]:
        monkeypatch.delitem(sys.modules, module)
    blocker = _BlockWebsockets()
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("websockets")
    yield blocker
    for name in [m for m in sys.modules if m.startswith("focusedgaze")]:
        del sys.modules[name]
    sys.modules.update(saved)


@pytest.mark.parametrize(
    "module",
    [
        "focusedgaze",
        "focusedgaze.core",
        "focusedgaze.core.filters",
        "focusedgaze.core.positioning",
        "focusedgaze.capture",
        "focusedgaze.config",
        "focusedgaze.calibration",
        "focusedgaze.assets",
        "focusedgaze.diagnostics",
    ],
)
def test_the_package_imports_without_websockets(no_websockets, module: str) -> None:
    """None of these may drag in an optional extra at import time."""
    for name in [m for m in sys.modules if m.startswith("focusedgaze")]:
        del sys.modules[name]
    importlib.import_module(module)
    assert "websockets" not in sys.modules


def test_the_server_module_itself_imports_without_websockets(no_websockets) -> None:
    """Referencing the classes must not require the extra.

    The deferred import is inside `serve_forever`, so everything a caller needs
    in order to construct a server, read its docstrings, or type-annotate
    against it is available in a base install.
    """
    for name in [m for m in sys.modules if m.startswith("focusedgaze")]:
        del sys.modules[name]
    server = importlib.import_module("focusedgaze.server")
    assert "websockets" not in sys.modules
    snapshot = server.GazeSnapshot(ok=True, x=0.1, y=0.2, t=1.0)
    assert server.gaze_message(snapshot)["type"] == "gaze"
    assert server.minimal_input_message(snapshot)["type"] == "input"


def test_running_the_server_without_websockets_names_the_remedy(no_websockets) -> None:
    """D8: never a bare ImportError. The message must carry the fix."""
    for name in [m for m in sys.modules if m.startswith("focusedgaze")]:
        del sys.modules[name]
    server_module = importlib.import_module("focusedgaze.server")
    # The freshly imported package has its OWN ServerError class object. Catching
    # the one imported at the top of this file would compare against a different
    # class with the same name and never match.
    fresh_error = importlib.import_module("focusedgaze.exceptions").ServerError

    class Source:
        def latest(self):
            return server_module.GazeSnapshot()

        def pause(self, timeout: float = 4.0) -> bool:
            return True

        def resume(self) -> bool:
            return True

    server = server_module.GazeServer(Source(), port=0)
    with pytest.raises(fresh_error) as excinfo:
        asyncio.run(server.serve_forever())
    message = str(excinfo.value)
    assert "focusedgaze[server]" in message
    assert "pip install" in message


def test_the_server_error_is_catchable_as_one_package_error() -> None:
    """Every error this package raises derives from GazeError (D7)."""
    assert issubclass(ServerError, GazeError)
