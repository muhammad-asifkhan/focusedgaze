"""Console entry point.

Full command set lands in Phase 6: download-models, calibrate, check, serve, demo,
export-onnx. Phase 1 ships only ``--version`` and a status line, so the entry point
declared in pyproject.toml resolves to something real rather than dangling.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``focusedgaze`` console script."""
    parser = argparse.ArgumentParser(
        prog="focusedgaze",
        description="Webcam eye-gaze tracking. In development — see MIGRATION_AUDIT.md.",
    )
    parser.add_argument("--version", action="version", version=f"focusedgaze {__version__}")
    parser.parse_args(argv)

    # Nothing to run yet. Say so plainly rather than pretending to be a tool.
    print(f"focusedgaze {__version__} — package skeleton (Phase 1).")
    print("No commands are implemented yet; they arrive in Phase 6.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
