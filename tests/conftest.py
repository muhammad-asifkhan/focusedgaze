"""Make `golden.adapters` importable from the tests directory."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
