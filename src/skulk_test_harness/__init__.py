"""Agent-controlled Skulk end-to-end test and benchmark harness."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

# Read the version from package metadata so pyproject.toml stays the single
# source of truth (a hardcoded string here drifted from it once already). The
# fallback covers running from a source tree that was never installed.
try:
    __version__ = version("skulk-test-harness")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
