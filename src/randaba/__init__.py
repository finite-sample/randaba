"""Randomized assignment-based anticlustering and randomization inference."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("randaba")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.0.0"
