"""Randomized balanced assignment by jittered shell anticlustering."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("shelldraft")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.0.0"
