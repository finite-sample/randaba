"""Finite Selection Model allocation at scale: many units, many groups."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fsm-scale")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.0.0"
