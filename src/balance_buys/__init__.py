"""What covariate balancing at assignment buys over regression, and what it costs."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("balance_buys")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.0.0"
