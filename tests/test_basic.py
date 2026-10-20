"""Basic package tests."""

from importlib.metadata import version

import randaba


def test_version() -> None:
    assert randaba.__version__ == version("randaba")
