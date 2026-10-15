"""Sphinx configuration — fleet standard via py-canon."""

from py_canon.sphinx import configure

configure(globals())

# Typeset math in the theory pages ($...$ and $$...$$).
myst_enable_extensions = [*globals().get("myst_enable_extensions", []), "dollarmath"]
