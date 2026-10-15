"""Two-group designs compared in the simulations."""

from collections.abc import Callable

import finite_selection as fs
import numpy as np

from shelldraft import designs


def _fsm(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    order = fs.som([n // 2, n // 2], method="scomars", rng=rng)
    return fs.fsm(np.atleast_2d(x.T).T, order, rng=rng).assignment - 1


DESIGNS: dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "CRD": lambda x, r: designs.crd(len(x), 2, r),
    "exchange": lambda x, r: designs.exchange_anticlust(x, 2, r),
    "rerand": lambda x, r: designs.rerandomize(x, r, accept=0.001),
    "FSM": _fsm,
    "shell": lambda x, r: designs.shell_draft(x, 2, r, jitter=1.0),
}
