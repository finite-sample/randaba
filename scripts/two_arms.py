"""Two arms: randomized ABA against the designs built for two arms.

N = 120 units, p in {5, 20, 30}, 20 covariate draws per cell; metrics and
per-design checkpoints as in many_arms.py. "Aggregated pairs" cuts the units
into N / 2 groups of two by randomized ABA, redrawn on every assignment, and
treats a random half of the groups.
"""

import zlib
from pathlib import Path

import many_arms as base
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from randaba import designs

OUT = Path("results/two_arms.parquet")
CELLS = Path("results/two_arms")
N = 120


def aggregated_pairs(x, rng):
    """Randomized ABA into pairs, then a random half of the pairs treated."""
    groups = designs.randomized_aba(x, len(x) // 2, rng)
    return rng.permutation(np.repeat([0, 1], len(x) // 4))[groups]


DESIGNS = {
    "complete randomization": lambda x, r: designs.crd(len(x), 2, r),
    "rerandomization (1%)": lambda x, r: designs.rerandomize(x, 2, r),
    "greedy pair-switching": lambda x, r: designs.greedy_switch(x, r),
    "greedy pair-switching, + squares": lambda x, r: designs.greedy_switch(
        x, r, features=base.squares(x)
    ),
    "sequential CAR": lambda x, r: designs.sequential_car(x, 2, r),
    "sequential CAR, + squares": lambda x, r: designs.sequential_car(
        base.squares(x), 2, r
    ),
    "randomized ABA": lambda x, r: designs.randomized_aba(x, 2, r),
    "randomized ABA, + squares": lambda x, r: designs.randomized_aba(
        base.squares(x), 2, r
    ),
    "randomized ABA, aggregated pairs": aggregated_pairs,
}


def run(p, rep, name):
    """One design on one covariate draw, saved on its own."""
    slug = name.replace(" ", "_").replace(",", "").replace("+", "plus")
    path = CELLS / f"{p}_{rep}_{slug}.parquet"
    if path.exists():
        return
    x, y0, z, tau = base.cell(2, N // 2, p, rep)
    rng = np.random.default_rng([p, rep, zlib.crc32(name.encode())])
    labels = np.array([DESIGNS[name](x, rng) for _ in range(base.DRAWS)])
    row = {"K": 2, "m": N // 2, "p": p, "rep": rep, "design": name}
    pd.DataFrame([row | base.evaluate(labels, x, 2, y0, z, tau)]).to_parquet(path)


if __name__ == "__main__":
    CELLS.mkdir(parents=True, exist_ok=True)
    todo = [(p, r, name) for p in (5, 20, 30) for r in range(20) for name in DESIGNS]
    Parallel(n_jobs=4)(delayed(run)(*j) for j in todo)
    frames = [pd.read_parquet(f) for f in sorted(CELLS.glob("*.parquet"))]
    pd.concat(frames).to_parquet(OUT)
