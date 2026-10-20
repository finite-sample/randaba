"""What the jitter buys: balance against randomness of the partition.

Randomized ABA at jitter in {0, 0.1, 0.3, 1, 3, 10}; at 10 the sort order is
close to a uniformly random permutation, so that level isolates what the sort
contributes. Randomness is summarized by co-assignment: q_ij, the probability
that units i and j share an arm. Complete randomization gives every pair
(m - 1) / (N - 1); a fixed partition gives every pair 0 or 1. Reported: the
largest q_ij and the root mean square gap from the complete-randomization
value (whose own Monte Carlo floor is reported alongside), with the metrics of
many_arms.py. p = 5, 10 covariate draws per cell.
"""

from pathlib import Path

import many_arms as base
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from randaba import designs

OUT = Path("results/jitter.parquet")
CELLS = Path("results/jitter")
JITTERS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)


def coassignment(labels, k):
    """Largest off-diagonal co-assignment probability and RMS gap from CRD."""
    d, n = labels.shape
    onehot = np.eye(k, dtype=np.float32)[labels]
    q = np.einsum("dik,djk->ij", onehot, onehot) / d
    off = q[~np.eye(n, dtype=bool)]
    target = (n / k - 1) / (n - 1)
    return float(off.max()), float(np.sqrt(((off - target) ** 2).mean()))


def run(k, m, rep):
    """Every jitter level on one covariate draw, saved together."""
    path = CELLS / f"{k}_{m}_{rep}.parquet"
    if path.exists():
        return
    x, y0, z, tau = base.cell(k, m, 5, rep)
    rng = np.random.default_rng([k, m, rep, 2030])
    n = k * m
    levels = [("complete randomization", lambda: designs.crd(n, k, rng))] + [
        (f"jitter {j:g}", lambda j=j: designs.randomized_aba(x, k, rng, jitter=j))
        for j in JITTERS
    ]
    rows = []
    for name, draw in levels:
        labels = np.array([draw() for _ in range(base.DRAWS)])
        q_max, q_gap = coassignment(labels, k)
        row = {"K": k, "m": m, "p": 5, "rep": rep, "design": name}
        rows.append(
            row
            | {"q_max": q_max, "q_gap": q_gap}
            | base.evaluate(labels, x, k, y0, z, tau)
        )
    pd.DataFrame(rows).to_parquet(path)


if __name__ == "__main__":
    CELLS.mkdir(parents=True, exist_ok=True)
    todo = [(k, m, r) for k, m in ((10, 10), (10, 40), (50, 10)) for r in range(10)]
    Parallel(n_jobs=4)(delayed(run)(*j) for j in todo)
    frames = [pd.read_parquet(f) for f in sorted(CELLS.glob("*.parquet"))]
    pd.concat(frames).to_parquet(OUT)
