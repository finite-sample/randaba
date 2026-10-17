"""Proposition 9: MSE from omitted squares and interactions under G(K).

For each omitted component g (a square or a pairwise product of covariates,
residualized on X as OLS leaves it), the ratio of its MSE under G(K) to complete
randomization. Partitions balance either the covariates only or the covariates
plus their squares and pairwise products.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from balance_buys import grouped

OUT = Path("results/gk_omitted.parquet")
N = 120


def features(x: np.ndarray) -> np.ndarray:
    """Squares and pairwise products."""
    p = x.shape[1]
    cols = [x[:, j] ** 2 for j in range(p)]
    cols += [x[:, a] * x[:, b] for a, b in combinations(range(p), 2)]
    return np.column_stack(cols)


def one(p: int, k: int, rep: int) -> list[dict]:
    """Relative MSE of each omitted component for both partitions."""
    rng = np.random.default_rng([p, k, rep, 11])
    x = rng.normal(size=(N, p))
    design = np.column_stack([np.ones(N), x])
    phi = features(x)
    resid = phi - design @ np.linalg.lstsq(design, phi, rcond=None)[0]
    out = []
    for name, basis in (("means", x), ("means+moments", np.column_stack([x, phi]))):
        groups = grouped.partition(basis, k, rng)
        ratios = [
            grouped.relative_mse(resid[:, j], groups) for j in range(phi.shape[1])
        ]
        out.append(
            {
                "p": p,
                "K": k,
                "m": N // k,
                "rep": rep,
                "partition": name,
                "EQ": grouped.expected_imbalance(x, groups),
                "R_mean": float(np.mean(ratios)),
                "R_max": float(np.max(ratios)),
            }
        )
    return out


if __name__ == "__main__":
    jobs = [(p, k, r) for p in (3, 5) for k in (30, 40, 60) for r in range(20)]
    rows = [
        row
        for res in Parallel(n_jobs=-1)(delayed(one)(*j) for j in jobs)
        for row in res
    ]
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    df.to_parquet(OUT)
    tab = df.groupby(["p", "K", "partition"])[["EQ", "R_mean", "R_max"]].agg(
        ["mean", "sem"]
    )
    pd.set_option("display.width", 160)
    print(tab.round(3).to_string())
