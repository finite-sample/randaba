"""How G(K)'s expected imbalance E[Q] scales with group size m.

Prediction (theory.md, Proposition 7): exchange anticlustering run to a local
optimum gives E[Q] of order m^-(1 + 2/p) N^(-2/p), while a random partition
gives about p.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from balance_buys import grouped
from balance_buys.designs import crd

OUT = Path("results/gk_rate.parquet")
SIZES = [2, 4, 6, 8, 10, 12, 20, 24, 30, 40, 60, 120]


def one(n: int, p: int, m: int, rep: int) -> dict:
    """E[Q] for one covariate draw, anticlustered and randomly partitioned."""
    rng = np.random.default_rng([n, p, m, rep])
    x = rng.normal(size=(n, p))
    k = n // m
    groups = grouped.partition(x, k, rng)
    random_groups = crd(n, k, rng)
    return {
        "N": n,
        "p": p,
        "m": m,
        "K": k,
        "rep": rep,
        "EQ_anticluster": grouped.expected_imbalance(x, groups),
        "EQ_random": grouped.expected_imbalance(x, random_groups),
    }


if __name__ == "__main__":
    jobs = [
        (n, p, m, rep)
        for n in (240, 480, 960)
        for p in (1, 2, 5, 10)
        for m in SIZES
        if (n // m) % 2 == 0
        for rep in range(20)
    ]
    rows = Parallel(n_jobs=-1)(delayed(one)(*j) for j in jobs)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    df.to_parquet(OUT)
    agg = df.groupby(["N", "p", "m"])[["EQ_anticluster", "EQ_random"]].mean()
    pd.set_option("display.width", 160)
    print(agg.unstack("N").round(5).to_string())
    print("\nlog E[Q] on log m and log N (m >= 6), against the prediction:")
    for p, g in df[df.m >= 6].groupby("p"):
        mean = g.groupby(["N", "m"]).EQ_anticluster.mean().reset_index()
        design = np.column_stack([np.ones(len(mean)), np.log(mean.m), np.log(mean.N)])
        coef = np.linalg.lstsq(design, np.log(mean.EQ_anticluster), rcond=None)[0]
        print(
            f"p={p:2d}  log m {coef[1]:6.2f} (predicted {-(1 + 2 / p):5.2f})"
            f"  log N {coef[2]:6.2f} (predicted {-2 / p:5.2f})"
        )
