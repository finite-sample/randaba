"""Check Proposition 7a: the local-optimum bound on E[Q] holds in every partition."""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from balance_buys import grouped

OUT = Path("results/gk_bound.parquet")


def one(p: int, m: int, rep: int, n: int = 240) -> dict:
    """Bound and E[Q] for one covariate draw."""
    rng = np.random.default_rng([p, m, rep, 7])
    x = rng.normal(size=(n, p))
    groups = grouped.partition(x, n // m, rng)
    bound = grouped.local_optimum_bound(x, groups)
    return {
        "p": p,
        "m": m,
        "rep": rep,
        "EQ": grouped.expected_imbalance(x, groups),
        "bound": bound,
    }


if __name__ == "__main__":
    jobs = [
        (p, m, r) for p in (1, 2, 5, 10) for m in (2, 4, 8, 20, 60) for r in range(20)
    ]
    df = pd.DataFrame(Parallel(n_jobs=-1)(delayed(one)(*j) for j in jobs))
    OUT.parent.mkdir(exist_ok=True)
    df.to_parquet(OUT)
    print(f"partitions: {len(df)}; bound violated: {(df.bound < df.EQ).sum()}")
    summary = df.assign(ratio=df.EQ / df.bound).groupby(["p", "m"])
    print(summary[["EQ", "bound", "ratio"]].mean().round(5).to_string())
