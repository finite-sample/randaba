"""Pilot simulation: designs x analyses x data-generating processes (docs/pilot.md).

One Parquet file per (N, p, design) cell in results/pilot/, skipped if present,
so an interrupted run resumes where it stopped.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from registry import DESIGNS

from shelldraft import inference

OUT = Path(__file__).resolve().parents[1] / "results" / "pilot"
GRID = [
    (20, 1),
    (20, 5),
    (50, 1),
    (50, 5),
    (50, 20),
    (100, 1),
    (100, 5),
    (100, 20),
    (300, 1),
    (300, 5),
    (300, 20),
]


def signal(x: np.ndarray, kind: str) -> np.ndarray:
    """Return the control-outcome signal f(X), scaled to variance 1."""
    lin = x.sum(axis=1)
    lin = (lin - lin.mean()) / lin.std()
    if kind == "linear":
        return lin
    x3 = x[:, :3]
    quad = np.column_stack(
        [
            x3**2 - 1,
            x3[:, [0]] * x3[:, [1]],
            x3[:, [0]] * x3[:, [2]],
            x3[:, [1]] * x3[:, [2]],
        ]
    ).sum(1)
    quad = (quad - quad.mean()) / quad.std()
    return (lin + quad) / np.sqrt(2)


def one_dataset(n: int, p: int, design: str, s: int, m: int) -> list[dict]:
    """Draw one covariate set, one observed split and m reference splits; test."""
    rng = np.random.default_rng([n, p, s])
    x = rng.normal(size=(n, p))
    noise = {k: rng.normal(size=n) for k in ("linear", "nonlinear")}
    drng = np.random.default_rng([n, p, s, list(DESIGNS).index(design)])
    zs = np.array([DESIGNS[design](x, drng) for _ in range(m + 1)])
    z, ref = zs[0], zs[1:]
    can_ols = p + 2 < n
    tau1 = 1.96 * np.sqrt(8 / n)
    rows = []
    for kind in ("linear", "nonlinear") if p >= 3 else ("linear",):
        y0 = signal(x, kind) + noise[kind]
        for tau in (0.0, tau1):
            y = y0 + tau * z
            row = {
                "N": n,
                "p": p,
                "design": design,
                "dataset": s,
                "dgp": kind,
                "tau": tau,
                "est_dm": float(inference.diff_means(y, z)),
                "p_t": inference.t_test(y, z),
            }
            row["p_ri_dm"] = inference.ri_pvalue(
                row["est_dm"], inference.diff_means(y, ref)
            )
            if can_ols:
                row["est_ols"] = float(inference.ols_coef(y, z, x))
                row["p_ols"] = inference.ols_test(y, z, x)
                row["p_ri_ols"] = inference.ri_pvalue(
                    row["est_ols"], inference.ols_coef(y, ref, x)
                )
            rows.append(row)
    return rows


def main() -> None:
    """Run every missing (N, p, design) cell."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=int, default=400)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=9)
    ap.add_argument("--designs", nargs="*", default=list(DESIGNS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for n, p in GRID:
        for design in args.designs:
            path = OUT / f"N{n}_p{p}_{design}.parquet"
            if path.exists():
                continue
            t = time.perf_counter()
            res = Parallel(n_jobs=args.jobs)(
                delayed(one_dataset)(n, p, design, s, args.draws)
                for s in range(args.datasets)
            )
            pd.DataFrame([r for rows in res for r in rows]).to_parquet(path)
            print(
                f"N={n:3d} p={p:2d} {design:10s} {time.perf_counter() - t:7.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
