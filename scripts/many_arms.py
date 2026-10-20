"""Many-arm comparison: randomized ABA against designs that run at K arms.

Grid: K in {4, 10, 20, 50} arms, m in {10, 40} units per arm, p in {5, 20}
Gaussian covariates, 20 covariate draws per cell. Each design draws DRAWS
assignments with its own seed; every design in a cell faces the same
covariates, outcome and unmeasured noise. ARMM runs only at K = 4, since it
scores all K! allocations of each block, and on 10 of the 20 draws (not at
m = 40, p = 20), since its R implementation takes about 15 minutes per cell
at N = 160. All metrics concern the contrast
between arms 0 and 1, which by label symmetry stands for any pair.

- imbalance: sum_g n_g |wbar_g|^2 on whitened covariates (and on standardized
  squares); its complete-randomization mean is about (K - 1) p.
- OLS error over unmeasured noise z ~ N(0, I): the squared error of the OLS
  contrast (arm dummies plus covariates) on z, relative to 2 / m, its value
  under exact balance; mean and 95th percentile over z.
- mse_dm, mse_ols: squared error of the difference in means and of OLS on an
  outcome with linear and quadratic parts plus noise.
- Size and power of the randomization test that redraws the design, for the
  difference in means and OLS, at an effect of 1.96 standard deviations of the
  complete-randomization difference in means.
"""

import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from randaba import designs

OUT = Path("results/many_arms.parquet")
CELLS = Path("results/many_arms")
R_SCRIPT = Path(__file__).with_name("armm_draws.R")
RSCRIPT = shutil.which("Rscript") or "Rscript"
DRAWS = 400
OBSERVED = 200
ZDRAWS = 2000


def outcomes(x, rng):
    """Control outcome: scaled linear and quadratic signals plus normal noise."""
    lin = x.sum(axis=1)
    lin = (lin - lin.mean()) / lin.std()
    x3 = x[:, :3]
    quad = np.column_stack(
        [
            x3**2,
            x3[:, [0]] * x3[:, [1]],
            x3[:, [0]] * x3[:, [2]],
            x3[:, [1]] * x3[:, [2]],
        ]
    ).sum(axis=1)
    quad = (quad - quad.mean()) / quad.std()
    return (lin + quad) / np.sqrt(2) + rng.normal(size=len(x))


def cell(k, m, p, rep):
    """Covariates, outcome, unmeasured noise and test effect for one cell."""
    rng = np.random.default_rng([k, m, p, rep, 2027])
    n = k * m
    x = rng.normal(size=(n, p))
    y0 = outcomes(x, rng)
    z = rng.normal(size=(ZDRAWS, n))
    crd = np.array([designs.crd(n, k, rng) for _ in range(2000)])
    tau = 1.96 * np.std(((crd == 0).astype(float) - (crd == 1)) @ y0 / m)
    return x, y0, z, tau


def armm(x, k, seed):
    """DRAWS ARMM assignments from the R package CARM."""
    with tempfile.TemporaryDirectory() as tmp:
        cov, out = Path(tmp, "x.csv"), Path(tmp, "out.csv")
        pd.DataFrame(x).to_csv(cov, index=False)
        # Arguments are paths and integers built here, not user input.
        subprocess.run(  # noqa: S603
            [RSCRIPT, str(R_SCRIPT), str(cov), str(k), str(DRAWS), str(seed), str(out)],
            check=True,
            capture_output=True,
        )
        return pd.read_csv(out).to_numpy()


def squares(x):
    """Covariates with their squares appended."""
    return np.column_stack([x, x**2])


DESIGNS = {
    "complete randomization": lambda x, k, r: designs.crd(len(x), k, r),
    "blocked (matched k-tuples)": lambda x, k, r: designs.blocked(x, k, r),
    "rerandomization (1%)": lambda x, k, r: designs.rerandomize(x, k, r),
    "exchange anticlustering": lambda x, k, r: designs.exchange_anticlust(
        designs.whiten(x), k, r
    ),
    "sequential CAR": lambda x, k, r: designs.sequential_car(x, k, r),
    "sequential CAR, + squares": lambda x, k, r: designs.sequential_car(
        squares(x), k, r
    ),
    "randomized ABA": lambda x, k, r: designs.randomized_aba(x, k, r),
    "randomized ABA, + squares": lambda x, k, r: designs.randomized_aba(
        squares(x), k, r
    ),
}


def draws_for(name, x, k, rng):
    """DRAWS label vectors from a design."""
    if name == "ARMM":
        return armm(x, k, seed=int(rng.integers(1 << 30)))
    if name == "ABA, no jitter":
        part = designs.randomized_aba(x, k, rng, jitter=0.0)
        return np.array([rng.permutation(k)[part] for _ in range(DRAWS)])
    return np.array([DESIGNS[name](x, k, rng) for _ in range(DRAWS)])


def evaluate(labels, x, k, y0, z, tau):
    """All metrics for one design's draws; arms may differ in size."""
    m = len(x) // k
    w = designs.whiten(x)
    sq = (x**2 - (x**2).mean(axis=0)) / (x**2).std(axis=0)
    onehot = np.eye(k)[labels]
    sizes = onehot.sum(axis=1)
    imb = ((np.einsum("dnk,np->dkp", onehot, w) ** 2).sum(axis=2) / sizes).sum(axis=1)
    imb2 = ((np.einsum("dnk,np->dkp", onehot, sq) ** 2).sum(axis=2) / sizes).sum(axis=1)
    contrast = onehot[:, :, 0] / sizes[:, [0]] - onehot[:, :, 1] / sizes[:, [1]]
    basis = np.linalg.qr(x - x.mean(axis=0))[0]
    resid = onehot - np.einsum(
        "np,dpk->dnk", basis, np.einsum("np,dnk->dpk", basis, onehot)
    )
    gram = np.einsum("dnk,dnl->dkl", resid, resid)
    e = np.zeros(k)
    e[0], e[1] = 1.0, -1.0
    rhs = np.broadcast_to(e, (len(labels), k))[..., None]
    coef = np.linalg.solve(gram, rhs)[..., 0]
    ols_w = np.einsum("dnk,dk->dn", resid, coef)
    mse_z = ((ols_w @ z.T) ** 2).mean(axis=0) / (2 / m)
    out = {
        "imbalance": imb.mean(),
        "imbalance_squares": imb2.mean(),
        "ols_z_mean": mse_z.mean(),
        "ols_z_q95": np.quantile(mse_z, 0.95),
        "mse_dm": float(((contrast @ y0) ** 2).mean()),
        "mse_ols": float(((ols_w @ y0) ** 2).mean()),
    }
    obs = np.arange(OBSERVED)
    for label, effect in (("size", 0.0), ("power", tau)):
        yobs = y0[None, :] + effect * onehot[obs, :, 0]
        for stat_name, wts in (("dm", contrast), ("ols", ols_w)):
            ref = yobs @ wts.T
            stat = ref[obs, obs]
            exceed = (np.abs(ref) >= np.abs(stat)[:, None] - 1e-12).sum(axis=1) - 1
            pval = (1 + exceed) / len(labels)
            out[f"ri_{label}_{stat_name}"] = float((pval <= 0.05).mean())
    return out


def run(k, m, p, rep, name):
    """One design in one cell, saved on its own."""
    slug = name.replace(" ", "_").replace(",", "").replace("+", "plus")
    path = CELLS / f"{k}_{m}_{p}_{rep}_{slug}.parquet"
    if path.exists():
        return
    x, y0, z, tau = cell(k, m, p, rep)
    rng = np.random.default_rng([k, m, p, rep, zlib.crc32(name.encode())])
    labels = draws_for(name, x, k, rng)
    row = {"K": k, "m": m, "p": p, "rep": rep, "design": name}
    pd.DataFrame([row | evaluate(labels, x, k, y0, z, tau)]).to_parquet(path)


def jobs():
    """Every (cell, design) pair, smaller cells first."""
    names = [*DESIGNS, "ABA, no jitter", "ARMM"]
    return [
        (k, m, p, rep, name)
        for m in (10, 40)
        for k in (4, 10, 20, 50)
        for p in (5, 20)
        for rep in range(20)
        for name in names
        if not (name == "ARMM" and (k != 4 or rep >= 10 or (m, p) == (40, 20)))
        and not ("squares" in name and 2 * p >= k * m - 1)
    ]


if __name__ == "__main__":
    CELLS.mkdir(parents=True, exist_ok=True)
    todo = jobs()
    if "--count" in sys.argv:
        print(len(todo))
        sys.exit()
    # Large cells hold about 1 GB per worker; four fit in 16 GB.
    Parallel(n_jobs=4)(delayed(run)(*j) for j in todo)
    frames = [pd.read_parquet(f) for f in sorted(CELLS.glob("*.parquet"))]
    pd.concat(frames).to_parquet(OUT)
