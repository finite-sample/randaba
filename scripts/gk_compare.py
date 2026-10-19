"""Compare G(K) with other balancing designs where it can win: many covariates.

N = 120 and p in {5, 20, 30}; 20 covariate draws per p. Every design is
analyzed by OLS on (1, T, X). For each covariate draw and design, 4000
assignments are drawn and reused for every metric:

- MSE over unobserved z ~ N(0, I): mean and 95% quantile, relative to 4/N
  (Kapelner et al.'s tail criterion; theory.md, Proposition 8).
- MSE from omitted structure: the squares and interactions of three
  covariates, with no noise, after OLS on X (theory.md, Proposition 9). The
  expected MSE of an outcome with this structure plus noise of variance
  sigma^2 is this term plus sigma^2 times the mean over z.
- Size and power at the 5% level of the randomization test that redraws the
  design, and of the classical OLS t-test, from 200 observed assignments. The
  effect is 1.96 standard deviations of the CRD + OLS estimate.
"""

from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from balance_buys import grouped
from balance_buys.designs import exchange_anticlust, whiten

OUT = Path("results/gk_compare.parquet")
N = 120
DRAWS = 4000
OBSERVED = 200
ZDRAWS = 2000


def crd(_x, rng):
    """Complete randomization."""
    base = np.repeat([1.0, -1.0], N // 2)
    return rng.permuted(np.tile(base, (DRAWS, 1)), axis=1)


def rerand(x, rng, accept=0.001):
    """Morgan-Rubin rerandomization on the Mahalanobis distance."""
    w = whiten(x)
    threshold = stats.chi2.ppf(accept, x.shape[1])
    base = np.repeat([1.0, -1.0], N // 2)
    kept = []
    while sum(len(k) for k in kept) < DRAWS:
        cand = rng.permuted(np.tile(base, (8192, 1)), axis=1)
        dist = ((cand @ w) ** 2).sum(axis=1) / (N - 1)
        kept.append(cand[dist <= threshold])
    return np.concatenate(kept)[:DRAWS]


def greedy(x, rng, basis=None):
    """Greedy pair-switching: two-group exchange to a local optimum."""
    w = whiten(x if basis is None else basis)
    out = np.empty((DRAWS, N))
    for r in range(DRAWS):
        out[r] = 2.0 * exchange_anticlust(w, 2, rng, local_max=True) - 1
    return out


def matched_pairs(x, rng):
    """Optimal non-bipartite Mahalanobis matching, randomized within pairs."""
    w = whiten(x)
    dist = np.linalg.norm(w[:, None] - w[None], axis=-1)
    graph = nx.Graph()
    big = dist.max() + 1
    for i, j in combinations(range(N), 2):
        graph.add_edge(i, j, weight=big - dist[i, j])
    pairs = np.array(sorted(nx.max_weight_matching(graph, maxcardinality=True)))
    flips = rng.choice([1.0, -1.0], size=(DRAWS, len(pairs)))
    out = np.empty((DRAWS, N))
    out[:, pairs[:, 0]] = flips
    out[:, pairs[:, 1]] = -flips
    return out


def gk(x, rng, m, basis=None):
    """G(K) with groups of m, partitioned on ``basis`` (default the covariates)."""
    groups = grouped.partition(x if basis is None else basis, N // m, rng)
    k = N // m
    labels = rng.permuted(np.tile(np.repeat([1.0, -1.0], k // 2), (DRAWS, 1)), axis=1)
    return labels[:, groups]


def squares(x):
    """Covariates and their squares."""
    return np.column_stack([x, x**2])


def second_moments(x):
    """Covariates, squares and pairwise products, or None if too many for N."""
    p = x.shape[1]
    if 2 * p + p * (p - 1) // 2 > N // 4:
        return None
    prods = [x[:, a] * x[:, b] for a, b in combinations(range(p), 2)]
    return np.column_stack([x, x**2, *prods])


DESIGNS = {
    "CRD": (crd, 1.0),
    "rerandomization": (rerand, np.nan),
    "greedy pair-switching": (greedy, np.nan),
    "matched pairs": (matched_pairs, 2 * (N - 1) / N),
    "G(K), pairs": (lambda x, r: gk(x, r, 2), (N - 1) / (N // 2 - 1)),
    "G(K), groups of 3": (lambda x, r: gk(x, r, 3), (N - 1) / (N // 3 - 1)),
    "G(K), groups of 4": (lambda x, r: gk(x, r, 4), (N - 1) / (N // 4 - 1)),
    "G(K), groups of 4, + squares": (
        lambda x, r: gk(x, r, 4, squares(x)),
        (N - 1) / (N // 4 - 1),
    ),
    "greedy pair-switching, + squares": (
        lambda x, r: greedy(x, r, squares(x)),
        np.nan,
    ),
    "greedy pair-switching, + squares and products": (
        lambda x, r: (
            None if second_moments(x) is None else greedy(x, r, second_moments(x))
        ),
        np.nan,
    ),
}


def outcomes(x, rng):
    """Linear and nonlinear control outcomes with noise, and the omitted signal."""
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
    noise = rng.normal(size=N)
    return {
        "linear": lin + noise,
        "nonlinear": (lin + quad) / np.sqrt(2) + noise,
        "signal": quad / np.sqrt(2),
    }


def evaluate(w, x, ys, tau, rng):
    """Every metric for one design's assignment draws."""
    xc = x - x.mean(axis=0)
    basis = np.linalg.qr(xc)[0]
    proj = w @ basis
    denom = N - (proj**2).sum(axis=1)
    resid_w = w - proj @ basis.T
    z = rng.normal(size=(ZDRAWS, N))
    mse_z = ((2 * (z @ resid_w.T) / denom) ** 2).mean(axis=1) * N / 4
    out = {"tail_mean": mse_z.mean(), "tail_q95": np.quantile(mse_z, 0.95)}
    design1 = np.column_stack([np.ones(N), x])
    m1 = np.eye(N) - design1 @ np.linalg.pinv(design1)
    for name, y0 in ys.items():
        err = 2 * (resid_w @ y0) / denom
        out[f"mse_{name}"] = float((err**2).mean())
        if name == "signal":
            continue
        for label, effect in (("size", 0.0), ("power", tau)):
            obs = np.arange(OBSERVED)
            yobs = y0[None, :] + effect * (w[obs] + 1) / 2
            ref = 2 * (yobs @ resid_w.T) / denom[None, :]
            stat = ref[obs, obs]
            exceed = (np.abs(ref) >= np.abs(stat)[:, None] - 1e-12).sum(axis=1) - 1
            p_ri = (1 + exceed) / DRAWS
            y1 = yobs @ m1
            t1 = resid_w[obs] / 2
            rss = (y1**2).sum(axis=1) - stat**2 * (t1**2).sum(axis=1)
            se = np.sqrt(rss / (N - x.shape[1] - 2) / (t1**2).sum(axis=1))
            p_t = 2 * stats.t.sf(np.abs(stat) / se, N - x.shape[1] - 2)
            out[f"ri_{label}_{name}"] = float((p_ri <= 0.05).mean())
            out[f"t_{label}_{name}"] = float((p_t <= 0.05).mean())
    return out


def one(p, rep):
    """All designs on one covariate draw."""
    rng = np.random.default_rng([p, rep, 2026])
    x = rng.normal(size=(N, p))
    ys = outcomes(x, rng)
    w_crd = crd(x, rng)
    xc = x - x.mean(axis=0)
    basis = np.linalg.qr(xc)[0]
    proj = w_crd @ basis
    resid_w = w_crd - proj @ basis.T
    sd = np.std(2 * (resid_w @ ys["linear"]) / (N - (proj**2).sum(axis=1)))
    tau = 1.96 * sd
    rows = []
    for name, (draw, worst) in DESIGNS.items():
        w = w_crd if name == "CRD" else draw(x, rng)
        if w is None:
            continue
        rows.append(
            {"p": p, "rep": rep, "design": name, "worst_case": worst}
            | evaluate(w, x, ys, tau, rng)
        )
    return rows


if __name__ == "__main__":
    jobs = [(p, r) for p in (5, 20, 30) for r in range(20)]
    res = Parallel(n_jobs=-1)(delayed(one)(*j) for j in jobs)
    df = pd.DataFrame([row for rows in res or [] for row in rows])
    OUT.parent.mkdir(exist_ok=True)
    df.to_parquet(OUT)
    cols = [
        "worst_case",
        "tail_mean",
        "tail_q95",
        "ri_size_nonlinear",
        "ri_power_linear",
        "ri_power_nonlinear",
        "t_size_linear",
        "t_power_linear",
    ]
    order = list(DESIGNS)
    pd.set_option("display.width", 250)
    for p, g in df.groupby("p"):
        present = [d for d in order if d in set(g.design)]
        mean = g.groupby("design")[cols].mean().loc[present]
        sem = g.groupby("design")[cols].sem().loc[present]
        for name in ("linear", "nonlinear", "signal"):
            mse = g.pivot(index="rep", columns="design", values=f"mse_{name}")
            present = [d for d in order if d in mse]
            mean[f"ess_{name}"] = mse["CRD"].mean() / mse[present].mean()
        print(f"\np = {p}: mean across 20 covariate draws (ESS: ratio of mean MSEs)")
        print(mean.round(3).to_string())
        print("standard error")
        print(sem.round(3).to_string())
