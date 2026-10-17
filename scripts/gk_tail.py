"""Choosing K: the tail criterion of Kapelner et al. (2021) for OLS under G(K).

For each K the MSE of the OLS treatment estimate over unobserved z ~ N(0, I) is
sum_i lambda_i chi2_1 (theory.md, Proposition 8). Reported relative to 4/N, the
MSE of an exactly balanced design, next to the closed-form approximations
(1 + E[Q]/N) chi2_{K-1}(q)/(K-1) for G(K) and
(N-3)/(N-p-3) chi2_{N-p-1}(q)/(N-p-1) for complete randomization with OLS.
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter
from scipy import stats

from shelldraft import grouped

OUT = Path("results/gk_tail.parquet")
FIG = Path("docs/_static/gk_tail.png")
N = 120
KS = [2, 4, 6, 8, 10, 12, 20, 24, 30, 40, 60]
Q = 0.95


def direct_quantile(x, groups, rng, reps=40000):
    """Quantile of the MSE over z by brute force: no eigenvalues, no reduction."""
    n = len(x)
    design = np.column_stack([np.ones(n), x])
    resid = np.eye(n) - design @ np.linalg.pinv(design)
    k = groups.max() + 1
    ws = []
    for treated in combinations(range(k), k // 2):
        s = -np.ones(k)
        s[list(treated)] = 1
        ws.append(s[groups])
    ws = np.array(ws)
    denom = n - np.einsum("ij,jk,ik->i", ws, np.eye(n) - resid, ws)
    z = rng.normal(size=(reps, n))
    err = 2 * (z @ resid @ ws.T) / denom
    return float(np.quantile((err**2).mean(axis=1), Q))


if __name__ == "__main__":
    rows = []
    for p in (5, 20):
        for rep in range(5):
            rng = np.random.default_rng([p, rep])
            x = rng.normal(size=(N, p))
            crd = grouped.crd_ols_error_eigenvalues(x, rng, draws=40000)
            base = {"p": p, "rep": rep}
            rows.append(
                base
                | {
                    "K": N,
                    "design": "CRD+OLS",
                    "mean": crd.sum() * N / 4,
                    "q95": grouped.mse_quantile(crd, Q, rng) * N / 4,
                    "EQ": p * N / (N - 1),
                }
            )
            for k in KS:
                groups = grouped.partition(x, k, rng)
                lam = grouped.ols_error_eigenvalues(x, groups, rng, draws=40000)
                rows.append(
                    base
                    | {
                        "K": k,
                        "design": "G(K)+OLS",
                        "mean": lam.sum() * N / 4,
                        "q95": grouped.mse_quantile(lam, Q, rng) * N / 4,
                        "EQ": grouped.expected_imbalance(x, groups),
                    }
                )
            if rep == 0:
                groups = grouped.partition(x, 10, rng)
                lam = grouped.ols_error_eigenvalues(x, groups)
                print(
                    f"p={p} K=10 check: eigenvalue route "
                    f"{grouped.mse_quantile(lam, Q, rng) * N / 4:.4f}, "
                    f"brute force over z {direct_quantile(x, groups, rng) * N / 4:.4f}"
                )
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    df.to_parquet(OUT)
    tab = df.groupby(["p", "design", "K"])[["mean", "q95", "EQ"]].mean()
    approx = []
    for (p, design, k), row in tab.iterrows():
        if design == "G(K)+OLS":
            approx.append((1 + row.EQ / N) * stats.chi2.ppf(Q, k - 1) / (k - 1))
        else:
            dof = N - p - 1
            approx.append((N - 3) / (N - p - 3) * stats.chi2.ppf(Q, dof) / dof)
    tab["q95 closed form"] = approx
    pd.set_option("display.width", 160)
    print(tab.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, p in zip(axes, (5, 20), strict=True):
        g = tab.loc[(p, "G(K)+OLS")]
        c = tab.loc[(p, "CRD+OLS")].iloc[0]
        ax.plot(g.index, g.q95, "o-", label="G(K)+OLS, 95% quantile")
        ax.plot(g.index, g["q95 closed form"], "--", color="grey", label="closed form")
        ax.plot(g.index, g["mean"], "s-", label="G(K)+OLS, mean")
        ax.axhline(c.q95, color="C3", label="CRD+OLS, 95% quantile")
        ax.axhline(c["mean"], color="C3", ls=":", label="CRD+OLS, mean")
        ax.set_xscale("log")
        ax.set_xticks(KS, [str(k) for k in KS])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("K (number of groups), N = 120")
        ax.set_title(f"p = {p} covariates")
    axes[0].set_ylabel("MSE of OLS over unobserved z,\nrelative to exact balance (4/N)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=150)
