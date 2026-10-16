"""Closed-form predictions (docs/theory.md) next to the pilot's simulated values."""

import numpy as np
from scipy import stats
from summarize import ess, load, rates


def df_gain(n: int, p: int) -> tuple[float, float]:
    """ESS of an exactly balanced design over CRD+OLS: Jensen bound, exact Gaussian.

    The exact value uses Q / N ~ Beta(p/2, (N-1-p)/2) under Gaussian covariates.
    """
    bound = (n - 1) / (n - 1 - p)
    exact = (n - 3) / (n - p - 3)
    return bound, exact


def gaussian_check(
    n: int, p: int, reps: int = 20000, seed: int = 0
) -> tuple[float, float]:
    """Monte Carlo N E[1/(N-Q)] under Gaussian X and CRD, and its standard error."""
    rng = np.random.default_rng(seed)
    w = np.repeat([1.0, -1.0], n // 2)
    vals = np.empty(reps)
    for r in range(reps):
        x = rng.normal(size=(n, p))
        x -= x.mean(axis=0)
        wr = rng.permutation(w)
        coef = np.linalg.lstsq(x, wr, rcond=None)[0]
        vals[r] = n / (n - wr @ x @ coef)
    return float(vals.mean()), float(vals.std() / np.sqrt(reps))


def ri_vs_t(kappa: float) -> tuple[float, float, float]:
    """t-test size and power, and RI power, when true SE = kappa x the t-test's SE.

    The effect is 1.96 of the t-test's standard errors, as in the pilot. The
    estimate over the t-test's SE is N(1.96, kappa^2) under the alternative.
    """
    size_t = 2 * stats.norm.sf(1.96 / kappa)
    pow_t = stats.norm.sf(0.0) + stats.norm.cdf(-2 * 1.96 / kappa)
    pow_ri = stats.norm.sf(1.96 - 1.96 / kappa) + stats.norm.cdf(-1.96 - 1.96 / kappa)
    return float(size_t), float(pow_t), float(pow_ri)


if __name__ == "__main__":
    d = load()
    e = ess(d)
    print("ESS of a balanced design + OLS over CRD + OLS (linear outcome)")
    print(
        f"{'N':>4} {'p':>3} {'bound':>7} {'exact':>6}   simulated rerand/exchange/shell"
    )
    for n, p in [
        (20, 1),
        (20, 5),
        (50, 1),
        (50, 5),
        (50, 20),
        (100, 5),
        (100, 20),
        (300, 5),
        (300, 20),
    ]:
        b, v = df_gain(n, p)
        sim = [
            e.loc[(n, p, "linear", g), "ESS OLS vs CRD+OLS"]
            for g in ("rerand", "exchange", "shell")
        ]
        print(
            f"{n:4d} {p:3d} {b:7.2f} {v:6.2f}   " + " / ".join(f"{s:.2f}" for s in sim)
        )

    print("\nExact Gaussian formula against Monte Carlo (no design involved)")
    for n, p in [(20, 5), (50, 20), (100, 20)]:
        mc, se = gaussian_check(n, p)
        print(f"{n:4d} {p:3d} formula {df_gain(n, p)[1]:.4f}   MC {mc:.4f} ± {se:.4f}")

    print("\nFSM, nonlinear outcome: heuristic (1 + V_g / sigma^2) x df gain")
    for n, p in [(50, 5), (100, 5), (300, 5), (300, 20)]:
        _, v = df_gain(n, p)
        sim = e.loc[(n, p, "nonlinear", "FSM"), "ESS OLS vs CRD+OLS"]
        print(f"{n:4d} {p:3d} predicted {1.5 * v:.2f}   simulated {sim:.2f}")

    r0, r1 = rates(d, True), rates(d, False)
    print("\nt-test vs RI on the difference in means under a mean-balancing design")
    for dgp, kappa in [("linear", np.sqrt(1 / 2)), ("nonlinear", np.sqrt(1.5 / 2))]:
        s, pt, pr = ri_vs_t(kappa)
        cells = [(n, p) for n, p in [(50, 5), (100, 5), (300, 5)]]
        sim = [
            (
                r0.loc[(n, p, dgp, "exchange"), "t-test"],
                r1.loc[(n, p, dgp, "exchange"), "t-test"],
                r1.loc[(n, p, dgp, "exchange"), "RI diff"],
            )
            for n, p in cells
        ]
        print(
            f"{dgp:9s} kappa {kappa:.3f}: predicted t size {s:.3f}, "
            f"t power {pt:.2f}, RI power {pr:.2f}"
        )
        for (n, p), (a, b, c) in zip(cells, sim, strict=True):
            print(
                f"           simulated exchange N={n} p={p}: t size {a:.3f}, "
                f"t power {b:.2f}, RI power {c:.2f}"
            )
