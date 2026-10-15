"""Closed-form predictions (docs/theory.md) next to the pilot's simulated values."""

import numpy as np
from scipy import integrate, stats
from summarize import ess, load, rates


def df_gain(n: int, p: int) -> tuple[float, float]:
    """ESS of an exactly balanced design over CRD+OLS: Jensen bound and chi2 value."""
    bound = (n - 1) / (n - 1 - p)
    c = n / (n - 1)

    def integrand(q: float) -> float:
        return n / (n - c * q) * stats.chi2.pdf(q, p)

    val, _ = integrate.quad(integrand, 0, (n - 1) / c * 0.999)
    return bound, val


def ri_vs_t(kappa: float) -> tuple[float, float, float]:
    """t-test size and power, and RI power, when true SE = kappa x the t-test's SE.

    The effect is 1.96 of the t-test's standard errors, as in the pilot. The
    estimate over the t-test's SE is N(1.96, kappa^2) under the alternative.
    """
    size_t = 2 * stats.norm.sf(1.96 / kappa)
    pow_t = stats.norm.sf(0.0) + stats.norm.cdf(-2 * 1.96 / kappa)
    pow_ri = stats.norm.sf(1.96 - 1.96 / kappa) + stats.norm.cdf(-1.96 - 1.96 / kappa)
    return size_t, pow_t, pow_ri


if __name__ == "__main__":
    d = load()
    e = ess(d)
    print("ESS of a balanced design + OLS over CRD + OLS (linear outcome)")
    print(
        f"{'N':>4} {'p':>3} {'bound':>7} {'chi2':>6}   simulated rerand/exchange/shell"
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

    print("\nFSM, nonlinear outcome: predicted (1 + V_g / sigma^2) x df gain")
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
