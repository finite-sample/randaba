"""Summarize the pilot: size, power and effective sample size per cell."""

from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results" / "pilot"
ORDER = ["CRD", "rerand", "exchange", "shell", "FSM"]
TESTS = {"p_t": "t-test", "p_ols": "OLS", "p_ri_dm": "RI diff", "p_ri_ols": "RI OLS"}


def load() -> pd.DataFrame:
    """Read every pilot cell."""
    d = pd.concat(pd.read_parquet(f) for f in sorted(RES.glob("*.parquet")))
    d["design"] = pd.Categorical(d["design"], ORDER, ordered=True)
    return d


def rates(d: pd.DataFrame, null: bool) -> pd.DataFrame:
    """Rejection rate at 0.05 for each test, under the null or the alternative."""
    sub = d[(d.tau == 0) == null]
    cols = [c for c in TESTS if c in sub]
    out = sub.groupby(["N", "p", "dgp", "design"], observed=True)[cols].agg(
        lambda s: (s < 0.05).mean()
    )
    return out.rename(columns=TESTS)


def ess(d: pd.DataFrame) -> pd.DataFrame:
    """MSE of CRD analyzed with OLS, divided by each design-estimator MSE."""
    sub = d[d.tau > 0].copy()
    sub["se_dm"] = (sub.est_dm - sub.tau) ** 2
    sub["se_ols"] = (sub.est_ols - sub.tau) ** 2
    mse = sub.groupby(["N", "p", "dgp", "design"], observed=True)[
        ["se_dm", "se_ols"]
    ].mean()
    ref = mse.xs("CRD", level="design")["se_ols"]
    out = pd.DataFrame(
        {
            "ESS diff vs CRD+OLS": ref.reindex(mse.index.droplevel("design")).to_numpy()
            / mse["se_dm"].to_numpy(),
            "ESS OLS vs CRD+OLS": ref.reindex(mse.index.droplevel("design")).to_numpy()
            / mse["se_ols"].to_numpy(),
        },
        index=mse.index,
    )
    return out.replace([np.inf, -np.inf], np.nan)


if __name__ == "__main__":
    d = load()
    pd.set_option("display.width", 200)
    per_cell = d.groupby(["N", "p", "design"], observed=True).dataset.nunique()
    print(f"{len(d)} rows; datasets per cell: {per_cell.min()}")
    print(
        "\nSIZE (rejection rate when tau = 0; target 0.05, simulation SE about 0.011)"
    )
    print(rates(d, null=True).round(3).to_string())
    print("\nPOWER (tau = 1.96 sqrt(8/N))")
    print(rates(d, null=False).round(3).to_string())
    print("\nEFFECTIVE SAMPLE SIZE relative to CRD analyzed with OLS (>1 is better)")
    print(ess(d).round(2).to_string())
