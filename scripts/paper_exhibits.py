"""Build the note's table and figure from saved simulation results.

Reads results/gk_compare.parquet and results/gk_tail.parquet; writes
paper/comparison.tex (pandas Styler, booktabs) and paper/tail.pdf (matplotlib).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullFormatter

PAPER = Path("paper")
LABELS = {
    "CRD": "Complete randomization",
    "rerandomization": "Rerandomization",
    "greedy pair-switching": "Greedy pair-switching",
    "greedy pair-switching, + squares": "Greedy, + squares",
    "greedy pair-switching, + squares and products": ("Greedy, + squares, products"),
    "matched pairs": "Matched pairs",
    "G(K), pairs": "Grouped (pairs)",
    "G(K), groups of 4": "Grouped (4)",
    "G(K), groups of 4, + squares": "Grouped (4), + squares",
}


def comparison_table() -> None:
    """MSE over z, omitted-signal ESS, randomization-test power and size, by p."""
    df = pd.read_parquet("results/gk_compare.parquet")
    df = df[df.design.isin(LABELS)]
    rows = []
    for (p, design), g in df.groupby(["p", "design"]):
        crd = df[(df.p == p) & (df.design == "CRD")].set_index("rep")
        sig = g.set_index("rep").mse_signal
        rows.append(
            {
                "$p$": p,
                "Design": LABELS[design],
                "Mean": g.tail_mean.mean(),
                "95th": g.tail_q95.mean(),
                "ESS": crd.mse_signal.loc[sig.index].mean() / sig.mean(),
                "Lin.": g.ri_power_linear.mean(),
                "Nonlin.": g.ri_power_nonlinear.mean(),
                "Size": (g.ri_size_linear.mean() + g.ri_size_nonlinear.mean()) / 2,
            }
        )
    order = {name: i for i, name in enumerate(LABELS.values())}
    tab = (
        pd.DataFrame(rows)
        .assign(_o=lambda d: d.Design.map(order))
        .sort_values(["$p$", "_o"])
        .drop(columns="_o")
        .set_index(["$p$", "Design"])
    )
    latex = (
        tab.style.format(precision=3, subset=["Mean", "95th"])
        .format(
            precision=2,
            subset=["ESS", "Lin.", "Nonlin.", "Size"],
        )
        .to_latex(
            hrules=True,
            clines=None,
            column_format="llrrrrrr",
            multirow_align="t",
        )
    )
    (PAPER / "comparison.tex").write_text(latex)


def tail_figure() -> None:
    """MSE mean and 95th percentile against K, against complete randomization."""
    tab = pd.read_parquet("results/gk_tail.parquet")
    tab = tab.groupby(["p", "design", "K"])[["mean", "q95"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    for ax, p in zip(axes, (5, 20), strict=True):
        g = tab.loc[(p, "G(K)+OLS")]
        c = tab.loc[(p, "CRD+OLS")].iloc[0]
        ax.plot(g.index, g.q95, "o-", ms=3, label="Grouped, 95th percentile")
        ax.plot(g.index, g["mean"], "s-", ms=3, label="Grouped, mean")
        ax.axhline(c.q95, color="C3", label="Complete rand., 95th percentile")
        ax.axhline(c["mean"], color="C3", ls=":", label="Complete rand., mean")
        ax.set_xscale("log")
        ticks = [2, 4, 8, 12, 20, 30, 60]
        ax.set_xticks(ticks, [str(k) for k in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Number of groups $K$ ($N = 120$)")
        ax.set_title(f"$p = {p}$")
    axes[0].set_ylabel("OLS MSE over $z$\n(relative to exact balance)")
    axes[1].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(PAPER / "tail.pdf")


if __name__ == "__main__":
    comparison_table()
    tail_figure()
