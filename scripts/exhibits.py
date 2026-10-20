"""Tables and figures for the note and README, from the saved study results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

PAPER = Path("paper")
README_DATA = Path("docs/_static/readme_comparison.csv")
README_DESIGNS = {
    "complete randomization": "Complete randomization",
    "blocked (matched k-tuples)": "Blocking",
    "rerandomization (1%)": "Rerandomization (1%)",
    "exchange anticlustering": "Exchange anticlustering",
    "sequential CAR": "Sequential CAR",
    "sequential CAR, + squares": "Sequential CAR + squares",
    "randomized ABA": "Randomized ABA",
    "randomized ABA, + squares": "Randomized ABA + squares",
}
ORDER = [
    "complete randomization",
    "rerandomization (1%)",
    "blocked (matched k-tuples)",
    "sequential CAR",
    "sequential CAR, + squares",
    "exchange anticlustering",
    "ARMM",
    "randomized ABA",
    "randomized ABA, + squares",
    "ABA, no jitter",
    "greedy pair-switching",
    "greedy pair-switching, + squares",
    "randomized ABA, aggregated pairs",
]
METRICS = {
    "imbalance": "Imbalance",
    "imbalance_squares": "Imbalance sq.",
    "ols_z_q95": "Tail",
    "mse_dm": "MSE",
    "ri_power_dm": "Power",
    "ri_size_dm": "Size",
}


def load(name):
    """Cell results, or every per-cell file if the run has not been combined."""
    path = Path(f"results/{name}.parquet")
    if path.exists():
        return pd.read_parquet(path)
    return pd.concat(
        pd.read_parquet(f) for f in Path(f"results/{name}").glob("*.parquet")
    )


def relative(df, keys):
    """Imbalance and MSE as ratios to complete randomization in the same cell."""
    base = df[df.design == "complete randomization"].set_index(keys)
    out = df.set_index(keys).copy()
    for col in ("imbalance", "imbalance_squares", "mse_dm", "mse_ols"):
        out[col] = out[col] / base[col].reindex(out.index).to_numpy()
    return out.reset_index()


# One color per design, fixed across panels even when a design is absent.
COLORS = dict(
    zip(
        ORDER,
        plt.get_cmap("tab20").colors[::2] + plt.get_cmap("tab20").colors[1::2],
        strict=False,
    )
)

SHORT = {
    "complete randomization": "complete",
    "rerandomization (1%)": "rerandomization",
    "blocked (matched k-tuples)": "blocked",
    "exchange anticlustering": "exchange",
    "greedy pair-switching": "greedy switching",
    "greedy pair-switching, + squares": "greedy switching, + sq.",
    "sequential CAR, + squares": "sequential CAR, + sq.",
    "randomized ABA, + squares": "randomized ABA, + sq.",
    "randomized ABA, aggregated pairs": "randomized ABA, pairs",
}


def latex(frame, path, caption, label, precision=2):
    """Write a booktabs table through pandas' Styler."""
    frame = frame.rename(index=SHORT)
    styler = frame.style.format(precision=precision, escape="latex")
    fine = [c for c in frame.columns if str(c).startswith("Imbalance")]
    styler = styler.format(precision=3, subset=fine)
    styler = styler.format_index(escape="latex", axis=0).format_index(
        escape="latex", axis=1
    )
    path.write_text(
        styler.to_latex(
            caption=caption,
            label=label,
            position="t",
            position_float="centering",
            hrules=True,
            clines="skip-last;data",
            multirow_align="t",
        )
    )


def table(df, by, path, caption, label):
    """Mean of each metric by design within the ``by`` cells."""
    rel = relative(df, [*by, "rep"])
    summary = rel.groupby([*by, "design"])[list(METRICS)].mean()
    summary = summary.rename(columns=METRICS).reset_index()
    summary["design"] = pd.Categorical(summary.design, ORDER, ordered=True)
    summary = summary.sort_values([*by, "design"]).set_index([*by, "design"])
    summary.index = summary.index.rename([*by, "Design"])
    latex(summary, path, caption, label)
    return summary


COST = pd.DataFrame(
    [
        ("Complete randomization", "$O(N)$", "$O(N)$"),
        ("Blocked on an index", "$O(Np^2 + N \\log N)$", "$O(N)$"),
        ("Rerandomization", "$O(Np/\\alpha)$ expected", "$O(N)$ per candidate"),
        ("Sequential CAR", "$O(NKq)$", "$O(Kq)$, streaming"),
        ("ARMM", "$O(m\\,K!\\,K^2 p)$", "$O(K!\\,K)$"),
        ("Exchange, per pass", "$O(N^2 p)$", "$O(Kp)$; $O(N^2)$ with distances"),
        (
            "Randomized ABA",
            "$O(Np^2 + p^3 + N\\log N + NK(K+p))$",
            "$O(Np + p^2 + K^2p)$",
        ),
    ],
    columns=["Design", "Time", "Memory"],
).set_index("Design")


def cost_table(path):
    """Time and memory of one draw, from the counts in the note."""
    styler = COST.style.format_index(escape="latex", axis=0)
    path.write_text(
        styler.to_latex(
            caption="Cost of one draw. $N$ units, $K$ arms of $m = N/K$, $p$"
            " covariates, $q$ balancing features, $\\alpha$ the rerandomization"
            " acceptance rate. Memory excludes the covariates themselves.",
            label="tab:cost",
            position="t",
            position_float="centering",
            hrules=True,
            column_format="lll",
        )
    )


def jitter_table(jit, path):
    """Balance, tail and co-assignment by jitter level."""
    rel = relative(jit, ["K", "m", "rep"])
    cols = {
        "imbalance": "Imbalance",
        "ols_z_q95": "OLS tail",
        "q_max": "Max co-assignment",
        "q_gap": "RMS gap",
    }
    out = rel.groupby(["K", "m", "design"])[list(cols)].mean().rename(columns=cols)
    order = ["complete randomization"] + [
        f"jitter {j:g}" for j in (0, 0.1, 0.3, 1, 3, 10)
    ]
    out = out.reset_index()
    out["design"] = pd.Categorical(out.design, order, ordered=True)
    out = out.sort_values(["K", "m", "design"]).set_index(["K", "m", "design"])
    out.index = out.index.rename(["K", "m", "Design"])
    latex(
        out,
        path,
        "Randomized ABA by jitter $\\sigma$, $p = 5$. Imbalance is relative to"
        " complete randomization; co-assignment is the probability that two units"
        " share an arm, whose RMS gap from complete randomization's value includes"
        " Monte Carlo noise of about complete randomization's own gap.",
        "tab:jitter",
        precision=3,
    )
    return out


def figure(df, path):
    """Imbalance and OLS tail against K, by design, for each (m, p)."""
    rel = relative(df, ["K", "m", "p", "rep"])
    mean = rel.groupby(["m", "p", "K", "design"])[["imbalance", "ols_z_q95"]].mean()
    mean = mean.reset_index()
    cells = sorted(mean[["m", "p"]].drop_duplicates().itertuples(index=False))
    fig, axes = plt.subplots(
        2, len(cells), figsize=(3.2 * len(cells), 5.6), sharex=True, squeeze=False
    )
    for j, (m, p) in enumerate(cells):
        sub = mean[(mean.m == m) & (mean.p == p)]
        for name in ORDER:
            line = sub[sub.design == name].sort_values("K")
            if line.empty or name == "complete randomization":
                continue
            look = {
                "linestyle": "-" if "ABA" in name else "--",
                "color": COLORS[name],
                "marker": "o",
                "ms": 3,
            }
            axes[0, j].plot(line.K, line.imbalance, label=SHORT.get(name, name), **look)
            axes[1, j].plot(line.K, line.ols_z_q95, **look)
        crd = sub[sub.design == "complete randomization"].sort_values("K")
        axes[1, j].plot(crd.K, crd.ols_z_q95, ":", color="black", label="complete")
        axes[0, j].set_yscale("log")
        axes[0, j].set_title(f"m = {m} per arm, p = {p}", fontsize=9)
        axes[1, j].set_xscale("log")
        axes[1, j].set_xticks([4, 10, 20, 50], ["4", "10", "20", "50"])
        axes[1, j].minorticks_off()
        axes[1, j].set_xlabel("arms, K")
    axes[0, 0].set_ylabel("imbalance / complete randomization")
    axes[1, 0].set_ylabel("OLS error, 95th pct over z")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    extra, extra_labels = axes[1, 0].get_legend_handles_labels()
    handles, labels = handles + extra, labels + extra_labels
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def readme_ratios(frame, metric):
    """Paired ratios for the complete 20-dataset README comparison."""
    if not (frame[["K", "m", "p"]].to_numpy() == [20, 40, 5]).all():
        raise ValueError("README data must use K=20, m=40, p=5")
    wide = frame.pivot(index="rep", columns="design", values=metric)
    if (
        set(wide.columns) != set(README_DESIGNS)
        or list(wide.index) != list(range(20))
        or not np.isfinite(wide.to_numpy()).all()
        or not (wide > 0).all().all()
    ):
        raise ValueError("Expected 20 complete datasets with positive finite metrics")
    return wide.div(wide["complete randomization"], axis=0)[list(README_DESIGNS)]


def readme_figure(*, refresh=False):
    """Plot paired balance and MSE ratios, with intervals across datasets."""
    if refresh:
        source = load("many_arms")
        frame = source.loc[
            (source.K == 20)
            & (source.m == 40)
            & (source.p == 5)
            & source.design.isin(README_DESIGNS),
            ["K", "m", "p", "rep", "design", "imbalance", "mse_dm"],
        ].sort_values(["rep", "design"])
    else:
        frame = pd.read_csv(README_DATA)
    ratios = [readme_ratios(frame, metric) for metric in ("imbalance", "mse_dm")]
    if refresh:
        frame.to_csv(README_DATA, index=False)

    with plt.rc_context({"font.size": 10, "svg.hashsalt": "randaba-readme"}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
        fig.subplots_adjust(left=0.25, right=0.93, bottom=0.18, top=0.75, wspace=0.4)
        for ax, values in zip(axes, ratios, strict=True):
            mean = 100 * values.mean()
            half = 100 * t.ppf(0.975, len(values) - 1) * values.sem()
            for row, name in enumerate(README_DESIGNS):
                color = {
                    "randomized ABA": "#007f86",
                    "randomized ABA, + squares": "#b04a16",
                }.get(name, "#657486")
                ax.errorbar(
                    mean[name],
                    row,
                    xerr=half[name],
                    fmt="o",
                    color=color,
                    markersize=6,
                    capsize=3,
                    linewidth=1.5,
                )
                ax.text(
                    1.03,
                    row,
                    f"{mean[name]:.0f}%" if mean[name] >= 10 else f"{mean[name]:.2g}%",
                    va="center",
                    color=color,
                    transform=ax.get_yaxis_transform(),
                    fontsize=10,
                )
            ax.axvline(100, color="#aab2bb", linestyle="--", linewidth=1, zorder=0)
            ax.grid(axis="x", alpha=0.15)
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0, pad=12)
            ax.set_xlabel("% of complete randomization · lower is better", labelpad=12)
        axes[0].set_yticks(range(len(README_DESIGNS)), README_DESIGNS.values())
        axes[0].set_ylim(len(README_DESIGNS) - 0.4, -0.6)
        axes[0].set_xscale("log")
        axes[0].set_xlim(0.05, 160)
        axes[0].set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
        axes[0].minorticks_off()
        axes[0].set_title("Baseline mean imbalance", loc="left", pad=16, weight="bold")
        axes[1].set_xlim(0, 110)
        axes[1].set_xticks([0, 25, 50, 75, 100])
        axes[1].set_title("Estimation error (MSE)", loc="left", pad=16, weight="bold")
        fig.text(
            0.025,
            0.94,
            "Better mean balance is only part of the story",
            fontsize=19,
            weight="bold",
            color="#22313f",
        )
        fig.text(
            0.025,
            0.87,
            "800 participants · 20 arms · nonlinear outcome · 20 simulated datasets",
            fontsize=12,
            color="#526170",
        )
        fig.text(
            0.025,
            0.035,
            "Points: mean paired ratios. Bars: 95% t intervals across datasets. "
            "400 assignments per design per dataset.",
            fontsize=9,
            color="#526170",
        )
        for extension in ("png", "svg"):
            metadata = {"Date": None} if extension == "svg" else None
            path = README_DATA.with_suffix(f".{extension}")
            fig.savefig(
                path,
                dpi=180,
                facecolor="white",
                metadata=metadata,
            )
            if extension == "svg":
                lines = (line.rstrip() for line in path.read_text().splitlines())
                path.write_text("\n".join(lines) + "\n")
        plt.close(fig)
    print(
        pd.DataFrame(
            {"imbalance_pct": 100 * ratios[0].mean(), "mse_pct": 100 * ratios[1].mean()}
        )
        .round(2)
        .to_string()
    )


def paper_exhibits():
    """Write all tables and the many-arm figure for the research note."""
    many = load("many_arms")
    print(
        table(
            many[(many.m == 40) & many.K.isin([4, 50])],
            ["K", "p"],
            PAPER / "tab_many_arms.tex",
            "Many arms, $m = 40$ units per arm, $K = 4$ and $50$. Imbalance,"
            " on the covariates and on their squares, and MSE of the"
            " difference in means are ratios to complete randomization in the"
            " same draw. Tail is the 95th percentile of OLS error over an"
            " unmeasured covariate, relative to exact balance. Power and size"
            " are for the randomization test of the difference in means.",
            "tab:many",
        ).to_string()
    )
    figure(many, PAPER / "fig_many_arms.pdf")
    cost_table(PAPER / "tab_cost.tex")
    print(
        table(
            load("two_arms"),
            ["p"],
            PAPER / "tab_two_arms.tex",
            "Two arms, $N = 120$; columns as in Table~\\ref{tab:many}.",
            "tab:two",
        ).to_string()
    )
    print(jitter_table(load("jitter"), PAPER / "tab_jitter.tex").to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--readme", action="store_true", help="Plot the README comparison"
    )
    mode.add_argument(
        "--all", action="store_true", help="Write all research-note exhibits"
    )
    parser.add_argument("--refresh", action="store_true", help="Refresh the README CSV")
    args = parser.parse_args()
    if args.refresh and not args.readme:
        parser.error("--refresh requires --readme")
    if args.readme:
        readme_figure(refresh=args.refresh)
    else:
        paper_exhibits()
