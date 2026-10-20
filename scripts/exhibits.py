"""Tables and figures for the note, from the results of the three studies."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PAPER = Path("paper")
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
        ("Randomized ABA", "$O(Np^2 + N\\log N + NK(K+p))$", "$O(N + Kp + K^2)$"),
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


if __name__ == "__main__":
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
