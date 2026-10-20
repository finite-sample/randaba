# Simulation setup and reproduction

The README figure shows one setting from the many-arm study: 800 participants,
20 arms, and five baseline covariates. It compares eight designs on the same
20 independent datasets, with 400 independently drawn assignments per design
and dataset. Randomized ABA uses `jitter=1`.

## Data and outcome model

Each entry of the covariate matrix `X` is an independent standard-normal draw.
For each dataset, construct:

- `L`: the sum of all five covariates, centered and scaled to empirical
  variance one.
- `Q`: the sum of the first three covariates' squares and their three distinct
  pairwise products, also centered and scaled to empirical variance one.
- `epsilon`: independent standard-normal noise for each participant.

The outcome is `Y = (L + Q) / sqrt(2) + epsilon`, held fixed across all
assignments and designs within a dataset. Each scaled signal component has
empirical variance one-half; their sum need not have variance exactly one
because their empirical covariance need not be zero. The true treatment
contrast in the figure is zero.

The “+ squares” variants balance all five covariates and their five squares.
They do not explicitly balance the interactions in `Q`. Blocking matches
consecutive participants ordered by the first principal component.
Rerandomization uses the lower 1% chi-square balance threshold. Exchange
anticlustering runs one pass on whitened covariates. Sequential CAR uses
Zhang's allocation rule with `k0=3` and allows unequal arm sizes. See
[scripts/many_arms.py](https://github.com/finite-sample/randaba/blob/main/scripts/many_arms.py)
for the implementation of every design and metric.

## What the figure measures

**Baseline imbalance** is the sum, over arms, of the squared norm of each
arm's mean whitened covariate vector, weighted by its size. The figure
always evaluates the original five covariates, even for designs that also
balance squares.

**Estimation error** is the mean squared difference in outcomes between
arms 0 and 1 across 400 assignments. With a true effect of zero, this is
the mean squared error of the unadjusted difference-in-means estimator.

For each metric, divide each design's value by complete randomization's
value on the same dataset. Plot the mean of those 20 ratios as a percentage.
Error bars are approximate 95% Student-t confidence intervals for the mean
ratio: `mean ± t(0.975, 19) × sample_sd / sqrt(20)`. The independent units
for these intervals are the datasets, not the 8,000 assignments. The
complete-randomization ratio is exactly one by construction.

## Rebuild the README figure

The [saved CSV](_static/readme_comparison.csv) contains one row per dataset
and design, extracted from `results/many_arms.parquet` at `K=20, m=40, p=5`.
It includes the raw imbalance and MSE values before normalization.
From the repository root:

```bash
uv run --locked --group sims python scripts/exhibits.py --readme
```

This uses the committed CSV and writes the PNG and SVG into `docs/_static/`.
To refresh that CSV after rerunning the many-arm simulations:

```bash
uv run --locked --group sims python scripts/exhibits.py --readme --refresh
```

This requires the local simulation results. It checks that all eight plotted
methods have exactly the 20 expected datasets before updating the artifact.

## Reproduce the full study

The full grid uses 4, 10, 20, and 50 arms; 10 or 40 participants per arm;
5 or 20 baseline covariates; and 20 independent datasets per setting.
The separate two-arm study uses 120 participants. The jitter study varies
the ordering noise to study the tradeoff between balance and random variation.

Install R and its `CARM` package for the ARMM comparison. ARMM enumerates
`K!` allocations; it is evaluated only at four arms, on 10 datasets per
setting, excluding 40 participants per arm with 20 covariates. It therefore
does not appear in the README's 20-arm figure.

```bash
uv sync --locked --group sims
uv run --group sims python scripts/many_arms.py
uv run --group sims python scripts/two_arms.py
uv run --group sims python scripts/jitter.py
uv run --group sims python scripts/exhibits.py --all
cd paper && latexmk -pdf note.tex
```

The simulation scripts save each completed cell and skip it on later runs.
Remove or move old results before rerunning a changed simulation specification.
The many-arm dataset seed is `[K, m, p, rep, 2027]`; each design has its own
seed `[K, m, p, rep, crc32(design_name)]`. Record dependency versions when
reproducing results. The full study can take substantial time and memory;
rebuilding the README figure from the CSV does not rerun it.

The studies cover Gaussian covariates, a single nonlinear outcome model,
and at most 2,000 participants. They measure balance, estimation error, and
test behavior. Computational scaling is established separately by the
[operation counts](method.md).
