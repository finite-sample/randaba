# balance_buys

[![CI](https://github.com/finite-sample/randaba/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/randaba/actions/workflows/ci.yml)
[![Docs](https://github.com/finite-sample/randaba/actions/workflows/docs.yml/badge.svg)](https://github.com/finite-sample/randaba/tree/main/docs)

What covariate balancing at assignment buys over regression adjustment in a
randomized experiment, and what it costs. This repository holds the designs,
their closed forms, randomization inference, and every simulation behind the
note in `paper/`.

## Findings

- **What balance buys.** Exact balance beats complete randomization analyzed
  by OLS by a degrees-of-freedom factor, `(N-3)/(N-p-3)` with Gaussian
  covariates. That is under 5% with a handful of covariates, and 20 to 35% at
  17 to 25 covariates per 100 units.
- **What it costs.** The cost lies in the distribution of assignments. A
  design that draws from one coarse partition pays it in the tail: grouped
  randomization's error over unmeasured covariates has only `K - 1` degrees of
  freedom. Small groups balanced on means make omitted squares and
  interactions worse than complete randomization.
- **What to do.** With few covariates per unit, randomize and adjust. With
  many, balance the covariates and their squares by local search from random
  starts (greedy pair-switching), and test by redrawing the design. In the
  note's comparison this came within 2.5% of the exact-balance optimum and
  roughly tripled protection against omitted nonlinear terms.

## Installation

```bash
pip install git+https://github.com/finite-sample/randaba
```

The simulations need extra packages: `uv sync --group sims`.

## Usage

Assign by greedy pair-switching on the covariates and their squares, then test
by redrawing the design:

```python
import numpy as np

from balance_buys import designs, grouped
from balance_buys.inference import ols_coef, ri_pvalue

rng = np.random.default_rng(0)
x = rng.normal(size=(120, 20))  # 120 units, 20 baseline covariates
features = np.column_stack([x, x**2])  # balance the covariates and their squares

z = designs.greedy_switch(x, rng, features=features)  # the assignment to run

# After the experiment, test by redrawing the design.
y = x.sum(axis=1) + x[:, 0] ** 2 + 0.5 * z + rng.normal(size=120)  # stand-in outcome
reference = np.array(
    [designs.greedy_switch(x, rng, features=features) for _ in range(1000)]
)
ols_coef(y, z, x)  # 0.214
ri_pvalue(ols_coef(y, z, x), ols_coef(y, reference, x))  # 0.226
```

For grouped randomization, the closed forms report what a partition costs
before the experiment runs:

```python
groups = grouped.partition(x, k=30, rng=rng)
grouped.expected_imbalance(x, groups)  # 6.06; about p = 20 for a random split
grouped.local_optimum_bound(x, groups)  # 16.37, a certificate on the line above
grouped.relative_mse(x[:, 0] ** 2, groups)  # 1.21: an omitted x1^2 costs 21% more
```

## Reproducing the note

| Script | What it does |
|---|---|
| `scripts/gk_rate.py` | How balance scales with group size (Proposition 4's heuristic rate) |
| `scripts/gk_bound.py` | Checks the local-optimum certificate on 400 partitions |
| `scripts/gk_tail.py` | The error tail of grouped randomization against `K` |
| `scripts/gk_omitted.py` | Omitted squares and interactions under grouped randomization |
| `scripts/gk_compare.py` | The comparison of designs |
| `scripts/paper_exhibits.py` | Writes `paper/comparison.tex` and `paper/tail.pdf` from the results |

```bash
uv sync --group sims
uv run python scripts/gk_compare.py   # about 15 minutes on 8 cores
uv run python scripts/gk_tail.py
uv run python scripts/paper_exhibits.py
cd paper && latexmk -pdf note.tex
```

The documentation site holds the longer project report and the full theory,
including the pilot that preceded the note.

## License

MIT.
