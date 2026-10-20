# randaba

[![CI](https://github.com/finite-sample/randaba/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/randaba/actions/workflows/ci.yml)
[![Docs](https://github.com/finite-sample/randaba/actions/workflows/docs.yml/badge.svg)](https://finite-sample.github.io/randaba/)

Randomized assignment-based anticlustering: random assignment to many arms
that keeps the arms alike on baseline covariates. Assignment-based
anticlustering (ABA; Baumann, Goldschmidt, Hochbaum and Yang 2026) splits
units into equal groups that are as similar as possible, fast enough for
millions of units, but it is deterministic. Jittering the sort that defines
its batches and shuffling the group labels turn it into an experimental
design: every unit has probability `1/K` of each arm, contrasts between arms
are unbiased, and the randomization test that redraws the design is exact.
The note in `paper/` gives the theory and a comparison with the designs that
run at many arms.

## Findings

- **ABA's step is ARMM's step, without the K! cost.** Each batch of ABA is
  assigned by the allocation that most reduces the multi-arm Mahalanobis
  imbalance, the same choice the adaptive design ARMM makes by scoring all
  `K!` allocations. ABA finds it with one assignment problem, `O(K^3)`, so
  the design runs at 50 arms, where ARMM cannot.
- **Best balance among designs linear in N.** With 4 to 50 arms, randomized
  ABA left 0.4 to 39 percent of complete randomization's imbalance on the
  covariate means, less than rerandomization, blocking, ARMM and sequential
  covariate-adaptive randomization (Zhang 2023) in every setting, and its
  lead grew with the number of arms. The exchange method of anticlustering
  balanced better but costs `O(N^2)` per pass.
- **Balance the squares too.** On the covariates alone, randomized ABA left
  imbalance on the omitted squares at up to 1.44 times complete
  randomization's. Run on the covariates and their squares, it gave the
  smallest error and the most power for a nonlinear outcome in nearly every
  many-arm setting (at 40 units per arm, power 0.64 to 0.72 against 0.48 to
  0.51 for complete randomization), with exact test size.
- **The jitter is what makes the test work.** Without it, only the labels
  are random; at 4 arms the test could not reject at all. A little jitter
  restores the test and the protection against unmeasured covariates while
  keeping most of the balance.
- **Not for two arms or streaming units.** With two arms, greedy
  pair-switching balanced 5 to 14 times better. When units arrive one at a
  time, use a sequential design such as Zhang's.

## Installation

Requires Python 3.12 or newer. Until the first PyPI release, install from
the repository:

```bash
pip install git+https://github.com/finite-sample/randaba
```

After publication, use `pip install randaba`. The distribution and import
name are both `randaba`.

The simulations need extra packages, `uv sync --group sims`, and the ARMM
comparison needs R with the `CARM` package.

## Usage

Assign 400 units to 20 arms, then use an arm contrast to test the sharp null
of no effect for any unit under any arm:

```python
import numpy as np

from randaba import designs, inference

rng = np.random.default_rng(0)
x = rng.normal(size=(400, 8))  # 400 units, 8 baseline covariates
k = 20  # arms

z = designs.randomized_aba(x, k, rng)  # the assignment to run; 20 units per arm

# After the experiment, use the arm 3 versus arm 0 contrast as the statistic.
y = x.sum(axis=1) + 0.6 * (z == 3) + rng.normal(size=400)  # stand-in outcome
reference = np.array([designs.randomized_aba(x, k, rng) for _ in range(2000)])
est = inference.contrast(y, z, 3, 0)  # 0.44
inference.ri_pvalue(est, inference.contrast(y, reference, 3, 0))  # 0.207
```

The reference assignments must use the same covariates, arm count, design,
and tuning parameters as the assignment that was run. Redrawing all arms
tests the global sharp null; it does not isolate a null about arms 3 and 0
while allowing arbitrary effects in the other arms. The Monte Carlo
p-value includes the observed draw, so it is never zero.

With no effect, that contrast has a standard deviation of 0.34 under this
design and 1.00 under complete randomization of the same units.

To balance the squares of the covariates as well, pass
`np.column_stack([x, x**2])` in place of `x`. `jitter` (default 1) sets how
much the partition varies between draws; `jitter=0` gives one partition,
randomized only by the label shuffle.

Equal-size designs require the number of units to be divisible by the number
of arms, with at least two arms. Supply finite numeric covariates and drop
constant or collinear columns before using designs that whiten them. Save
the assignment actually used in the experiment. A seed reproduces a draw
with the same dependency versions; record those versions too.

The other designs in the comparison are in `randaba.designs`: complete
randomization, blocking on an index, rerandomization for `K` arms, the
sequential covariate-adaptive randomization of Zhang (2023), and the exchange
method of anticlustering.

Rerandomization stops after `max_draws` candidates (default 100,000) and
raises `RuntimeError` if none meets the balance threshold. This can happen
when a threshold is unattainable for the supplied covariates.

## Reproducing the note

| Script | What it does |
|---|---|
| `scripts/many_arms.py` | The comparison at 4 to 50 arms (calls `scripts/armm_draws.R` for ARMM) |
| `scripts/two_arms.py` | The comparison at two arms |
| `scripts/jitter.py` | Balance against randomness as the jitter varies |
| `scripts/exhibits.py` | Writes the tables and figure in `paper/` |

```bash
uv sync --group sims
uv run python scripts/many_arms.py
uv run python scripts/two_arms.py
uv run python scripts/jitter.py
uv run python scripts/exhibits.py --all
cd paper && latexmk -pdf note.tex
```

Each script saves every cell as it finishes and skips finished cells, so an
interrupted run resumes where it stopped.

## Development

```bash
uv sync --locked --group docs
make ci
make ci-docker
make ci-docker PYTHON_VERSION=3.14
```

The checks cover linting, formatting, types, docstrings, tests with at least
95% coverage, documentation, distribution metadata, and tests against an
installed wheel. Docker uses the standard Python image and a temporary copy
of the checkout. The documentation includes this README directly.

See the [release guide](docs/releasing.md)
for the first public release. Report bugs through
[GitHub Issues](https://github.com/finite-sample/randaba/issues).

## License

MIT.
