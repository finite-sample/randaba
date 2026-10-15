# Theory: what balance buys over regression adjustment

The pilot's numbers follow from five short results. Each is stated for two
equal arms; `scripts/theory_check.py` sets the predictions beside the
simulations.

## Setup

There are `N` units and `N/2` go to each arm. Write the assignment as
`W` in `{-1, +1}^N` with `1'W = 0`. Unit `i` has potential outcomes
`Y_i(1)` and `Y_i(0)`; let `mu_i = (Y_i(1) + Y_i(0)) / 2` and
`tau = mean(Y(1) - Y(0))`. The difference in means is `tau_hat`. A design is a
probability distribution over `W`, and `Sigma = E[W W']`.

## 1. Error identity

**Lemma.** `tau_hat - tau = (2/N) W' mu`, exactly, for any potential outcomes.

*Proof.* Write `Y_i(1) = mu_i + delta_i / 2` and `Y_i(0) = mu_i - delta_i / 2`.
The treated sum minus the control sum is `W' mu + (1/2) sum_i delta_i`.
Multiplying by `2/N` gives `(2/N) W' mu + tau`.

Everything about a design's precision is therefore in `Sigma`.

## 2. Unbiasedness and the variance budget

**Proposition 1.** If the design is unchanged by swapping the labels
(`W` and `-W` have the same distribution), then `E W = 0`, `tau_hat` is
unbiased, and `MSE = (4/N^2) mu' Sigma mu`. Every such design has
`diag(Sigma) = 1`, so `trace(Sigma) = N`.

*Proof.* Symmetry gives `E W = -E W`. Each `W_i^2 = 1`.

Shuffling labels at the end of any procedure makes it symmetric, which is why
the shell draft does it.

**Corollary (the price of balance).** Write `mu = c 1 + X_c beta + r`, with
`X_c` the centred covariates and `r` orthogonal to `1` and `X_c`. If a design
balances exactly (`X_c' W = 0` always), then `MSE = (4/N^2) r' Sigma r`.

- *Averaged* over directions of `r` in the `N - p - 1`-dimensional space it
  can occupy, this is `(4/N^2) |r|^2 N / (N - p - 1)`. Complete randomization
  gives `N / (N - 1)` in place of `N / (N - p - 1)`. Balance on what you
  measured costs a factor `(N - 1) / (N - p - 1)` on average for what you did
  not measure.
- The *worst* direction gives `(4/N^2) |r|^2 lambda_max`, the largest
  eigenvalue of `Sigma` on that space. Nothing bounds it except the design's
  own spread. This is Savje's critique of the FSM.

*Proof.* The projection `P` onto `span(1, X_c)` satisfies `P W = 0`, so
`Sigma P = 0` and the trace of `Sigma` on the complement is all of `N`.

## 3. Balance against OLS: a degrees-of-freedom result

**Proposition 2.** Suppose `Y(0) = alpha + X beta + e` with `e` independent of
`W`, mean zero, variance `sigma^2`, and a constant effect. Let `Q = W' P W`,
where `P` projects onto `X_c`. Then OLS of `Y` on `(1, T, X)` is unbiased given
`W`, and

```text
MSE(design, OLS) = 4 sigma^2 E[ 1 / (N - Q) ].
```

The minimum over all designs, `4 sigma^2 / N`, is reached exactly when `Q = 0`
always, that is, under exact balance. Then OLS and the difference in means
coincide.

*Proof.* Conditional on `W`, OLS has variance `sigma^2 / |T_res|^2`, where
`T_res` is the residual of the treatment indicator `T = (W + 1) / 2` on
`(1, X)`. Since `1'W = 0`, `T_res = (W - P W) / 2`, and
`|T_res|^2 = (N - Q) / 4`.

**Corollary.** Under complete randomization `E[Q] = p N / (N - 1)`. By Jensen's
inequality, the effective sample size of an exactly balanced design over
complete randomization analysed by OLS is

```text
ESS = N E_CRD[ 1 / (N - Q) ]  >=  (N - 1) / (N - 1 - p).
```

It does not depend on `beta` or on how well the covariates predict the
outcome. Regression adjustment spends about `p` degrees of freedom estimating
`beta`; exact balance makes the estimate unnecessary and so returns them. With
one covariate, the case Papenberg and Angelike simulate, the gain is at most
`(N - 1) / (N - 2)`: 6% at `N = 20`, under 1% at `N = 300`. That is why they
found no power gain once they adjusted. The gain exceeds 10% roughly when
`p > 0.09 N`.

## 4. Balancing what the regression leaves out

**Proposition 3.** Add a component `g` to the model:
`Y(0) = alpha + X beta + g + e`, with `g` orthogonal to `span(1, X)` in the
sample and `V_g = |g|^2 / N`. OLS on `X` leaves `g` in the residual, so to
first order complete randomization with OLS has
`MSE = 4 (sigma^2 + V_g) / N` times the degrees-of-freedom factor. A design
that also balances features `Phi` with `g` in `span(Phi)` (`Phi' W = 0`)
removes `g`:

```text
ESS = (1 + V_g / sigma^2) x [degrees-of-freedom factor].
```

A design that balances only the means of `X` leaves `g`, and its ESS is the
degrees-of-freedom factor alone.

The FSM's D-optimal criterion pushes each group's `X_g' X_g` towards half of
`X' X`. That balances every square and cross-product, so the FSM falls in the
first case when `N` is large enough for its groups to match second moments.
Mean-balancing designs (rerandomization, exchange anticlustering, the
centroid-cost shell draft) fall in the second. The same gain is available to
any design that balances the squares and cross-products explicitly.

## 5. Inference

**Proposition 4 (the t-test under balance).** Let `kappa^2` be the share of the
complete-randomization variance of `tau_hat` that the design leaves:
`kappa^2 = (sigma^2 + V_unbalanced) / (sigma^2 + V_total)`. The pooled t-test
estimates the complete-randomization variance whatever the design, so its
statistic is shrunk by `kappa`. Its size is `2 Phi_bar(1.96 / kappa)` instead
of 0.05, and its power at an effect of `1.96` of its own standard errors stays
at about one half. A randomization test that redraws the actual design
measures the true variance, so its power is
`Phi_bar(1.96 - 1.96 / kappa)`.

With half the outcome variance balanced away (`kappa = 0.71`), this gives a
t-test size of 0.006, t-test power 0.50 and randomization-test power 0.79.

**Proposition 5 (resolution of a randomization test).** Under the sharp null
the difference in means satisfies `S(-w) = -S(w)`. So the exact two-sided
randomization p-value is at least `P(W in {w_obs, -w_obs})`. A design can
reject at level `alpha` only if it puts at most `alpha` probability on the
observed partition. At `alpha = 0.05` that takes at least 20 roughly equally
likely partitions (40 assignments). A deterministic design such as plain ABA
has one partition and can never reject. The FSM at `N = 100` with 20
covariates produced 24 distinct partitions in 2,000 draws, which is why its
randomization tests had little power.

## Predictions against the pilot

Simulated values carry roughly ±0.1 to 0.15 of noise (400 datasets per cell).

ESS of a mean-balancing design with OLS over complete randomization with OLS,
for the linear outcome (Proposition 2, chi-squared approximation for `Q`):

| N | p | predicted | simulated: rerand / exchange / shell |
|---|---|---|---|
| 20 | 1 | 1.06 | 0.91 / 0.82 / 0.85 |
| 20 | 5 | 1.47 | 1.52 / 1.33 / 1.48 |
| 50 | 5 | 1.12 | 1.15 / 1.18 / 1.13 |
| 50 | 20 | 1.81 | 1.60 / 1.79 / 1.66 |
| 100 | 20 | 1.26 | 1.09 / 1.32 / 1.36 |
| 300 | 20 | 1.07 | 0.90 / 0.98 / 0.92 |

ESS of the FSM for the nonlinear outcome, where `V_g / sigma^2 = 0.5`
(Proposition 3):

| N | p | predicted | simulated |
|---|---|---|---|
| 50 | 5 | 1.68 | 1.43 |
| 100 | 5 | 1.58 | 1.63 |
| 300 | 5 | 1.53 | 1.62 |
| 300 | 20 | 1.61 | 1.36 |

The two shortfalls are where the FSM's groups are too small to match second
moments (`N = 50`) or there are too many of them to match (`p = 20`, 231
squares and cross-products).

t-test against randomization test on the difference in means, exchange
anticlustering, `p = 5` (Proposition 4):

| outcome | kappa | predicted size / power / RI power | simulated, N = 50, 100, 300 |
|---|---|---|---|
| linear | 0.71 | 0.006 / 0.50 / 0.79 | size 0.003 to 0.007; power 0.43 to 0.51; RI 0.76 to 0.81 |
| nonlinear | 0.87 | 0.024 / 0.50 / 0.62 | size 0.022 to 0.028; power 0.45 to 0.51; RI 0.56 to 0.61 |

## What follows

1. Against a correctly specified regression, balance is worth exactly the
   degrees of freedom the regression spends: `(N - 1) / (N - 1 - p)` at best.
   Design matters when `p / N` is not small.
2. Beyond that, a design is worth `1 + V_g / sigma^2` for each outcome
   component it balances that the analysis does not model. Balancing second
   moments (as the FSM does) pays when outcomes are nonlinear.
3. Balance must be paid for with randomness the analysis can use: label
   symmetry for unbiasedness, at least `2 / alpha` partitions for the test to
   reject, and a bounded largest eigenvalue for robustness to what was not
   measured.
4. The classical t-test is the wrong analysis for any balanced design. A
   randomization test that redraws the design recovers the lost power at
   exact size.
