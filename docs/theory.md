# Theory: what balance buys over regression adjustment

The pilot's numbers follow from five short results, and a sixth gives a design with an explicit dial. Each is stated for two
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

**Proposition 4 (the t-test under balance).** This holds under a constant
treatment effect, with a Gaussian approximation to the sampling distribution of
`tau_hat`.

The pooled (or Neyman) variance estimator targets
`S_1^2 / n_1 + S_0^2 / n_0`, built from the within-arm variances. Two facts
qualify what that is.

- Under complete randomization it exceeds the true variance by `S_tau^2 / N`,
  where `S_tau^2` is the variance of the unit-level effects. It is exact only
  when effects are constant, and otherwise conservative.
- Balancing covariates barely changes the within-arm variances, so under a
  balanced design the estimator still targets roughly that quantity. The true
  variance, though, has lost the balanced part.

Let `kappa^2 = (sigma^2 + V_unbalanced) / (sigma^2 + V_total)`. Under a
constant effect, the t statistic is shrunk by `kappa`. If `tau_hat` is
approximately Gaussian, its size is `2 Phi_bar(1.96 / kappa)` instead of 0.05.
Its power at an effect of `1.96` of its own standard errors stays at about one
half. A randomization test that redraws the actual design uses the true
randomization distribution, so its power is about `Phi_bar(1.96 - 1.96 / kappa)`.
With half the outcome variance balanced away (`kappa = 0.71`), these give a
t-test size of 0.006, t-test power of 0.50 and randomization-test power of 0.79.

Two qualifications apply.

- *Heterogeneous effects* make the variance estimator more conservative still,
  under every design (Branson, Li and Ding 2024).
- *The Gaussian approximation is design-specific.* Under thresholded
  rerandomization the limit of `N^{1/2}(tau_hat - tau)` is
  `V^{1/2} {(1 - R^2)^{1/2} e_0 + R L_{K,a}}`: a normal plus a truncated
  component (Li, Ding and Rubin 2018). That limit is less dispersed than the
  complete-randomization normal `V^{1/2} e_0` in every quantile gap
  (Branson, Li and Ding 2024, Theorem 3). Under near-exact balance (`Q` close
  to 0) the truncated part vanishes, and the limit is the normal of the
  residual part.
- *For rerandomization, the fix already exists analytically.* Li, Ding and
  Rubin (2018) replace `z_{1-alpha/2}` with the quantile of that mixture at an
  estimated `R^2`. Branson, Li and Ding (2024) derive power and sample size
  for this test.
  - They show that the Neyman estimator overestimates the variance by the
    same `S_tau^2` under both designs, so it is proportionally more
    conservative under rerandomization.
  - With heterogeneity and small effects, that can make rerandomization less
    powerful than complete randomization (their Theorems 4 and 5).
  - The less conservative estimator `V_N - s^2_{tau|X}` (Ding, Feller and
    Miratrix 2019) narrows the gap.

  No analytic analogue exists for anticlustering designs (Papenberg and
  Angelike 2026), the shell draft or the FSM. There, a randomization test that
  redraws the design is the available exact option. Its size is exact in finite
  samples for the sharp null, whatever the outcome distribution. For the weak
  null of zero average effect, it inherits the same heterogeneity issues.

**Proposition 5 (resolution of a randomization test).** Under the sharp null
the difference in means satisfies `S(-w) = -S(w)`. So the exact two-sided
randomization p-value is at least `P(W in {w_obs, -w_obs})`. A design can
reject at level `alpha` only if it puts at most `alpha` probability on the
observed partition. At `alpha = 0.05` that takes at least 20 roughly equally
likely partitions (40 assignments). A deterministic design such as plain ABA
has one partition and can never reject. The FSM at `N = 100` with 20
covariates produced 24 distinct partitions in 2,000 draws, which is why its
randomization tests had little power.

## 6. Grouped randomization: a dial between ABA and complete randomization

**Design `G(K)`.**

1. Partition the `N` units into `K` groups of `m = N / K` with anticlustering
   on the covariates (and on any features to be balanced, per Proposition 3).
2. Assign `K / 2` groups to treatment by complete randomization over groups.

The partition depends on `X` alone, so inference can condition on it.

**Proposition 6.** Let `G` be the `N x K` group-membership matrix. Then:

1. *Unbiasedness.* Every unit has probability 1/2 of treatment, so `tau_hat` is
   unbiased under arbitrary effects.
2. *Spread.* `Sigma = (K / (K - 1)) G (I - 11'/K) G'`. Its nonzero eigenvalues
   all equal `N / (K - 1)`, on the `K - 1` group contrasts, and it is zero on
   within-group directions. The worst-case variance factor for unmeasured
   components is exactly `(N - 1) / (K - 1)` relative to complete
   randomization. Averaged over all directions it equals complete
   randomization, because the trace is still `N`.
3. *Resolution.* There are `C(K, K/2) / 2` equally likely partitions, so the
   randomization p-value can go down to `2 / C(K, K/2)`. That is 0.029 at
   `K = 8`, 0.008 at `K = 10` and about 1e-5 at `K = 20`. Rejection at 5%
   needs `K >= 8`.
4. *Balance.* The treated mean minus the overall mean is the average of `K/2`
   group deviations. Anticlustering makes each deviation small, so `Q` is
   small. It grows as `K` approaches `N`, because the groups shrink.
5. *Standard errors.* This is a cluster-randomized experiment with `K` equal
   clusters. The Neyman estimator on group means applies and is conservative,
   with `K - 2` degrees of freedom for a t reference. This gives a closed-form
   variance the shell draft lacks.

*Proof of 2.* `W = G s`, where `s` is a complete randomization of `K` labels
with `E[s s'] = (K / (K - 1)) (I - 11'/K)`. The matrix `G (I - 11'/K) G'` has
eigenvalue `m` on the group contrasts.

`K` is the dial. `K = 2` is ABA itself: one partition, a worst case of
`N - 1`, and no test. `K = N` is complete randomization. Choosing `K` sets
three things explicitly:

- the worst case, `(N - 1) / (K - 1)`;
- the resolution, `C(K, K/2)`;
- the balance, through the group size `m`.

The worst case is real only for unmeasured components that line up with the
grouping. The grouping is a function of `X`, so the danger lies in functions of
`X` that the anticlustering did not balance. Plain ABA on means makes second
moments worse than random, which is why the partition should balance the
squares and cross-products too.

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
4. The classical t-test is conservative under any balanced design (and, with
   heterogeneous effects, under complete randomization too). A randomization
   test that redraws the design recovers the lost power at exact size for the
   sharp null.
5. Grouped randomization `G(K)` turns the balance-robustness trade-off into
   one integer, with a closed-form worst case, test resolution and cluster
   standard errors.
