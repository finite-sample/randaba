# Pilot: does balanced randomized assignment beat complete randomization plus regression?

## Question

Covariate-balancing assignment (anticlustering, the FSM, rerandomization) makes
treatment groups alike on measured covariates. Two studies find that regression
adjustment recovers most of that benefit (Green and Mikhaeil 2026; Papenberg and
Angelike 2026). Papenberg and Angelike also find that the unadjusted t-test is
conservative under anticlustering, and they leave a randomization test matched to
the assignment procedure to future work.

The pilot asks two things.

1. Does randomization inference (RI) that redraws the actual design recover the
   power the t-test leaves on the table, at exact size?
2. Is there a realistic setting where a balanced design, analysed well, beats
   complete randomization analysed with regression by a meaningful margin?

## Designs (two equal groups)

| Design | Definition |
|---|---|
| CRD | Uniform random split |
| Exchange anticlustering | Random equal split, then one pass of the `anticlust` exchange method on the k-means (variance) objective, as in Papenberg and Angelike |
| Rerandomization | Draw CRD splits until the Mahalanobis distance between group means falls below its 0.1% quantile (Morgan and Rubin 2012) |
| FSM | `finite_selection.fsm` with a SCOMARS order |
| Shell draft | Whiten covariates, sort units by squared distance from the centre after multiplying by `exp(N(0,1))` (jitter 1), cut into shells of 2, assign each shell by the optimal matching on centroid distance, shuffle the two labels |

## Data-generating processes

Covariates are independent standard normal. The control outcome is
`Y(0) = f(X) + e` with `e ~ N(0, 1)`, and `Y(1) = Y(0) + tau`.

| Name | `f(X)` |
|---|---|
| `pa` | `beta_c * x_1` with `beta_c = 1`, one covariate: the Papenberg and Angelike cell with the largest covariate effect |
| `linear` | Equal weights on all `p` covariates, scaled so `Var f = 1` |
| `nonlinear` | Linear part plus squares and pairwise products of the first three covariates, scaled so `Var f = 1` and the two parts carry equal variance |

`tau` is 0 for size and `1.96 * sqrt(8 / N)` for power, which gives about 50% power
to an unadjusted test under CRD.

Grid: `N` in {20, 50, 100, 300}; `p` in {1, 5, 20} (with `p = 20` only where
`N >= 50`).

## Analyses

| Analysis | Test of `tau = 0` |
|---|---|
| t-test | Student two-sample t-test |
| OLS | Treatment coefficient from OLS on treatment and all covariates, classical standard error (where `p < N - 2`) |
| RI, difference in means | Sharp null; reference statistics from 200 fresh draws of the same design on the same covariates |
| RI, OLS coefficient | The same reference draws, statistic is the OLS treatment coefficient |

The reference draws depend only on the covariates and the design, so one set
serves every outcome model and both statistics.

## What counts as success

- **Correctness.** Under CRD every test has size 0.05 within simulation error.
  Every RI test has size 0.05 within simulation error under every design. The
  exchange-anticlustering replica reproduces R `anticlust` given the same initial
  split.
- **Question 1** is answered yes if RI on the difference in means has size 0.05
  and more power than the t-test under the balanced designs.
- **Question 2** is answered yes if some cell gives a balanced design an effective
  sample size at least 10% above CRD with OLS, both analysed with their best
  valid test.

## Out of scope for the pilot

Many groups, large `N`, heavy-tailed covariates, worst-case outcome directions,
and closed-form standard errors for shell designs.
