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
2. Is there a realistic setting where a balanced design, analyzed well, beats
   complete randomization analyzed with regression by a meaningful margin?

## Designs (two equal groups)

| Design | Definition |
|---|---|
| CRD | Uniform random split |
| Exchange anticlustering | Random equal split, then one pass of the `anticlust` exchange method on the k-means (variance) objective, as in Papenberg and Angelike |
| Rerandomization | Draw CRD splits until the Mahalanobis distance between group means falls below its 0.1% quantile (Morgan and Rubin 2012) |
| FSM | `finite_selection.fsm` with a SCOMARS order |
| Shell draft | Whiten covariates, sort units by squared distance from the center after multiplying by `exp(N(0,1))` (jitter 1), cut into shells of 2, assign each shell by the optimal matching on centroid distance, shuffle the two labels |

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
  sample size at least 10% above CRD with OLS, both analyzed with their best
  valid test.

## Out of scope for the pilot

Many groups, large `N`, heavy-tailed covariates, worst-case outcome directions,
and closed-form standard errors for shell designs.

## Results (400 datasets per cell, 200 reference draws)

Reproduce with `uv run python scripts/pilot.py` then `scripts/headline.py`.
Power is at `tau = 1.96 sqrt(8/N)`. ESS is the MSE of CRD analyzed by OLS
divided by the MSE of the design analyzed by OLS; with 400 datasets it carries
roughly ±0.1 to 0.15 of simulation noise.

```text
                  CRD+OLS  rerand RI-OLS  ESS rerand  exchange RI-OLS  ESS exchange  shell RI-OLS  ESS shell  FSM RI-OLS  ESS FSM
N   p  dgp                                                                                                                       
20  1  linear        0.77           0.68        0.91             0.75          0.82          0.70       0.85        0.36     0.94
    5  linear        0.60           0.60        1.52             0.69          1.33          0.70       1.48        0.07     1.36
       nonlinear     0.43           0.45        1.44             0.49          1.39          0.53       1.56        0.06     1.42
50  1  linear        0.73           0.77        1.15             0.76          1.17          0.73       1.04        0.62     1.10
    5  linear        0.72           0.79        1.15             0.76          1.18          0.75       1.13        0.40     1.13
       nonlinear     0.57           0.57        1.00             0.58          0.93          0.60       1.02        0.40     1.43
    20 linear        0.57           0.71        1.60             0.74          1.79          0.68       1.66        0.22     1.13
       nonlinear     0.38           0.52        1.65             0.51          1.74          0.54       1.77        0.16     1.37
100 1  linear        0.81           0.73        0.91             0.75          0.91          0.80       0.83        0.76     0.95
    5  linear        0.74           0.80        1.17             0.78          1.23          0.81       1.12        0.62     1.05
       nonlinear     0.60           0.62        1.01             0.62          0.97          0.63       1.05        0.63     1.63
    20 linear        0.71           0.74        1.09             0.79          1.32          0.79       1.36        0.31     1.15
       nonlinear     0.50           0.57        1.01             0.57          1.29          0.58       1.16        0.26     1.28
300 1  linear        0.77           0.77        1.08             0.80          1.22          0.81       1.17        0.77     1.03
    5  linear        0.77           0.79        1.05             0.81          1.13          0.80       1.08        0.78     1.09
       nonlinear     0.60           0.62        1.03             0.56          1.03          0.60       1.02        0.77     1.62
    20 linear        0.78           0.79        0.90             0.78          0.98          0.76       0.92        0.46     1.07
       nonlinear     0.57           0.55        1.08             0.62          1.28          0.60       1.15        0.45     1.36

RI size across all cells: 0.005 to 0.075
```
