# Theory: what a balancing design buys, and what it costs

Results are for two equal arms. Section 3 gives the regression benchmark,
section 6 the grouped design `G(K)`, section 7 how its balance scales with
group size, section 8 how to choose `K` and section 9 what happens to omitted
structure. `scripts/theory_check.py` sets the
predictions beside the pilot's simulations.

Two kinds of risk appear below, and they are not the same comparison.
*Randomization risk* holds the potential outcomes fixed and averages over the
design only. *Model risk* also averages over an outcome model (section 3). Each
result says which it uses.

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

A design's randomization risk is therefore all in `Sigma`.

## 2. Unbiasedness and the variance budget

**Proposition 1.** If the design is unchanged by swapping the labels
(`W` and `-W` have the same distribution), then `E W = 0`, `tau_hat` is
unbiased, and `MSE = (4/N^2) mu' Sigma mu`. Every such design has
`diag(Sigma) = 1`, so `trace(Sigma) = N`, and `lambda_max(Sigma) <= N`.

*Proof.* Symmetry gives `E W = -E W`. Each `W_i^2 = 1`.

Shuffling labels at the end of any procedure makes it symmetric, which is why
the shell draft does it.

**Corollary (the price of balance).** Write `mu = c 1 + X_c beta + r`, with
`X_c` the centred covariates of rank `p` and `r` orthogonal to `1` and `X_c`.
If a design balances exactly (`X_c' W = 0` always, which requires such
assignments to exist), then `MSE = (4/N^2) r' Sigma r`.

- *Averaged* over directions of `r` drawn uniformly from the sphere in the
  `N - p - 1`-dimensional space it can occupy, this is
  `(4/N^2) |r|^2 N / (N - p - 1)`. Complete randomization gives `N / (N - 1)`
  in place of `N / (N - p - 1)`. So under that isotropic average, balance on
  what was measured costs a factor `(N - 1) / (N - p - 1)` on what was not.
  Real omitted components need not be isotropic.
- The *worst* direction gives `(4/N^2) |r|^2 lambda_max`, the largest
  eigenvalue of `Sigma` on that space. Only `lambda_max <= N` bounds it, so a
  single partition can be up to `N - 1` times worse than complete
  randomization (Kallus 2018, Theorem 1; Kallus 2020). This is Efron's (1971)
  accidental-bias bound `z' Sigma z <= lambda_max |z|^2`, as Kapelner et al.
  (2021) use it.

*Proof.* The projection `P` onto `span(1, X_c)` satisfies `P W = 0`, so
`Sigma P = 0` and the trace of `Sigma` on the complement is all of `N`.

## 3. A regression benchmark

**Proposition 2.** Suppose `Y(0) = alpha + X beta + e` with a constant effect,
`E[e | X, W] = 0` and `Cov(e | X, W) = sigma^2 I`. Normality is not needed. Let
`Q = W' P W`, where `P` projects onto `X_c`, and suppose `N - Q > 0` on every
assignment the design can produce. Then OLS of `Y` on `(1, T, X)` is unbiased
given `W`, and its model risk is

```text
MSE(design, OLS) = 4 sigma^2 E_W[ 1 / (N - Q) ].
```

The formula does not involve `beta`. The minimum, `4 sigma^2 / N`, is reached
when `Q = 0` on every assignment (exact balance), where OLS and the difference
in means coincide.

*Proof.* By Frisch–Waugh–Lovell, `tau_hat_OLS - tau = 2 W' M e / (N - Q)` with
`M = I - 11'/N - P`. Given `W` its variance is `4 sigma^2 W'MW / (N - Q)^2`, and
`W'MW = N - Q`.

The mechanism is collinearity between treatment and the covariates: `Q / N` is
the share of the treatment vector that the covariates explain, and OLS
discards it.

None of this is new. The conditional variance is the textbook OLS formula, and
Krieger, Azriel and Kapelner (2019, Section 2.3) write it in terms of the
Mahalanobis imbalance. Kapelner, Krieger, Sklar and Azriel (arXiv:1905.03337,
eq. 6) average it over a design to third order:
`sigma^2/n + (sigma^2/n^2) trace(X_perp' Sigma X_perp)`, where `X_perp` is the
orthogonalized covariate matrix. That depends on the design only through
`E[Q] = trace(P Sigma)`. The corollaries below put these pieces in exact form.

**Corollary 2a (lower bound, any fixed `X`).** Under complete randomization
`E[Q | X] = p N / (N - 1)`. By Jensen's inequality, the gain of an exactly
balanced design over complete randomization analysed by OLS is

```text
N E_CRD[ 1 / (N - Q) | X ]  >=  (N - 1) / (N - 1 - p).
```

Equality needs `Q` constant across assignments. There is no upper bound that
depends only on `N` and `p`. Take `N = 8`, `w0 = (1,1,1,1,-1,-1,-1,-1)`,
`v = (1,-1,1,-1,1,-1,1,-1)` and the single covariate `x = w0 + 0.01 v`. An exactly
balanced assignment exists, and enumerating all 70 assignments gives a gain of
287 against the bound's 7/6. Shrinking the perturbation makes it arbitrarily
large.

**Corollary 2b (exact, Gaussian covariates).** If the rows of `X` are
independent `N(0, I_p)`, the centred covariate space is a uniformly oriented
`p`-dimensional subspace of `1-perp`, so averaging over `X` and complete
randomization, `Q / N ~ Beta(p/2, (N - 1 - p)/2)`. For `N > p + 3`,

```text
N E_{X, CRD}[ 1 / (N - Q) ]  =  (N - 3) / (N - p - 3).
```

The expectation is infinite when `N <= p + 3`. This is the classical
inverse-Beta (equivalently inverse-Wishart) moment for regression on Gaussian
regressors, applied to the treatment column.

**Scope.** Exact balance is generally unattainable with continuous
covariates, so `4 sigma^2 / N` is an ideal that real designs approach. The
comparison is with unregularized OLS on all `p` covariates. It says nothing
about shrinkage, prediction-based adjustment or external prognostic scores.
Kallus (2018, Corollary 3 and Section 2.4.4) shows the other side: an optimally
balancing design's variance relative to complete randomization is at least
`1 - R^2` and approaches it, with no finite-sample bias, as the minimal
imbalance shrinks. Proposition 2 adds what unregularized OLS under complete
randomization pays for reaching the same `1 - R^2`. To first order,
rerandomization brings no gain once the analysis uses an optimal regression
(Li and Ding 2020). Proposition 2 describes the smaller
finite-sample term that this first-order result leaves out. It matters when
`p / N` is not small (Lei and Ding 2021 study adjustment in that regime). With
one covariate, the case Papenberg and Angelike simulate, Corollary 2b gives
`17/16` at `N = 20` and under 1% at `N = 300`.

## 4. Balancing what the regression leaves out

**Proposition 3.** Add a component `g` to the model,
`Y(0) = alpha + X beta + g + e`, with `g` orthogonal to `span(1, X)` in the
sample. The OLS error picks up `2 W' g / (N - Q)`, so the model risk is

```text
4 sigma^2 E[ 1 / (N - Q) ]  +  4 E[ (W' g)^2 / (N - Q)^2 ].
```

- A design that exactly balances both `X` and `g` has risk `4 sigma^2 / N`.
- A design that exactly balances `X` alone has risk
  `4 sigma^2 / N + (4 / N^2) g' Sigma g`. That second term can be larger or
  smaller than its complete-randomization counterpart, so balancing means
  can make an omitted component worse.

**Approximation.** Under complete randomization
`E[(W'g)^2] = N |g|^2 / (N - 1)`. When `Q` is small relative to `N` and
roughly independent of `W'g`, the second term is about `4 V_g / N` with
`V_g = |g|^2 / N`. A design balancing `X` and `g` then gains about
`(1 + V_g / sigma^2)` times the Proposition 2 factor. This is a heuristic, not
a bound.

The FSM's D-optimal criterion pushes each group's `X_g' X_g` towards half of
`X' X`. That suggests it balances squares and cross-products, but no theorem
here shows it. Whether it does is measured, not proved. Balancing a feature map
`Phi` that contains the squares and cross-products does it explicitly. That is
the design of Bertsimas et al. (2015), as Kallus (2018, Section 2.3.4)
describes it, and Kallus's quadratic-kernel
allocation. The k-plus criterion (Papenberg 2024) appends each covariate's
squared (and higher) deviations from its mean, so it balances marginal
variances but not covariances. Balancing `Phi` is feasible only while its
dimension is well below `N`: with `p = 20` there are 230 squares and
cross-products.

## 5. Inference

**Proposition 4 (the t-test under a balanced design).** The pooled and Neyman
variance estimators target `S_1^2 / n_1 + S_0^2 / n_0`, built from the within-arm
variances.

- Under complete randomization the true variance is that quantity minus
  `S_tau^2 / N`. So the Neyman estimator is conservative *in expectation*,
  exact in expectation only under constant effects. A realized estimate can
  still fall below the true variance.
- A balancing design changes the true variance to `(4/N^2) mu' Sigma mu` while
  barely changing the within-arm variances. The t-test is conservative when
  the design *reduces* that quadratic form relative to complete randomization,
  which is the intended case. It need not reduce it. With the eight units
  above, the design `W = +/- w0`, the covariate `x = v` (exactly balanced) and
  sharp-null outcomes `Y = w0 + 0.01 v`, the t-test rejects on every draw. A
  balanced design can push variance into an unmeasured direction (Proposition 1).

*Illustration.* Suppose the effect is constant, `tau_hat` is approximately
Gaussian, the design removes a known share of outcome variance, and the
estimated standard error is close to its target. Let
`kappa^2 = (sigma^2 + V_unbalanced) / (sigma^2 + V_total)`. The t statistic is
shrunk by `kappa`, so its size is `2 Phi_bar(1.96 / kappa)` rather than 0.05.
Its power at an effect of 1.96 of its own standard errors stays at about one
half. A randomization test that uses the true randomization distribution has
power of about `Phi_bar(1.96 - 1.96 / kappa)`. At `kappa = 0.71` these give
0.006, 0.50 and 0.79.

Three qualifications apply.

- *Heterogeneous effects* add `S_tau^2 / N` of conservativeness under complete
  randomization and rerandomization alike (Branson, Li and Ding 2024).
- *The limiting distribution depends on the design.* Under thresholded
  rerandomization, `N^{1/2}(tau_hat - tau)` tends to
  `V^{1/2} {(1 - R^2)^{1/2} e_0 + R L_{K,a}}` (Li, Ding and Rubin 2018). That
  limit is less dispersed than the complete-randomization normal in every
  quantile gap (Branson, Li and Ding 2024, Theorem 3). For anticlustering no
  limit theorem is available. A small `Q` alone does not give a central limit
  theorem.
- *For rerandomization an analytic fix exists.* Li, Ding and Rubin (2018)
  replace `z_{1-alpha/2}` with a quantile of that mixture. With heterogeneity
  and small effects, the extra conservativeness can make rerandomization less
  powerful than complete randomization (Branson, Li and Ding 2024, Theorems 4
  and 5).

For anticlustering, the shell draft and the FSM, a randomization test that
redraws the design controls level in finite samples for the **sharp** null.
This is the standard recipe (Morgan and Rubin 2012, Section 2.2). Krieger,
Azriel and Kapelner (2019, Section 5.2) already apply it to greedy
pair-switching. That is the two-group exchange algorithm run to a local
optimum, the same family as `anticlust`'s exchange method. So a randomization
test for exchange anticlustering is not new, although Papenberg and Angelike
(2026) list it as future work.
Discreteness and ties can hold its rejection rate well below `alpha`. It does
not test the weak null of zero average effect. The statistics used here (the
unstudentized difference in means and OLS coefficient) need extra
justification for that. Studentizing is the standard route (Wu and Ding 2021).

**Proposition 5 (resolution of a randomization test).** Under the sharp null
the difference in means satisfies `S(-w) = -S(w)`. So the two-sided
randomization p-value is at least `P(W in {w_obs, -w_obs})`, the probability of
the observed partition.

- For a design uniform over `M` partitions the smallest p-value is at least
  `1 / M`, so rejection at 5% needs at least 20 partitions (40 assignments).
  Ties at the extreme can prevent rejection even then. This is the constraint
  behind Kallus's (2020) inference-constrained design: no assignment gets more
  than `alpha / 2` probability.
- A non-uniform design can reject on a rare partition with fewer partitions
  in all.
- A deterministic design such as plain ABA, label shuffle or not, has one
  partition. Its p-value is always 1.

The FSM at `N = 100` with 20 covariates produced 24 distinct partitions in
2,000 draws. Its randomization tests had little power. A small support is
consistent with that, but the partition probabilities and the statistic's
distribution over them would be needed to show it is the cause.

## 6. Grouped randomization `G(K)`

**Design `G(K)`.**

1. Partition the `N` units into `K` groups of `m = N / K` (`K` even and
   dividing `N`) by anticlustering on the covariates.
2. Assign `K / 2` groups to treatment by complete randomization over groups.

The partition depends on `X` alone, and everything below conditions on it.
Once the partition is fixed, step 2 is ordinary equal-size cluster
randomization. What may be new is constructing many clusters that are diverse
within and alike between, then randomizing them. Its neighbours:

- **`K` equal to the number of arms** is the design of Bertsimas, Johnson and
  Kallus (2015). It builds one optimal partition by mixed-integer programming
  and randomizes which group gets which treatment. That is a single partition,
  so they test with a bootstrap, not a randomization test.
- **Cluster rerandomization** (Lu, Liu, Liu and Ding 2023) takes natural
  clusters as given and discards cluster assignments with poor covariate
  balance. Its theory needs many clusters. In their simulations, normal-based
  intervals were valid from about 24 clusters with one covariate and needed
  more than 44 with five (their Appendix A). Constrained randomization of
  group-randomized trials (Moulton 2004) is the same idea, as Lu et al.
  describe it. Moulton's paper itself was not obtained.

In Kallus's (2018, 2020) framework, an outcome class given by a kernel matrix
`K_ker` gives a design the worst-case risk `lambda_max(K_ker^{1/2} Sigma K_ker^{1/2})`.
His minimax design over all `Sigma`, constrained so that no assignment exceeds
`alpha / 2`, is solved heuristically:

1. Find the top `T` balanced partitions by repeated mixed-integer quadratic
   programs.
2. Weight those partitions with an `N x N` semidefinite program.

`G(K)` is a member of the same constrained class, with three differences:

- Its `Sigma` is closed form.
- Every assignment has probability `1 / C(K, K/2)`.
- Its partition comes from a scalable anticlustering heuristic.

How close its risk comes to the constrained optimum is open.

**Proposition 6.** Let `G` be the `N x K` membership matrix, `s` the `K`-vector
of group labels and `W = G s`.

1. *Unbiasedness.* Every unit has probability 1/2 of treatment, so `tau_hat` is
   unbiased under arbitrary effects.
2. *Spread.* `Sigma = (K / (K - 1)) G (I - 11'/K) G'`. Its `K - 1` nonzero
   eigenvalues all equal `N / (K - 1)`, on the group contrasts, and it is zero
   on within-group directions. Over all centred outcome directions, the
   worst-case variance relative to complete randomization is
   `(N - 1) / (K - 1)`. Over directions orthogonal to `X` this is an upper
   bound, attained if some group contrast is orthogonal to `X`. The isotropic
   average equals complete randomization, because the trace is still `N`. The
   squared Frobenius norm is `||Sigma||_F^2 = N^2 / (K - 1)`. That runs from
   about `N` under complete randomization to `N^2` for a single partition.
   It is the randomness term in the tail criterion of Kapelner et al. (2021;
   arXiv:1905.03337), which those authors must estimate from tens of
   thousands of sampled assignments.
3. *Resolution.* There are `C(K, K/2) / 2` equally likely partitions, so the
   randomization p-value is at least `2 / C(K, K/2)`. That is 0.029 at `K = 8`,
   0.008 at `K = 10` and about 1e-5 at `K = 20`. Rejection at 5% needs
   `K >= 8`. That only makes rejection possible. It does not give useful power.
4. *Balance.* Let `D` be the `K x p` matrix of group-mean deviations
   `xbar_g - xbar` and `A = X_c' X_c`. Then `X_c' W = m D' s`, and

   ```text
   E[Q | G, X] = (K m^2 / (K - 1)) trace(A^{-1} D' D).
   ```

   Expected treatment imbalance is the partition's own between-group
   imbalance, scaled. `K` does not determine it by itself. For two groups the
   rates are known:
   - The optimal partition's imbalance is exponentially small in `N`
     (Bertsimas et al. 2015; Kallus 2018, Section 3.3, a heuristic
     argument).
   - Greedy pair-switching from a random start reaches
     `O_p(n^{-(1 + 2/p)})` (Krieger et al. 2019).

   For `K` groups of size `m` built by exchange, see Proposition 7.
5. *Variance estimation.* Conditional on the partition, the group mean
   potential outcomes `Ybar_g(t)` are fixed numbers, and with equal groups the
   unit-level difference in means equals the difference in group means. So

   ```text
   Var(tau_hat | G) = S^2_{Ybar(1)} / (K/2) + S^2_{Ybar(0)} / (K/2) - S^2_{taubar} / K,
   ```

   and the Neyman estimator on group means exceeds it in expectation by
   `S^2_{taubar} / K >= 0`. Forming the groups from `X` does not break this.
   Independent sampling of groups is not needed, because the randomness is
   the assignment of known group labels. A conservative variance does not
   make a `t_{K-2}` test valid in finite samples, though. With `K = 8`, no
   effect and group outcomes `(1,1,1,0,0,0,0,0)`, the pooled t-test rejects at
   5% on 10 of the 70 assignments (14%). Normal or `t` references need a
   group-level central limit theorem, meaning many groups and none dominant.
   Lu et al.'s simulations suggest that means a few dozen groups. With that
   many, their Theorem 6 also covers regression adjustment on group-level
   covariates with robust standard errors (Su and Ding 2021). For few groups,
   use the randomization test, which is valid for the sharp null.

*Proof of 2.* `E[s s'] = (K / (K - 1)) (I - 11'/K)`, and because `G'G = m I`,
`G (I - 11'/K) G'` has eigenvalue `m` on the group contrasts.

*Proof of 4.* `G' X_c = m D`, so `X_c' W = m D' s` and
`Q = m^2 s' D A^{-1} D' s`. The rows of `D` sum to zero, so taking expectations
with `E[s s']` gives the formula.

**The constraint that matters.** `K = 2` with a fixed partition is a single
partition. Its worst case is `N - 1` and its test cannot reject. `K = N` is
complete randomization. Keeping the worst case within a factor `C` needs

```text
(N - 1) / (K - 1) <= C   <=>   K >= 1 + (N - 1) / C,
```

so `K` must grow in proportion to `N`, and groups stay small. At `N = 300` and
`K = 8` the worst case is 42.7. The open question is whether anticlustering
still makes `D` small when `m` is held to `C` or so units.

The worst case is Efron's criterion, and it always picks complete
randomization. Kapelner et al. (2021) argue for a quantile of the MSE over
unobserved covariates instead. For OLS (arXiv:1905.03337, eq. 13) its
design-dependent part is approximately

```text
trace(X_perp' Sigma X_perp)  +  c (2 ||(I - P) Sigma||_F^2 + ...)^{1/2}.
```

Under `G(K)` the first term is `E[Q]` from item 4. The second is at most
`N^2 / (K - 1)` from item 2. So `K` can be chosen by minimizing a
one-dimensional criterion in near closed form, where Kapelner et al. tune a
rerandomization threshold by sampling. Section 8 computes that criterion
exactly.

## 7. How balance scales with group size

Partition `N` units with `p` whitened covariates into `K` groups of `m`, by the
exchange method run to a local optimum (`balance_buys.grouped.partition`). Let
`d_g` be group `g`'s mean deviation.

**Proposition 7a (a certificate at any local optimum).** For groups `g` and `h`,
let `delta = x_j - x_i` (`i` in `g`, `j` in `h`), and let `theta` be its angle
to `d_h - d_g`. Then

```text
|d_g - d_h|  <=  rho_gh / m,     rho_gh = min over pairs with cos theta > 0 of |delta| / cos theta,
E[Q | partition]  <=  sum_{g<h} rho_gh^2 / ((K - 1)(N - 1)).
```

*Proof.* The exchange objective is `sum_g m |d_g|^2`. Swapping `i` and `j`
changes it by `2 delta'(d_g - d_h) + 2 |delta|^2 / m`. At a local optimum that
change is non-negative for every pair. So `|delta| |d_g - d_h| cos theta <=
|delta|^2 / m`, which is the first line. A pair with `cos theta > 0` always
exists, because the average of `x_j - x_i` over all pairs is `d_h - d_g`
itself. For the second line:

- Since `sum_g d_g = 0`, `sum_g |d_g|^2 = (1/K) sum_{g<h} |d_g - d_h|^2`.
- Whitening gives `A = (N - 1) I`.
- Substitute both into Proposition 6, item 4.

The bound depends only on the data and the partition, so it can be reported
with any `G(K)` partition. Across 400 partitions (`scripts/gk_bound.py`;
`N = 240`, `p` from 1 to 10, `m` from 2 to 60) it never failed. The ratio of
`E[Q]` to the bound was 0.1 to 0.66 for `p >= 5`, and smaller for few
covariates, where the bound is loose.

**Proposition 7b (rate; heuristic, checked by simulation).** The same partition
gives

```text
E[Q | partition]  ~  c_p m^{-(1 + 2/p)} N^{-2/p},
```

against about `p` for a random partition or for complete randomization.

*Argument.* By 7a, a swap helps when `x_j - x_i` falls in a ball of radius
about `m |d_g - d_h| / 2` that touches the origin. Group `g` can trade with
every unit outside it, so there are about `m N` candidate pairs. Treat their
differences as draws with a density bounded near zero; Krieger et al. (2019)
make the same assumption for `p > 1`, and the algorithm's choice of groups
makes it untrue in a strict sense. At a local optimum no pair falls in the
ball, so `m N (m |d_g - d_h|)^p` is of order 1, giving
`|d| ~ m^{-1} (m N)^{-1/p}`. A proof would need to handle that dependence.
Item 4 of Proposition 6 then gives `E[Q] ~ (K / (K - 1)) m |d|^2`. With
`K = 2` this is Krieger, Azriel and Kapelner's (2019) `O_p(n^{-(1 + 2/p)})`.
Counting only swaps between one pair of groups (`m^2` pairs) would predict
`m^{-(1 + 4/p)}` instead, and the simulation rejects that.

*Check* (`scripts/gk_rate.py`). The design was:

- `N` in {240, 480, 960}, `p` in {1, 2, 5, 10};
- `m` from 6 to 120;
- 20 covariate draws per cell.

Regressing `log E[Q]` on `log m` and `log N`:

| p | `log m` (predicted) | `log N` (predicted) |
|---|---|---|
| 1 | -2.99 (-3.00) | -2.12 (-2.00) |
| 2 | -1.96 (-2.00) | -1.07 (-1.00) |
| 5 | -1.37 (-1.40) | -0.49 (-0.40) |
| 10 | -1.18 (-1.20) | -0.31 (-0.20) |

Random partitions gave `E[Q]` of about `p` at every `m`. The `N` exponent runs
about 0.1 steeper than predicted, which the heuristic does not explain.

*The robustness frontier.* For large `K`, the worst-case factor
`(N - 1)/(K - 1)` is about `m`. So holding the worst case to `C` caps the
imbalance at about `E[Q] ~ c_p C^{-(1 + 2/p)} N^{-2/p}`. Balance improves
polynomially in the robustness one gives up, and more slowly the more
covariates there are.

## 8. Choosing K by the tail of the MSE

Kapelner et al. (2021; arXiv:1905.03337) treat unobserved covariates as a
random vector `z` and judge a design by a high quantile of its MSE over `z`.
That is less pessimistic than the worst case.

**Proposition 8.** Let `Y = alpha + tau T + X beta + z` with a constant effect,
and estimate `tau` by OLS on `(1, T, X)`. Given `z`, the design MSE is
`z' M z` with

```text
M = 4 (I - P) E[W W' / (N - Q)^2] (I - P).
```

If `z` has independent `N(0, sigma^2)` entries, the MSE over `z` is distributed
as `sigma^2 sum_i lambda_i chi2_1`, where `lambda_i` are the eigenvalues of `M`.
Its mean is Proposition 2's `4 sigma^2 E[1/(N - Q)]`.

1. *Under `G(K)` this is a `K x K` computation, exact given the partition.*
   The nonzero eigenvalues are those of `4 H^{1/2} (m I - m^2 B) H^{1/2}`,
   with `B = D A^{-1} D'` and `H = E[s s' / (N - m^2 s' B s)^2]`. The
   expectation runs over the `C(K, K/2)` label assignments.
2. *Exact balance gives `K - 1` degrees of freedom.* If `D = 0`, the `K - 1`
   eigenvalues all equal `4 / (N (K - 1))`, and the MSE is
   `(4 sigma^2 / N) chi2_{K-1} / (K - 1)`. Its mean is optimal, but its
   spread is that of a chi-squared with `K - 1` degrees of freedom.
3. *Closed-form approximations.* Write `chi2_k(q)` for the `q` quantile of a
   chi-squared with `k` degrees of freedom. Relative to `4 sigma^2 / N`, the
   `q` quantile of the MSE is about

   ```text
   G(K) + OLS:  (1 + E[Q]/N) chi2_{K-1}(q) / (K - 1)
   CRD  + OLS:  ((N - 3)/(N - p - 3)) chi2_{N-p-1}(q) / (N - p - 1)
   ```

   The second form averages over Gaussian covariates (Corollary 2b).

*Checks* (`tests/test_grouped.py`, `scripts/gk_tail.py`):

- The `K x K` eigenvalues equal those of the `N x N` matrix built by brute
  force over all assignments.
- The quantile from the eigenvalues matches a brute-force quantile over
  40,000 draws of `z` within 0.5%.
- The closed forms match the exact computation within 1.3% at `N = 120` and
  `p` = 5 and 20, for every `K` from 2 to 60.

*Consequence.* `G(K)` with OLS beats complete randomization with OLS on the
mean for every `K` (Proposition 2). At the 95% quantile it does so only when
`chi2_{K-1}(q)/(K - 1)` falls below the complete-randomization term. Its MSE
over unobserved covariates has `K - 1` degrees of freedom, against `N - p - 1`
for complete randomization. At `N = 120`:

- With `p = 5`, complete randomization has the better tail for every `K` up
  to 60.
- With `p = 20`, `G(K)` matches it at `K = 40` (groups of 3) and beats it at
  `K = 60` (pairs).

Choosing `K` is minimizing the first closed form, with `E[Q]` from
Proposition 7. The answer favours many small groups, and only when `p/N` is
large.

![Tail criterion against K](_static/gk_tail.png)

## 9. Omitted structure that is not noise

Proposition 8 treats what the analysis omits as independent noise. A real
omission is usually a function of the covariates, such as a square or an
interaction. Those line up with the partition.

**Proposition 9.** Let `g` be an outcome component the analysis omits; for OLS
under exact balance, take `g` residualized on `(1, X)`. Its contribution to the
MSE under `G(K)`, relative to complete randomization, is exactly

```text
g' Sigma_G g / g' Sigma_CRD g  =  ((N - 1)/N) m s_b^2 / V_g  =:  R_g,
```

where:

- `s_b^2` is the variance of `g`'s `K` group means (divisor `K - 1`);
- `V_g` is the variance of `g` (divisor `N`).

A random partition gives `R_g` near 1. A `g` constant within groups gives the
worst case `(N - 1)/(K - 1)`. `R_g` can be computed for any candidate omission
before the experiment runs (`balance_buys.grouped.relative_mse`).

*Proof.* `g' Sigma_G g = (K/(K - 1)) m^2 sum_k (gbar_k - gbar)^2` from
Proposition 6, and `g' Sigma_CRD g = (N/(N - 1)) |g - gbar|^2`. The test
suite checks the identity against enumeration.

*What it shows* (`scripts/gk_omitted.py`).

- **Setup.** `N = 120`, `p` = 3 or 5, 20 draws. The candidates `g` are every
  square and pairwise product, residualized on `X`.
- **Mean-only anticlustering into small groups makes these omissions worse
  than complete randomization.** Mean `R_g` was 1.33 to 1.76, rising as
  groups shrink: 1.76 for pairs at `p = 3`. To make a pair's mean equal the
  sample mean, the partition pairs a high unit with a low one, so the pair
  means of `x^2` spread out.
- **Balancing the squares and products as well cut `R_g` to 0.41 to 0.81.**
  The cost is mean balance: `E[Q]` rose from 0.04–1.0 to 0.9–4.3, against
  `p` under complete randomization.

So which structure the partition balances decides whether small groups help or
hurt on nonlinear outcomes. Anticluster on the functions whose omission you
fear, and report `R_g` for the rest.

## Predictions against the pilot

Simulated values carry roughly ±0.1 to 0.15 of noise (400 datasets per cell).
The pilot never ran `G(K)`.

ESS of a mean-balancing design with OLS over complete randomization with OLS,
linear outcome. The prediction is Corollary 2b's exact value for an exactly
balanced design. The simulated designs only approximate exact balance.

| N | p | Jensen bound (2a) | exact Gaussian (2b) | simulated: rerand / exchange / shell |
|---|---|---|---|---|
| 20 | 1 | 1.06 | 1.06 | 0.91 / 0.82 / 0.85 |
| 20 | 5 | 1.36 | 1.42 | 1.52 / 1.33 / 1.48 |
| 50 | 5 | 1.11 | 1.12 | 1.15 / 1.18 / 1.13 |
| 50 | 20 | 1.69 | 1.74 | 1.60 / 1.79 / 1.66 |
| 100 | 20 | 1.25 | 1.26 | 1.09 / 1.32 / 1.36 |
| 300 | 20 | 1.07 | 1.07 | 0.90 / 0.98 / 0.92 |

ESS of the FSM for the nonlinear outcome against the Proposition 3 heuristic
with `V_g / sigma^2 = 0.5`. The pilot standardized the quadratic part but did
not residualize it on `X` in each sample, so it is orthogonal to `X` only in
the population.

| N | p | heuristic | simulated |
|---|---|---|---|
| 50 | 5 | 1.68 | 1.43 |
| 100 | 5 | 1.58 | 1.63 |
| 300 | 5 | 1.53 | 1.62 |
| 300 | 20 | 1.61 | 1.36 |

t-test against randomization test on the difference in means, exchange
anticlustering, `p = 5` (the Proposition 4 illustration):

| outcome | kappa | predicted size / power / RI power | simulated, N = 50, 100, 300 |
|---|---|---|---|
| linear | 0.71 | 0.006 / 0.50 / 0.79 | size 0.003 to 0.007; power 0.43 to 0.51; RI 0.76 to 0.81 |
| nonlinear | 0.87 | 0.024 / 0.50 / 0.62 | size 0.022 to 0.028; power 0.45 to 0.51; RI 0.56 to 0.61 |

## What follows

1. A design's precision and validity depend on its whole assignment
   distribution, `Sigma` and its support, not on the balance it achieves.
2. Against unregularized OLS, exact balance gains at least
   `(N - 1) / (N - 1 - p)`, exactly `(N - 3) / (N - p - 3)` with Gaussian
   covariates. It matters when `p / N` is not small.
3. Balancing an outcome component the analysis omits can help or hurt. It
   helps when that component is itself balanced.
4. Balance is paid for in randomness: label symmetry for unbiasedness, enough
   probability spread (at least `1 / alpha` partitions for a uniform design)
   for a test to reject, and a bounded `lambda_max` for robustness.
5. Under `G(K)` all three are explicit given the partition, and so is the
   Frobenius randomness term of Kapelner et al.'s tail criterion. The worst
   case forces `K` to grow with `N`, which puts the burden on balancing small
   groups.
6. Any local optimum of the exchange method carries a checkable bound on its
   imbalance (Proposition 7a). The typical rate is about
   `m^{-(1 + 2/p)} N^{-2/p}` (Proposition 7b): fast for few covariates, slow
   for many.
7. Under the tail criterion, `G(K)`'s MSE behaves like a chi-squared with
   `K - 1` degrees of freedom (Proposition 8). It beats complete
   randomization with OLS at the 95% quantile only with many small groups
   and a large `p/N`.
8. Small groups balanced on means alone make omitted squares and interactions
   worse than complete randomization. Balancing those functions too reverses
   that, at a cost in mean balance (Proposition 9).
9. In a direct comparison (report, section 5), local search from random starts
   (greedy pair-switching) reaches Proposition 2's optimum on average and has
   the lightest tail. So the randomness that protects against unmeasured
   covariates is best bought with many near-optimal partitions, not with one
   coarse partition randomized at the group level.

Items 1 to 4 are known, mostly from Kallus (2018, 2020), Kapelner et al.
(2021), Krieger et al. (2019) and Li, Ding and Rubin (2018). They are stated
here so that `G(K)` can be judged against them.
