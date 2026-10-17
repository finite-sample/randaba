# Anticlustering for experimental assignment: balance, robustness and valid inference

**Finding.** Anticlustering (splitting units into groups that look alike on
covariates) can build the assignment of a randomized experiment. How well that
works depends on the whole distribution of assignments the procedure can
produce, not on the balance of the draw you get. Three properties of that
distribution decide it:

- **Unbiasedness**, from label symmetry.
- **Robustness**, the largest eigenvalue of the assignment covariance, which
  bounds how badly an unmeasured outcome component can be hit.
- **Testability**, the probability mass on the observed partition, which
  bounds the smallest randomization p-value.

This framing is not new: it is Kallus's (2018, 2020) and that of Krieger,
Kapelner and Azriel (2019, 2021). What may be new is grouped randomization `G(K)`. It
anticlusters into `K` groups, then randomizes whole groups to arms, and it
makes all three properties explicit:

- the worst case `(N-1)/(K-1)`;
- the Frobenius randomness term `N^2/(K-1)` of Kapelner et al.'s tail
  criterion;
- the test resolution `2/C(K, K/2)`.

Holding the worst case to a factor `C` of complete randomization requires
`K >= 1 + (N-1)/C`. So the question that decides whether the design is useful
is how well anticlustering balances small groups. That question is open, and
the pilot did not run `G(K)`.

## 1. The question

Random assignment balances covariates on average, not in the draw you get. Two
remedies compete:

- **Balance at design time:** blocking, rerandomization, the Finite Selection
  Model (FSM), anticlustering.
- **Adjust at analysis time:** regression on the covariates.

To first order, adjustment recovers what design-time balance buys (Li and Ding
2020). Simulations agree (Green and Mikhaeil 2026; Papenberg and Angelike 2026).
The questions here are what balance buys beyond that, and what it costs in
randomization.

## 2. What already exists

| Approach | Balance | Randomness | Inference |
|---|---|---|---|
| Complete randomization | none by design | maximal | standard |
| Blocking, matched pairs and tuples (Bai 2022; Bai, Liu and Tabord-Meehan 2024) | similar units grouped, treatment varied within | within blocks | design-specific asymptotics |
| Rerandomization (Morgan and Rubin 2012) | Mahalanobis below a threshold | large | non-normal limit (Li, Ding and Rubin 2018); power (Branson, Li and Ding 2024) |
| Cluster rerandomization (Lu, Liu, Liu and Ding 2023) | arm balance over given clusters | restricted cluster assignments | asymptotic, needs a few dozen clusters; Fisher test for fewer |
| FSM (Morris 1979; Chattopadhyay, Morris and Zubizarreta 2026) | D-optimal sequential selection | random turn order; few partitions at small `N` with many covariates | randomization test |
| Gram-Schmidt Walk (Harshaw et al. 2024) | tunable | tunable, with a worst-case bound | design-based |
| Optimized partition, arms randomized (Bertsimas, Johnson and Kallus 2015) | means and second moments, cross-products included, by integer programming | one partition | bootstrap |
| Optimal designs (Kallus 2018, 2020) | minimax under an outcome class | mixed strategies, optionally constrained for Fisher's test | design-based |
| Greedy pair-switching (Krieger, Azriel and Kapelner 2019) | exchange to a local optimum from a random start; imbalance `O_p(n^{-(1+2/p)})` | close to complete randomization by pairwise entropy | randomization test over replicate designs |
| Matching, then greedy switching across pairs (Krieger, Azriel and Kapelner, arXiv:2012.03330) | pairs for nonlinear outcomes, plus mean balance | as random as matched pairs | randomization test |
| Tail-criterion rerandomization (Kapelner et al. 2021; arXiv:1905.03337) | threshold chosen by a quantile of MSE over unobserved covariates | tuned by that criterion | randomization test |
| Anticlustering for assignment (Papenberg and Angelike 2026) | best of the methods they compared | a random start and one exchange pass | t-test and OLS; a randomization test listed as future work |
| Assignment-based anticlustering, ABA (Baumann et al. 2026) | near-perfect on means; scales to millions | none (deterministic) | not designed for it |

The closest work is Kallus's.

- **Kallus (2018)** treats a design as a game against an adversarial outcome
  function from a stated class. Without structure, complete randomization is
  minimax and a single partition can be `N - 1` times worse. Under a correctly
  specified parametric class, optimal balancing reaches the regression-adjusted
  variance `1 - R^2` without bias. Its pure-strategy designs (a single best
  partition) cannot be tested by randomization, so he proposes a bootstrap
  test.
- **Kallus (2020)** constrains the minimax design so that no assignment exceeds
  probability `alpha / 2`, which keeps Fisher's test usable. He computes it with
  repeated mixed-integer programs and an `N x N` semidefinite program.

- **Kapelner, Krieger and coauthors** settle much of the rest.
  - Greedy pair-switching is the two-group exchange algorithm, and they
    already test it by redrawing the design (Krieger et al. 2019). So a
    randomization test for exchange anticlustering, which Papenberg and
    Angelike list as future work, exists.
  - Their pairwise-entropy measure of randomness is the pair check used in
    section 3.
  - Their tail criterion (Kapelner et al. 2021) penalizes `||Sigma||_F^2`,
    which favours designs between complete randomization and a single
    optimal partition.

`G(K)` sits in Kallus's constrained class. With `K` equal to the number of arms
it is Bertsimas, Johnson and Kallus's design. Its covariance is closed form and
it needs no solver. That makes Kapelner et al.'s criterion a one-dimensional
problem in `K` rather than something estimated by sampling. How close `G(K)`
comes to Kallus's constrained optimum is unknown. The specific construction,
many anticlusters randomized whole, was not found in the literature read, which
does not establish that it is new.

## 3. How anticlustering balances, and how to randomize it

ABA sorts units by distance from the centre and deals them out in shells of
`K`. Each shell goes one unit per group, by the matching that maximizes total
squared distance to the groups' current centroids. Expanding that objective
shows the matching minimizes `sum x_i . c_k`: each group receives the unit
pointing most against the way its centroid currently leans. That is the logic
of online vector balancing, which keeps drift bounded in that literature. No
such bound is proved here for ABA's shells. The FSM is the same family with a
different distance (leverage in the group's own covariance) and a random turn
order instead of a joint matching.

An ablation on 20,000 units, 8 covariates and 200 groups:

| Variant | Mean imbalance | Second-moment imbalance | Same-group pairs repeated across runs |
|---|---|---|---|
| Random partition | 8 | 36 | 0.005 |
| ABA | 0.005 | 56 | 1.0 (deterministic) |
| ABA, sort order jittered, labels shuffled | 0.005 | 56 | 0.005 |
| ABA given squares and cross-products as features | 0.013 | 0.08 | 1.0 |
| FSM distance in shells, random turn order | 0.035 | 0.19 | 0.005 |

The **shell draft** is ABA made random:

1. Whiten the covariates.
2. Sort units by distance from the centre times `exp(sigma Z)`, with `Z`
   standard normal (the jitter).
3. Cut the sorted list into shells of `K`.
4. Deal each shell by ABA's matching.
5. Shuffle the group labels at the end.

The pilot used `sigma = 1` and centroid distance. Its code is
`shelldraft.designs.shell_draft`.

What the ablation shows, and what it does not:

1. **Jittering the sort order left balance unchanged** and made pair
   co-membership look random. Pair co-membership is Krieger et al.'s (2019)
   pairwise-entropy measure. Neither the probability mass on each partition
   nor the assignment covariance (its `lambda_max` or Frobenius norm) was
   measured. Those decide robustness and testability.
2. **Shuffling labels at the end is mandatory.** Without it, jitter left some
   units in group 0 in 83% of draws. The pairs check could not see this,
   because it ignores labels.
3. **Balancing means can unbalance spread.** Two fixes exist. The k-plus
   criterion (Papenberg 2024) appends each covariate's squared and higher
   deviations, which balances marginal variances but not covariances.
   Balancing squares and cross-products together is the design of Bertsimas,
   Johnson and Kallus (2015), whose multivariate formulation includes the
   cross-moments, and Kallus's (2018) quadratic-kernel allocation. Either
   works only while the feature count stays well below `N`.

## 4. Theory

Full statements and proofs are in [Theory](theory.md). Two equal arms; `W` is
the `+/-1` assignment vector; `Sigma = E[WW']`.

1. **Error identity.** `tau_hat - tau = (2/N) W' mu`, with `mu_i` the average of
   unit `i`'s two potential outcomes. A design's randomization risk is all in
   `Sigma`.
2. **Variance budget.** Under label symmetry `tau_hat` is unbiased and
   `trace Sigma = N` for every design. Balance moves variance off the measured
   directions onto the rest. Averaged isotropically, the cost is
   `(N-1)/(N-p-1)`. In the worst direction it is `lambda_max`, up to `N`.
3. **A regression benchmark.** In a linear model with a constant effect and
   homoskedastic errors, OLS has `MSE = 4 sigma^2 E[1/(N-Q)]`, where `Q = W'P_X W`
   measures treatment-covariate collinearity. The pieces are known: Kapelner
   et al. (arXiv:1905.03337, eq. 6) give the design-averaged form to third
   order, and the Gaussian value is a classical inverse-Beta moment. Exact
   balance gains *at least*
   `(N-1)/(N-1-p)` over complete randomization with OLS, and exactly
   `(N-3)/(N-p-3)` with Gaussian covariates. That gain is 6% with one
   covariate at `N = 20`, consistent with Papenberg and Angelike's null. It is
   a finite-sample term beneath Li and Ding's first-order result, and it
   matters when `p/N` is not small. Papenberg and Angelike's own noise
   condition is consistent with it. With five irrelevant covariates added to
   the regression, adjusted power was 0.720 after anticlustering against 0.706
   after random assignment, compared with 0.722 and 0.719 with the one
   covariate alone.
4. **Omitted components.** If the design balances an outcome component that
   the regression omits, the gain grows by about `1 + V_g/sigma^2`. If it
   balances only the means, that component can get worse, not better.
5. **Inference.** The t-test is conservative when the design reduces the true
   variance, which is the intended case but is not guaranteed. A
   randomization test that redraws the design controls level for the sharp
   null (Morgan and Rubin 2012; Krieger et al. 2019). The weak null needs a
   studentized statistic. A design uniform over `M` partitions can reach
   p-values no smaller than `1/M` (Kallus 2020, Section 7).
6. **Grouped randomization `G(K)`**, conditional on the partition:
   - Every unit has probability 1/2.
   - The worst case is `(N-1)/(K-1)` times complete randomization.
   - `||Sigma||_F^2 = N^2/(K-1)`, the randomness term of Kapelner et al.'s
     criterion.
   - The smallest p-value is `2/C(K, K/2)`.
   - Expected imbalance is `E[Q] = (K m^2/(K-1)) trace(A^{-1} D'D)`, with `D` the
     group-mean deviations.
   - The group-level Neyman variance is conservative in expectation. A
     `t_{K-2}` reference is only an approximation: with `K = 8`, one null
     configuration rejects 14% of the time at a nominal 5%. Lu et al.'s
     simulations suggest normal-based intervals need a few dozen groups.

## 5. Evidence

The pilot ([details](pilot.md)) ran 400 datasets per cell with two arms and
200 reference draws per dataset. It compared complete randomization,
rerandomization, exchange anticlustering, the shell draft and the FSM. It did
not run `G(K)`. Gains are effective sample size relative to complete
randomization analysed with OLS, and carry about ±0.1 to 0.15 of simulation
noise.

| N | p | Exact balance (Gaussian `X`) | Rerandomization | Exchange anticlustering | Shell draft |
|---|---|---|---|---|---|
| 20 | 5 | 1.42 | 1.52 | 1.33 | 1.48 |
| 50 | 20 | 1.74 | 1.60 | 1.79 | 1.66 |
| 100 | 20 | 1.26 | 1.09 | 1.32 | 1.36 |
| 300 | 20 | 1.07 | 0.90 | 0.98 | 0.92 |

With 400 datasets the simulated values cannot separate the designs from each
other or from the formula. On the nonlinear outcome the FSM reached 1.6 at
`N = 100` and `300`. That is in line with the omitted-component heuristic, but
the pilot's quadratic term was not orthogonal to `X` in each sample, so this
is not a test of it. At small `N` with many covariates the FSM produced a few
dozen distinct allocations, and its randomization tests had little power.

The t-test against the randomization test, under exchange anticlustering with
5 covariates, where half the outcome variance is balanced away
(`kappa = 0.71`):

| | t-test size | t-test power | Randomization-test power |
|---|---|---|---|
| Predicted | 0.006 | 0.50 | 0.79 |
| Simulated | 0.003 to 0.007 | 0.43 to 0.51 | 0.76 to 0.81 |

## 6. What has to be shown next

`G(K)` is a proposal, not a recommendation, until three things are done.

1. **A balance theorem for small groups.** Bound the partition's group-mean
   deviation `D` as a function of `N`, `K` and `p` for a specific
   anticlustering algorithm. For two groups the analogues exist: exponential
   for the optimum (Bertsimas et al. 2015) and `O_p(n^{-(1+2/p)})` for greedy
   switching (Krieger et al. 2019). Through `E[Q]` above, a `K`-group bound
   becomes a risk bound at a fixed worst case `C`.
2. **Choosing `K`.** Minimize Kapelner et al.'s tail criterion over `K`, using
   the closed-form `E[Q]` and `||Sigma||_F^2`. Check the choice against their
   sampled threshold for rerandomization on the same data.
3. **A comparison at matched robustness.**
   - Sweep `K`.
   - Compare `G(K)` against:
     - tail-criterion rerandomization;
     - greedy pair-switching and its matched-pairs hybrid;
     - matched tuples;
     - the Gram-Schmidt Walk;
     - complete randomization with good adjustment, which should include the
       same nonlinear features the design is given.
   - At small `N`, also compare against Kallus's inference-constrained
     design, where his solver runs.
   - Report MSE, power, interval coverage and length, and sharp-null and
     weak-null rejection rates separately.
   - Include heterogeneous effects and heteroskedasticity, plus outcome
     components aligned with group membership.
   - Use enough replications to attach uncertainty to differences between
     designs.

Also open: more than two arms, the many-group case where ABA's speed matters,
and a real dataset (Lalonde, with 1974 and 1975 earnings as prognostic
covariates).

## Code and status

- **This repository** (`finite-sample/shelldraft`, private) holds the shell
  draft, a replica of `anticlust`'s exchange method checked exactly against R,
  rerandomization, inference code, the pilot and the theory checks. `G(K)` is
  not yet implemented.
- **`finite-sample/finite-selection`** is the Python FSM with parity to R.
