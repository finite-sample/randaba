# Anticlustering for randomized experiments: report

**Finding.** Anticlustering (splitting units into groups that look alike on
covariates) can serve as the assignment step of a randomized experiment, as long
as two conditions hold. The design has to keep enough randomness for
inference, and the analysis has to account for the design. Under those
conditions:

- **Against regression adjustment,** balance is worth the degrees of freedom
  the regression would spend, about $(N-1)/(N-1-p)$, plus whatever variance the
  design balances that the analysis does not model.
- **On the pilot,** that meant 30 to 80% more effective sample size than complete
  randomization analysed with OLS when the covariates were many relative to
  $N$, and roughly nothing with one covariate.
- **One integer controls the trade-off.** Anticluster into $K$ groups, then
  randomize groups to arms. $K$ sets the worst case, the resolution of the
  test and the balance, each in closed form.

## 1. The question

Random assignment balances covariates on average, not in the draw you get. Two
remedies compete:

- **Balance at design time:** blocking, rerandomization, the Finite Selection
  Model (FSM), anticlustering.
- **Adjust at analysis time:** regression on the covariates.

Green and Mikhaeil (2026) and Papenberg and Angelike (2026) both find that
adjustment recovers most of what design-time balance buys. The questions here
are when design still adds something, and how to keep inference valid when it
does.

## 2. What already exists

| Approach | Balance | Randomness | Inference |
|---|---|---|---|
| Complete randomization | none by design | maximal | standard |
| Blocking, matched pairs (Bai 2022) | on block variables | within blocks | standard, blocked |
| Rerandomization (Morgan and Rubin 2012) | Mahalanobis below a threshold | large | asymptotic non-normal limit (Li, Ding and Rubin 2018); power and sample size (Branson, Li and Ding 2024) |
| FSM (Morris 1979; Chattopadhyay, Morris and Zubizarreta 2026) | D-optimal, means and second moments | random turn order; collapses at small $N$ with many covariates | randomization test |
| Gram-Schmidt Walk (Harshaw et al. 2024) | tunable | tunable, with a worst-case bound | design-based |
| Kernel allocation (Kallus 2018) | optimal under a kernel | two allocations | model-based only |
| Anticlustering for assignment (Papenberg and Angelike 2026) | best of the methods they compared | a random start and one exchange pass | t-test and OLS; they leave a matching randomization test to future work |
| Assignment-based anticlustering, ABA (arXiv 2601.06351, 2026) | near-perfect on means; scales to millions of units | none (deterministic) | not designed for it |

Papenberg and Angelike are closest to this report. With one covariate and a
linear outcome, they found that balance did not raise power once the analysis
adjusted. Section 4 explains why that was expected.

## 3. How anticlustering balances, and how to randomize it

ABA sorts units by distance from the centre and deals them out in shells of
$K$. Each shell goes one unit per group, by the matching that maximizes total
squared distance to the groups' current centroids. Expanding that objective
shows the matching minimizes $\sum x_i \cdot c_k$: each group receives the unit
pointing most directly against the way its centroid currently leans. So the
groups' drift stays bounded instead of growing with group size, which is the
classic online vector-balancing result. The FSM is the same family with a
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

Three lessons:

1. **Randomness is nearly free if it goes in the right place.** Jittering the
   sort order changes which units compete in a shell but leaves the matching
   optimal, so balance is unchanged. Noise in the matching itself costs
   balance.
2. **Shuffling labels at the end is mandatory.** Without it, jitter left some
   units in group 0 in 83% of draws. The pairs check could not see this,
   because it ignores labels.
3. **Balancing means can unbalance spread.** ABA on means made second moments
   worse than random. Feed it the squares and cross-products if the outcome may
   be nonlinear.

## 4. Theory

Full statements and proofs are in [Theory](theory.md). Two equal arms; $W$ is
the $\pm1$ assignment vector; $\Sigma = E[WW']$.

1. **Error identity.** $\hat\tau - \tau = \tfrac{2}{N} W'\mu$, with $\mu_i$ the
   average of unit $i$'s two potential outcomes. A design's precision is all
   in $\Sigma$.
2. **Variance budget.** Under label symmetry $\hat\tau$ is unbiased and
   $\operatorname{tr}\Sigma = N$ for every design. Balance moves variance off
   the measured directions onto the rest. On average the cost is a factor
   $(N-1)/(N-p-1)$; in the worst direction it is $\lambda_{\max}$.
3. **Balance against OLS.** In a linear model with a constant effect,
   $\mathrm{MSE} = 4\sigma^2\, E[1/(N-Q)]$, where $Q = W'P_X W$ measures
   imbalance. Exact balance is optimal, and its advantage over complete
   randomization with OLS is at least $(N-1)/(N-1-p)$, whatever the
   coefficients. With one covariate that is at most 6%, which is why
   Papenberg and Angelike saw nothing.
4. **Balancing what the analysis omits.** A component $g$ that the regression
   does not model but the design balances adds a factor $1 + V_g/\sigma^2$.
   The FSM balances second moments, so it collects this on quadratic outcomes.
5. **Inference.** The pooled t-test ignores the design, so it is conservative
   under balance. Even under complete randomization it is conservative when
   effects vary. For rerandomization an analytic correction exists (Li, Ding
   and Rubin 2018). For anticlustering, a randomization test that redraws the
   design is exact for the sharp null. It can reject at level $\alpha$ only if
   no partition carries more than $\alpha$ probability, so at least 20 roughly
   equally likely partitions at 5%.
6. **Grouped randomization $G(K)$.** Anticluster into $K$ groups and randomize
   $K/2$ groups to treatment.
   - Every unit has probability 1/2 of treatment.
   - The worst-case variance factor is exactly $(N-1)/(K-1)$.
   - The smallest randomization p-value is $2/\binom{K}{K/2}$, so $K \ge 8$
     is needed to reject at 5%.
   - Cluster-level Neyman standard errors apply.

## 5. Evidence

The pilot ([details](pilot.md)) ran 400 datasets per cell with two arms and
200 reference draws per dataset. Gains are effective sample size relative to
complete randomization analysed with OLS, and carry about $\pm 0.1$ to $0.15$
of simulation noise.

| N | p | Predicted (Prop. 3) | Rerandomization | Exchange anticlustering | Shell draft |
|---|---|---|---|---|---|
| 20 | 5 | 1.47 | 1.52 | 1.33 | 1.48 |
| 50 | 20 | 1.81 | 1.60 | 1.79 | 1.66 |
| 100 | 20 | 1.26 | 1.09 | 1.32 | 1.36 |
| 300 | 20 | 1.07 | 0.90 | 0.98 | 0.92 |

On the nonlinear outcome the FSM reached 1.6 at $N = 100$ and $300$, against
1.5 to 1.6 predicted. At small $N$ with many covariates it collapsed to a few
dozen distinct allocations, and its randomization tests lost their power.

The t-test against the randomization test, under exchange anticlustering with
5 covariates: half the outcome variance is balanced away, so $\kappa = 0.71$.

| | t-test size | t-test power | Randomization-test power |
|---|---|---|---|
| Predicted | 0.006 | 0.50 | 0.79 |
| Simulated | 0.003 to 0.007 | 0.43 to 0.51 | 0.76 to 0.81 |

## 6. Recommended design

Use $G(K)$:

1. Anticluster on the covariates plus their squares and cross-products.
2. Pick the smallest $K$ that gives the test resolution you need and a
   tolerable worst case $(N-1)/(K-1)$.
3. Randomize groups to arms.
4. Analyse with the difference in means or OLS, using cluster-level Neyman
   standard errors or a randomization test that redraws step 3.

Expect gains over complete randomization with OLS when $p/N$ is not small or
the outcome is nonlinear in ways the analysis will not model. Expect little
otherwise.

## 7. Open questions

- **Prior art for $G(K)$.** It resembles counterbalanced stimulus lists and
  cluster randomization of matched sets; this needs a literature check before
  any claim of novelty.
- **Balance as a function of group size.** How fast does an anticlustered
  group's mean error shrink with $m = N/K$? That would make $K$'s choice a
  formula.
- **Heterogeneous effects and the weak null.** The randomization test is exact
  only for the sharp null. Branson, Li and Ding show conservative variance
  estimation can make balanced designs lose power for small effects.
- **The robustness frontier.** What is the best balance achievable at a given
  $\lambda_{\max}$? $G(K)$ is one curve; the Gram-Schmidt Walk is another.
- **More arms and larger scale.** More than two arms, and the many-group case
  (minibatches, benchmark subsets) where ABA's speed matters.
- **A real dataset.** Lalonde, with the 1974 and 1975 earnings as prognostic
  covariates.

## Code and status

- **This repository** (`finite-sample/shelldraft`, private) holds the designs:
  the shell draft, a replica of `anticlust`'s exchange method checked exactly
  against R, and rerandomization. It also holds the inference code, the pilot
  and the theory checks.
- **`finite-sample/finite-selection`** is the Python FSM with parity to R.
