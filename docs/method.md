# How randomized ABA works

Assignment-based anticlustering (ABA), introduced by
[Baumann, Goldschmidt, Hochbaum, and Yang](https://arxiv.org/abs/2601.06351),
constructs groups whose baseline characteristics are similar. `randaba`
adds random variation in the batches and random treatment labels.

## One assignment

1. Center and whiten the baseline covariates so distances account for their
   different scales and correlations.
2. Compute each participant's squared distance from the center. Multiply it
   by `exp(jitter * Z)`, where each `Z` is an independent standard-normal
   draw. Sort by this score and form consecutive batches of `K` participants.
3. Put one participant from each batch in each of the `K` groups. Solve an
   assignment problem that maximizes the total squared distance from the
   participants to the groups' current centroids. This reduces differences
   between the updated group means.
4. Uniformly shuffle the treatment labels attached to the finished groups.

The default `jitter=1` varies who is grouped together between assignments.
At `jitter=0`, the partition is fixed and only treatment labels change.
That still gives unbiased contrasts and valid randomization tests, but the
restricted set of comparisons can severely limit power. With two arms,
label shuffling alone only reverses the sign of a difference in means, so
a two-sided test cannot reject.

## The connection to ARMM

For a fixed batch and equal current group sizes, maximizing the sum of
squared distances to current centroids also minimizes the sum of squared
updated group totals. This is the greedy allocation criterion used by
[ARMM (Yang et al., 2023)](https://doi.org/10.1016/j.csda.2022.107642).
An assignment solver finds that allocation without enumerating `K!`
permutations. ARMM's biased coin and batch formation differ from randomized
ABA's, so the full assignment distributions are different.

## Operation counts

Let `n` be the number of participants, `K` the number of arms, and `p` the
number of supplied balancing features, including any appended squares.
There are `n/K` batches. Count the work in one call:

| Step | Work |
|---|---|
| Center and whiten | `O(np² + p³)` |
| Generate scores and sort | `O(np + n log n)` |
| Form the `K × K` cost matrix for each batch | `(n/K) × O(K²p) = O(nKp)` |
| Solve each assignment problem | `(n/K) × O(K³) = O(nK²)` |
| Update group totals and shuffle labels | `O(np + n + K)` |

The total is `O(np² + p³ + n log n + nKp + nK²)`.
With `K` and `p` held fixed, the assignment pass is `O(n)` and the full
procedure is `O(n log n)`. Doubling the participant count doubles the
number of batch solves. Increasing `K` is more expensive: the solve cost
across batches is quadratic in `K`.

The implementation uses NumPy's
[stable indirect sort](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html)
and SciPy's
[linear assignment solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html).
It keeps an `n × p` whitened array and creates a `K × K × p` temporary when
forming distances. Additional memory is `O(np + p² + K²p)`, hence linear
in `n` at fixed `K, p`.

For comparison, an exchange pass visits each of `n` participants and
considers swaps with order `n` others, computing `p`-dimensional changes:
`O(n²p)` per pass. Sequential adaptive allocation scores `K` arms for each
participant, taking `O(nKp)` for the allocation pass. These counts establish
how work grows; they do not give elapsed seconds or a measured speed ratio.

The [research note](https://github.com/finite-sample/randaba/blob/main/paper/note.tex)
gives the proofs for unbiasedness, randomization-test validity, the greedy
step equivalence, and balance on the sort key.
