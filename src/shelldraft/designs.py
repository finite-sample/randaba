"""Assignment designs: each returns one group label (0 to k - 1) per unit."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

Cost = Literal["centroid", "leverage"]


def _check(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if k < 2 or len(x) % k:
        raise ValueError("the number of units must be a multiple of k >= 2")
    return x


def whiten(x: np.ndarray) -> np.ndarray:
    """Centre the covariates and rotate them to identity sample covariance.

    Euclidean distance on the result is Mahalanobis distance on ``x``.

    Args:
        x: Covariates, one row per unit.

    Returns:
        The whitened covariates.

    Raises:
        ValueError: If the sample covariance is singular.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    centred = x - x.mean(axis=0)
    try:
        chol = np.linalg.cholesky(np.atleast_2d(np.cov(centred, rowvar=False)))
    except np.linalg.LinAlgError as err:
        raise ValueError("covariates are collinear; drop or combine columns") from err
    return np.linalg.solve(chol, centred.T).T


def crd(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Complete randomization into ``k`` equal groups.

    Args:
        n: Number of units, a multiple of ``k``.
        k: Number of groups.
        rng: Random number generator.

    Returns:
        Group labels.

    Raises:
        ValueError: If ``n`` is not a multiple of ``k`` or ``k < 2``.
    """
    if k < 2 or n % k:
        raise ValueError("n must be a multiple of k >= 2")
    return rng.permutation(np.repeat(np.arange(k), n // k))


def shell_draft(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    jitter: float = 0.05,
    cost: Cost = "centroid",
    eps: float = 1e-3,
) -> np.ndarray:
    """Randomized shell anticlustering into ``k`` equal groups.

    Units are whitened, sorted by squared distance from the centre after
    multiplying it by ``exp(jitter * N(0, 1))``, and cut into shells of ``k``.
    Each shell goes one unit per group by the matching that maximizes the total
    cost. With ``cost="centroid"`` the cost is the squared distance from a unit
    to the group's current centroid (as in assignment-based anticlustering);
    with ``cost="leverage"`` it is ``d'(D_g'D_g + ridge)^-1 d`` on the design
    row ``d = (1, w)`` (the FSM's D-optimal criterion). Labels are shuffled at
    the end, so every unit has probability ``1 / k`` of each group.

    Args:
        x: Covariates, one row per unit.
        k: Number of groups.
        rng: Random number generator.
        jitter: Log-scale noise on the sort key; 0 gives a deterministic
            partition (before the label shuffle).
        cost: Matching cost, ``"centroid"`` or ``"leverage"``.
        eps: Ridge scale for ``cost="leverage"``; the ridge is
            ``eps * D'D / n``.

    Returns:
        Group labels.
    """
    x = _check(x, k)
    w = whiten(x)
    n, p = w.shape
    key = (w**2).sum(axis=1)
    if jitter:
        key = key * np.exp(jitter * rng.normal(size=n))
    order = np.argsort(-key, kind="stable")
    z = np.empty(n, dtype=int)
    sums = np.zeros((k, p))
    counts = np.zeros(k)
    d = np.column_stack([np.ones(n), w])
    ainv = np.repeat(np.linalg.inv(eps * (d.T @ d) / n)[None], k, axis=0)
    for start in range(0, n, k):
        batch = order[start : start + k]
        if cost == "centroid":
            centroids = sums / np.maximum(counts, 1)[:, None]
            c = ((w[batch][:, None, :] - centroids[None]) ** 2).sum(axis=-1)
        else:
            db = d[batch]
            c = np.einsum("ij,kjl,il->ik", db, ainv, db)
        rows, cols = linear_sum_assignment(c, maximize=True)
        for i, g in zip(rows, cols, strict=True):
            u = batch[i]
            z[u] = g
            sums[g] += w[u]
            counts[g] += 1
            v = ainv[g] @ d[u]
            ainv[g] -= np.outer(v, v) / (1.0 + d[u] @ v)
    return rng.permutation(k)[z]


def exchange_anticlust(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    init: np.ndarray | None = None,
) -> np.ndarray:
    """One pass of the exchange method on the k-means anticlustering objective.

    Starting from a random equal split (or ``init``), each unit in turn is
    swapped with the unit in another group whose swap most reduces the spread
    of the group centroids, if any swap reduces it. This is the procedure
    Papenberg and Angelike (2026) use, after R ``anticlust``'s ``exchange``
    method with ``objective = "variance"``.

    Args:
        x: Covariates, one row per unit, used as given (not standardized).
        k: Number of groups.
        rng: Random number generator, used for the initial split.
        init: Optional initial labels.

    Returns:
        Group labels.
    """
    x = _check(x, k)
    n = len(x)
    z = crd(n, k, rng) if init is None else np.asarray(init, dtype=int).copy()
    sums = np.array([x[z == g].sum(axis=0) for g in range(k)])
    sizes = np.bincount(z, minlength=k).astype(float)
    for i in range(n):
        a = z[i]
        others = np.flatnonzero(z != a)
        b = z[others]
        new_a = sums[a] - x[i] + x[others]
        new_b = sums[b] - x[others] + x[i]
        delta = ((new_a**2).sum(axis=1) - (sums[a] ** 2).sum()) / sizes[a] + (
            (new_b**2).sum(axis=1) - (sums[b] ** 2).sum(axis=1)
        ) / sizes[b]
        j = int(np.argmin(delta))
        if delta[j] < 0:
            u, g = others[j], z[others[j]]
            sums[a] += x[u] - x[i]
            sums[g] += x[i] - x[u]
            z[i], z[u] = g, a
    return z


def rerandomize(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    accept: float = 0.001,
    batch: int = 4096,
) -> np.ndarray:
    """Two-group rerandomization on the Mahalanobis distance (Morgan and Rubin).

    Complete randomizations are drawn until the Mahalanobis distance between
    the group means, scaled by ``n / 4``, falls below the ``accept`` quantile
    of its chi-squared approximation.

    Args:
        x: Covariates, one row per unit.
        rng: Random number generator.
        accept: Target acceptance probability.
        batch: Number of candidate splits drawn at a time.

    Returns:
        Group labels (0 or 1).
    """
    x = _check(x, 2)
    n, p = x.shape
    sinv = np.linalg.inv(np.atleast_2d(np.cov(x, rowvar=False)))
    threshold = chi2.ppf(accept, p)
    base = np.repeat([0, 1], n // 2)
    while True:
        cand = rng.permuted(np.tile(base, (batch, 1)), axis=1)
        diff = (cand @ x - (1 - cand) @ x) / (n // 2)
        m = np.einsum("ij,jk,ik->i", diff, sinv, diff) * n / 4
        hit = np.flatnonzero(m <= threshold)
        if hit.size:
            return cand[hit[0]]
