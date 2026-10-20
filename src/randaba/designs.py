"""Assignment designs: each returns one group label (0 to k - 1) per unit."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2, norm


def _covariates(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] == 0:
        raise ValueError("x must have at least two rows and one covariate column")
    if not np.isfinite(x).all():
        raise ValueError("covariates must be finite")
    return x


def _positive_int(value: int, name: str) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")


def _check(x: np.ndarray, k: int) -> np.ndarray:
    x = _covariates(x)
    _positive_int(k, "k")
    if k < 2 or len(x) % k:
        raise ValueError("the number of units must be a multiple of k >= 2")
    return x


def whiten(x: np.ndarray) -> np.ndarray:
    """Center the covariates and rotate them to identity sample covariance.

    Euclidean distance on the result is Mahalanobis distance on ``x``.

    Args:
        x: Covariates, one row per unit.

    Returns:
        The whitened covariates.

    Raises:
        ValueError: If covariates are nonfinite, have invalid dimensions, or
            their sample covariance is singular.
    """
    x = _covariates(x)
    centered = x - x.mean(axis=0)
    try:
        chol = np.linalg.cholesky(np.atleast_2d(np.cov(centered, rowvar=False)))
    except np.linalg.LinAlgError as err:
        raise ValueError("covariates are collinear; drop or combine columns") from err
    return np.linalg.solve(chol, centered.T).T


def crd(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Complete randomization into ``k`` equal groups.

    Args:
        n: Number of units, a multiple of ``k``.
        k: Number of groups.
        rng: Random number generator.

    Returns:
        Group labels.

    Raises:
        ValueError: If sizes are not positive integers, ``n`` is not a
            multiple of ``k``, or ``k < 2``.
    """
    _positive_int(n, "n")
    _positive_int(k, "k")
    if k < 2 or n % k:
        raise ValueError("n must be a multiple of k >= 2")
    return rng.permutation(np.repeat(np.arange(k), n // k))


def randomized_aba(
    x: np.ndarray, k: int, rng: np.random.Generator, *, jitter: float = 1.0
) -> np.ndarray:
    """Randomized assignment-based anticlustering into ``k`` equal groups.

    Units are whitened and sorted by squared distance from the center, after
    multiplying each distance by ``exp(jitter * N(0, 1))``. The sorted list is
    cut into shells of ``k`` consecutive units, and each shell is spread one
    unit per group by the assignment that maximizes the total squared distance
    from units to their group's current centroid, as in assignment-based
    anticlustering (Baumann et al. 2026). The group labels are then shuffled,
    so every unit has probability ``1 / k`` of each group.

    Args:
        x: Covariates, one row per unit.
        k: Number of groups.
        rng: Random number generator.
        jitter: Log-scale noise on the sort key; 0 gives a single partition,
            randomized only by the label shuffle.

    Returns:
        Group labels.

    Raises:
        ValueError: If ``jitter`` is negative or nonfinite.
    """
    x = _check(x, k)
    if not np.isfinite(jitter) or jitter < 0:
        raise ValueError("jitter must be finite and nonnegative")
    w = whiten(x)
    n, p = w.shape
    key = (w**2).sum(axis=1) * np.exp(jitter * rng.normal(size=n))
    order = np.argsort(-key, kind="stable")
    z = np.empty(n, dtype=int)
    sums = np.zeros((k, p))
    for shell, start in enumerate(range(0, n, k)):
        batch = order[start : start + k]
        centroids = sums / max(shell, 1)
        cost = ((w[batch][:, None, :] - centroids[None]) ** 2).sum(axis=-1)
        rows, cols = linear_sum_assignment(cost, maximize=True)
        z[batch[rows]] = cols
        np.add.at(sums, cols, w[batch[rows]])
    return rng.permutation(k)[z]


def sequential_car(
    x: np.ndarray, k: int, rng: np.random.Generator, *, k0: float = 3.0
) -> np.ndarray:
    """Covariate-adaptive randomization over ``k`` arms (Zhang 2023, section 4).

    Units arrive in a random order. Each goes to arm ``t`` with probability
    proportional to ``1 - Phi(clip(d_t, -k0, k0))``, where ``d_t`` is the
    imbalance after assigning the unit to ``t`` minus the average of that
    imbalance over arms (Zhang's allocation function 4.3). The imbalance is
    ``sum_t |S_t - S / k|^2``, with ``S_t`` the sum of the features over arm
    ``t`` and ``S`` over all units so far; the features are a constant and the
    whitened ``x``, so arm sizes are balanced too, though not forced equal.
    The centered difference ``d_t`` reduces to ``2 <S_t - S / k, phi>`` for the
    arriving unit's features ``phi``.

    Args:
        x: Covariates, one row per unit.
        k: Number of arms.
        rng: Random number generator.
        k0: Clipping constant of the allocation function.

    Returns:
        Arm labels; sizes may differ slightly.

    Raises:
        ValueError: If ``k0`` is negative or nonfinite.
    """
    x = _check(x, k)
    if not np.isfinite(k0) or k0 < 0:
        raise ValueError("k0 must be finite and nonnegative")
    phi = np.column_stack([np.ones(len(x)), whiten(x)])
    lam = np.zeros((k, phi.shape[1]))
    z = np.empty(len(x), dtype=int)
    for i in rng.permutation(len(x)):
        d = 2.0 * lam @ phi[i]
        prob = norm.sf(np.clip(d, -k0, k0))
        t = rng.choice(k, p=prob / prob.sum())
        lam -= phi[i] / k
        lam[t] += phi[i]
        z[i] = t
    return z


def exchange_anticlust(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    init: np.ndarray | None = None,
    local_max: bool = False,
) -> np.ndarray:
    """The exchange method on the k-means anticlustering objective.

    Starting from a random equal split (or ``init``), each unit in turn is
    swapped with the unit in another group whose swap most reduces the spread
    of the group centroids, if any swap reduces it. One pass is the procedure
    Papenberg and Angelike (2026) use, after R ``anticlust``'s ``exchange``
    method with ``objective = "variance"``. With ``local_max`` the passes
    repeat until one makes no swap, as in ``anticlust``'s ``local-maximum``
    method; for two groups this is greedy pair-switching (Krieger, Azriel and
    Kapelner 2019) with the unweighted Euclidean objective.

    Args:
        x: Covariates, one row per unit, used as given (not standardized).
        k: Number of groups.
        rng: Random number generator, used for the initial split.
        init: Optional integer labels, with equal counts of 0 through ``k - 1``.
        local_max: Repeat passes until a local optimum.

    Returns:
        Group labels.

    Raises:
        ValueError: If initial labels do not form ``k`` equal groups.
    """
    x = _check(x, k)
    n = len(x)
    if init is not None:
        init = np.asarray(init)
        if (
            init.shape != (n,)
            or not np.issubdtype(init.dtype, np.integer)
            or np.any((init < 0) | (init >= k))
            or not np.all(np.bincount(init, minlength=k) == n // k)
        ):
            raise ValueError("init must contain integer labels for k equal groups")
    z = crd(n, k, rng) if init is None else np.asarray(init, dtype=int).copy()
    sums = np.array([x[z == g].sum(axis=0) for g in range(k)])
    sizes = np.bincount(z, minlength=k).astype(float)
    while _exchange_pass(x, z, sums, sizes) and local_max:
        pass
    return z


def _exchange_pass(
    x: np.ndarray, z: np.ndarray, sums: np.ndarray, sizes: np.ndarray
) -> bool:
    swapped = False
    for i in range(len(x)):
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
            swapped = True
    return swapped


def greedy_switch(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    features: np.ndarray | None = None,
) -> np.ndarray:
    """Two equal arms by greedy pair-switching from a random start.

    A random equal split is improved by swapping units between the arms until
    no swap reduces the distance between the arms' means (Krieger, Azriel and
    Kapelner 2019). Distances are Mahalanobis on ``features`` (default
    ``x``); pass the covariates with their squares to balance spread as well.
    Each call starts from a fresh random split, so repeated calls draw from
    the design, which is what a randomization test needs.

    Args:
        x: Covariates, one row per unit.
        rng: Random number generator.
        features: Optional matrix to balance instead of ``x``.

    Returns:
        Treatment indicators (0 or 1), half of each, with labels shuffled.

    Raises:
        ValueError: If ``features`` does not have one row per unit in ``x``.
    """
    x = _check(x, 2)
    basis = x if features is None else _covariates(features)
    if len(basis) != len(x):
        raise ValueError("features must have one row per unit in x")
    z = exchange_anticlust(whiten(basis), 2, rng, local_max=True)
    return z if rng.random() < 0.5 else 1 - z


def blocked(
    x: np.ndarray, k: int, rng: np.random.Generator, *, index: np.ndarray | None = None
) -> np.ndarray:
    """Blocked randomization: sort on an index, randomize within blocks of ``k``.

    Units are sorted on ``index`` (default the first principal component of the
    standardized covariates), cut into consecutive blocks of ``k``, and each
    block's units are randomly permuted across the ``k`` groups: matched
    ``k``-tuples on a one-dimensional index.

    Args:
        x: Covariates, one row per unit.
        k: Number of groups.
        rng: Random number generator.
        index: Optional sorting index, one value per unit.

    Returns:
        Group labels.

    Raises:
        ValueError: If the index is not a finite vector with one value per
            unit, or a covariate is constant when computing the default index.
    """
    x = _check(x, k)
    if index is None:
        if np.any(x.std(axis=0) == 0):
            raise ValueError("drop constant covariates or supply an index")
        z = (x - x.mean(axis=0)) / x.std(axis=0)
        key = np.linalg.svd(z, full_matrices=False)[2][0] @ z.T
    else:
        key = np.asarray(index, dtype=float)
        if key.shape != (len(x),) or not np.isfinite(key).all():
            raise ValueError("index must be a finite vector with one value per unit")
    order = np.argsort(key, kind="stable")
    labels = np.empty(len(x), dtype=int)
    for start in range(0, len(x), k):
        labels[order[start : start + k]] = rng.permutation(k)
    return labels


def rerandomize(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    accept: float = 0.01,
    batch: int = 64,
    max_draws: int = 100_000,
) -> np.ndarray:
    """Rerandomization for ``k`` equal groups on a multi-group Mahalanobis criterion.

    Complete randomizations are drawn until
    ``B = sum_g m |wbar_g - wbar|^2`` on whitened covariates, which is
    approximately chi-squared with ``(k - 1) p`` degrees of freedom under
    complete randomization, falls below that distribution's ``accept`` quantile.

    Args:
        x: Covariates, one row per unit.
        k: Number of groups.
        rng: Random number generator.
        accept: Target acceptance probability.
        batch: Candidate splits drawn at a time.
        max_draws: Maximum candidate splits before raising ``RuntimeError``.

    Returns:
        Group labels.

    Raises:
        ValueError: If ``accept`` is outside (0, 1), or ``batch`` or
            ``max_draws`` is not a positive integer.
        RuntimeError: If no acceptable assignment is found within ``max_draws``.
    """
    x = _check(x, k)
    if not 0 < accept < 1:
        raise ValueError("accept must be between 0 and 1, exclusive")
    _positive_int(batch, "batch")
    _positive_int(max_draws, "max_draws")
    w = whiten(x)
    n, p = w.shape
    m = n // k
    threshold = chi2.ppf(accept, (k - 1) * p)
    labels = np.repeat(np.arange(k), m)
    for start in range(0, max_draws, batch):
        count = min(batch, max_draws - start)
        perms = rng.permuted(np.tile(np.arange(n), (count, 1)), axis=1)
        sums = w[perms].reshape(count, k, m, p).sum(axis=2)
        hit = np.flatnonzero((sums**2).sum(axis=(1, 2)) / m <= threshold)
        if hit.size:
            z = np.empty(n, dtype=int)
            z[perms[hit[0]]] = labels
            return z
    raise RuntimeError("no acceptable assignment found; increase accept or max_draws")
