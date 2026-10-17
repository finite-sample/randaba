"""Grouped randomization G(K): anticluster into K groups, then randomize groups.

Everything here conditions on the partition. Writing ``W = G s`` with ``G`` the
``N x K`` membership matrix and ``s`` a complete randomization of the ``K``
group labels, every design quantity reduces to a ``K x K`` computation.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

from shelldraft.designs import exchange_anticlust, whiten

ENUMERATE_MAX = 20


def partition(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Anticluster units into ``k`` equal groups.

    Runs the exchange method to a local optimum on the whitened covariates,
    so balance is measured in Mahalanobis distance.

    Args:
        x: Covariates, one row per unit.
        k: Number of groups, even and dividing the number of units.
        rng: Random number generator, used for the initial split.

    Returns:
        Group labels (0 to ``k - 1``).

    Raises:
        ValueError: If ``k`` is odd.
    """
    if k % 2:
        raise ValueError("k must be even so that half the groups are treated")
    return exchange_anticlust(whiten(x), k, rng, local_max=True)


def assign(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Treat a random half of the groups.

    Args:
        groups: Group labels from ``partition``.
        rng: Random number generator.

    Returns:
        Treatment indicators (0 or 1).
    """
    k = int(groups.max()) + 1
    treated = rng.permutation(np.repeat([0, 1], k // 2))
    return treated[groups]


def _membership(groups: np.ndarray) -> np.ndarray:
    k = int(groups.max()) + 1
    return (groups[:, None] == np.arange(k)[None]).astype(float)


def _label_draws(k: int, rng: np.random.Generator | None, draws: int) -> np.ndarray:
    if k <= ENUMERATE_MAX:
        out = -np.ones((comb(k, k // 2), k))
        for row, treated in enumerate(combinations(range(k), k // 2)):
            out[row, list(treated)] = 1.0
        return out
    if rng is None:
        raise ValueError(f"k > {ENUMERATE_MAX} needs an rng for Monte Carlo")
    base = np.repeat([1.0, -1.0], k // 2)
    return rng.permuted(np.tile(base, (draws, 1)), axis=1)


def covariance(groups: np.ndarray) -> np.ndarray:
    """Assignment covariance ``Sigma = E[W W']`` in closed form.

    Args:
        groups: Group labels.

    Returns:
        The ``N x N`` matrix ``(K / (K - 1)) G (I - 11'/K) G'``.
    """
    g = _membership(groups)
    k = g.shape[1]
    return k / (k - 1) * g @ (np.eye(k) - 1.0 / k) @ g.T


def imbalance_matrix(x: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, int]:
    """The ``K x K`` matrix ``B`` with ``Q = W' P W = m^2 s' B s``.

    Args:
        x: Covariates, one row per unit.
        groups: Group labels.

    Returns:
        ``B = D A^{-1} D'`` (``D`` the group-mean deviations, ``A = X_c'X_c``)
        and the group size ``m``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    xc = x - x.mean(axis=0)
    g = _membership(groups)
    m = int(g[:, 0].sum())
    d = g.T @ xc / m
    b = d @ np.linalg.solve(xc.T @ xc, d.T)
    return b, m


def expected_imbalance(x: np.ndarray, groups: np.ndarray) -> float:
    """``E[Q | partition] = (K m^2 / (K - 1)) trace(A^{-1} D'D)``.

    Under complete randomization the same quantity is ``p N / (N - 1)``.

    Args:
        x: Covariates, one row per unit.
        groups: Group labels.

    Returns:
        The expected treatment-covariate collinearity ``E[Q]``.
    """
    b, m = imbalance_matrix(x, groups)
    k = len(b)
    return float(k * m**2 / (k - 1) * np.trace(b))


def ols_error_eigenvalues(
    x: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator | None = None,
    draws: int = 20000,
) -> np.ndarray:
    """Eigenvalues of the OLS error matrix ``M`` under G(K).

    For an outcome ``Y = alpha + tau T + X beta + z`` and OLS of ``Y`` on
    ``(1, T, X)``, the error is ``2 W'(I - P) z / (N - Q)``, so its mean square
    over the design is ``z' M z`` with
    ``M = 4 (I - P) E[W W' / (N - Q)^2] (I - P)``. With ``z`` independent with
    variance ``sigma^2``, the MSE over ``z`` is distributed as
    ``sigma^2 sum_i lambda_i chi2_1``. Under G(K) the nonzero eigenvalues are
    those of the ``K x K`` matrix ``4 H^{1/2} (m I - m^2 B) H^{1/2}`` with
    ``H = E[s s' / (N - m^2 s' B s)^2]``, computed exactly by enumerating the
    label assignments when ``K <= 20`` and by Monte Carlo otherwise.

    Args:
        x: Covariates, one row per unit.
        groups: Group labels.
        rng: Random number generator, needed only when ``K > 20``.
        draws: Monte Carlo draws when ``K > 20``.

    Returns:
        The nonzero eigenvalues, largest first.
    """
    b, m = imbalance_matrix(x, groups)
    k = len(b)
    n = k * m
    s = _label_draws(k, rng, draws)
    q = m**2 * np.einsum("ij,jk,ik->i", s, b, s)
    h = (s / (n - q)[:, None]).T @ (s / (n - q)[:, None]) / len(s)
    hval, hvec = np.linalg.eigh(h)
    hroot = hvec * np.sqrt(np.clip(hval, 0, None)) @ hvec.T
    core = 4 * hroot @ (m * np.eye(k) - m**2 * b) @ hroot
    lam = np.linalg.eigvalsh((core + core.T) / 2)[::-1]
    return lam[lam > 1e-12 * lam[0]]


def crd_ols_error_eigenvalues(
    x: np.ndarray, rng: np.random.Generator, draws: int = 20000
) -> np.ndarray:
    """Eigenvalues of the OLS error matrix under complete randomization.

    Same ``M`` as ``ols_error_eigenvalues``, estimated from ``draws`` complete
    randomizations.

    Args:
        x: Covariates, one row per unit.
        rng: Random number generator.
        draws: Number of complete randomizations.

    Returns:
        The nonzero eigenvalues, largest first.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = len(x)
    xc = x - x.mean(axis=0)
    basis = np.linalg.qr(xc)[0]
    w = rng.permuted(np.tile(np.repeat([1.0, -1.0], n // 2), (draws, 1)), axis=1)
    proj = w @ basis
    q = (proj**2).sum(axis=1)
    resid = (w - proj @ basis.T) / (n - q)[:, None]
    sing = np.linalg.svd(resid, compute_uv=False)
    lam = 4 * sing**2 / draws
    return lam[lam > 1e-12 * lam[0]]


def mse_quantile(
    eigenvalues: np.ndarray,
    q: float,
    rng: np.random.Generator,
    draws: int = 200000,
) -> float:
    """Quantile over ``z`` of the MSE ``sum_i lambda_i chi2_1`` (``sigma = 1``).

    Args:
        eigenvalues: Eigenvalues from ``ols_error_eigenvalues``.
        q: Quantile level, for example 0.95.
        rng: Random number generator.
        draws: Monte Carlo draws of the chi-squared variables.

    Returns:
        The ``q`` quantile.
    """
    lam = np.asarray(eigenvalues)
    total = np.zeros(draws)
    for chunk in np.array_split(lam, max(1, len(lam) // 64)):
        total += rng.chisquare(1, size=(draws, len(chunk))) @ chunk
    return float(np.quantile(total, q))
