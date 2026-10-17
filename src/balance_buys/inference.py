"""Two-group tests of no treatment effect: classical and randomization-based."""

from __future__ import annotations

import numpy as np
from scipy import stats

_TIE_RTOL = 1e-10


def diff_means(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Difference in means, treated (1) minus control (0), for one or many splits.

    Args:
        y: Outcomes.
        z: One split (n,) or many (m, n), with labels 0 and 1.

    Returns:
        The statistic for each split.
    """
    z = np.asarray(z, dtype=float)
    n1 = z.sum(axis=-1)
    return (z @ y) / n1 - ((1 - z) @ y) / (z.shape[-1] - n1)


def ols_coef(y: np.ndarray, z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS coefficient on treatment, adjusting for an intercept and ``x``.

    Uses the Frisch-Waugh-Lovell form, so many splits cost one projection.

    Args:
        y: Outcomes.
        z: One split (n,) or many (m, n), with labels 0 and 1.
        x: Covariates, one row per unit.

    Returns:
        The coefficient for each split.
    """
    xc = np.column_stack([np.ones(len(y)), x])
    q, _ = np.linalg.qr(xc)
    y_res = y - q @ (q.T @ y)
    z = np.asarray(z, dtype=float)
    z_res = z - (z @ q) @ q.T
    return (z_res @ y_res) / (z_res**2).sum(axis=-1)


def t_test(y: np.ndarray, z: np.ndarray) -> float:
    """Two-sided p-value of the Student two-sample t-test.

    Args:
        y: Outcomes.
        z: Split with labels 0 and 1.

    Returns:
        The p-value.
    """
    result = stats.ttest_ind(y[z == 1], y[z == 0])
    return float(result.pvalue)  # pyright: ignore[reportAttributeAccessIssue]


def ols_test(y: np.ndarray, z: np.ndarray, x: np.ndarray) -> float:
    """Two-sided p-value for the treatment coefficient, classical OLS errors.

    Args:
        y: Outcomes.
        z: Split with labels 0 and 1.
        x: Covariates, one row per unit.

    Returns:
        The p-value.
    """
    design = np.column_stack([np.ones(len(y)), z, x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    df = len(y) - design.shape[1]
    sigma2 = resid @ resid / df
    se = np.sqrt(sigma2 * np.linalg.inv(design.T @ design)[1, 1])
    return float(2 * stats.t.sf(abs(coef[1] / se), df))


def ri_pvalue(stat_obs: float, stat_ref: np.ndarray) -> float:
    """Two-sided randomization p-value under the sharp null of no effect.

    ``stat_ref`` holds the statistic recomputed on the observed outcomes for
    fresh draws of the same design. The observed split counts as one draw.

    Args:
        stat_obs: Statistic for the observed split.
        stat_ref: Statistics for the reference splits.

    Returns:
        ``(1 + #{|ref| >= |obs|}) / (1 + m)``, counting ties within a relative
        tolerance.
    """
    tol = _TIE_RTOL * max(1.0, abs(stat_obs))
    hits = np.count_nonzero(np.abs(stat_ref) >= abs(stat_obs) - tol)
    return (1 + hits) / (1 + len(stat_ref))
