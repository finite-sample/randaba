"""Randomization inference for contrasts between arms."""

from __future__ import annotations

import numpy as np

_TIE_RTOL = 1e-10


def contrast(y: np.ndarray, z: np.ndarray, a: int, b: int) -> np.ndarray:
    """Difference in means between arms ``a`` and ``b``, for one or many draws.

    Args:
        y: Outcomes.
        z: Arm labels, one draw (n,) or many (d, n).
        a: First arm.
        b: Second arm.

    Returns:
        Mean outcome in arm ``a`` minus mean in arm ``b``, per draw.

    Raises:
        ValueError: If dimensions do not match, outcomes are nonfinite, or
            either requested arm is absent from any draw.
    """
    y = np.asarray(y, dtype=float)
    z = np.asarray(z)
    if y.ndim != 1 or z.ndim not in (1, 2) or z.shape[-1] != len(y):
        raise ValueError("y must be a vector and z must have one label per outcome")
    if not np.isfinite(y).all():
        raise ValueError("outcomes must be finite")
    in_a, in_b = (z == a).astype(float), (z == b).astype(float)
    if np.any(in_a.sum(axis=-1) == 0) or np.any(in_b.sum(axis=-1) == 0):
        raise ValueError("both requested arms must be present in every draw")
    return (in_a @ y) / in_a.sum(axis=-1) - (in_b @ y) / in_b.sum(axis=-1)


def ri_pvalue(stat_obs: float, stat_ref: np.ndarray) -> float:
    """Two-sided randomization p-value under the sharp null of no effect.

    ``stat_ref`` holds the statistic recomputed on the observed outcomes for
    fresh draws of the same design. The observed draw counts as one draw.

    Args:
        stat_obs: Statistic for the observed draw.
        stat_ref: Statistics for the reference draws.

    Returns:
        ``(1 + #{|ref| >= |obs|}) / (1 + d)``, counting ties within a relative
        tolerance.

    Raises:
        ValueError: If the observed statistic is not a finite scalar, or the
            reference statistics are not a nonempty finite vector.
    """
    stat_ref = np.asarray(stat_ref, dtype=float)
    if np.ndim(stat_obs) != 0 or not np.isfinite(stat_obs):
        raise ValueError("stat_obs must be a finite scalar")
    if stat_ref.ndim != 1 or not stat_ref.size or not np.isfinite(stat_ref).all():
        raise ValueError("stat_ref must be a nonempty finite vector")
    tol = _TIE_RTOL * max(1.0, abs(stat_obs))
    hits = np.count_nonzero(np.abs(stat_ref) >= abs(stat_obs) - tol)
    return (1 + hits) / (1 + len(stat_ref))
