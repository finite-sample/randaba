from itertools import combinations

import numpy as np
import pytest
from scipy import stats

from shelldraft import grouped


def _all_assignments(groups):
    k = groups.max() + 1
    for treated in combinations(range(k), k // 2):
        s = -np.ones(k)
        s[list(treated)] = 1
        yield s[groups]


@pytest.fixture
def setup():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(24, 2))
    groups = grouped.partition(x, 6, rng)
    return x, groups


def test_partition_equal_groups_and_even_k(setup):
    _, groups = setup
    assert np.bincount(groups).tolist() == [4] * 6
    with pytest.raises(ValueError, match="even"):
        grouped.partition(np.arange(9.0)[:, None], 3, np.random.default_rng(0))


def test_assign_is_half_and_unbiased(setup):
    _, groups = setup
    rng = np.random.default_rng(0)
    draws = np.array([grouped.assign(groups, rng) for _ in range(4000)])
    assert (draws.sum(axis=1) == 12).all()
    np.testing.assert_allclose(draws.mean(axis=0), 0.5, atol=0.04)


def test_covariance_matches_enumeration(setup):
    _, groups = setup
    w = np.array(list(_all_assignments(groups)))
    np.testing.assert_allclose(grouped.covariance(groups), w.T @ w / len(w))
    sigma = grouped.covariance(groups)
    assert np.isclose((sigma**2).sum(), 24**2 / 5)
    assert np.isclose(np.linalg.eigvalsh(sigma).max(), 24 / 5)


def test_expected_imbalance_matches_enumeration(setup):
    x, groups = setup
    xc = x - x.mean(axis=0)
    proj = xc @ np.linalg.solve(xc.T @ xc, xc.T)
    q = [w @ proj @ w for w in _all_assignments(groups)]
    assert np.isclose(grouped.expected_imbalance(x, groups), np.mean(q))


def test_ols_error_eigenvalues_match_brute_force(setup):
    x, groups = setup
    n = len(x)
    design = np.column_stack([np.ones(n), x])
    resid = np.eye(n) - design @ np.linalg.pinv(design)
    acc = np.zeros((n, n))
    ws = list(_all_assignments(groups))
    for w in ws:
        acc += np.outer(w, w) / (w @ w - w @ (np.eye(n) - resid) @ w) ** 2
    m = 4 * resid @ (acc / len(ws)) @ resid
    brute = np.sort(np.linalg.eigvalsh(m))[::-1]
    closed = grouped.ols_error_eigenvalues(x, groups)
    np.testing.assert_allclose(closed, brute[: len(closed)], rtol=1e-9)
    assert brute[len(closed)] < 1e-12


def test_exact_balance_gives_k_minus_1_equal_eigenvalues():
    values = np.array([[0.0, 1.0], [1.0, -2.0], [3.0, 0.5], [-1.0, 4.0]])
    k = 6
    x = np.tile(values, (k, 1))
    groups = np.repeat(np.arange(k), len(values))
    assert np.isclose(grouped.expected_imbalance(x, groups), 0)
    lam = grouped.ols_error_eigenvalues(x, groups)
    np.testing.assert_allclose(lam, 4 / (len(x) * (k - 1)))
    assert len(lam) == k - 1


def test_crd_eigenvalues_trace_is_prop2():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(10, 2))
    xc = x - x.mean(axis=0)
    proj = xc @ np.linalg.solve(xc.T @ xc, xc.T)
    exact = []
    for treated in combinations(range(10), 5):
        w = -np.ones(10)
        w[list(treated)] = 1
        exact.append(1 / (10 - w @ proj @ w))
    lam = grouped.crd_ols_error_eigenvalues(x, rng, draws=200000)
    assert np.isclose(lam.sum(), 4 * np.mean(exact), rtol=0.01)


def test_relative_mse_matches_enumeration(setup):
    x, groups = setup
    g = x[:, 0] ** 2
    gc = g - g.mean()
    grouped_ms = np.mean([(w @ gc) ** 2 for w in _all_assignments(groups)])
    n = len(g)
    rng = np.random.default_rng(5)
    base = np.repeat([1.0, -1.0], n // 2)
    crd_ms = np.mean([(rng.permutation(base) @ gc) ** 2 for _ in range(200000)])
    assert np.isclose(grouped.relative_mse(g, groups), grouped_ms / crd_ms, rtol=0.02)
    assert np.isclose(
        grouped_ms / (n / (n - 1) * gc @ gc),
        grouped.relative_mse(g, groups),
        rtol=1e-9,
    )


def test_relative_mse_worst_case_is_constant_within_groups(setup):
    _, groups = setup
    g = groups.astype(float) ** 2
    k = groups.max() + 1
    assert np.isclose(grouped.relative_mse(g, groups), (len(g) - 1) / (k - 1))


def test_local_optimum_bound_holds(setup):
    x, groups = setup
    bound = grouped.local_optimum_bound(x, groups)
    assert grouped.expected_imbalance(x, groups) <= bound


def test_mse_quantile_single_eigenvalue_is_chi2():
    rng = np.random.default_rng(2)
    got = grouped.mse_quantile(np.array([1.0]), 0.95, rng)
    assert np.isclose(got, stats.chi2.ppf(0.95, 1), rtol=0.02)
