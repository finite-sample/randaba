import json
from pathlib import Path

import numpy as np
import pytest

from randaba import designs

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def x():
    return np.random.default_rng(0).normal(size=(60, 3))


@pytest.mark.parametrize(
    "case", json.loads((FIXTURES / "anticlust_exchange.json").read_text())["cases"]
)
def test_exchange_matches_r_anticlust(case):
    x = np.array(case["x"], dtype=float)
    out = designs.exchange_anticlust(
        x,
        case["k"],
        np.random.default_rng(0),
        init=np.array(case["init"]),
        local_max=case["method"] == "local-maximum",
    )
    np.testing.assert_array_equal(out, case["out"])


@pytest.mark.parametrize(
    "draw",
    [
        lambda x, r: designs.crd(len(x), 3, r),
        lambda x, r: designs.randomized_aba(x, 3, r),
        lambda x, r: designs.exchange_anticlust(x, 3, r),
    ],
)
def test_equal_group_sizes(x, draw):
    z = draw(x, np.random.default_rng(1))
    np.testing.assert_array_equal(np.bincount(z), [20, 20, 20])


def test_whiten_gives_identity_covariance(x):
    w = designs.whiten(x @ np.array([[1, 0.5, 0], [0, 1, 0.3], [0, 0, 2.0]]))
    np.testing.assert_allclose(np.cov(w, rowvar=False), np.eye(3), atol=1e-12)


def test_randomized_aba_without_jitter_is_one_partition_up_to_labels(x):
    a = designs.randomized_aba(x, 2, np.random.default_rng(1), jitter=0)
    b = designs.randomized_aba(x, 2, np.random.default_rng(2), jitter=0)
    assert np.array_equal(a, b) or np.array_equal(a, 1 - b)


def test_randomized_aba_balances_better_than_crd(x):
    def gap(z):
        d = x[z == 1].mean(0) - x[z == 0].mean(0)
        return d @ np.linalg.inv(np.cov(x, rowvar=False)) @ d

    rng = np.random.default_rng(3)
    shell = np.mean([gap(designs.randomized_aba(x, 2, rng)) for _ in range(50)])
    crd = np.mean([gap(designs.crd(len(x), 2, rng)) for _ in range(50)])
    assert shell < crd / 20


def test_randomized_aba_inclusion_is_uniform():
    x = np.random.default_rng(4).normal(size=(12, 2))
    rng = np.random.default_rng(5)
    share = np.mean([designs.randomized_aba(x, 2, rng) for _ in range(4000)], axis=0)
    np.testing.assert_allclose(share, 0.5, atol=0.035)  # 4.4 SE


def test_bad_sizes_raise():
    with pytest.raises(ValueError, match="multiple"):
        designs.randomized_aba(np.zeros((7, 1)), 2, np.random.default_rng(0))


def test_greedy_switch_balances_and_is_label_symmetric(x):
    rng = np.random.default_rng(4)
    draws = np.array([designs.greedy_switch(x, rng) for _ in range(400)])
    assert (draws.sum(axis=1) == 30).all()
    np.testing.assert_allclose(draws.mean(axis=0), 0.5, atol=0.1)
    gap = np.abs(x[draws[0] == 1].mean(0) - x[draws[0] == 0].mean(0)).max()
    crd_gap = np.abs(x[:30].mean(0) - x[30:].mean(0)).max()
    assert gap < crd_gap / 10
    with_sq = designs.greedy_switch(x, rng, features=np.column_stack([x, x**2]))
    assert with_sq.sum() == 30


def test_blocked_is_one_per_group_per_block(x):
    rng = np.random.default_rng(6)
    index = x[:, 0]
    z = designs.blocked(x, 3, rng, index=index)
    order = np.argsort(index)
    for start in range(0, 60, 3):
        assert sorted(z[order[start : start + 3]]) == [0, 1, 2]
    draws = np.array([designs.blocked(x, 3, rng) for _ in range(600)])
    np.testing.assert_allclose((draws == 0).mean(axis=0), 1 / 3, atol=0.08)


def test_rerandomize_meets_threshold(x):
    z = designs.rerandomize(x, 3, np.random.default_rng(7), accept=0.05)
    assert np.bincount(z).tolist() == [20, 20, 20]
    w = designs.whiten(x)
    stat = sum(((w[z == g].sum(0)) ** 2).sum() for g in range(3)) / 20
    assert stat <= 1.6354  # chi2(6) 5% quantile


@pytest.mark.parametrize("k", [3, 5])
def test_randomized_aba_balances_sort_key_within_range_over_m(k):
    # One unit per group per shell, shells consecutive in the sorted key, so
    # group means of the key differ by at most (max key - min key) / m.
    rng = np.random.default_rng(11)
    x = rng.standard_t(3, size=(20 * k, 4))
    key = (designs.whiten(x) ** 2).sum(axis=1)
    z = designs.randomized_aba(x, k, rng, jitter=0.0)
    means = np.array([key[z == g].mean() for g in range(k)])
    assert means.max() - means.min() <= (key.max() - key.min()) / 20 + 1e-12
    spreads = [
        np.ptp([key[c == g].mean() for g in range(k)])
        for c in (designs.crd(len(x), k, rng) for _ in range(300))
    ]
    assert max(spreads) > (key.max() - key.min()) / 20


def test_randomized_aba_labels_are_exchangeable(x):
    rng = np.random.default_rng(12)
    draws = np.array(
        [designs.randomized_aba(x, 4, rng, jitter=0.0) for _ in range(2000)]
    )
    np.testing.assert_allclose((draws == 2).mean(axis=0), 0.25, atol=0.05)


def test_sequential_car_score_is_centered_potential_imbalance():
    # Zhang's d_t: imbalance if the unit joins arm t, minus the mean over arms.
    rng = np.random.default_rng(13)
    k, q = 5, 4
    sums = rng.normal(size=(k, q))
    lam = sums - sums.sum(axis=0) / k
    phi = rng.normal(size=q)
    imb = []
    for t in range(k):
        new = sums.copy()
        new[t] += phi
        imb.append(((new - new.sum(axis=0) / k) ** 2).sum())
    np.testing.assert_allclose(np.array(imb) - np.mean(imb), 2 * lam @ phi)


def test_sequential_car_balances_and_is_label_symmetric(x):
    rng = np.random.default_rng(14)
    w = designs.whiten(x)

    def imbalance(z):
        return sum((w[z == g].sum(0) ** 2).sum() / (z == g).sum() for g in range(3))

    draws = [designs.sequential_car(x, 3, rng) for _ in range(300)]
    crd = [designs.crd(60, 3, rng) for _ in range(300)]
    assert np.mean([imbalance(z) for z in draws]) < 0.3 * np.mean(
        [imbalance(z) for z in crd]
    )
    sizes = np.array([np.bincount(z, minlength=3) for z in draws])
    assert np.abs(sizes - 20).max() <= 3
    np.testing.assert_allclose((np.array(draws) == 1).mean(axis=0), 1 / 3, atol=0.1)


def test_aba_shell_step_minimizes_armm_criterion_over_all_allocations():
    # With equal counts, the max-distance-to-centroid assignment of a shell is
    # the allocation, among all K!, with the smallest summed squared deviation
    # of the arm sums from their mean (ARMM's mean pairwise criterion).
    from itertools import permutations

    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(15)
    k, p, t = 4, 3, 5
    sums = rng.normal(size=(k, p)) * 3
    shell = rng.normal(size=(k, p))
    cost = ((shell[:, None, :] - (sums / t)[None]) ** 2).sum(axis=-1)
    rows, cols = linear_sum_assignment(cost, maximize=True)

    def criterion(arm_of_unit):
        new = sums.copy()
        new[list(arm_of_unit)] += shell
        return ((new - new.mean(axis=0)) ** 2).sum()

    best = min(permutations(range(k)), key=criterion)
    np.testing.assert_array_equal(cols[np.argsort(rows)], best)


@pytest.mark.parametrize(
    ("n", "k"), [(0, 2), (-4, 2), (4, 1), (5, 2), (4.0, 2), (4, True)]
)
def test_crd_rejects_invalid_sizes(n, k):
    with pytest.raises(ValueError, match=r"positive integer|multiple"):
        designs.crd(n, k, np.random.default_rng(0))


@pytest.mark.parametrize(
    "bad", [0, [], [[]], np.zeros((2, 2, 2)), [0, np.nan], [0, np.inf]]
)
def test_whiten_rejects_invalid_covariates(bad):
    with pytest.raises(ValueError, match=r"rows|finite"):
        designs.whiten(bad)


def test_whiten_univariate_and_singular():
    w = designs.whiten(np.arange(6))
    assert w.shape == (6, 1)
    np.testing.assert_allclose(w.mean(axis=0), 0, atol=1e-15)
    np.testing.assert_allclose(np.var(w, ddof=1), 1)
    with pytest.raises(ValueError, match="collinear"):
        designs.whiten(np.ones((6, 2)))


@pytest.mark.parametrize("value", [-1, np.inf, np.nan])
def test_randomization_parameters_must_be_nonnegative_and_finite(x, value):
    with pytest.raises(ValueError, match="jitter"):
        designs.randomized_aba(x, 3, np.random.default_rng(0), jitter=value)
    with pytest.raises(ValueError, match="k0"):
        designs.sequential_car(x, 3, np.random.default_rng(0), k0=value)


@pytest.mark.parametrize(
    "init", [[0, 1], [0.5] * 60, [0] * 60, [-1] * 60, [3] * 60, [[0, 1, 2]] * 20]
)
def test_exchange_rejects_invalid_initial_assignments(x, init):
    with pytest.raises(ValueError, match="init"):
        designs.exchange_anticlust(x, 3, np.random.default_rng(0), init=init)


def test_greedy_switch_rejects_mismatched_features(x):
    with pytest.raises(ValueError, match="features"):
        designs.greedy_switch(x, np.random.default_rng(0), features=x[:-2])


@pytest.mark.parametrize(
    "index", [np.zeros(59), np.zeros((60, 1)), np.full(60, np.nan)]
)
def test_blocked_rejects_invalid_index(x, index):
    with pytest.raises(ValueError, match="index"):
        designs.blocked(x, 3, np.random.default_rng(0), index=index)


def test_blocked_constant_covariates_require_index():
    x = np.ones((6, 1))
    with pytest.raises(ValueError, match="constant"):
        designs.blocked(x, 2, np.random.default_rng(0))
    z = designs.blocked(x, 2, np.random.default_rng(0), index=np.arange(6))
    np.testing.assert_array_equal(np.bincount(z), [3, 3])


@pytest.mark.parametrize("accept", [0, -0.1, 1, np.inf, np.nan])
def test_rerandomize_rejects_invalid_acceptance(x, accept):
    with pytest.raises(ValueError, match="accept"):
        designs.rerandomize(x, 3, np.random.default_rng(0), accept=accept)


@pytest.mark.parametrize("parameter", ["batch", "max_draws"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_rerandomize_rejects_invalid_draw_counts(x, parameter, value):
    with pytest.raises(ValueError, match=parameter):
        designs.rerandomize(x, 3, np.random.default_rng(0), **{parameter: value})


def test_rerandomize_stops_when_threshold_is_unattainable():
    # Each group contains one unit, so every allocation has the same imbalance.
    with pytest.raises(RuntimeError, match="no acceptable assignment"):
        designs.rerandomize(
            np.array([0.0, 1.0]), 2, np.random.default_rng(0), batch=3, max_draws=5
        )


@pytest.mark.parametrize(
    "name", ["randomized_aba", "blocked", "rerandomize", "sequential_car"]
)
def test_designs_reproduce_from_seed_and_preserve_covariates(x, name):
    original = x.copy()
    draw = getattr(designs, name)
    a = draw(x, 3, np.random.default_rng(42))
    b = draw(x, 3, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(x, original)
