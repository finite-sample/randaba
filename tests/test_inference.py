import numpy as np
import pytest

from randaba import designs, inference


def test_contrast_matches_group_means_for_many_draws():
    rng = np.random.default_rng(0)
    y = rng.normal(size=30)
    z = np.array([designs.crd(30, 3, rng) for _ in range(5)])
    got = inference.contrast(y, z, 2, 0)
    want = [y[d == 2].mean() - y[d == 0].mean() for d in z]
    np.testing.assert_allclose(got, want)
    assert np.isclose(inference.contrast(y, z[0], 2, 0), want[0])


def test_ri_pvalue_counts_observed_and_ties():
    assert inference.ri_pvalue(1.0, np.array([0.5, -1.0, 2.0])) == 0.75
    assert inference.ri_pvalue(3.0, np.array([0.5, -1.0, 2.0])) == 0.25


def test_ri_test_has_size_under_sharp_null():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 3))
    y = x.sum(axis=1) + rng.normal(size=40)
    ref = np.array([designs.randomized_aba(x, 4, rng) for _ in range(199)])
    stats = inference.contrast(y, ref, 0, 1)
    pvals = [inference.ri_pvalue(s, np.delete(stats, i)) for i, s in enumerate(stats)]
    # Exchangeable draws: p-values are uniform on {1/199, ..., 1}.
    assert abs(np.mean(np.array(pvals) <= 0.05) - 0.05) < 0.02


@pytest.mark.parametrize(
    ("y", "z"),
    [
        ([1, 2], [0]),
        ([[1, 2]], [0, 1]),
        ([1, np.nan], [0, 1]),
        ([1, 2], [0, 0]),
        ([1, 2], [[0, 1], [1, 1]]),
        ([1, 2], 0),
    ],
)
def test_contrast_rejects_invalid_outcomes_or_assignments(y, z):
    with pytest.raises(ValueError, match=r"outcome|arms"):
        inference.contrast(y, z, 0, 1)


@pytest.mark.parametrize(
    ("observed", "reference"),
    [(np.nan, [1]), (np.inf, [1]), ([1], [1]), (1, []), (1, [[1, 2]]), (1, [np.nan])],
)
def test_ri_pvalue_rejects_invalid_statistics(observed, reference):
    with pytest.raises(ValueError, match=r"stat_obs|stat_ref"):
        inference.ri_pvalue(observed, reference)


def test_ri_pvalue_accepts_list_and_counts_near_ties():
    assert inference.ri_pvalue(1.0, [1 - 1e-12, -1.0, 0.0]) == 0.75
