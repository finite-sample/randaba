import numpy as np

from shelldraft import inference


def test_ols_coef_matches_lstsq_for_many_splits():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    y = rng.normal(size=40)
    zs = np.array([rng.permutation(np.repeat([0, 1], 20)) for _ in range(5)])
    fast = inference.ols_coef(y, zs, x)
    slow = [
        np.linalg.lstsq(np.column_stack([np.ones(40), z, x]), y, rcond=None)[0][1]
        for z in zs
    ]
    np.testing.assert_allclose(fast, slow, rtol=1e-10)


def test_diff_means_vectorized():
    y = np.arange(4.0)
    z = np.array([[1, 1, 0, 0], [0, 1, 0, 1]])
    np.testing.assert_allclose(inference.diff_means(y, z), [-2.0, 1.0])


def test_ri_pvalue_counts_observed_and_ties():
    assert inference.ri_pvalue(1.0, np.array([0.5, -1.0, 2.0, 0.1])) == 3 / 5
    assert inference.ri_pvalue(10.0, np.zeros(99)) == 1 / 100


def test_ols_test_matches_statsmodels_formula():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 2))
    z = np.repeat([0, 1], 15)
    y = x @ [1.0, -1.0] + 0.5 * z + rng.normal(size=30)
    p = inference.ols_test(y, z, x)
    assert 0 < p < 1
    assert inference.t_test(y, z) != p
