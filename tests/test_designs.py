import json
from pathlib import Path

import numpy as np
import pytest

from shelldraft import designs

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
        lambda x, r: designs.shell_draft(x, 3, r),
        lambda x, r: designs.shell_draft(x, 3, r, cost="leverage"),
        lambda x, r: designs.exchange_anticlust(x, 3, r),
    ],
)
def test_equal_group_sizes(x, draw):
    z = draw(x, np.random.default_rng(1))
    np.testing.assert_array_equal(np.bincount(z), [20, 20, 20])


def test_rerandomize_sizes_and_threshold(x):
    z = designs.rerandomize(x, np.random.default_rng(1), accept=0.01)
    assert np.bincount(z).tolist() == [30, 30]
    d = x[z == 1].mean(0) - x[z == 0].mean(0)
    m = d @ np.linalg.inv(np.cov(x, rowvar=False)) @ d * len(x) / 4
    assert m <= 0.2971  # chi2(3) 1% quantile


def test_whiten_gives_identity_covariance(x):
    w = designs.whiten(x @ np.array([[1, 0.5, 0], [0, 1, 0.3], [0, 0, 2.0]]))
    np.testing.assert_allclose(np.cov(w, rowvar=False), np.eye(3), atol=1e-12)


def test_shell_draft_without_jitter_is_one_partition_up_to_labels(x):
    a = designs.shell_draft(x, 2, np.random.default_rng(1), jitter=0)
    b = designs.shell_draft(x, 2, np.random.default_rng(2), jitter=0)
    assert np.array_equal(a, b) or np.array_equal(a, 1 - b)


def test_shell_draft_balances_better_than_crd(x):
    def gap(z):
        d = x[z == 1].mean(0) - x[z == 0].mean(0)
        return d @ np.linalg.inv(np.cov(x, rowvar=False)) @ d

    rng = np.random.default_rng(3)
    shell = np.mean([gap(designs.shell_draft(x, 2, rng)) for _ in range(50)])
    crd = np.mean([gap(designs.crd(len(x), 2, rng)) for _ in range(50)])
    assert shell < crd / 20


def test_shell_draft_inclusion_is_uniform():
    x = np.random.default_rng(4).normal(size=(12, 2))
    rng = np.random.default_rng(5)
    share = np.mean([designs.shell_draft(x, 2, rng) for _ in range(4000)], axis=0)
    np.testing.assert_allclose(share, 0.5, atol=0.035)  # 4.4 SE


def test_bad_sizes_raise():
    with pytest.raises(ValueError, match="multiple"):
        designs.shell_draft(np.zeros((7, 1)), 2, np.random.default_rng(0))
