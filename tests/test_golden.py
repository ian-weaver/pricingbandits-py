"""Golden tests: Python components vs values exported from the validated R package.

The deterministic components (kernels, basis functions, aggregation, likelihood,
posterior) must match R to near machine precision (LAPACK-level differences
only). The optimizer and samplers are checked at looser, statistically
appropriate tolerances.
"""

import json
import os

import numpy as np
import pytest

import pricingbandits as pb

GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "golden.json")


@pytest.fixture(scope="module")
def gold():
    with open(GOLDEN) as fh:
        return json.load(fh)


def test_kernels(gold):
    g = gold["kernels"]
    xi, xj = np.array(g["x_i"]), np.array(g["x_j"])
    sf, l = g["sigma_f"], g["l"]
    np.testing.assert_allclose(pb.rbf_kernel(xi, xj, sf, l), g["k00"], rtol=1e-12)
    np.testing.assert_allclose(pb.rbf_kernel_01(xi, xj, sf, l), g["k01"], rtol=1e-12)
    np.testing.assert_allclose(pb.rbf_kernel_11(xi, xj, sf, l), g["k11"], rtol=1e-12)


def test_basis_functions(gold):
    g = gold["basis"]
    xs = np.array(g["x"])
    for key, J in (("J5", 5), ("J11", 11)):
        ref = np.array(g[key])
        got = np.vstack([pb.basis_function(x, J) for x in xs])
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-15)


def test_aggregation(gold):
    g = gold["aggregate"]
    prices, dec, tx = np.array(g["prices"]), np.array(g["decisions"]), np.array(g["test_x"])
    s, n = pb.aggregate_data_ucb(prices, dec, tx)
    np.testing.assert_allclose(s, g["ucb_s"], rtol=0)
    np.testing.assert_allclose(n, g["ucb_n"], rtol=0)
    s, n = pb.aggregate_data_ts(prices, dec, tx)
    np.testing.assert_allclose(s, g["ts_s"], rtol=0)
    np.testing.assert_allclose(n, g["ts_n"], rtol=0)
    gx, gy, gs = pb.aggregate_data_gp(prices, dec, tx, np.full(11, 0.25), 10)
    np.testing.assert_allclose(gx, g["gp_x"], rtol=0)
    np.testing.assert_allclose(gy, g["gp_y"], rtol=1e-15)
    np.testing.assert_allclose(gs, g["gp_sig"], rtol=1e-15)


def test_nlml_values(gold):
    g = gold["nlml"]
    tx, ty, sg = np.array(g["train_x"]), np.array(g["train_y"]), np.array(g["sigma2"])
    for theta, ref in zip(g["thetas"], g["values"]):
        assert pb.nlml(np.array(theta), tx, ty, sg) == pytest.approx(ref, rel=1e-10)


def test_optimal_hyperparameters(gold):
    g = gold["nlml"]
    tx, ty, sg = np.array(g["train_x"]), np.array(g["train_y"]), np.array(g["sigma2"])
    # different optimizer implementations; agreement to ~1e-4 is what matters
    got = pb.optimal_hyperparameters(tx, ty, sg)
    np.testing.assert_allclose(got, g["opt"], atol=2e-4)


def test_posterior_prediction(gold):
    g = gold["posterior"]
    a = gold["nlml"]
    tx, ty, sg = np.array(a["train_x"]), np.array(a["train_y"]), np.array(a["sigma2"])
    test_x = np.arange(1, 11) / 10

    m, c = pb.posterior_prediction(tx, ty, test_x, None, 0.7, 0.2, sg)
    np.testing.assert_allclose(m, g["mean_plain"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(c, np.array(g["cov_plain"]), rtol=0, atol=1e-10)

    m, c = pb.posterior_prediction(tx, ty, test_x, test_x, 0.7, 0.2, sg)
    np.testing.assert_allclose(m, g["mean_joint"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(c, np.array(g["cov_joint"]), rtol=0, atol=1e-10)

    m, c = pb.posterior_prediction(tx, ty, [0.0], np.concatenate([[0.0], test_x]), 0.7, 0.2, sg)
    np.testing.assert_allclose(m, g["mean_mono"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(c, np.array(g["cov_mono"]), rtol=0, atol=1e-10)


def test_make_pos_definite(gold):
    g = gold["posterior"]
    a = gold["nlml"]
    tx, ty, sg = np.array(a["train_x"]), np.array(a["train_y"]), np.array(a["sigma2"])
    test_x = np.arange(1, 11) / 10
    _, c = pb.posterior_prediction(tx, ty, test_x, None, 0.7, 0.2, sg)
    fixed = pb.make_pos_definite(c, 1e-10)
    np.testing.assert_allclose(fixed, np.array(g["cov_plain_fixed"]), rtol=0, atol=1e-8)
    _, cj = pb.posterior_prediction(tx, ty, test_x, test_x, 0.7, 0.2, sg)
    fixed_j = pb.make_pos_definite(cj, 1e-10)
    np.testing.assert_allclose(fixed_j, np.array(g["cov_joint_fixed"]), rtol=0, atol=1e-8)


def test_ucb_scores(gold):
    g = gold["aggregate"]
    prices, dec, tx = np.array(g["prices"]), np.array(g["decisions"]), np.array(g["test_x"])
    scores = pb.ucb(prices, dec, tx)
    np.testing.assert_allclose(scores, gold["ucb_scores"]["scores"], rtol=1e-12)


@pytest.mark.parametrize("key", ["beta_2_9", "beta_2_2", "beta_9_2"])
def test_ucb_full_trajectory(gold, key):
    """UCB is deterministic given valuations: entire 2500-consumer trajectories
    must match the R package exactly."""
    g = gold["ucb_traj"][key]
    v = np.array(g["valuations"])
    p, d = pb.mab_experiment(v, "UCB", np.arange(1, 11) / 10, 2500, 10,
                             rng=np.random.default_rng(0))
    np.testing.assert_array_equal(p, np.array(g["prices"]))
    np.testing.assert_array_equal(d, np.array(g["decisions"]))


def test_tmvn_moments(gold):
    """Truncated-MVN sampler: per-coordinate means/sds/quantiles must agree with
    50k R draws within Monte-Carlo error."""
    t = gold["tmvn"]
    mu, cov = np.array(t["mu"]), np.array(t["cov"])
    lb = np.full(20, -np.inf)
    ub = np.concatenate([np.full(10, 2.0), np.zeros(10)])
    rng = np.random.default_rng(123)
    dr = pb.rtmvnorm(50_000, mu, cov, lb, ub, rng)
    ref_m, ref_s = np.array(t["ref_mean"]), np.array(t["ref_sd"])
    se = ref_s * np.sqrt(2.0 / 50_000)  # MC standard error of a mean difference
    assert np.all(np.abs(dr.mean(0) - ref_m) < 5 * se)
    assert np.all(np.abs(dr.std(0, ddof=1) - ref_s) / ref_s < 0.05)
    np.testing.assert_allclose(np.quantile(dr, 0.05, axis=0), t["ref_q05"], atol=6 * se.max())
    np.testing.assert_allclose(np.quantile(dr, 0.95, axis=0), t["ref_q95"], atol=6 * se.max())


def test_tmvn_moments_mono_case(gold):
    t = gold["tmvn_mono"]
    mu, cov = np.array(t["mu"]), np.array(t["cov"])
    d = len(mu)
    lb = np.full(d, -np.inf)
    ub = np.concatenate([[2.0], np.zeros(d - 1)])
    rng = np.random.default_rng(124)
    dr = pb.rtmvnorm(50_000, mu, cov, lb, ub, rng)
    ref_m, ref_s = np.array(t["ref_mean"]), np.array(t["ref_sd"])
    se = ref_s * np.sqrt(2.0 / 50_000)
    assert np.all(np.abs(dr.mean(0) - ref_m) < 5 * se)
    assert np.all(np.abs(dr.std(0, ddof=1) - ref_s) / ref_s < 0.05)


def test_tmvn_respects_bounds(gold):
    t = gold["tmvn"]
    mu, cov = np.array(t["mu"]), np.array(t["cov"])
    lb = np.full(20, -np.inf)
    ub = np.concatenate([np.full(10, 2.0), np.zeros(10)])
    rng = np.random.default_rng(5)
    dr = pb.rtmvnorm(2000, mu, cov, lb, ub, rng)
    assert np.all(dr <= ub + 1e-12)


def test_wrapper_validation():
    with pytest.raises(ValueError):
        pb.pricing_bandit([0.5, 0.6], prices=[0.5, 1.5], policy="UCB")
    with pytest.raises(ValueError):
        pb.pricing_bandit([0.5], prices=[0.5, 1.0], policy="NOPE")
    with pytest.raises(ValueError):
        pb.pricing_bandit([0.5], prices=[0.5, 1.0], policy="GP-TS-M", timeout=0)


def test_custom_timeout_runs():
    rng = np.random.default_rng(3)
    v = rng.beta(2, 2, 40)
    out = pb.pricing_bandit(v, np.arange(1, 11) / 10, policy="GP-TS-M",
                            timeout=2.0, rng=rng)
    assert len(out.prices_tested) == 40


class _PinnedRng(np.random.Generator):
    """Generator whose first permutation() puts a pinned price first (used to
    align the single random choice GP policies make - the first price)."""

    def __init__(self, pinned):
        super().__init__(np.random.PCG64(0))
        self._pinned = pinned
        self._used = False

    def permutation(self, x):
        if not self._used:
            self._used = True
            x = np.asarray(x, dtype=float)
            return np.concatenate([[self._pinned], x[x != self._pinned]])
        return super().permutation(x)


@pytest.mark.parametrize("key", ["beta_2_9", "beta_2_2", "beta_9_2"])
def test_gpucb_trajectory_pinned_first_price(key):
    """GP-UCB is deterministic after the (random) first price. With the first
    price pinned to R's draw and R's valuations, trajectories should agree
    almost everywhere (occasional argmax flips from optimizer/LAPACK noise are
    tolerated up to 0.5%)."""
    path = os.path.join(os.path.dirname(__file__), "golden", "gpucb_traj.json")
    with open(path) as fh:
        g = json.load(fh)[key]
    v = np.array(g["valuations"])
    rng = _PinnedRng(g["first_price"])
    p, _ = pb.mab_experiment(v, "GP-UCB", np.arange(1, 11) / 10, 2500, 10, rng=rng)
    agree = float((p == np.array(g["prices"])).mean())
    assert agree >= 0.995
