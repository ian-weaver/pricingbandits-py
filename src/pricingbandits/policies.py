"""Bandit policies: UCB, TS, GP variants, monotonic GP variants, and the policy
evaluation wrappers with their fallback chains.

Line-for-line port of the R package's ``policies.R`` (including its quirks; see
the package README). RNG is threaded through as a ``numpy.random.Generator``.
"""

from __future__ import annotations

import numpy as np

from .aggregate import aggregate_data_gp, aggregate_data_ts, aggregate_data_ucb
from .diagnostics import bump
from .gp import make_pos_definite, mvrnorm, nlml, optimal_hyperparameters, posterior_prediction
from .truncated_mvn import rtmvnorm

DEFAULT_TIMEOUT = 5.0  # seconds per truncated-sampling attempt (user-overridable)


def ucb(prices_tested, purchase_decisions, test_x, rng=None):
    """Upper Confidence Bound policy (deterministic given the data)."""
    test_x = np.asarray(test_x, dtype=float)
    s_t, n_t = aggregate_data_ucb(prices_tested, purchase_decisions, test_x)
    t = n_t.sum()
    v_t = (s_t * test_x**2) / n_t - (test_x * (s_t / n_t)) ** 2 + np.sqrt(2 * np.log(t) / n_t)
    scores = test_x * ((s_t / n_t) + np.sqrt((np.log(t) / n_t) * np.minimum(0.25, v_t)))
    return scores


def ts(prices_tested, purchase_decisions, test_x, rng):
    """Thompson Sampling with Beta posteriors per arm."""
    test_x = np.asarray(test_x, dtype=float)
    s_t, n_t = aggregate_data_ts(prices_tested, purchase_decisions, test_x)
    draws = rng.beta(s_t, n_t - s_t)
    return test_x * draws


def gpucb(prices_tested, purchase_decisions, test_x, sigma2_y, batch_size, rng=None):
    """Gaussian Process UCB policy."""
    test_x = np.asarray(test_x, dtype=float)
    train_x, train_y, sig = aggregate_data_gp(prices_tested, purchase_decisions, test_x,
                                              sigma2_y, batch_size)
    opt = optimal_hyperparameters(train_x, train_y, sig)
    t = len(prices_tested) + batch_size
    mean, cov = posterior_prediction(train_x, train_y, test_x, None, opt[0], opt[1], sig)
    cov = make_pos_definite(cov, 1e-10)
    sd = np.sqrt(np.abs(np.diag(cov)))
    d = len(test_x)
    delta = 0.1
    beta = 2 * np.log((d * t**2 * np.pi**2) / (6 * delta)) / 5
    return test_x * (mean + np.sqrt(beta) * sd)


def gpts(prices_tested, purchase_decisions, test_x, sigma2_y, batch_size, rng=None):
    """Gaussian Process Thompson Sampling policy."""
    test_x = np.asarray(test_x, dtype=float)
    train_x, train_y, sig = aggregate_data_gp(prices_tested, purchase_decisions, test_x,
                                              sigma2_y, batch_size)
    opt = optimal_hyperparameters(train_x, train_y, sig)
    mean, cov = posterior_prediction(train_x, train_y, test_x, None, opt[0], opt[1], sig)
    cov = make_pos_definite(cov, 1e-10)
    draw = mvrnorm(1, mean, cov, rng)
    return draw * test_x


def gpucb_mono(prices_tested, purchase_decisions, test_x, knots, sigma2_y, batch_size,
               basis_functions, lb1, lb2, ub, rng, timeout=None):
    """Monotonic GP-UCB: 1e4 truncated draws over [f(0), f'(knots)], demand curves
    reconstructed through the basis functions (monotone by construction)."""
    test_x = np.asarray(test_x, dtype=float)
    train_x, train_y, sig = aggregate_data_gp(prices_tested, purchase_decisions, test_x,
                                              sigma2_y, batch_size)
    opt = optimal_hyperparameters(train_x, train_y, sig)
    t = len(prices_tested) + batch_size

    mean, cov = posterior_prediction(train_x, train_y, [0.0], knots, opt[0], opt[1], sig)
    cov = make_pos_definite(cov, 1e-10)
    n_knots = len(knots)
    draws = rtmvnorm(10_000, mean, cov,
                     lb=np.concatenate([[lb1], np.full(n_knots, lb2)]),
                     ub=np.concatenate([[ub], np.zeros(n_knots)]),
                     rng=rng, timeout=timeout)
    demand = draws[:, 1:] @ basis_functions.T + draws[:, [0]]
    mean_d = demand.mean(axis=0)
    sd_d = demand.std(axis=0, ddof=1)
    d = len(test_x)
    delta = 0.1
    beta = 2 * np.log((d * t**2 * np.pi**2) / (6 * delta)) / 5
    return test_x * (mean_d + np.sqrt(beta) * sd_d)


def gpts_mono(prices_tested, purchase_decisions, test_x, knots, sigma2_y, batch_size,
              basis_functions, lb1, lb2, ub, rng, timeout=None):
    """Monotonic GP-TS: one truncated draw over [f(0), f'(knots)], reconstructed
    through the basis functions."""
    test_x = np.asarray(test_x, dtype=float)
    train_x, train_y, sig = aggregate_data_gp(prices_tested, purchase_decisions, test_x,
                                              sigma2_y, batch_size)
    opt = optimal_hyperparameters(train_x, train_y, sig)

    mean, cov = posterior_prediction(train_x, train_y, [0.0], knots, opt[0], opt[1], sig)
    cov = make_pos_definite(cov, 1e-10)
    n_knots = len(knots)
    draw = rtmvnorm(1, mean, cov,
                    lb=np.concatenate([[lb1], np.full(n_knots, lb2)]),
                    ub=np.concatenate([[ub], np.zeros(n_knots)]),
                    rng=rng, timeout=timeout)
    demand = basis_functions @ draw[1:] + draw[0]
    return demand * test_x


def noise_sample(prices_tested, purchase_decisions, test_x, batch_size, rng):
    """Heterogeneous-noise sampler (port of the R package's NoiseSample)."""
    test_x = np.asarray(test_x, dtype=float)
    sigma2_y = np.full(len(test_x) + 1, 0.25)
    train_x, train_y, sig = aggregate_data_gp(prices_tested, purchase_decisions, test_x,
                                              sigma2_y, batch_size)
    opt = optimal_hyperparameters(train_x, train_y, sig)
    mean, cov = posterior_prediction(train_x, train_y, np.concatenate([[0.0], test_x]),
                                     None, opt[0], opt[1], sig)
    cov = make_pos_definite(cov, 1e-10)
    draws = mvrnorm(1, mean, cov, rng)
    draws = np.clip(draws, 0.01, 0.99)
    return draws * (1 - draws)


def non_mono_policy_eval(policy, prices_tested, purchase_decisions, test_x, sigma2_y,
                         batch_size, gp, prev_scores, rng):
    """Evaluate a non-monotonic policy; on error fall back to the previous action
    scores (mirrors the R NonMonoPolicyEval, silently for GP failures)."""
    try:
        if gp:
            return policy(prices_tested, purchase_decisions, test_x, sigma2_y, batch_size, rng=rng)
        return policy(prices_tested, purchase_decisions, test_x, rng=rng)
    except Exception:
        bump("GPErrorPrevAS")
        return prev_scores


def policy_evaluation(policy, prices_tested, purchase_decisions, test_x, knots, sigma2_y,
                      batch_size, basis_functions, gp, mono, prev_scores, rng,
                      timeout=DEFAULT_TIMEOUT):
    """Policy evaluation with the R fallback chains.

    Basis-monotonic chain (mirroring the R PolicyEvaluation quirks exactly):
    attempt 1 with a `timeout`-second budget; on timeout, retry once with the
    *same* bounds and budget; if the retry fails, fall back to the previous
    action scores.
    """
    if not mono:
        return non_mono_policy_eval(policy, prices_tested, purchase_decisions, test_x,
                                    sigma2_y, batch_size, gp, prev_scores, rng)
    try:
        return policy(prices_tested, purchase_decisions, test_x, knots, sigma2_y, batch_size,
                      basis_functions, lb1=-np.inf, lb2=-np.inf, ub=2.0, rng=rng,
                      timeout=timeout)
    except TimeoutError:
        bump("MonoTimeout1")
        try:
            # second attempt (same bounds, mirroring the R implementation)
            return policy(prices_tested, purchase_decisions, test_x, knots, sigma2_y, batch_size,
                          basis_functions, lb1=-np.inf, lb2=-np.inf, ub=2.0, rng=rng,
                          timeout=timeout)
        except Exception:
            bump("MonoRetryError")
            # last resort in R calls the policy with the non-mono signature, which
            # errors and yields the previous action scores; net effect:
            bump("GPErrorPrevAS")
            return prev_scores
