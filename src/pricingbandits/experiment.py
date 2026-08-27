"""Multi-armed bandit experiment loop and the user-facing ``pricing_bandit`` wrapper.

Mirrors the R package's ``MABExperiment`` / ``PricingBandit`` behavior exactly
(update timing, priors, anchoring, purchase rule, reset semantics).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import policies as pol
from .basis import basis_matrix
from .diagnostics import bump, get_diagnostics, reset_diagnostics

POLICIES = {
    "UCB": pol.ucb,
    "TS": pol.ts,
    "GP-UCB": pol.gpucb,
    "GP-TS": pol.gpts,
    "GP-UCB-M": pol.gpucb_mono,
    "GP-TS-M": pol.gpts_mono,
}

_GP_POLICIES = {"GP-UCB", "GP-TS", "GP-UCB-M", "GP-TS-M"}
_MONO_POLICIES = {"GP-UCB-M", "GP-TS-M"}


@dataclass
class BanditResult:
    """Result of a single pricing-bandit experiment."""

    prices_tested: np.ndarray
    purchase_decisions: np.ndarray
    policy: str
    diagnostics: dict = field(default_factory=dict)


def mab_experiment(valuations, policy_name, test_x, num_iter, batch_size,
                   num_knots=None, hetero_noise=False, reset=None,
                   rng=None, timeout=5.0):
    """Run one MAB experiment (mirror of the R ``MABExperiment``)."""
    if rng is None:
        rng = np.random.default_rng()
    valuations = np.asarray(valuations, dtype=float)
    test_x = np.asarray(test_x, dtype=float)
    policy = POLICIES[policy_name]

    reset_diagnostics()

    prices_tested = np.full(num_iter, np.nan)
    purchase_decisions = np.full(num_iter, np.nan)
    sigma2_y = np.full(len(test_x) + 1, 0.25)
    action_scores = np.full(len(test_x), 0.5)
    gp = policy_name in _GP_POLICIES
    mono = policy_name in _MONO_POLICIES

    knots = None
    basis_functions = None
    if mono:
        knots = np.concatenate([[0.0], np.arange(1, num_knots) / (num_knots - 1)])
        basis_functions = basis_matrix(test_x, num_knots)

    price_to_test = np.nan
    for t0 in range(num_iter):  # t0 is 0-based; R's t = t0 + 1
        if t0 % batch_size == 0:
            # amend training data if the algorithm is resetting
            if reset is not None:
                m = (t0 + 1) % reset
                to_keep = 0 if m == 1 else m - 1
                if to_keep == 0:
                    train_x = np.array([])
                    train_y = np.array([])
                elif to_keep > 0:
                    train_x = prices_tested[t0 - to_keep : t0]
                    train_y = purchase_decisions[t0 - to_keep : t0]
                else:
                    # R quirk: ToKeep == -1 yields the reversed range (t+1):(t-1);
                    # after na.omit only the observation at t-1 survives.
                    train_x = prices_tested[max(t0 - 1, 0) : t0]
                    train_y = purchase_decisions[max(t0 - 1, 0) : t0]
                train_x = train_x[~np.isnan(train_x)]
                train_y = train_y[~np.isnan(train_y)]
            else:
                train_x = prices_tested[~np.isnan(prices_tested)]
                train_y = purchase_decisions[~np.isnan(purchase_decisions)]

            if len(train_x) == 0 and gp:
                # first price chosen randomly for GP variants (R: sample(TestX)[1])
                price_to_test = rng.permutation(test_x)[0]
            else:
                if hetero_noise:
                    sigma2_y = pol.noise_sample(train_x, train_y, test_x, batch_size, rng)

                bump("PolicyUpdates")
                action_scores = pol.policy_evaluation(
                    policy, train_x, train_y, test_x, knots, sigma2_y, batch_size,
                    basis_functions, gp, mono, action_scores, rng, timeout=timeout)
                a = int(np.argmax(action_scores))
                price_to_test = test_x[a]

        purchase = 1.0 if price_to_test < valuations[t0] + 1e-10 else 0.0
        prices_tested[t0] = price_to_test
        purchase_decisions[t0] = purchase

    return prices_tested, purchase_decisions


def pricing_bandit(valuations, prices, policy="GP-TS-M", batch_size=10,
                   num_knots=None, hetero=False, reset=None,
                   rng=None, seed=None, timeout=5.0):
    """Run a single pricing-bandit experiment (mirror of the R ``PricingBandit``).

    Parameters
    ----------
    valuations : array-like
        One WTP draw per consumer (any user-specified distribution). Its length
        determines the number of iterations.
    prices : array-like
        Candidate prices (the arms), values in (0, 1].
    policy : str
        One of ``"UCB"``, ``"TS"``, ``"GP-UCB"``, ``"GP-TS"``, ``"GP-UCB-M"``,
        ``"GP-TS-M"``.
    batch_size : int
        Consumers per policy update (default 10).
    num_knots : int, optional
        Knot count for the "-M" variants; defaults to 11 (the paper's value)
        regardless of the number of arms - many more knots put the truncated
        sampler in a near-degenerate high-dimensional space and can break it,
        so 11 is recommended even for dense price grids.
    hetero : bool
        Heterogeneous-noise toggle (default False).
    reset : int, optional
        Wipe the history every ``reset`` consumers (default: never).
    rng : numpy.random.Generator, optional
        Random generator; ``seed`` builds one when omitted.
    timeout : float
        Seconds allowed for each truncated-sampling attempt in the monotonic
        fallback chain before the next fallback is tried (default 5). Increase
        on slow machines to give the exact sampler more time; decrease to fail
        over to the cheaper approximations sooner.

    Returns
    -------
    BanditResult
        ``prices_tested``, ``purchase_decisions``, plus the fallback-counter
        ``diagnostics``.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; choose from {sorted(POLICIES)}")
    valuations = np.asarray(valuations, dtype=float)
    prices_arr = np.asarray(prices, dtype=float)
    if valuations.ndim != 1 or len(valuations) < 1:
        raise ValueError("`valuations` must be a non-empty 1-D array (one WTP draw per consumer).")
    if prices_arr.ndim != 1 or len(prices_arr) < 2 or np.any(prices_arr <= 0) or np.any(prices_arr > 1):
        raise ValueError("`prices` must contain at least two prices in (0, 1].")
    if batch_size < 1:
        raise ValueError("`batch_size` must be a positive integer.")
    if timeout <= 0:
        raise ValueError("`timeout` must be a positive number of seconds.")
    if rng is None:
        rng = np.random.default_rng(seed)
    if num_knots is None:
        num_knots = 11

    p, d = mab_experiment(valuations, policy, prices_arr, len(valuations), batch_size,
                          num_knots=num_knots, hetero_noise=hetero, reset=reset,
                          rng=rng, timeout=timeout)
    return BanditResult(prices_tested=p, purchase_decisions=d, policy=policy,
                        diagnostics=get_diagnostics())
