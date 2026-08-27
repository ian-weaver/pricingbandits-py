"""Data-aggregation helpers for the bandit policies.

Direct port of the R package's ``datapreprocessing.R``. Grouping mirrors
dplyr's behavior: groups are formed on exact float values and returned in
ascending order of price (prices in an experiment are always exact copies of
the test-grid values, so exact-equality grouping is well defined).
"""

from __future__ import annotations

import numpy as np


def aggregate_data_ucb(prices_tested, purchase_decisions, test_x):
    """Aggregate for UCB: per-price successes and counts with a (1, 1) prior on
    every untested price.

    Returns ``(s_t, n_t)`` aligned with ``test_x`` in ascending price order
    (mirrors the R left-join onto the test grid).
    """
    prices = np.asarray(prices_tested, dtype=float)
    dec = np.asarray(purchase_decisions, dtype=float)
    test_x = np.asarray(test_x, dtype=float)

    order = np.argsort(test_x, kind="stable")
    s_t = np.ones(len(test_x))
    n_t = np.ones(len(test_x))
    for k in order:
        m = prices == test_x[k]
        if np.any(m):
            s_t[k] = dec[m].sum()
            n_t[k] = m.sum()
    # R returns rows ordered by the MissingData frame (test_x order as given)
    return s_t, n_t


def aggregate_data_ts(prices_tested, purchase_decisions, test_x):
    """Aggregate for TS: appends one success and one failure per arm, then groups
    by price in ascending order (mirrors dplyr group_by sorting).

    Returns ``(s_t, n_t)`` in ascending order of the distinct prices (which is
    the full test grid, since every arm gets pseudo-observations).
    """
    prices = np.concatenate([np.asarray(prices_tested, dtype=float),
                             np.asarray(test_x, dtype=float),
                             np.asarray(test_x, dtype=float)])
    dec = np.concatenate([np.asarray(purchase_decisions, dtype=float),
                          np.zeros(len(test_x)),
                          np.ones(len(test_x))])
    uniq = np.unique(prices)
    s_t = np.empty(len(uniq))
    n_t = np.empty(len(uniq))
    for i, p in enumerate(uniq):
        m = prices == p
        s_t[i] = dec[m].sum()
        n_t[i] = m.sum()
    return s_t, n_t


def aggregate_data_gp(prices_tested, purchase_decisions, test_x, sigma2_y, batch_size):
    """Aggregate for GP policies: anchors demand with ``batch_size`` fake purchases
    at price 0, groups to per-price purchase rates, and scales the observation
    noise by the per-price counts.

    Returns ``(train_x, train_y, sigma2_scaled)`` in ascending price order.
    ``sigma2_y`` must have length ``len(test_x) + 1`` (index 0 is the price-0
    anchor), exactly as in the R implementation.
    """
    prices = np.concatenate([np.asarray(prices_tested, dtype=float), np.zeros(batch_size)])
    dec = np.concatenate([np.asarray(purchase_decisions, dtype=float), np.ones(batch_size)])
    test_x = np.asarray(test_x, dtype=float)
    sigma2_y = np.asarray(sigma2_y, dtype=float)

    uniq = np.unique(prices)
    train_y = np.empty(len(uniq))
    n_t = np.empty(len(uniq))
    for i, p in enumerate(uniq):
        m = prices == p
        train_y[i] = dec[m].mean()
        n_t[i] = m.sum()

    grid = np.concatenate([[0.0], test_x])
    tested = np.isin(np.round(grid, 2), np.round(uniq, 2))
    sigma2_scaled = sigma2_y[tested] / n_t
    return uniq, train_y, sigma2_scaled
