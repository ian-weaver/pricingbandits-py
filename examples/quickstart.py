"""Quickstart: every pricingbandits algorithm on one shared consumer sequence.

Mirrors the R package vignette: 1,000 consumers with Beta(2, 9) willingness to
pay, ten candidate prices, all six policies plus the heterogeneous-noise
versions of the two monotonic ones. Requires only numpy/scipy;
matplotlib is optional (used for the final figure if installed).

Run:  python examples/quickstart.py
"""

import time

import numpy as np
from scipy.stats import beta as beta_dist

import pricingbandits as pb

# --- the demand environment: one WTP draw per consumer, any source you like
rng_env = np.random.default_rng(29)
valuations = rng_env.beta(2, 9, 1000)
prices = np.arange(1, 11) / 10  # ten candidate prices 0.1 ... 1.0

# --- scoring: expected revenue of each price under the true WTP distribution
expected_reward = prices * (1 - beta_dist.cdf(prices, 2, 9))
grid = np.arange(1, 1_000_001) / 1_000_000
true_optimal = np.max(grid * (1 - beta_dist.cdf(grid, 2, 9)))
price_to_er = {round(p, 10): er for p, er in zip(prices, expected_reward)}


def score(out):
    er = np.array([price_to_er[round(p, 10)] for p in out.prices_tested])
    return np.cumsum(er) / (np.arange(1, len(er) + 1) * true_optimal) * 100


RUNS = [
    ("UCB", False), ("TS", False),
    ("GP-UCB", False), ("GP-TS", False),
    ("GP-UCB-M", False), ("GP-TS-M", False),
    ("GP-UCB-M", True), ("GP-TS-M", True),
]

curves = {}
for policy, hetero in RUNS:
    label = policy + (" (hetero)" if hetero else "")
    t0 = time.time()
    out = pb.pricing_bandit(valuations, prices, policy=policy, batch_size=10,
                            hetero=hetero, rng=np.random.default_rng(1))
    curves[label] = score(out)
    print(f"{label:<20} {curves[label][-1]:5.1f}% of optimal   "
          f"({time.time() - t0:5.1f}s)")

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for label, c in curves.items():
        ax = axes[0] if "TS" in label.split()[0] else axes[1]
        ax.plot(c, label=label, linestyle="--" if "hetero" in label else "-")
    axes[0].set_title("Thompson Sampling family")
    axes[1].set_title("UCB family")
    for ax in axes:
        ax.set_xlabel("Consumers")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Cumulative revenue (% of true optimal)")
    fig.suptitle("All algorithms on the same 1,000 Beta(2,9) consumers")
    fig.tight_layout()
    fig.savefig("quickstart_results.png", dpi=150)
    print("figure saved to quickstart_results.png")
except ImportError:
    print("(install matplotlib to also get the figure)")
