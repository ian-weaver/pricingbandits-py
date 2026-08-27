"""pricingbandits: multi-armed bandit approaches to pricing experiments.

Python port of the PricingBandits R package (Weaver, Kumar & Jain,
"Nonparametric Pricing Bandits Leveraging Informational Externalities to Learn
the Demand Curve"). The willingness-to-pay distribution is fully user-specified
via a vector of consumer valuations.
"""

from .aggregate import aggregate_data_gp, aggregate_data_ts, aggregate_data_ucb
from .basis import basis_function, basis_matrix
from .diagnostics import get_diagnostics, reset_diagnostics
from .experiment import POLICIES, BanditResult, mab_experiment, pricing_bandit
from .gp import (
    make_pos_definite,
    mvrnorm,
    nlml,
    optimal_hyperparameters,
    posterior_prediction,
)
from .kernels import (
    covariance_from_kernel,
    joint_cov_from_kernel,
    rbf_kernel,
    rbf_kernel_01,
    rbf_kernel_11,
    rbf_kernel_all,
)
from .policies import gpts, gpts_mono, gpucb, gpucb_mono, noise_sample, ts, ucb
from .truncated_mvn import rtmvnorm

__version__ = "2.0.0"

__all__ = [
    "pricing_bandit", "mab_experiment", "BanditResult", "POLICIES",
    "ucb", "ts", "gpucb", "gpts", "gpucb_mono", "gpts_mono", "noise_sample",
    "aggregate_data_ucb", "aggregate_data_ts", "aggregate_data_gp",
    "basis_function", "basis_matrix",
    "nlml", "optimal_hyperparameters", "posterior_prediction",
    "make_pos_definite", "mvrnorm", "rtmvnorm",
    "rbf_kernel", "rbf_kernel_01", "rbf_kernel_11", "rbf_kernel_all",
    "covariance_from_kernel", "joint_cov_from_kernel",
    "get_diagnostics", "reset_diagnostics",
]
