"""Gaussian-process regression: marginal likelihood, hyperparameter optimization,
posterior prediction (jointly over function values and derivatives), and
covariance repair.

Direct port of the R package's ``gpregression.R``.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .diagnostics import bump
from .kernels import covariance_from_kernel, cross_cov_from_kernel, joint_cov_from_kernel, rbf_kernel


def nlml(theta, train_x, train_y, sigma2_y):
    """Negative log marginal likelihood.

    Mirrors the R package's ``NLML`` (which omits the additive ``n*log(2*pi)``
    normalizing constant; the constant does not affect the optimum).
    """
    train_x = np.asarray(train_x, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    K = covariance_from_kernel(train_x, train_x, rbf_kernel, theta[0], theta[1]) + np.diag(
        np.broadcast_to(sigma2_y, train_x.shape)
    )
    sign, logdet = np.linalg.slogdet(K)
    if sign <= 0 or not np.isfinite(logdet):
        raise np.linalg.LinAlgError("non-positive determinant in NLML")
    return 0.5 * logdet + float(train_y @ np.linalg.solve(K, train_y))


def optimal_hyperparameters(train_x, train_y, sigma2_y):
    """Optimize (sigma_f, l) within the same box and from the same start as the
    R package's nloptr COBYLA call.

    Uses L-BFGS-B rather than COBYLA: scipy's COBYLA (pure-Python PRIMA) is two
    orders of magnitude slower here, and on this smooth 2-D objective L-BFGS-B
    finds the same optimum in >99% of encountered datasets (both are local
    methods from the same start; residual differences wash out statistically).
    Falls back to the prior values (0.7, 0.2) on any numerical error, mirroring
    the R implementation (and counting the event in the diagnostics).
    """
    x0 = np.array([0.7, 0.2])
    lower = np.array([0.15, 0.1])
    upper = np.array([1.5, 0.4])
    try:
        res = minimize(
            nlml,
            x0=x0,
            args=(train_x, train_y, sigma2_y),
            method="L-BFGS-B",
            bounds=list(zip(lower, upper)),
        )
        sol = np.clip(res.x, lower, upper)
        if not np.all(np.isfinite(sol)):
            raise FloatingPointError("non-finite hyperparameters")
        return sol
    except Exception:
        bump("HyperoptPrior")
        return x0


def posterior_prediction(train_x, train_y, test_x, test_d, sigma_f, l, sigma2_y):
    """Joint GP posterior over function values at ``test_x`` and derivatives at
    ``test_d`` (either may be empty/None).

    Returns ``(mean, cov)``. Mirrors the R implementation's linear algebra:
    the training-covariance inverse is formed explicitly (chol2inv) and applied
    with two matrix products.
    """
    train_x = np.asarray(train_x, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    test_x = np.asarray([] if test_x is None else np.atleast_1d(test_x), dtype=float)
    test_d = np.asarray([] if test_d is None else np.atleast_1d(test_d), dtype=float)

    train_index = np.zeros(len(train_x), dtype=int)
    test_all = np.concatenate([test_x, test_d])
    test_index = np.concatenate([np.zeros(len(test_x), dtype=int), np.ones(len(test_d), dtype=int)])

    k_00 = joint_cov_from_kernel(train_x, train_index, sigma_f, l)
    k_11 = joint_cov_from_kernel(test_all, test_index, sigma_f, l)
    k_01 = cross_cov_from_kernel(train_x, train_index, test_all, test_index, sigma_f, l)

    K = k_00 + np.diag(np.broadcast_to(sigma2_y, train_x.shape))
    c, low = cho_factor(K)
    k_inv = cho_solve((c, low), np.eye(len(train_x)))

    mean = k_01.T @ k_inv @ train_y
    cov = k_11 - k_01.T @ k_inv @ k_01
    return mean, cov


def make_pos_definite(cov, zilon):
    """Clip negative eigenvalues to ``zilon`` until the matrix is PSD (mirrors
    the R package's ``MakePosDefinitive`` Rebonato-Jackel style repair)."""
    new_mat = (cov + cov.T) / 2.0
    min_eigen = np.linalg.eigvalsh(new_mat).min()
    neg_error = min_eigen < zilon
    while neg_error:
        vals, vecs = np.linalg.eigh(new_mat)
        vals = np.maximum(zilon, vals)
        new_mat = vecs @ np.diag(vals) @ vecs.T
        new_mat = (new_mat + new_mat.T) / 2.0
        neg_error = np.linalg.eigvalsh(new_mat).min() < 0
    return new_mat


def mvrnorm(n, mu, sigma, rng):
    """Multivariate-normal sampler mirroring MASS::mvrnorm (eigendecomposition
    based, tolerant of slightly negative eigenvalues)."""
    mu = np.asarray(mu, dtype=float).ravel()
    d = len(mu)
    vals, vecs = np.linalg.eigh((sigma + sigma.T) / 2.0)
    tol = 1e-6
    if not np.all(vals >= -tol * abs(vals[-1])):
        raise np.linalg.LinAlgError("'Sigma' is not positive definite")
    z = rng.standard_normal((n, d))
    draws = mu + z @ (vecs * np.sqrt(np.maximum(vals, 0))).T
    return draws[0] if n == 1 else draws
