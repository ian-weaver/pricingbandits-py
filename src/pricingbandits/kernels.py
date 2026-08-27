"""RBF kernel functions and covariance-matrix builders.

Direct port of the R package's ``kernels.R``. All kernels operate on scalars or
numpy arrays (vectorized via broadcasting).
"""

from __future__ import annotations

import numpy as np


def rbf_kernel(x_i, x_j, sigma_f, l):
    """Standard RBF kernel between two points."""
    return sigma_f**2 * np.exp(-((x_i - x_j) ** 2) / (2 * l**2))


def rbf_kernel_01(x_i, x_j, sigma_f, l):
    """RBF kernel between a point and a derivative."""
    return sigma_f**2 / l**2 * (x_i - x_j) * np.exp(-((x_i - x_j) ** 2) / (2 * l**2))


def rbf_kernel_11(x_i, x_j, sigma_f, l):
    """RBF kernel between two derivatives."""
    return (
        sigma_f**2 / l**4 * (l**2 - (x_i - x_j) ** 2) * np.exp(-((x_i - x_j) ** 2) / (2 * l**2))
    )


def rbf_kernel_all(x_i, x_j, d_i, d_j, sigma_f, l):
    """Generalized RBF kernel dispatching on derivative orders (0 or 1)."""
    x_i = np.asarray(x_i, dtype=float)
    x_j = np.asarray(x_j, dtype=float)
    d_i = np.asarray(d_i)
    d_j = np.asarray(d_j)
    out = np.empty(np.broadcast(x_i, x_j, d_i, d_j).shape, dtype=float)
    m00 = (d_i == 0) & (d_j == 0)
    m01 = (d_i == 0) & (d_j == 1)
    m10 = (d_i == 1) & (d_j == 0)
    m11 = (d_i == 1) & (d_j == 1)
    bx_i, bx_j, _, _ = np.broadcast_arrays(x_i, x_j, d_i, d_j)
    out[m00] = rbf_kernel(bx_i[m00], bx_j[m00], sigma_f, l)
    out[m01] = rbf_kernel_01(bx_i[m01], bx_j[m01], sigma_f, l)
    out[m10] = rbf_kernel_01(bx_j[m10], bx_i[m10], sigma_f, l)
    out[m11] = rbf_kernel_11(bx_i[m11], bx_j[m11], sigma_f, l)
    return out


def covariance_from_kernel(x1, x2, kernel, sigma_f, l):
    """Covariance matrix between two sets of points (outer product over the kernel)."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    return kernel(x1[:, None], x2[None, :], sigma_f, l)


def joint_cov_from_kernel(x, index, sigma_f, l):
    """Joint covariance matrix for a set of points with derivative indicators."""
    x = np.asarray(x, dtype=float)
    index = np.asarray(index)
    return rbf_kernel_all(x[:, None], x[None, :], index[:, None], index[None, :], sigma_f, l)


def cross_cov_from_kernel(x1, index1, x2, index2, sigma_f, l):
    """Cross covariance between two sets of (point, derivative-indicator) pairs."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    index1 = np.asarray(index1)
    index2 = np.asarray(index2)
    return rbf_kernel_all(x1[:, None], x2[None, :], index1[:, None], index2[None, :], sigma_f, l)
