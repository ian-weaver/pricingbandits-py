"""RBF kernel (including derivative cross-covariances) and covariance-matrix
builders.

Mirror of the R package's ``kernels.R``: one kernel function whose derivative
orders ``d_i``/``d_j`` (0 = function value, 1 = first derivative) select the
value-value, value-derivative, derivative-value, or derivative-derivative
covariance. Vectorized via broadcasting.
"""

from __future__ import annotations

import numpy as np


def rbf_kernel(x_i, x_j, sigma_f, l, d_i=0, d_j=0):
    """RBF kernel between two (possibly derivative) points.

    ``d_i`` and ``d_j`` give the derivative order of each argument (0 or 1,
    default 0). With the defaults this is the plain RBF kernel.
    """
    x_i = np.asarray(x_i, dtype=float)
    x_j = np.asarray(x_j, dtype=float)
    # fast path for the common all-values case (e.g. likelihood evaluations)
    if np.isscalar(d_i) and np.isscalar(d_j) and d_i == 0 and d_j == 0:
        return sigma_f**2 * np.exp(-((x_i - x_j) ** 2) / (2 * l**2))

    d_i = np.asarray(d_i)
    d_j = np.asarray(d_j)
    out = np.empty(np.broadcast(x_i, x_j, d_i, d_j).shape, dtype=float)
    bx_i, bx_j, bd_i, bd_j = np.broadcast_arrays(x_i, x_j, d_i, d_j)

    m00 = (bd_i == 0) & (bd_j == 0)
    m01 = (bd_i == 0) & (bd_j == 1)
    m10 = (bd_i == 1) & (bd_j == 0)
    m11 = (bd_i == 1) & (bd_j == 1)
    out[m00] = sigma_f**2 * np.exp(-((bx_i[m00] - bx_j[m00]) ** 2) / (2 * l**2))
    out[m01] = (sigma_f**2 / l**2 * (bx_i[m01] - bx_j[m01])
                * np.exp(-((bx_i[m01] - bx_j[m01]) ** 2) / (2 * l**2)))
    out[m10] = (sigma_f**2 / l**2 * (bx_j[m10] - bx_i[m10])
                * np.exp(-((bx_j[m10] - bx_i[m10]) ** 2) / (2 * l**2)))
    out[m11] = (sigma_f**2 / l**4 * (l**2 - (bx_i[m11] - bx_j[m11]) ** 2)
                * np.exp(-((bx_i[m11] - bx_j[m11]) ** 2) / (2 * l**2)))
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
    return rbf_kernel(x[:, None], x[None, :], sigma_f, l, index[:, None], index[None, :])


def cross_cov_from_kernel(x1, index1, x2, index2, sigma_f, l):
    """Cross covariance between two sets of (point, derivative-indicator) pairs."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    index1 = np.asarray(index1)
    index2 = np.asarray(index2)
    return rbf_kernel(x1[:, None], x2[None, :], sigma_f, l, index1[:, None], index2[None, :])
