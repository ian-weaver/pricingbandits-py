"""Piecewise-linear-derivative basis functions for the monotonic GP policies.

Direct port of the R package's ``basisfunctions.R``.
"""

from __future__ import annotations

import numpy as np


def basis_function(x, J):
    """Values of the J basis functions at test point ``x``.

    Mirrors the R implementation exactly, including its knot layout
    ``(0, 1/(J-1), 2/(J-1), ..., 1)``.
    """
    delta_j = 1.0 / (J - 1)
    knots = np.concatenate(([0.0], np.arange(1, J) / (J - 1)))
    output = np.zeros(J)
    i = int(np.max(np.nonzero(knots <= x + 0.0)[0]))  # R: max(which(Knots <= x)) - 1-indexed
    # translate to R's 1-based index for the branch structure below
    i_r = i + 1

    if i_r == 1:
        output[0] = x - 0.5 * (x**2) / delta_j
        output[1] = x - knots[1] * x / delta_j + 0.5 * x**2 / delta_j
    if i_r == 2:
        output[0] = delta_j / 2
        output[1] = delta_j / 2 + (x - knots[1]) * (1 + knots[1] / delta_j) - 0.5 * (
            x**2 - knots[1] ** 2
        ) / delta_j
        output[2] = (x - knots[1]) * (1 - knots[2] / delta_j) + 0.5 * (
            x**2 - knots[1] ** 2
        ) / delta_j
    if i_r == J:
        output[0] = delta_j / 2
        output[1 : J - 1] = delta_j
        output[J - 1] = delta_j / 2
    if i_r not in (1, 2, J):
        output[0] = delta_j / 2
        output[1 : i_r - 1] = delta_j
        output[i_r - 1] = delta_j / 2 + (x - knots[i_r - 1]) * (1 + knots[i_r - 1] / delta_j) - 0.5 * (
            x**2 - knots[i_r - 1] ** 2
        ) / delta_j
        output[i_r] = (x - knots[i_r - 1]) * (1 - knots[i_r] / delta_j) + 0.5 * (
            x**2 - knots[i_r - 1] ** 2
        ) / delta_j
    return output


def basis_matrix(test_x, J):
    """Basis-function matrix with one row per test price (mirrors MABExperiment setup)."""
    test_x = np.asarray(test_x, dtype=float)
    return np.vstack([basis_function(x, J) for x in test_x])
