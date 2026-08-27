"""Truncated multivariate normal sampling via minimax exponential tilting.

Self-contained port of Botev (2017), "The normal law under linear restrictions:
simulation and estimation via minimax tilting" (JRSS-B) - the algorithm behind
the R package ``TruncatedNormal`` used by the reference implementation. Only
numpy/scipy are required.

The sampler accepts an optional wall-clock ``timeout``; when exceeded a
``TimeoutError`` is raised, mirroring ``R.utils::withTimeout`` semantics in the
R policies' fallback chains (Windows-safe: no signals involved).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erfc, erfcinv, erfcx

_SQRT_2PI = np.sqrt(2 * np.pi)
_SQRT2 = np.sqrt(2.0)


def _ln_phi(x):
    """log of the upper tail: log P(Z > x), stable for large x."""
    with np.errstate(divide="ignore"):  # log(0) -> -inf is the correct limit
        return -0.5 * x**2 - np.log(2.0) + np.log(erfcx(x / _SQRT2))


def ln_normal_pr(a, b):
    """log P(a < Z < b) for standard normal, elementwise, numerically stable."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p = np.zeros(np.broadcast(a, b).shape)
    a_b, b_b = np.broadcast_arrays(a, b)

    # case: both bounds in the upper tail
    upper = a_b > 0
    if np.any(upper):
        pa = _ln_phi(a_b[upper])
        pb = _ln_phi(b_b[upper])
        p[upper] = pa + np.log1p(-np.exp(pb - pa))
    # case: both bounds in the lower tail
    lower = b_b < 0
    lower &= ~upper
    if np.any(lower):
        pa = _ln_phi(-b_b[lower])
        pb = _ln_phi(-a_b[lower])
        p[lower] = pa + np.log1p(-np.exp(pb - pa))
    # case: straddling zero
    mid = ~(upper | lower)
    if np.any(mid):
        pa = erfc(-a_b[mid] / _SQRT2) / 2  # lower tail P(Z < a)
        pb = erfc(b_b[mid] / _SQRT2) / 2   # upper tail P(Z > b)
        p[mid] = np.log1p(-pa - pb)
    return p


def _ntail(l, u, rng, timeout_at=None):
    """Rayleigh accept-reject sampler for the upper tail (l > 0), vectorized."""
    c = l**2 / 2
    n = len(l)
    f = np.expm1(c - u**2 / 2)
    x = np.empty(n)
    todo = np.ones(n, dtype=bool)
    while np.any(todo):
        if timeout_at is not None and time.monotonic() > timeout_at:
            raise TimeoutError("truncated-normal tail sampling timed out")
        idx = np.nonzero(todo)[0]
        cand = c[idx] - np.log1p(rng.uniform(size=len(idx)) * f[idx])
        accept = rng.uniform(size=len(idx)) ** 2 * cand < c[idx]
        x[idx[accept]] = cand[accept]
        todo[idx[accept]] = False
    return np.sqrt(2 * x)


def _trnd(l, u, rng, timeout_at=None):
    """Plain normal rejection sampler on [l, u], vectorized."""
    n = len(l)
    x = np.empty(n)
    todo = np.ones(n, dtype=bool)
    while np.any(todo):
        if timeout_at is not None and time.monotonic() > timeout_at:
            raise TimeoutError("truncated-normal sampling timed out")
        idx = np.nonzero(todo)[0]
        cand = rng.standard_normal(len(idx))
        accept = (cand > l[idx]) & (cand < u[idx])
        x[idx[accept]] = cand[accept]
        todo[idx[accept]] = False
    return x


def trandn(l, u, rng, timeout_at=None):
    """Sample truncated standard normal on [l, u], elementwise (Botev's trandn)."""
    l = np.asarray(l, dtype=float)
    u = np.asarray(u, dtype=float)
    x = np.empty(l.shape)
    thresh = 0.66

    hi = l > thresh
    lo = u < -thresh
    mid = ~(hi | lo)
    if np.any(hi):
        x[hi] = _ntail(l[hi], u[hi], rng, timeout_at)
    if np.any(lo):
        x[lo] = -_ntail(-u[lo], -l[lo], rng, timeout_at)
    if np.any(mid):
        lm, um = l[mid], u[mid]
        xm = np.empty(lm.shape)
        wide = np.abs(um - lm) > 2.0
        if np.any(wide):
            xm[wide] = _trnd(lm[wide], um[wide], rng, timeout_at)
        narrow = ~wide
        if np.any(narrow):
            pl = erfc(lm[narrow] / _SQRT2) / 2
            pu = erfc(um[narrow] / _SQRT2) / 2
            uni = rng.uniform(size=narrow.sum())
            xm[narrow] = _SQRT2 * erfcinv(2 * (pl + (pu - pl) * uni))
        x[mid] = xm
    return x


def _cholperm(sigma, l, u):
    """Gibson-Glasbey-Elston ordered Cholesky. Returns (L, l, u, perm)."""
    d = len(l)
    sigma = sigma.copy()
    l = l.copy()
    u = u.copy()
    L = np.zeros((d, d))
    z = np.zeros(d)
    perm = np.arange(d)
    eps = 1e-10

    for j in range(d):
        pr = np.full(d, np.inf)
        i_range = np.arange(j, d)
        s_part = L[i_range, :j] @ z[:j] if j > 0 else np.zeros(len(i_range))
        denom2 = np.diag(sigma)[i_range] - np.sum(L[i_range, :j] ** 2, axis=1)
        denom2 = np.maximum(denom2, eps)
        denom = np.sqrt(denom2)
        tl = (l[i_range] - s_part) / denom
        tu = (u[i_range] - s_part) / denom
        pr[i_range] = ln_normal_pr(tl, tu)
        k = int(np.argmin(pr[j:]) + j)

        # symmetric swap of j and k
        if k != j:
            sigma[[j, k], :] = sigma[[k, j], :]
            sigma[:, [j, k]] = sigma[:, [k, j]]
            L[[j, k], :] = L[[k, j], :]
            l[[j, k]] = l[[k, j]]
            u[[j, k]] = u[[k, j]]
            perm[[j, k]] = perm[[k, j]]

        s2 = sigma[j, j] - np.sum(L[j, :j] ** 2)
        if s2 < -0.01:
            raise np.linalg.LinAlgError("sigma is not positive semi-definite")
        L[j, j] = np.sqrt(max(s2, eps))
        if j < d - 1:
            L[j + 1 :, j] = (sigma[j + 1 :, j] - L[j + 1 :, :j] @ L[j, :j]) / L[j, j]

        tl_j = (l[j] - (L[j, :j] @ z[:j] if j > 0 else 0.0)) / L[j, j]
        tu_j = (u[j] - (L[j, :j] @ z[:j] if j > 0 else 0.0)) / L[j, j]
        w = ln_normal_pr(tl_j, tu_j)
        z[j] = (np.exp(-0.5 * tl_j**2 - w) - np.exp(-0.5 * tu_j**2 - w)) / _SQRT_2PI
    return L, l, u, perm


def _gradpsi(y, L, l, u):
    """Gradient and Jacobian of the tilting objective (Botev's gradpsi)."""
    d = len(u)
    c = np.zeros(d)
    x = np.zeros(d)
    mu = np.zeros(d)
    x[: d - 1] = y[: d - 1]
    mu[: d - 1] = y[d - 1 :]

    c[1:] = L[1:, :] @ x
    lt = l - mu - c
    ut = u - mu - c

    w = ln_normal_pr(lt, ut)
    pl = np.exp(-0.5 * lt**2 - w) / _SQRT_2PI
    pu = np.exp(-0.5 * ut**2 - w) / _SQRT_2PI
    P = pl - pu

    dfdx = -mu[: d - 1] + (P @ L)[: d - 1]
    dfdm = mu - x + P
    grad = np.concatenate([dfdx, dfdm[: d - 1]])

    lt = np.where(np.isinf(lt), 0.0, lt)
    ut = np.where(np.isinf(ut), 0.0, ut)
    dP = -(P**2) + lt * pl - ut * pu
    DL = dP[:, None] * L
    mx = -np.eye(d) + DL
    xx = L.T @ DL
    mx = mx[: d - 1, : d - 1]
    xx = xx[: d - 1, : d - 1]
    jac = np.block([[xx, mx.T], [mx, np.diag(1 + dP[: d - 1])]])
    return grad, jac


def _psy(x, L, l, u, mu):
    """Tilting log-acceptance bound psi(x, mu)."""
    d = len(u)
    x = x.copy()
    mu = mu.copy()
    x[d - 1] = 0.0
    mu[d - 1] = 0.0
    c = np.zeros(d)
    c[1:] = L[1:, :] @ x
    lt = l - mu - c
    ut = u - mu - c
    return float(np.sum(ln_normal_pr(lt, ut) + 0.5 * mu**2 - x * mu))


def _mvnrnd(n, L, l, u, mu, rng, timeout_at=None):
    """Tilted sequential sampler: returns (logpr, Z) with Z of shape (d, n)."""
    d = len(l)
    mu_full = np.zeros(d)
    mu_full[: d - 1] = mu[: d - 1]
    Z = np.zeros((d, n))
    logpr = np.zeros(n)
    for k in range(d):
        col = L[k, :k] @ Z[:k] if k > 0 else np.zeros(n)
        tl = l[k] - mu_full[k] - col
        tu = u[k] - mu_full[k] - col
        Z[k] = mu_full[k] + trandn(tl, tu, rng, timeout_at)
        logpr += ln_normal_pr(tl, tu) + 0.5 * mu_full[k] ** 2 - mu_full[k] * Z[k]
    return logpr, Z


def rtmvnorm(n, mu, sigma, lb, ub, rng, timeout=None):
    """Draw ``n`` samples from N(mu, sigma) truncated to [lb, ub].

    Mirrors ``TruncatedNormal::rtmvnorm``: returns an (n, d) array (a 1-D array
    of length d when n == 1). Raises ``TimeoutError`` when ``timeout`` (seconds)
    is exceeded, and ``LinAlgError``/``RuntimeError`` on numerical failure -
    callers' fallback chains treat these like the R errors.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    lb = np.asarray(lb, dtype=float).ravel() - mu
    ub = np.asarray(ub, dtype=float).ravel() - mu
    sigma = np.asarray(sigma, dtype=float)
    d = len(lb)
    timeout_at = None if timeout is None else time.monotonic() + timeout

    L_full, l, u, perm = _cholperm(sigma, lb, ub)
    D = np.diag(L_full)
    if np.any(D < 1e-10):
        # method may fail; mirror the R warning path by proceeding anyway
        pass
    u_s = u / D
    l_s = l / D
    L = L_full / D[:, None] - np.eye(d)

    # find the optimal tilting parameters. A dogleg trust-region solve of the
    # nonlinear system (as least squares) mirrors the Powell-dogleg strategy the
    # R package uses via nleqslv and stays stable on the near-singular
    # covariances the GP posteriors produce.
    x0 = np.zeros(2 * d - 2)
    sol = least_squares(lambda y: _gradpsi(y, L, l_s, u_s)[0], x0,
                        jac=lambda y: _gradpsi(y, L, l_s, u_s)[1],
                        method="dogbox", xtol=1e-12, gtol=None)
    if not np.all(np.isfinite(sol.x)):
        raise RuntimeError("tilting optimization failed")
    soln = sol.x
    x_star = np.zeros(d)
    mu_star = np.zeros(d)
    x_star[: d - 1] = soln[: d - 1]
    mu_star[: d - 1] = soln[d - 1 :]
    psistar = _psy(x_star, L, l_s, u_s, mu_star)

    # accept-reject with exponential tilting
    accepted = np.zeros((d, 0))
    n_todo = n
    iter_guard = 0
    while n_todo > 0:
        if timeout_at is not None and time.monotonic() > timeout_at:
            raise TimeoutError("rtmvnorm timed out")
        iter_guard += 1
        if iter_guard > 10_000:
            raise RuntimeError("rtmvnorm acceptance rate too low")
        chunk = max(n_todo, int(np.ceil(n_todo / 0.5)))
        logpr, Z = _mvnrnd(chunk, L, l_s, u_s, mu_star, rng, timeout_at)
        accept = -np.log(rng.uniform(size=chunk)) > (psistar - logpr)
        keep = Z[:, accept][:, :n_todo]
        accepted = np.hstack([accepted, keep])
        n_todo = n - accepted.shape[1]

    # back-transform to original scale and order, then shift by the mean
    out = L_full @ accepted  # (d, n), permuted coordinate order
    result = np.empty((d, n))
    result[perm, :] = out
    result = result.T + mu[None, :]
    return result[0] if n == 1 else result
