"""
Corrected-MLE machinery for logistic regression with known per-direction
label flip rates (eps, delta). All functions accept eps and delta either as
scalars or as length-n arrays (the per-x case in Section 6 of the report).

Conventions
-----------
* X is the *raw* design matrix without an intercept column. Every routine
  prepends a constant internally (sm.add_constant), so callers should pass
  X with shape (n, p) and the resulting parameter vector has length p+1
  with the intercept first.
* eps_i = P(yhat = 0 | y = 1)     (false-negative rate)
  delta_i = P(yhat = 1 | y = 0)   (false-positive rate)
  a_i = 1 - eps_i - delta_i
  q_i(beta) = delta_i + a_i * sigmoid(x_i' beta)

This module is shared by the report's figure-generation script and by all
notebooks; this avoids the three different fit_corr signatures we used
during the project.
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import expit


# ---------------------------------------------------------------------------
# Project-wide seed.  Anyone constructing a new RNG should branch off this.
# ---------------------------------------------------------------------------
SEED = 6114


def H(u: np.ndarray) -> np.ndarray:
    """Numerically stable logistic CDF; equivalent to 1 / (1 + exp(-u))."""
    return expit(u)


# ---------------------------------------------------------------------------
# Likelihood, gradient, observed information.
# ---------------------------------------------------------------------------
def neg_logL(theta, Xd, yh, eps, delta):
    """Negative corrected log-likelihood of (yh | X, theta) given (eps, delta).

    Xd must already include the intercept column (use sm.add_constant or
    np.column_stack([np.ones, X])). eps and delta can be scalars or
    length-n arrays.
    """
    eta = Xd @ theta
    p = H(eta)
    a = 1.0 - eps - delta
    q = np.clip(delta + a * p, 1e-12, 1 - 1e-12)
    return -np.sum(yh * np.log(q) + (1 - yh) * np.log(1 - q))


def grad_L(theta, Xd, yh, eps, delta):
    """Analytical gradient of the corrected log-likelihood (returns the
    gradient of -neg_logL, i.e.\ the score with the sign convention used by
    scipy.optimize)."""
    eta = Xd @ theta
    p = H(eta)
    a = 1.0 - eps - delta
    q = np.clip(delta + a * p, 1e-12, 1 - 1e-12)
    w = a * p * (1.0 - p)
    r = (yh - q) / (q * (1.0 - q))
    return -Xd.T @ (r * w)


def numeric_hess(theta, Xd, yh, eps, delta, h: float = 1e-4):
    """Second-order central-difference Hessian of neg_logL at theta.

    Returns the *positive*-definite observed information J = -Hess(loglik) =
    Hess(neg_logL).
    """
    f = lambda t: neg_logL(t, Xd, yh, eps, delta)
    d = len(theta)
    Hm = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            tp = theta.copy(); tp[i] += h; tp[j] += h
            tm = theta.copy(); tm[i] -= h; tm[j] -= h
            tpm = theta.copy(); tpm[i] += h; tpm[j] -= h
            tmp = theta.copy(); tmp[i] -= h; tmp[j] += h
            v = (f(tp) - f(tpm) - f(tmp) + f(tm)) / (4 * h * h)
            Hm[i, j] = v
            Hm[j, i] = v
    return Hm


def hessian_se(theta, X, yh, eps, delta, on_bound=None):
    """Hessian-based standard errors at theta.

    By default returns sqrt(diag(J^{-1})) where J = -Hess(loglik).  When
    ``on_bound`` is supplied (length-len(theta) bool array marking
    coordinates pinned at the L-BFGS-B bound), we instead return the
    asymptotic se for the *unbounded* coordinates, computed as the
    inverse of the Hessian's submatrix on those coordinates.  The
    bound-hit coordinates' entries are NaN.

    Why this matters: at a constrained optimum the right object for
    inference on the free coordinates is the Hessian's restriction to
    the unbounded subspace, not the diagonal of the full inverse, since
    the latter accounts for "phantom" variability in the pinned
    coordinate that is not present at a constrained optimum.
    """
    Xd = sm.add_constant(X, has_constant="add")
    J = numeric_hess(theta, Xd, yh, eps, delta)
    p = len(theta)
    if on_bound is None:
        free = np.ones(p, dtype=bool)
    else:
        free = ~np.asarray(on_bound, dtype=bool)
    se = np.full(p, np.nan)
    if not np.any(free):
        return se
    J_sub = J[np.ix_(free, free)]
    try:
        cov_sub = np.linalg.inv(J_sub)
    except np.linalg.LinAlgError:
        return se
    diag_sub = np.diag(cov_sub)
    se_sub = np.where(diag_sub > 0, np.sqrt(diag_sub), np.nan)
    se[free] = se_sub
    return se


# ---------------------------------------------------------------------------
# Naive logistic baseline (used as a warm start for the corrected fit).
# ---------------------------------------------------------------------------
def fit_naive(X, y):
    """Standard logistic regression on (X, y).

    Returns (params, se, pvalues) all of length p+1 (intercept first).
    Returns NaN arrays on singular / non-converged fits.
    """
    Xd = sm.add_constant(X, has_constant="add")
    try:
        res = sm.GLM(y, Xd, family=sm.families.Binomial()).fit(disp=0)
        return (np.asarray(res.params),
                np.asarray(res.bse),
                np.asarray(res.pvalues))
    except Exception:
        k = Xd.shape[1]
        nan = np.full(k, np.nan)
        return nan, nan.copy(), nan.copy()


# ---------------------------------------------------------------------------
# Corrected MLE: bounded L-BFGS-B with the analytical gradient.
# ---------------------------------------------------------------------------
def fit_corr(X, yh, eps, delta, start=None, bound: float = 15.0):
    """Maximise the corrected log-likelihood by bounded L-BFGS-B.

    Parameters
    ----------
    X : (n, p) array
        Raw design matrix without an intercept column.
    yh : (n,) array
        Observed (noisy) labels.
    eps, delta : scalar or (n,) array
        Per-direction flip rates. Same broadcasting rules as in neg_logL.
    start : (p+1,) array or None
        Warm start. If None, defaults to the naive fit's parameters when
        available; otherwise zeros.
    bound : float
        Per-coordinate bound M; the optimiser is constrained to [-M, M]^{p+1}.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Standard scipy result; on_bound can be checked via
        np.max(np.abs(res.x)) >= bound - tol for the bound-hit filter.
    """
    Xd = sm.add_constant(X, has_constant="add")
    if start is None:
        try:
            start, _, _ = fit_naive(X, yh)
            if np.any(np.isnan(start)):
                start = np.zeros(Xd.shape[1])
        except Exception:
            start = np.zeros(Xd.shape[1])
    bnds = [(-bound, bound)] * len(start)
    return minimize(
        neg_logL, x0=np.asarray(start), args=(Xd, yh, eps, delta),
        jac=grad_L, method="L-BFGS-B", bounds=bnds,
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 2000},
    )


# ---------------------------------------------------------------------------
# Symmetric label-flip simulator.
# ---------------------------------------------------------------------------
def flip_labels(y, eps, delta=None, rng: np.random.Generator | None = None):
    """Flip labels at per-direction rates (eps, delta).

    Parameters
    ----------
    y : (n,) integer array of {0, 1}
    eps, delta : scalar or (n,) array
        eps  = P(flip y=1 to 0); delta = P(flip y=0 to 1).
        If delta is None, uses the symmetric model delta = eps.
    rng : np.random.Generator
        Required (no fallback) so callers cannot accidentally produce
        non-reproducible flips.

    Returns
    -------
    yh : (n,) integer array
    """
    if rng is None:
        raise ValueError("flip_labels requires an explicit rng for reproducibility")
    if delta is None:
        delta = eps
    eps_arr = np.asarray(eps) if np.ndim(eps) else np.full(len(y), float(eps))
    del_arr = np.asarray(delta) if np.ndim(delta) else np.full(len(y), float(delta))
    flip1 = (y == 1) & (rng.uniform(size=len(y)) < eps_arr)
    flip0 = (y == 0) & (rng.uniform(size=len(y)) < del_arr)
    yh = y.copy()
    yh[flip1] = 0
    yh[flip0] = 1
    return yh
