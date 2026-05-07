"""
STEP 2-3: Wald coverage experiment with bounded L-BFGS-B.
Setup: (a, b) = (0.5, 1.2), n = 1000, B = 400 replicates, bound = 10.
For each (eps, delta) in main grid, report convergent fraction, MC mean/coverage, max |θ̂|, bound hits.

STEP 4: Sanity check with B = 60 bounded fits over dense eps grid.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import expit

# ========== helpers (copy from generate_report_figures.py) ==========
def H(u): return expit(u)

def simulate_ab(n, a, b, eps, delta, rng):
    x = rng.standard_normal(n)
    p = H(a + b * x)
    y = rng.binomial(1, p)
    flip1 = (y == 1) & (rng.uniform(size=n) < eps)
    flip0 = (y == 0) & (rng.uniform(size=n) < delta)
    yh = y.copy()
    yh[flip1] = 0
    yh[flip0] = 1
    return x, y, yh

def fit_naive(X, y):
    Xd = sm.add_constant(X, has_constant='add')
    try:
        r = sm.GLM(y, Xd, family=sm.families.Binomial()).fit(disp=0)
        return np.asarray(r.params), np.asarray(r.bse)
    except Exception:
        k = Xd.shape[1]
        return np.full(k, np.nan), np.full(k, np.nan)

def neg_logL(theta, Xd, yh, eps, delta):
    eta = Xd @ theta
    p = H(eta)
    c = 1.0 - eps - delta
    ps = np.clip(delta + c * p, 1e-12, 1 - 1e-12)
    return -np.sum(yh * np.log(ps) + (1 - yh) * np.log(1 - ps))

def grad_L(theta, Xd, yh, eps, delta):
    eta = Xd @ theta
    p = H(eta)
    c = 1.0 - eps - delta
    ps = np.clip(delta + c * p, 1e-12, 1 - 1e-12)
    w = c * p * (1.0 - p)
    r = (yh - ps) / (ps * (1.0 - ps))
    return -Xd.T @ (r * w)

def fit_corr(X, yh, eps, delta, start, bounded=False, bound=15.0):
    Xd = sm.add_constant(X, has_constant='add')
    if bounded:
        bnds = [(-bound, bound)] * len(start)
        return minimize(neg_logL, x0=start, args=(Xd, yh, eps, delta),
                        jac=grad_L, method='L-BFGS-B', bounds=bnds)
    return minimize(neg_logL, x0=start, args=(Xd, yh, eps, delta),
                    jac=grad_L, method='BFGS')

def numeric_hess(f, theta, h=1e-4):
    d = len(theta)
    Hm = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            tp  = theta.copy(); tp[i]  += h; tp[j]  += h
            tm  = theta.copy(); tm[i]  -= h; tm[j]  -= h
            tpm = theta.copy(); tpm[i] += h; tpm[j] -= h
            tmp = theta.copy(); tmp[i] -= h; tmp[j] += h
            v = (f(tp) - f(tpm) - f(tmp) + f(tm)) / (4 * h * h)
            Hm[i, j] = v; Hm[j, i] = v
    return Hm

def se_hess(theta, X, yh, eps, delta):
    Xd = sm.add_constant(X, has_constant='add')
    J = numeric_hess(lambda t: neg_logL(t, Xd, yh, eps, delta), theta)
    try:
        se = np.sqrt(np.diag(np.linalg.inv(J)))
    except Exception:
        se = np.full(len(theta), np.nan)
    return se

# ========== STEP 2-3: Wald coverage with B=400 ==========
print("=" * 80)
print("STEP 2-3: WALD COVERAGE & BOUND DIAGNOSTICS (B=400, bound=10)")
print("=" * 80)

A_TRUE, B_TRUE = 0.5, 1.2
N = 1000
B_MAIN = 400
BOUND = 10.0

EPS_MAIN = np.array([0.00, 0.10, 0.25, 0.40, 0.80, 0.95])

rng_main = np.random.default_rng(2026)

# Storage for convergent fits only (nan for diverged)
a_vals = np.full((len(EPS_MAIN), B_MAIN), np.nan)
b_vals = np.full((len(EPS_MAIN), B_MAIN), np.nan)
se_a_vals = np.full((len(EPS_MAIN), B_MAIN), np.nan)
se_b_vals = np.full((len(EPS_MAIN), B_MAIN), np.nan)

max_theta = np.zeros(len(EPS_MAIN))
n_bound_hits = np.zeros(len(EPS_MAIN), dtype=int)

for k, eps in enumerate(EPS_MAIN):
    for b in range(B_MAIN):
        X, y, yh = simulate_ab(N, A_TRUE, B_TRUE, eps, eps, rng_main)

        # Naive fit for initialization
        ab_n, _ = fit_naive(X, yh)
        if np.any(np.isnan(ab_n)):
            continue

        # Corrected fit (bounded)
        start = ab_n.copy() if eps < 0.5 else -ab_n.copy()
        rc = fit_corr(X, yh, eps, eps, start, bounded=True, bound=BOUND)

        # Check if hit bound: if so, count it and skip storage
        if np.max(np.abs(rc.x)) >= BOUND - 0.1:
            n_bound_hits[k] += 1
            continue

        # Store convergent fit
        a_vals[k, b] = rc.x[0]
        b_vals[k, b] = rc.x[1]
        max_theta[k] = max(max_theta[k], np.max(np.abs(rc.x)))

        # Compute Hessian-based SE
        se = se_hess(rc.x, X, yh, eps, eps)
        se_a_vals[k, b] = se[0]
        se_b_vals[k, b] = se[1]

# Report results
print("\neps  | conv_frac | mean_a  | mean_b  | cov_a | cov_b | width_a | width_b | max_|θ̂| | n_bound")
print("-" * 100)

for k, eps in enumerate(EPS_MAIN):
    conv_frac = np.mean(~np.isnan(a_vals[k, :]))

    # Convergent fits only
    a_conv = a_vals[k, ~np.isnan(a_vals[k, :])]
    b_conv = b_vals[k, ~np.isnan(b_vals[k, :])]
    se_a_conv = se_a_vals[k, ~np.isnan(se_a_vals[k, :])]
    se_b_conv = se_b_vals[k, ~np.isnan(se_b_vals[k, :])]

    if len(a_conv) == 0:
        print(f"{eps:.2f} | {0:.3f}     | N/A     | N/A     | N/A   | N/A   | N/A     | N/A     | N/A      | {n_bound_hits[k]}")
        continue

    mean_a = np.mean(a_conv)
    mean_b = np.mean(b_conv)
    mean_se_a = np.mean(se_a_conv)
    mean_se_b = np.mean(se_b_conv)

    # Wald coverage: check if a_true / b_true in [est ± 1.96*se]
    ci_a_lo = a_conv - 1.96 * se_a_conv
    ci_a_hi = a_conv + 1.96 * se_a_conv
    cov_a = np.mean((A_TRUE >= ci_a_lo) & (A_TRUE <= ci_a_hi))

    ci_b_lo = b_conv - 1.96 * se_b_conv
    ci_b_hi = b_conv + 1.96 * se_b_conv
    cov_b = np.mean((B_TRUE >= ci_b_lo) & (B_TRUE <= ci_b_hi))

    width_a = np.mean(ci_a_hi - ci_a_lo)
    width_b = np.mean(ci_b_hi - ci_b_lo)

    print(f"{eps:.2f} | {conv_frac:.3f}     | {mean_a:7.4f} | {mean_b:7.4f} | {cov_a:.3f} | {cov_b:.3f} | {width_a:7.4f} | {width_b:7.4f} | {max_theta[k]:8.4f} | {n_bound_hits[k]}")

# ========== STEP 4: Sanity check with B=60, dense eps grid ==========
print("\n" + "=" * 80)
print("STEP 4: SANITY CHECK - CORRECTED MEAN TRACKS TRUTH (B=60, dense eps)")
print("=" * 80)

EPS_DENSE = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.48, 0.52, 0.6, 0.8, 0.95])
B_DENSE = 60

a_dense = np.full((len(EPS_DENSE), B_DENSE), np.nan)
b_dense = np.full((len(EPS_DENSE), B_DENSE), np.nan)

rng_dense = np.random.default_rng(2027)

for k, eps in enumerate(EPS_DENSE):
    for b in range(B_DENSE):
        X, y, yh = simulate_ab(N, A_TRUE, B_TRUE, eps, eps, rng_dense)

        ab_n, _ = fit_naive(X, yh)
        if np.any(np.isnan(ab_n)):
            continue

        start = ab_n.copy() if eps < 0.5 else -ab_n.copy()
        rc = fit_corr(X, yh, eps, eps, start, bounded=True, bound=BOUND)

        # Drop bound hits
        if np.max(np.abs(rc.x)) >= BOUND - 0.1:
            continue

        a_dense[k, b] = rc.x[0]
        b_dense[k, b] = rc.x[1]

print("\neps  | conv_frac | mean_a  | sd_a   | mean_b  | sd_b")
print("-" * 60)

for k, eps in enumerate(EPS_DENSE):
    conv_frac = np.mean(~np.isnan(a_dense[k, :]))
    a_conv = a_dense[k, ~np.isnan(a_dense[k, :])]
    b_conv = b_dense[k, ~np.isnan(b_dense[k, :])]

    if len(a_conv) == 0:
        print(f"{eps:.2f} | {0:.3f}     | N/A     | N/A    | N/A     | N/A")
        continue

    mean_a = np.mean(a_conv)
    sd_a = np.std(a_conv, ddof=1)
    mean_b = np.mean(b_conv)
    sd_b = np.std(b_conv, ddof=1)

    print(f"{eps:.2f} | {conv_frac:.3f}     | {mean_a:7.4f} | {sd_a:6.4f} | {mean_b:7.4f} | {sd_b:6.4f}")

print("\nDone.")
