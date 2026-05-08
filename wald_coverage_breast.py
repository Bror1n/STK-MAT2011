"""
Per-coordinate Wald coverage on the breast-cancer training fold.

Sweeps a small grid of (eps, delta) scenarios; at each (eps, delta) and
each Monte-Carlo replicate, fits the bounded corrected MLE and computes
per-coordinate Wald intervals using the submatrix Hessian on the
unbounded coordinates.  No replicate-level filter is applied: a
coordinate at the bound contributes NaN to its own Wald se and does not
enter that coordinate's coverage average.  Per-coordinate interior-fit
fractions are reported.

Run: python3 wald_coverage_breast.py
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from helper_functions.corrected_mle import (
    SEED, fit_naive, fit_corr, hessian_se, flip_labels,
)


def main() -> None:
    print("=" * 80)
    print("BREAST-CANCER WALD COVERAGE  (B = 200, bound = 15, no replicate-level filter)")
    print("=" * 80)

    data = load_breast_cancer()
    X_train, _, y_train, _ = train_test_split(
        data.data, data.target, test_size=171, random_state=SEED,
        stratify=data.target,
    )
    keep = ["mean radius", "mean texture", "mean smoothness", "mean concave points"]
    idx = np.array([list(data.feature_names).index(k) for k in keep])
    X_tr = X_train[:, idx]
    mu = X_tr.mean(axis=0)
    sd_x = X_tr.std(axis=0, ddof=1)
    X_tr = (X_tr - mu) / sd_x
    n_tr, p = X_tr.shape
    P = p + 1   # plus intercept

    # Clean-labels target -- treated as the truth in this comparison
    theta_clean, _, _ = fit_naive(X_tr, y_train)

    eps_grid = np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.80, 0.95])
    B = 200
    BOUND = 15.0
    TAU = 0.1

    coord_names = ["intercept", "radius", "texture", "smoothness", "concave"]

    # Per-coordinate storage
    fits     = np.full((len(eps_grid), B, P), np.nan)
    ses      = np.full((len(eps_grid), B, P), np.nan)
    on_bnd   = np.zeros((len(eps_grid), B, P), dtype=bool)

    rng = np.random.default_rng(SEED + 1)
    for k, eps in enumerate(eps_grid):
        for b in range(B):
            yh = flip_labels(y_train, eps, eps, rng=rng)
            ab_n, _, _ = fit_naive(X_tr, yh)
            if np.any(np.isnan(ab_n)):
                ab_n = np.zeros(P)
            start = ab_n.copy() if eps < 0.5 else -ab_n.copy()
            res = fit_corr(X_tr, yh, eps, eps, start, bound=BOUND)
            fits[k, b]   = res.x
            on_bnd[k, b] = np.abs(res.x) >= BOUND - TAU
            try:
                se = hessian_se(res.x, X_tr, yh, eps, eps,
                                on_bound=on_bnd[k, b])
                ses[k, b] = se
            except Exception:
                pass

    # Per-coordinate coverage
    print()
    header = f"{'eps':>5}  " + "  ".join(f"{nm:>10}" for nm in coord_names)
    print("Per-coordinate p_conv:")
    print(header)
    for k, eps in enumerate(eps_grid):
        p_conv_per_coord = (~on_bnd[k]).mean(axis=0)
        row = "  ".join(f"{v:>10.3f}" for v in p_conv_per_coord)
        print(f"{eps:>5.2f}  {row}")

    print()
    print("Per-coordinate Wald coverage of clean-labels target:")
    print(header)
    for k, eps in enumerate(eps_grid):
        cov = np.full(P, np.nan)
        for j in range(P):
            keep_j = ~on_bnd[k, :, j] & ~np.isnan(ses[k, :, j])
            if not keep_j.any():
                continue
            ci_lo = fits[k, keep_j, j] - 1.96 * ses[k, keep_j, j]
            ci_hi = fits[k, keep_j, j] + 1.96 * ses[k, keep_j, j]
            cov[j] = float(((theta_clean[j] >= ci_lo) & (theta_clean[j] <= ci_hi)).mean())
        row = "  ".join(f"{v:>10.3f}" for v in cov)
        print(f"{eps:>5.2f}  {row}")


if __name__ == "__main__":
    main()
