"""
Wald coverage and convergence diagnostics on the single-covariate
synthetic experiment (a, b) = (0.5, 1.2), n = 1000, B = 400 replicates.

This script does NOT apply a bound-hit filter at the replicate level.
Instead, for each replicate it stores the bounded-MLE point estimate and
flags each coordinate's bound-hit status separately.  Wald inference is
performed per-coordinate using the submatrix Hessian on the unbounded
coordinates: a coordinate at the bound has no Wald interval (NaN), and
the corresponding coverage rate is averaged only over the replicates on
which that coordinate was interior.  The interior-fit fraction
\widehat p_conv is therefore reported per-coordinate as well.

This matches the per-coordinate convention used by the figures in
Sections 5.2 and 5.3 of the report.

Run: python3 wald_coverage_bounded.py
"""

from __future__ import annotations

import numpy as np

from helper_functions.corrected_mle import (
    SEED, H, fit_naive, fit_corr, hessian_se, flip_labels,
)


def simulate_ab(n: int, a: float, b: float, eps: float, delta: float,
                rng: np.random.Generator):
    x = rng.standard_normal(n)
    y = rng.binomial(1, H(a + b * x))
    yh = flip_labels(y, eps, delta, rng=rng)
    return x, y, yh


def main() -> None:
    print("=" * 80)
    print("WALD COVERAGE  (B = 400, bound = 10, no replicate-level filter)")
    print("=" * 80)

    A_TRUE, B_TRUE = 0.5, 1.2
    N = 1000
    B_MAIN = 400
    BOUND = 10.0
    TAU = 0.1

    eps_main = np.array([0.00, 0.10, 0.25, 0.40, 0.80, 0.95])
    rng_main = np.random.default_rng(SEED)

    # Per-coordinate storage. We keep the point estimate on every
    # replicate, including bound-hits; we keep the Wald se only when the
    # coordinate is interior (NaN otherwise).
    a_vals    = np.full((len(eps_main), B_MAIN), np.nan)
    b_vals    = np.full((len(eps_main), B_MAIN), np.nan)
    a_se      = np.full((len(eps_main), B_MAIN), np.nan)
    b_se      = np.full((len(eps_main), B_MAIN), np.nan)
    on_bnd_a  = np.zeros((len(eps_main), B_MAIN), dtype=bool)
    on_bnd_b  = np.zeros((len(eps_main), B_MAIN), dtype=bool)

    for k, eps in enumerate(eps_main):
        for b in range(B_MAIN):
            x, _, yh = simulate_ab(N, A_TRUE, B_TRUE, eps, eps, rng_main)

            ab_n, _, _ = fit_naive(x.reshape(-1, 1), yh)
            if np.any(np.isnan(ab_n)):
                ab_n = np.zeros(2)
            start = ab_n.copy() if eps < 0.5 else -ab_n.copy()

            res = fit_corr(x.reshape(-1, 1), yh, eps, eps,
                           start=start, bound=BOUND)
            a_vals[k, b], b_vals[k, b] = res.x
            on_bnd_a[k, b] = abs(res.x[0]) >= BOUND - TAU
            on_bnd_b[k, b] = abs(res.x[1]) >= BOUND - TAU

            try:
                on_bnd = np.array([on_bnd_a[k, b], on_bnd_b[k, b]])
                se = hessian_se(res.x, x.reshape(-1, 1), yh, eps, eps,
                                on_bound=on_bnd)
                a_se[k, b], b_se[k, b] = se   # NaN on the bound-hit coord
            except Exception:
                pass

    # Report per-coordinate convergence and coverage.
    print()
    print(f"{'eps':>5}  {'p_conv(a)':>10}  {'p_conv(b)':>10}  "
          f"{'cov(a)':>7}  {'cov(b)':>7}  "
          f"{'width(a)':>9}  {'width(b)':>9}")
    print("-" * 80)

    for k, eps in enumerate(eps_main):
        # Interior masks per coordinate
        keep_a = ~on_bnd_a[k] & ~np.isnan(a_se[k])
        keep_b = ~on_bnd_b[k] & ~np.isnan(b_se[k])
        p_conv_a = keep_a.mean()
        p_conv_b = keep_b.mean()

        if keep_a.any():
            ci_lo = a_vals[k, keep_a] - 1.96 * a_se[k, keep_a]
            ci_hi = a_vals[k, keep_a] + 1.96 * a_se[k, keep_a]
            cov_a = float(np.mean((A_TRUE >= ci_lo) & (A_TRUE <= ci_hi)))
            width_a = float((ci_hi - ci_lo).mean())
        else:
            cov_a, width_a = float("nan"), float("nan")

        if keep_b.any():
            ci_lo = b_vals[k, keep_b] - 1.96 * b_se[k, keep_b]
            ci_hi = b_vals[k, keep_b] + 1.96 * b_se[k, keep_b]
            cov_b = float(np.mean((B_TRUE >= ci_lo) & (B_TRUE <= ci_hi)))
            width_b = float((ci_hi - ci_lo).mean())
        else:
            cov_b, width_b = float("nan"), float("nan")

        print(f"{eps:>5.2f}  {p_conv_a:>10.3f}  {p_conv_b:>10.3f}  "
              f"{cov_a:>7.3f}  {cov_b:>7.3f}  "
              f"{width_a:>9.3f}  {width_b:>9.3f}")

    print()
    print("Bound-hit fractions (intercept, slope) per eps:")
    for k, eps in enumerate(eps_main):
        print(f"  eps = {eps:.2f}  hit(a) = {on_bnd_a[k].mean():.3f}, "
              f"hit(b) = {on_bnd_b[k].mean():.3f}")


if __name__ == "__main__":
    main()
