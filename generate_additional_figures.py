"""
Produce additional report figures that promote previously notebook-only
plots into the report and appendix.

Figures saved to BOTH STK-MAT2011/figures/ and MAT-STK2011-Project/figures/:

  fig_nocov_sd.pdf
      Plot 14 from the inventory.  No-covariate experiment: MC sd of the
      moment-inversion estimator p_hat as a function of eps, against the
      closed-form sqrt(p*(1-p*))/(c sqrt(n)) overlay.  Validates the
      Section 5.1 closed form numerically.

  fig_sim_multicov_sd.pdf
      Plot 15.  Ten-coefficient synthetic design with beta_j* varying
      from 0.1 to 3; left panel absolute MC sd, right panel relative
      inflation sd(eps)/sd(0) with 1/c theoretical overlay.  Direct
      empirical support for the "stronger predictors inflate faster
      than 1/|c|" prose in Section 6.2.

  fig_per_x_fisher_weight.pdf
      Plot 28.  Per-observation flip rate eps_i vs x_3 (left) and the
      per-observation Fisher weight w_i in the per-i vs constant-bar(eps)
      models (right), at the gamma_1 = 0.8 setup of Section 7.2.
      Visual mechanism for why the constant approximation under-reports
      its variance on the noise-driving coefficient.

  fig_shrinkage_rate_validation.pdf
      Plot 21.  Empirical local rate r_emp from a Monte-Carlo experiment
      versus the closed-form r_j of equation (eq:rj), plotted against
      the y = x identity line.  Numerical validation of the corrected
      population-level shrinkage rate formula.

  fig_per_x_which_coord.pdf
      Plot 27.  Four-panel bar plot showing the bias of each estimator
      (naive / corrected-const / corrected-per-i) as the choice of
      noise-driving covariate rotates through x_1, x_2, x_3, x_4.  The
      surgical-bias pattern: constant-bar(eps) bias lands on whichever
      x_j drives the noise.

Run: python3 generate_additional_figures.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from helper_functions.corrected_mle import (
    SEED, H, fit_naive, fit_corr, flip_labels,
)
from helper_functions.styling import set_latex_plot_style

set_latex_plot_style(use_tex=False, figure_size=(6.0, 3.6))


HERE = Path(__file__).parent.resolve()
OUT_LOCAL   = HERE / "figures"
OUT_PROJECT = (HERE.parent / "MAT-STK2011-Project" / "figures").resolve()
OUT_LOCAL.mkdir(exist_ok=True)
if OUT_PROJECT.parent.exists():
    OUT_PROJECT.mkdir(exist_ok=True)


def save(fig, name: str) -> None:
    out_local = OUT_LOCAL / name
    fig.savefig(out_local, bbox_inches="tight")
    print(f"  wrote {out_local}")
    if OUT_PROJECT.parent.exists():
        out_proj = OUT_PROJECT / name
        fig.savefig(out_proj, bbox_inches="tight")
        print(f"  wrote {out_proj}")


# =======================================================================
# 1. fig_nocov_sd.pdf -- no-covariate sd validation
# =======================================================================
def make_nocov_sd_figure() -> None:
    print("Generating fig_nocov_sd.pdf ...")
    p_true = 0.30
    n = 1000
    B = 2000
    eps_grid = np.linspace(0.0, 0.45, 31)

    sd_emp = np.zeros(len(eps_grid))
    rng = np.random.default_rng(SEED)
    for k, eps in enumerate(eps_grid):
        delta = eps   # symmetric flips
        c = 1.0 - eps - delta
        # Each replicate: draw n y_i ~ Bern(p), flip to noisy yh, invert.
        ys = rng.binomial(1, p_true, size=(B, n))
        flips_pos = (ys == 1) & (rng.uniform(size=ys.shape) < eps)
        flips_neg = (ys == 0) & (rng.uniform(size=ys.shape) < delta)
        yh = ys.copy()
        yh[flips_pos] = 0
        yh[flips_neg] = 1
        ybar = yh.mean(axis=1)
        # Moment inversion p_hat = (ybar - delta) / c  (handles c=0 edge case)
        if abs(c) < 1e-12:
            sd_emp[k] = np.nan
        else:
            phat = (ybar - delta) / c
            sd_emp[k] = phat.std(ddof=1)

    # Closed form: sd = sqrt(p*(1-p*))/(|c| sqrt(n))   with p* = delta + c*p
    p_star = eps_grid + (1.0 - 2 * eps_grid) * p_true
    c_grid = 1.0 - 2 * eps_grid
    sd_theory = np.sqrt(p_star * (1.0 - p_star)) / (np.abs(c_grid) * np.sqrt(n))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    axes[0].plot(eps_grid, sd_emp, "o", color="#1f77b4", markersize=4,
                 label=r"MC sd of $\widehat p$")
    axes[0].plot(eps_grid, sd_theory, "-", color="#ff7f0e",
                 label=r"$\sqrt{p^*(1-p^*)} \,/\, (|c|\sqrt{n})$")
    axes[0].set_xlabel(r"$\varepsilon = \delta$")
    axes[0].set_ylabel(r"sd of $\widehat p$")
    axes[0].set_title(r"No covariate: sd grows like $1/|c|$")
    axes[0].legend(fontsize=9)

    axes[1].semilogy(eps_grid, sd_emp, "o", color="#1f77b4", markersize=4,
                     label="MC sd")
    axes[1].semilogy(eps_grid, sd_theory, "-", color="#ff7f0e",
                     label="theory")
    axes[1].set_xlabel(r"$\varepsilon = \delta$")
    axes[1].set_ylabel(r"sd of $\widehat p$ (log scale)")
    axes[1].set_title("Same plot, log $y$-axis")
    axes[1].legend(fontsize=9)

    fig.suptitle(rf"No-covariate sd validation: $p = {p_true}$, $n = {n}$, "
                 f"$B = {B}$ replicates per $\\varepsilon$", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig_nocov_sd.pdf")
    plt.close(fig)


# =======================================================================
# 2. fig_sim_multicov_sd.pdf -- multi-covariate sd inflation
# =======================================================================
def make_multicov_sd_figure() -> None:
    print("Generating fig_sim_multicov_sd.pdf ...")
    n = 1000
    beta_star = np.array([3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0])
    p = len(beta_star)
    eps_grid = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    B = 120

    rng_design = np.random.default_rng(SEED)
    X = rng_design.standard_normal((n, p))
    p_true_X = H(X @ beta_star)

    betas = np.full((len(eps_grid), B, p + 1), np.nan)
    for k, eps in enumerate(eps_grid):
        for b in range(B):
            rng_b = np.random.default_rng(SEED + 100_000 * b + k)
            y_b = rng_b.binomial(1, p_true_X)
            yh = flip_labels(y_b, eps, rng=rng_b)
            try:
                Xd = sm.add_constant(X, has_constant="add")
                naive = sm.GLM(yh, Xd, family=sm.families.Binomial()).fit(disp=0)
                start = np.asarray(naive.params)
            except Exception:
                start = np.zeros(p + 1)
            res = fit_corr(X, yh, eps, eps, start)
            if np.max(np.abs(res.x)) >= 14.9:
                continue
            betas[k, b] = res.x
    sd_mat = np.nanstd(betas, axis=1, ddof=1)   # (n_eps, p+1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.0, 0.95, p))

    for j in range(p):
        axes[0].plot(eps_grid, sd_mat[:, j + 1], "o-", color=colors[j],
                     markersize=5, linewidth=1,
                     label=fr"$\beta_{{{j+1}}}^\star = {beta_star[j]:.2g}$")
    axes[0].set_xlabel(r"$\varepsilon$")
    axes[0].set_ylabel(r"MC sd of $\widehat\beta_j$")
    axes[0].set_title("Absolute MC standard deviation")
    axes[0].legend(fontsize=7, ncol=2, loc="upper left")

    inv_c = 1.0 / (1.0 - 2 * eps_grid)
    for j in range(p):
        sd0 = sd_mat[0, j + 1]
        if sd0 <= 1e-12 or np.isnan(sd0):
            continue
        axes[1].plot(eps_grid, sd_mat[:, j + 1] / sd0, "o-",
                     color=colors[j], markersize=5, linewidth=1,
                     label=fr"$\beta_{{{j+1}}}^\star = {beta_star[j]:.2g}$")
    axes[1].plot(eps_grid, inv_c, "k--", linewidth=2,
                 label=r"$1/c = 1/(1-2\varepsilon)$ (theory floor)")
    axes[1].set_xlabel(r"$\varepsilon$")
    axes[1].set_ylabel(r"$\mathrm{sd}(\widehat\beta_j(\varepsilon)) / \mathrm{sd}(\widehat\beta_j(0))$")
    axes[1].set_title(r"Relative inflation: $1/|c|$ is a lower bound")
    axes[1].legend(fontsize=7, ncol=2, loc="upper left")

    fig.suptitle(rf"Multi-covariate synthetic, $n = {n}$, "
                 rf"$B = {B}$ replicates per $\varepsilon$",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig_sim_multicov_sd.pdf")
    plt.close(fig)


# =======================================================================
# 3. fig_per_x_fisher_weight.pdf -- per-observation Fisher weight
# =======================================================================
def make_fisher_weight_figure() -> None:
    print("Generating fig_per_x_fisher_weight.pdf ...")
    n = 2000
    beta_star = np.array([-1.5, 1.2, 0.6, -0.3])
    alpha_star = -1.0
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n, len(beta_star)))
    p_true = H(alpha_star + X @ beta_star)
    del_const = np.full(n, 0.05)

    # Match Section 7.2 setup: gamma_1 = 0.8 calibrated to mean eps_i = 0.15
    g_lo, g_hi = -10.0, 10.0
    for _ in range(60):
        g_mid = 0.5 * (g_lo + g_hi)
        m = H(g_mid + 0.8 * X[:, 2]).mean()
        if m > 0.15: g_hi = g_mid
        else:        g_lo = g_mid
    g0 = 0.5 * (g_lo + g_hi)
    eps_per_i = H(g0 + 0.8 * X[:, 2])
    a_per_i = 1.0 - eps_per_i - del_const
    q_per_i = del_const + a_per_i * p_true
    w_per_i = a_per_i**2 * p_true**2 * (1 - p_true)**2 / (q_per_i * (1 - q_per_i))
    a_bar = 1.0 - eps_per_i.mean() - del_const.mean()
    q_bar = del_const + a_bar * p_true
    w_bar = a_bar**2 * p_true**2 * (1 - p_true)**2 / (q_bar * (1 - q_bar))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    order = np.argsort(X[:, 2])
    axes[0].plot(X[order, 2], eps_per_i[order], "-", color="#1f77b4",
                 linewidth=1.6,
                 label=r"$\varepsilon_i = H(\gamma_0 + 0.8\, x_{i,3})$")
    axes[0].axhline(eps_per_i.mean(), color="#d62728", linestyle="--",
                    label=fr"$\bar\varepsilon = {eps_per_i.mean():.3f}$")
    axes[0].set_xlabel(r"$x_{i,3}$")
    axes[0].set_ylabel(r"$\varepsilon_i$")
    axes[0].set_title(r"Per-observation flip rate vs. $x_3$")
    axes[0].legend(fontsize=9)

    axes[1].scatter(X[:, 2], w_per_i, s=10, alpha=0.45, color="#1f77b4",
                    label=r"per-$i$ weight $a_i^2 \cdot \frac{p_i^2(1-p_i)^2}{q_i(1-q_i)}$")
    axes[1].scatter(X[:, 2], w_bar, s=10, alpha=0.45, color="#d62728",
                    label=r"const $\bar\varepsilon$ weight $\bar a^2 \cdot \frac{p_i^2(1-p_i)^2}{\bar q_i(1-\bar q_i)}$")
    axes[1].set_xlabel(r"$x_{i,3}$")
    axes[1].set_ylabel(r"Fisher weight $w_i$")
    axes[1].set_title(r"Per-observation Fisher weight at $\beta^\star$")
    axes[1].legend(fontsize=8, loc="upper right")

    fig.suptitle(r"Per-$i$ vs constant-$\bar\varepsilon$ Fisher weights "
                 r"(Section~7.2 setup, $\gamma_1 = 0.8$)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig_per_x_fisher_weight.pdf")
    plt.close(fig)


# =======================================================================
# 4. fig_shrinkage_rate_validation.pdf -- r_emp vs r_theory
# =======================================================================
def make_shrinkage_validation_figure() -> None:
    print("Generating fig_shrinkage_rate_validation.pdf ...")
    n = 4000
    beta_star = np.array([3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0])
    p = len(beta_star)
    rng_design = np.random.default_rng(SEED)
    X = rng_design.standard_normal((n, p))
    p_true_X = H(X @ beta_star)

    eps_grid = np.linspace(0.0, 0.45, 31)
    B = 80
    betas = np.zeros((len(eps_grid), p + 1))
    for k, eps in enumerate(eps_grid):
        bb = np.zeros((B, p + 1))
        for b in range(B):
            rng_b = np.random.default_rng(SEED + 10_000 * b + k)
            y_b = rng_b.binomial(1, p_true_X)
            yh = flip_labels(y_b, eps, rng=rng_b)
            params, _, _ = fit_naive(X, yh)
            bb[b] = params
        betas[k] = bb.mean(axis=0)

    # Empirical local slope on the first 3 grid points (window eps in [0, 0.03])
    def local_slope(grid, vals, n_pts=3):
        e = grid[:n_pts]
        b = np.log(np.maximum(np.abs(vals[:n_pts]), 1e-12))
        s, _ = np.polyfit(e, b, 1)
        return s

    r_emp = np.array([-local_slope(eps_grid, betas[:, j + 1]) for j in range(p)])

    # Theoretical r_j = -[(X' W X)^{-1} X' (1 - 2 p*)]_j / beta_j*
    Xd = sm.add_constant(X, has_constant="add")
    full_truth = np.r_[0.0, beta_star]
    p_star = H(Xd @ full_truth)
    W = p_star * (1 - p_star)
    XWX = Xd.T @ (Xd * W[:, None])
    rhs = Xd.T @ (1 - 2 * p_star)
    dbeta = np.linalg.solve(XWX, rhs)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_theory_full = -dbeta / np.where(np.abs(full_truth) < 1e-9, np.nan, full_truth)
    r_theory = r_theory_full[1:]   # drop intercept

    keep = beta_star != 0

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(r_theory[keep], r_emp[keep], s=80,
               color="#1f77b4", edgecolor="black", linewidth=0.6)
    lo = float(min(r_theory[keep].min(), r_emp[keep].min())) - 0.5
    hi = float(max(r_theory[keep].max(), r_emp[keep].max())) + 0.5
    ax.plot([lo, hi], [lo, hi], color="#d62728", linestyle="--",
            label=r"identity $r_{\rm emp} = r_{\rm theory}$")
    for j in np.where(keep)[0]:
        ax.annotate(fr"$\beta_{{{j+1}}}^\star = {beta_star[j]:.2g}$",
                    (r_theory[j], r_emp[j]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel(r"theoretical $r_j$ from equation~(eq:rj)")
    ax.set_ylabel(r"empirical $r_j$ from MC slope on $\varepsilon \in [0, 0.03]$")
    ax.set_title(r"Closed-form local rate matches MC slope")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_shrinkage_rate_validation.pdf")
    plt.close(fig)


# =======================================================================
# 5. fig_per_x_which_coord.pdf -- surgical bias across drivers
# =======================================================================
def make_which_coord_figure() -> None:
    print("Generating fig_per_x_which_coord.pdf ...")
    n_e = 2000
    beta_star_e = np.array([-1.5, 1.2, 0.6, -0.3])
    alpha_star_e = -1.0
    truth_e = np.r_[alpha_star_e, beta_star_e]
    n_par = len(truth_e)

    rng_e = np.random.default_rng(SEED)
    X_e = rng_e.standard_normal((n_e, len(beta_star_e)))
    p_true_e = H(alpha_star_e + X_e @ beta_star_e)
    del_const_e = np.full(n_e, 0.05)

    def calibrate_gamma0(g1, x_drive, eps_mean=0.15):
        g_lo, g_hi = -10.0, 10.0
        for _ in range(60):
            g_mid = 0.5 * (g_lo + g_hi)
            m = H(g_mid + g1 * x_drive).mean()
            if m > eps_mean: g_hi = g_mid
            else:            g_lo = g_mid
        return 0.5 * (g_lo + g_hi)

    gamma1_fixed = 1.5
    B_e = 80
    J_choices = [0, 1, 2, 3]
    results_J = {}

    for j_drive in J_choices:
        g0 = calibrate_gamma0(gamma1_fixed, X_e[:, j_drive])
        eps_vec = H(g0 + gamma1_fixed * X_e[:, j_drive])
        bias_n = np.zeros(n_par); bias_c = np.zeros(n_par); bias_f = np.zeros(n_par)
        for b in range(B_e):
            rng_b = np.random.default_rng(SEED + 60_000 + 1000 * j_drive + b)
            y_b = rng_b.binomial(1, p_true_e)
            flip1 = (y_b == 1) & (rng_b.uniform(size=n_e) < eps_vec)
            flip0 = (y_b == 0) & (rng_b.uniform(size=n_e) < del_const_e)
            yh = y_b.copy(); yh[flip1] = 0; yh[flip0] = 1
            nv, _, _ = fit_naive(X_e, yh)
            if np.any(np.isnan(nv)): nv = np.zeros(n_par)
            eps_const = np.full(n_e, eps_vec.mean())
            rc = fit_corr(X_e, yh, eps_const, del_const_e, nv.copy())
            rf = fit_corr(X_e, yh, eps_vec,    del_const_e, nv.copy())
            bias_n += nv   - truth_e
            bias_c += rc.x - truth_e
            bias_f += rf.x - truth_e
        results_J[j_drive] = (bias_n / B_e, bias_c / B_e, bias_f / B_e)

    coord_names = ["intercept", r"$\beta_1$", r"$\beta_2$", r"$\beta_3$", r"$\beta_4$"]
    xpos = np.arange(len(coord_names))

    fig, axes = plt.subplots(1, len(J_choices), figsize=(15, 3.6), sharey=True)
    for ax, j_drive in zip(axes, J_choices):
        bn, bc, bf = results_J[j_drive]
        width = 0.27
        ax.bar(xpos - width, bn, width, color="#d62728", label="naive")
        ax.bar(xpos,         bc, width, color="#ff7f0e",
               label=r"corrected, const $\bar\varepsilon$")
        ax.bar(xpos + width, bf, width, color="#1f77b4",
               label=r"corrected, per-$i$ $\varepsilon_i$")
        ax.axhline(0, color="0.85", lw=0.8)
        ax.set_xticks(xpos)
        ax.set_xticklabels(coord_names, rotation=0, fontsize=8)
        ax.set_title(rf"$\varepsilon_i = H(\gamma_0 + 1.5\, x_{{i,{j_drive+1}}})$")
        if j_drive == J_choices[0]:
            ax.set_ylabel("mean bias")
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Wherever you put the noise dependence, "
                 "the constant-$\\bar\\varepsilon$ bias goes there",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig_per_x_which_coord.pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_nocov_sd_figure()
    print()
    make_multicov_sd_figure()
    print()
    make_fisher_weight_figure()
    print()
    make_shrinkage_validation_figure()
    print()
    make_which_coord_figure()
