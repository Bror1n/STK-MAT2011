"""
Produce report-ready PDFs from the per-x epsilon exploration in
per_x_epsilon.ipynb (Section 6).  Two figures:

  fig_per_x_spread.pdf       -- bias and sd of beta_hat as a function of
                                sd(eps_i), at fixed E[eps_i] = 0.15.
                                Side-by-side: beta_3 (drives noise) and
                                beta_1 (does not).
  fig_per_x_identification.pdf -- bound-hit fraction vs. eps_high under a
                                  two-population design, parameterised
                                  by the size of the un-poisoned subset.

Both use B_e = 80, B_id = 60 to keep the runtime moderate while still
producing visibly low-noise curves on the figure.

Run: python3 generate_per_x_exploration_figures.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from helper_functions.corrected_mle import (
    SEED, H, fit_naive, fit_corr,
)


# Output destinations: STK-MAT2011/figures/ + MAT-STK2011-Project/figures/
HERE = Path(__file__).parent.resolve()
OUT_LOCAL   = HERE / "figures"
OUT_PROJECT = (HERE.parent / "MAT-STK2011-Project" / "figures").resolve()
OUT_LOCAL.mkdir(exist_ok=True)
if OUT_PROJECT.parent.exists():
    OUT_PROJECT.mkdir(exist_ok=True)


# -----------------------------------------------------------------------
# Shared design (Section 3 of per_x_epsilon.ipynb)
# -----------------------------------------------------------------------
N_E = 2000
BETA_STAR_E = np.array([-1.5, 1.2, 0.6, -0.3])
ALPHA_STAR_E = -1.0
TRUTH_E = np.r_[ALPHA_STAR_E, BETA_STAR_E]

rng_design = np.random.default_rng(SEED)
X_E = rng_design.standard_normal((N_E, len(BETA_STAR_E)))
P_TRUE_E = H(ALPHA_STAR_E + X_E @ BETA_STAR_E)
DEL_CONST_E = np.full(N_E, 0.05)


def calibrate_gamma0(gamma1: float, x_drive: np.ndarray,
                     eps_mean: float = 0.15) -> float:
    """Bisect to find gamma_0 so that mean(H(g0 + g1 x)) = eps_mean."""
    g_lo, g_hi = -10.0, 10.0
    for _ in range(60):
        g_mid = 0.5 * (g_lo + g_hi)
        m = H(g_mid + gamma1 * x_drive).mean()
        if m > eps_mean:
            g_hi = g_mid
        else:
            g_lo = g_mid
    return 0.5 * (g_lo + g_hi)


# =======================================================================
# Figure 1: spread of eps_i vs bias and sd of beta_hat (Section 6.1)
# =======================================================================
def make_spread_figure() -> None:
    gamma1_grid = np.linspace(0.0, 3.0, 13)
    B_e = 80
    n_par = len(TRUTH_E)

    bias_naive = np.zeros((len(gamma1_grid), n_par))
    bias_const = np.zeros((len(gamma1_grid), n_par))
    bias_full  = np.zeros((len(gamma1_grid), n_par))
    sd_naive   = np.zeros((len(gamma1_grid), n_par))
    sd_const   = np.zeros((len(gamma1_grid), n_par))
    sd_full    = np.zeros((len(gamma1_grid), n_par))
    var_eps    = np.zeros(len(gamma1_grid))

    for k, g1 in enumerate(gamma1_grid):
        g0 = calibrate_gamma0(g1, X_E[:, 2], eps_mean=0.15)
        eps_vec_k = H(g0 + g1 * X_E[:, 2])
        var_eps[k] = float(np.var(eps_vec_k))

        bn = np.zeros((B_e, n_par))
        bc = np.zeros((B_e, n_par))
        bf = np.zeros((B_e, n_par))
        for b in range(B_e):
            rng_b = np.random.default_rng(SEED + 50_000 + 1000 * k + b)
            y_b = rng_b.binomial(1, P_TRUE_E)
            flip1 = (y_b == 1) & (rng_b.uniform(size=N_E) < eps_vec_k)
            flip0 = (y_b == 0) & (rng_b.uniform(size=N_E) < DEL_CONST_E)
            yh = y_b.copy(); yh[flip1] = 0; yh[flip0] = 1
            nv, _, _ = fit_naive(X_E, yh)
            if np.any(np.isnan(nv)): nv = np.zeros(n_par)
            eps_const_e = np.full(N_E, eps_vec_k.mean())
            rc = fit_corr(X_E, yh, eps_const_e, DEL_CONST_E, nv.copy())
            rf = fit_corr(X_E, yh, eps_vec_k,    DEL_CONST_E, nv.copy())
            bn[b] = nv
            bc[b] = rc.x
            bf[b] = rf.x

        bias_naive[k] = bn.mean(0) - TRUTH_E
        bias_const[k] = bc.mean(0) - TRUTH_E
        bias_full[k]  = bf.mean(0) - TRUTH_E
        sd_naive[k]   = bn.std(0, ddof=1)
        sd_const[k]   = bc.std(0, ddof=1)
        sd_full[k]    = bf.std(0, ddof=1)
        print(f"  gamma1 = {g1:.2f}  sd(eps_i) = {np.sqrt(var_eps[k]):.3f}  "
              f"|bias_const_3| = {abs(bias_const[k, 3]):.3f}", flush=True)

    sd_eps_grid = np.sqrt(var_eps)
    J_DRIVE = 3
    J_NONDRIVE = 1

    # 2 x 2 figure: top row = bias, bottom row = sd
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

    # Top-left: bias of beta_3 (drives noise)
    ax = axes[0, 0]
    ax.plot(sd_eps_grid, bias_naive[:, J_DRIVE], "o-", color="#d62728",
            label="naive")
    ax.plot(sd_eps_grid, bias_const[:, J_DRIVE], "o-", color="#ff7f0e",
            label=r"corrected, const $\bar\varepsilon$")
    ax.plot(sd_eps_grid, bias_full[:, J_DRIVE], "o-", color="#1f77b4",
            label=r"corrected, per-$i$ $\varepsilon_i$")
    ax.axhline(0, color="0.85", lw=0.8)
    ax.set_ylabel(r"mean bias of $\widehat\beta_3$")
    ax.set_title(r"$\beta_3^\star = 0.6$ (drives $\varepsilon_i$)")
    ax.legend(fontsize=8, loc="upper right")

    # Top-right: bias of beta_1 (does not drive noise)
    ax = axes[0, 1]
    ax.plot(sd_eps_grid, bias_naive[:, J_NONDRIVE], "o-", color="#d62728",
            label="naive")
    ax.plot(sd_eps_grid, bias_const[:, J_NONDRIVE], "o-", color="#ff7f0e",
            label=r"corrected, const $\bar\varepsilon$")
    ax.plot(sd_eps_grid, bias_full[:, J_NONDRIVE], "o-", color="#1f77b4",
            label=r"corrected, per-$i$ $\varepsilon_i$")
    ax.axhline(0, color="0.85", lw=0.8)
    ax.set_ylabel(r"mean bias of $\widehat\beta_1$")
    ax.set_title(r"$\beta_1^\star = -1.5$ (does not drive $\varepsilon_i$)")

    # Bottom-left: sd of beta_3
    ax = axes[1, 0]
    ax.plot(sd_eps_grid, sd_naive[:, J_DRIVE], "o-", color="#d62728")
    ax.plot(sd_eps_grid, sd_const[:, J_DRIVE], "o-", color="#ff7f0e")
    ax.plot(sd_eps_grid, sd_full[:, J_DRIVE],  "o-", color="#1f77b4")
    ax.set_xlabel(r"sd of $\varepsilon_i$ across observations")
    ax.set_ylabel(r"MC sd of $\widehat\beta_3$")

    # Bottom-right: sd of beta_1
    ax = axes[1, 1]
    ax.plot(sd_eps_grid, sd_naive[:, J_NONDRIVE], "o-", color="#d62728")
    ax.plot(sd_eps_grid, sd_const[:, J_NONDRIVE], "o-", color="#ff7f0e")
    ax.plot(sd_eps_grid, sd_full[:, J_NONDRIVE],  "o-", color="#1f77b4")
    ax.set_xlabel(r"sd of $\varepsilon_i$ across observations")
    ax.set_ylabel(r"MC sd of $\widehat\beta_1$")

    fig.suptitle(
        r"Spread of $\varepsilon_i$ vs. bias (top) and sd (bottom) of $\widehat\beta$.  "
        r"$\mathbb{E}\varepsilon_i \equiv 0.15$ held fixed.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_local = OUT_LOCAL / "fig_per_x_spread.pdf"
    fig.savefig(out_local, bbox_inches="tight")
    print(f"  wrote {out_local}")
    if OUT_PROJECT.parent.exists():
        out_proj = OUT_PROJECT / "fig_per_x_spread.pdf"
        fig.savefig(out_proj, bbox_inches="tight")
        print(f"  wrote {out_proj}")
    plt.close(fig)


# =======================================================================
# Figure 2: identification limits at high spread (Section 6.3)
# =======================================================================
def make_identification_figure() -> None:
    eps_low_id = 0.05
    eps_high_grid = np.array([0.30, 0.50, 0.70, 0.85, 0.90, 0.92, 0.94, 0.948])
    f_grid_id = [1.00, 0.99, 0.95, 0.50]
    B_id = 60
    SIM_BOUND_E = 15.0
    TAU_E = 0.1
    n_par = len(TRUTH_E)

    order_x3 = np.argsort(-X_E[:, 2])

    results = {}
    for f_h in f_grid_id:
        biases = np.zeros((len(eps_high_grid), n_par))
        hits = np.zeros(len(eps_high_grid))
        mean_a = np.zeros(len(eps_high_grid))
        for k, eps_high in enumerate(eps_high_grid):
            n_high = int(round(f_h * N_E))
            eps_vec_k = np.full(N_E, eps_low_id)
            if n_high > 0:
                eps_vec_k[order_x3[:n_high]] = eps_high
            mean_a[k] = (1.0 - eps_vec_k - DEL_CONST_E).mean()

            on_bnd_count = 0
            bias_acc = np.zeros(n_par)
            n_kept = 0
            for b in range(B_id):
                rng_b = np.random.default_rng(
                    SEED + 70_000 + 1000 * int(f_h * 100) + 100 * k + b)
                y_b = rng_b.binomial(1, P_TRUE_E)
                flip1 = (y_b == 1) & (rng_b.uniform(size=N_E) < eps_vec_k)
                flip0 = (y_b == 0) & (rng_b.uniform(size=N_E) < DEL_CONST_E)
                yh = y_b.copy(); yh[flip1] = 0; yh[flip0] = 1
                nv, _, _ = fit_naive(X_E, yh)
                if np.any(np.isnan(nv)): nv = np.zeros(n_par)
                rf = fit_corr(X_E, yh, eps_vec_k, DEL_CONST_E, nv.copy(),
                              bound=SIM_BOUND_E)
                if np.max(np.abs(rf.x)) >= SIM_BOUND_E - TAU_E:
                    on_bnd_count += 1
                else:
                    bias_acc += rf.x - TRUTH_E
                    n_kept += 1
            biases[k] = bias_acc / max(n_kept, 1)
            hits[k] = on_bnd_count / B_id
        results[f_h] = (biases, hits, mean_a)
        print(f"  f = {f_h:.2f}  hits at eps_high=0.94 = "
              f"{results[f_h][1][-2]:.2f}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
    for ix, f_h in enumerate(f_grid_id):
        _, hits, _ = results[f_h]
        n_clean = int(round((1 - f_h) * N_E))
        axes[0].plot(eps_high_grid, hits, "o-", color=colors[ix],
                     label=fr"$f = {f_h:.2f}$ ($n_{{\rm clean}} = {n_clean}$)")
    axes[0].axvline(1.0 - 0.05, color="black", linestyle="--",
                    label=r"per-$i$ pole at $\varepsilon_i = 0.95$")
    axes[0].set_xlabel(r"$\varepsilon_{\rm high}$")
    axes[0].set_ylabel("bound-hit fraction across replicates")
    axes[0].set_title(r"Identification limit as $\varepsilon_{\rm high} \to$ pole")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=8, loc="upper left")

    biases, hits, _ = results[1.0]
    coord_labels = ["intercept"] + [fr"$\beta_{{{j}}}$" for j in range(1, n_par)]
    for j in range(n_par):
        axes[1].plot(eps_high_grid, biases[:, j], "o-", label=coord_labels[j])
    axes[1].axhline(0, color="0.85", lw=0.8)
    axes[1].set_xlabel(r"$\varepsilon_{\rm high}$ (no clean subset, $f = 1.0$)")
    axes[1].set_ylabel("mean bias (interior fits only)")
    axes[1].set_title("Per-$i$ corrected bias on interior fits")
    axes[1].legend(fontsize=8, ncol=2)

    fig.suptitle(
        r"Two-population design: $\varepsilon_i = \varepsilon_{\rm high}$ on a fraction "
        r"$f$ of observations, $\varepsilon_i = 0.05$ otherwise; $\delta = 0.05$ throughout.",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_local = OUT_LOCAL / "fig_per_x_identification.pdf"
    fig.savefig(out_local, bbox_inches="tight")
    print(f"  wrote {out_local}")
    if OUT_PROJECT.parent.exists():
        out_proj = OUT_PROJECT / "fig_per_x_identification.pdf"
        fig.savefig(out_proj, bbox_inches="tight")
        print(f"  wrote {out_proj}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating fig_per_x_spread.pdf ...")
    make_spread_figure()
    print()
    print("Generating fig_per_x_identification.pdf ...")
    make_identification_figure()
