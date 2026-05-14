"""Regenerate figures for the MAT-STK2011 report.

Produces PDFs in MAT-STK2011-Project/figures/ :

    fig_sim_attenuation.pdf      - naive attenuation, full eps range
    fig_sim_corrected.pdf        - corrected MLE, full eps range
    fig_sim_sd_growth.pdf        - sd inflation and Hessian-se agreement
    fig_breast_paths.pdf         - breast-cancer coefficient paths (naive/corrected)
    fig_breast_overlay.pdf       - focal predictor naive vs corrected
    fig_breast_pvalues.pdf       - Wald p-values per coefficient vs epsilon
"""

from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# ---- styling (re-use the project helper) ----
HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
from helper_functions.styling import set_latex_plot_style, save_latex_figure
from helper_functions.corrected_mle import SEED  # 6114, used everywhere

set_latex_plot_style(use_tex=False, figure_size=(6.0, 3.6))

# Figures land next to the LaTeX source: ../MAT-STK2011-Project/figures.
# Falls back to a local figures/ dir if the report directory does not exist
# (e.g. running the script in isolation).
PROJECT_FIG_DIR = (HERE.parent / "MAT-STK2011-Project" / "figures").resolve()
LOCAL_FIG_DIR = (HERE / "figures").resolve()
OUT = PROJECT_FIG_DIR if PROJECT_FIG_DIR.parent.exists() else LOCAL_FIG_DIR
OUT.mkdir(parents=True, exist_ok=True)
print(f"Figures will be saved to: {OUT}")


# ========== keep-strip helpers ==========
# Every corrected-fit figure attaches a thin strip at the bottom that shows,
# for each epsilon on the x-axis, the fraction of Monte-Carlo replicates that
# survived the L-BFGS-B bound check. This makes clear what was dropped before
# the MC summary statistics were computed.

from matplotlib.gridspec import GridSpec

def _keep_fraction(samples):
    """Per-ε fraction of replicates retained (NaN == dropped)."""
    arr = samples if samples.ndim == 2 else samples[..., 0]
    return (~np.isnan(arr)).mean(axis=1)


def _fig_with_strip(n_rows_main=1, n_cols=1, figsize=None,
                    keep_frac_height=0.18, hspace=None):
    """Build a figure with an extra strip-row at the bottom spanning all cols.

    When `n_rows_main >= 2`, xtick labels are hidden on the upper main rows so
    they do not crash into the titles of the row below.
    """
    if figsize is None:
        figsize = (3.8 * n_cols, 3.0 * n_rows_main + 1.0)
    if hspace is None:
        hspace = 0.45 if n_rows_main >= 2 else 0.28
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows_main + 1, n_cols, figure=fig,
                  height_ratios=[1.0] * n_rows_main + [keep_frac_height * n_rows_main],
                  hspace=hspace)
    main_axes = np.empty((n_rows_main, n_cols), dtype=object)
    for r in range(n_rows_main):
        for c in range(n_cols):
            main_axes[r, c] = fig.add_subplot(gs[r, c])
    strip_ax = fig.add_subplot(gs[n_rows_main, :])
    # Hide xtick labels on the upper main rows so they cannot collide with the
    # titles of the row beneath. The bottom main row keeps its ticks.
    if n_rows_main >= 2:
        for r in range(n_rows_main - 1):
            for c in range(n_cols):
                plt.setp(main_axes[r, c].get_xticklabels(), visible=False)
    return fig, main_axes, strip_ax


def _draw_keep_strip(ax, grid, keep_frac):
    w = float(np.diff(grid).min()) * 0.7 if len(grid) > 1 else 0.02
    colors = [(0.65 - 0.45 * k, 0.65 - 0.45 * k, 0.75 - 0.45 * k, 0.85)
              for k in keep_frac]
    ax.bar(grid, keep_frac, width=w, color=colors, edgecolor='black', linewidth=0.3)
    ax.axhline(1.0, color='grey', lw=0.4, ls=':')
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(['0', '.5', '1'])
    ax.set_ylabel('kept', fontsize=8)
    ax.set_xlabel(r'$\varepsilon = \delta$')
    ax.grid(axis='y', alpha=0.25, lw=0.4)
    for x, k in zip(grid, keep_frac):
        if k < 0.5:
            ax.text(x, k + 0.04, f"{k:.0%}", ha='center', va='bottom',
                    fontsize=6, color='firebrick')
    ax.set_title('fraction of MC replicates retained after the bound check',
                 fontsize=8, loc='left', pad=2)


# ========== helpers (same as in the notebooks) ==========
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
        cov = np.linalg.inv(J)
    except np.linalg.LinAlgError:
        return np.full(len(theta), np.nan)
    diag = np.diag(cov)
    return np.where(diag > 0, np.sqrt(np.abs(diag)), np.nan)


# ========== TASK (b) SIMULATION ==========

A_TRUE, B_TRUE = 0.5, 1.2
N_B = 1000
B_MC = 120
EPS_GRID = np.concatenate([np.linspace(0.0, 0.48, 13),
                           np.linspace(0.52, 0.99, 13)])

naive_a = np.full((len(EPS_GRID), B_MC), np.nan)
naive_b = np.full((len(EPS_GRID), B_MC), np.nan)
corr_a  = np.full((len(EPS_GRID), B_MC), np.nan)
corr_b  = np.full((len(EPS_GRID), B_MC), np.nan)
corr_a_se = np.full((len(EPS_GRID), B_MC), np.nan)
corr_b_se = np.full((len(EPS_GRID), B_MC), np.nan)

SIM_BOUND = 10.0  # the corrected likelihood is flat in beta when c = 1-eps-delta
                  # is small; bound the optimizer so BFGS cannot wander to infinity
                  # and drop replicates that hit the boundary as non-identified.

rng_mc = np.random.default_rng(SEED)
for k, eps in enumerate(EPS_GRID):
    for b in range(B_MC):
        x, y, yh = simulate_ab(N_B, A_TRUE, B_TRUE, eps, eps, rng_mc)
        ab_n, _ = fit_naive(x.reshape(-1, 1), yh)
        if np.any(np.isnan(ab_n)):
            continue
        naive_a[k, b] = ab_n[0]; naive_b[k, b] = ab_n[1]
        start = ab_n.copy() if eps < 0.5 else -ab_n.copy()
        rc = fit_corr(x.reshape(-1, 1), yh, eps, eps, start,
                      bounded=True, bound=SIM_BOUND)
        if np.max(np.abs(rc.x)) >= SIM_BOUND - 0.1:
            continue  # non-identified replicate
        corr_a[k, b] = rc.x[0]; corr_b[k, b] = rc.x[1]
        se = se_hess(rc.x, x.reshape(-1, 1), yh, eps, eps)
        corr_a_se[k, b] = se[0]; corr_b_se[k, b] = se[1]

print("task-b MC done; overall convergent fraction = %.3f"
      % np.mean(~np.isnan(corr_a)))


# ---- Figure 1: naive attenuation ----
def _panel(ax, grid, samples, true_val, color):
    m = np.nanmean(samples, axis=1)
    s = np.nanstd(samples, axis=1, ddof=1)
    jit = np.random.default_rng(0).standard_normal(samples.shape) * 0.003
    xx = grid[:, None] + jit
    ax.plot(xx.ravel(), samples.ravel(), '.', ms=1.8, alpha=0.12, color=color)
    ax.fill_between(grid, m - s, m + s, alpha=0.22, color=color)
    ax.plot(grid, m, 'o-', color=color, ms=3.0, lw=1.1)
    ax.axhline(true_val, color='k', ls=':', lw=0.9)
    ax.axhline(0, color='grey', lw=0.4)
    ax.axvline(0.5, color='r', ls='--', lw=0.7, alpha=0.5)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
_panel(axes[0], EPS_GRID, naive_a, A_TRUE, 'tab:blue')
axes[0].set_xlabel(r'$\varepsilon = \delta$')
axes[0].set_ylabel(r'$\widehat{\beta}_0$')
axes[0].set_title('Naive GLM intercept')
_panel(axes[1], EPS_GRID, naive_b, B_TRUE, 'tab:orange')
axes[1].set_xlabel(r'$\varepsilon = \delta$')
axes[1].set_ylabel(r'$\widehat{\beta}_1$')
axes[1].set_title('Naive GLM slope')
plt.tight_layout()
save_latex_figure(str(OUT / 'fig_sim_attenuation.pdf'))
plt.close()


# ---- Figure 2: corrected MLE ----
fig, main_axes, strip_ax = _fig_with_strip(n_rows_main=1, n_cols=2, figsize=(10, 4.0))
axes = main_axes[0]
_panel(axes[0], EPS_GRID, corr_a, A_TRUE, 'tab:blue')
axes[0].set_xlabel('')
axes[0].set_ylabel(r'$\widehat{\beta}_0$')
axes[0].set_title('Corrected MLE intercept')
_panel(axes[1], EPS_GRID, corr_b, B_TRUE, 'tab:orange')
axes[1].set_xlabel('')
axes[1].set_ylabel(r'$\widehat{\beta}_1$')
axes[1].set_title('Corrected MLE slope')
_draw_keep_strip(strip_ax, EPS_GRID, _keep_fraction(corr_a))
plt.tight_layout()
save_latex_figure(str(OUT / 'fig_sim_corrected.pdf'))
plt.close()


# ---- Figure 3: sd growth + Hessian agreement ----
mc_sd_a = np.nanstd(corr_a, axis=1, ddof=1)
mc_sd_b = np.nanstd(corr_b, axis=1, ddof=1)
mean_se_a = np.nanmean(corr_a_se, axis=1)
mean_se_b = np.nanmean(corr_b_se, axis=1)

fig, main_axes, strip_ax = _fig_with_strip(n_rows_main=1, n_cols=2, figsize=(10, 4.0))
axes = main_axes[0]
axes[0].plot(EPS_GRID, mc_sd_a,  'o-',  color='tab:blue',   ms=3.5, label=r'MC sd of $\widehat{\beta}_0$')
axes[0].plot(EPS_GRID, mean_se_a,'s--', color='tab:blue',   ms=3.5, alpha=0.55, label=r'Hessian se of $\widehat{\beta}_0$')
axes[0].plot(EPS_GRID, mc_sd_b,  'o-',  color='tab:orange', ms=3.5, label=r'MC sd of $\widehat{\beta}_1$')
axes[0].plot(EPS_GRID, mean_se_b,'s--', color='tab:orange', ms=3.5, alpha=0.55, label=r'Hessian se of $\widehat{\beta}_1$')
axes[0].axvline(0.5, color='r', ls='--', lw=0.7, alpha=0.5)
axes[0].set_yscale('log')
axes[0].set_xlabel('')
axes[0].set_ylabel('standard deviation (log scale)')
axes[0].set_title('Absolute sd of the corrected MLE')
axes[0].legend(fontsize=8)

base_a = mc_sd_a[0]; base_b = mc_sd_b[0]
axes[1].plot(EPS_GRID, mc_sd_a / base_a, 'o-', color='tab:blue',   ms=3.5, label=r'$\widehat{\beta}_0$')
axes[1].plot(EPS_GRID, mc_sd_b / base_b, 'o-', color='tab:orange', ms=3.5, label=r'$\widehat{\beta}_1$')
axes[1].axhline(1, color='k', ls=':', lw=1)
axes[1].axvline(0.5, color='r', ls='--', lw=0.7, alpha=0.5)
axes[1].set_yscale('log')
axes[1].set_xlabel('')
axes[1].set_ylabel(r'sd / sd at $\varepsilon = 0$')
axes[1].set_title('Variance inflation vs. the clean case')
axes[1].legend(fontsize=8)
_draw_keep_strip(strip_ax, EPS_GRID, _keep_fraction(corr_a))
plt.tight_layout()
save_latex_figure(str(OUT / 'fig_sim_sd_growth.pdf'))
plt.close()


# ========== TASK (c) BREAST CANCER ==========

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=171, random_state=SEED, stratify=data.target
)
SEL = ['mean radius', 'mean texture', 'mean smoothness', 'mean concave points']
idx = np.array([list(data.feature_names).index(f) for f in SEL])
X_tr = X_train[:, idx]
mu = X_tr.mean(axis=0); sd = X_tr.std(axis=0, ddof=1)
X_tr_z = (X_tr - mu) / sd

ab_clean, se_clean = fit_naive(X_tr_z, y_train)
param_names = ['intercept'] + SEL
print("clean baseline:", dict(zip(param_names, np.round(ab_clean, 3))))

EPS_C = np.concatenate([np.linspace(0.0, 0.48, 13), np.linspace(0.52, 0.99, 13)])
B_C = 120
rng_c = np.random.default_rng(SEED + 1)
P = X_tr_z.shape[1] + 1

from scipy.stats import norm

naive_p = np.full((len(EPS_C), B_C, P), np.nan)
corr_p  = np.full((len(EPS_C), B_C, P), np.nan)
se_p    = np.full((len(EPS_C), B_C, P), np.nan)
pval_p  = np.full((len(EPS_C), B_C, P), np.nan)

def flip(y, eps, delta, rng):
    yh = y.copy().astype(int)
    flip1 = (y == 1) & (rng.uniform(size=len(y)) < eps)
    flip0 = (y == 0) & (rng.uniform(size=len(y)) < delta)
    yh[flip1] = 0; yh[flip0] = 1
    return yh

for k, eps in enumerate(EPS_C):
    for b in range(B_C):
        yn = flip(y_train, eps, eps, rng_c)
        ab_n, _ = fit_naive(X_tr_z, yn)
        if np.any(np.isnan(ab_n)):
            continue
        naive_p[k, b] = ab_n
        start = ab_n.copy() if eps < 0.5 else -ab_n.copy()
        # L-BFGS-B with explicit bounds: prevents divergence into the flat plateau
        # of the corrected likelihood when the logistic model is near-separable
        # (strong predictors + moderate flip rate).
        rc = fit_corr(X_tr_z, yn, eps, eps, start, bounded=True, bound=15.0)
        # Drop replicates that hit the bound (the likelihood has no finite MLE for
        # those labels) so the MC summary reflects convergent fits only.
        if np.max(np.abs(rc.x)) >= 14.9:
            continue
        corr_p[k, b] = rc.x
        se = se_hess(rc.x, X_tr_z, yn, eps, eps)
        if np.any(np.isnan(se)):
            continue
        se_p[k, b] = se
        z = rc.x / se
        pval_p[k, b] = 2.0 * norm.sf(np.abs(z))

print("task-c MC done; convergent fraction = %.3f" %
      (np.mean(~np.isnan(corr_p[:, :, 0]))))


# ---- Figure 4: breast-cancer coefficient paths (naive top, corrected bottom) ----
fig, main_axes, strip_ax = _fig_with_strip(n_rows_main=2, n_cols=P,
                                           figsize=(3.0 * P, 5.6))
axes = main_axes
colors_row = ['tab:orange', 'tab:green']

for j, name in enumerate(param_names):
    for row, (samples, color) in enumerate([(naive_p[:, :, j], colors_row[0]),
                                             (corr_p[:, :, j],  colors_row[1])]):
        ax = axes[row, j]
        m = np.nanmean(samples, axis=1); s = np.nanstd(samples, axis=1, ddof=1)
        jit = np.random.default_rng(1).standard_normal(samples.shape) * 0.003
        xx = EPS_C[:, None] + jit
        ax.plot(xx.ravel(), samples.ravel(), '.', ms=1.4, alpha=0.10, color=color)
        ax.fill_between(EPS_C, m - s, m + s, alpha=0.22, color=color)
        ax.plot(EPS_C, m, 'o-', color=color, ms=2.6, lw=1.0)
        ax.axhline(ab_clean[j], color='k', ls='--', lw=0.8)
        ax.axhline(0, color='grey', lw=0.4)
        ax.axvline(0.5, color='r', ls='--', lw=0.6, alpha=0.5)
        ax.set_title(f'{"naive" if row==0 else "corrected"}: {name}', fontsize=9)
    axes[1, j].set_xlabel('')

axes[0, 0].set_ylabel(r'$\widehat{\theta}_j$')
axes[1, 0].set_ylabel(r'$\widehat{\theta}_j$')
_draw_keep_strip(strip_ax, EPS_C, _keep_fraction(corr_p))
plt.tight_layout()
save_latex_figure(str(OUT / 'fig_breast_paths.pdf'))
plt.close()


# ---- Figure 5: focal overlay for mean concave points ----
j_star = param_names.index('mean concave points')

fig, main_axes, strip_ax = _fig_with_strip(n_rows_main=1, n_cols=1, figsize=(7.5, 4.4))
ax = main_axes[0, 0]
mn = np.nanmean(naive_p[:, :, j_star], axis=1); sn = np.nanstd(naive_p[:, :, j_star], axis=1, ddof=1)
mc = np.nanmean(corr_p[:,  :, j_star], axis=1); sc = np.nanstd(corr_p[:,  :, j_star], axis=1, ddof=1)

ax.fill_between(EPS_C, mn - sn, mn + sn, alpha=0.2, color='tab:orange')
ax.plot(EPS_C, mn, 'o-', color='tab:orange', ms=3.0, label='naive')
ax.fill_between(EPS_C, mc - sc, mc + sc, alpha=0.2, color='tab:green')
ax.plot(EPS_C, mc, 's-', color='tab:green', ms=3.0, label='corrected')
ax.axhline(ab_clean[j_star], color='k', ls='--', lw=1, label=f'clean baseline = {ab_clean[j_star]:.2f}')
ax.axhline(0, color='grey', lw=0.4)
ax.axvline(0.5, color='r', ls='--', lw=0.7, alpha=0.5, label=r'pole $\varepsilon + \delta = 1$')
ax.set_xlabel('')
ax.set_ylabel(r'coefficient on mean concave points')
ax.set_title('Naive vs corrected coefficient, breast-cancer data')
ax.legend(fontsize=8)
_draw_keep_strip(strip_ax, EPS_C, _keep_fraction(corr_p))
plt.tight_layout()
save_latex_figure(str(OUT / 'fig_breast_overlay.pdf'))
plt.close()


# ---- Figure 6: Wald p-values for the corrected MLE vs eps ----
# One curve per coefficient: median p-value across convergent replicates,
# with an IQR band. Log y-scale so we can see the jump from "strong evidence"
# to "no evidence". Legend is placed outside the plot so it cannot crash into
# the curves or the title.
ALPHA = 0.05

with np.errstate(invalid='ignore'):
    med_p = np.nanmedian(pval_p, axis=1)
    q1_p  = np.nanpercentile(pval_p, 25, axis=1)
    q3_p  = np.nanpercentile(pval_p, 75, axis=1)

fig, main_axes, strip_ax = _fig_with_strip(n_rows_main=1, n_cols=1,
                                           figsize=(11.0, 5.4))
ax = main_axes[0, 0]

palette = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
for j, name in enumerate(param_names):
    c = palette[j % len(palette)]
    ax.fill_between(EPS_C, q1_p[:, j], q3_p[:, j], alpha=0.12, color=c)
    ax.plot(EPS_C, med_p[:, j], 'o-', color=c, ms=3.0, lw=1.1, label=name)

ax.axhline(ALPHA, color='k', ls='--', lw=0.9, label=fr'$\alpha = {ALPHA}$')
ax.axvline(0.5, color='r', ls='--', lw=0.7, alpha=0.5, label='pole')
ax.set_yscale('log')
ax.set_ylim(1e-4, 1.2)
ax.set_xlabel('')
ax.set_ylabel('median Wald $p$-value (log scale)')
ax.set_title(r'Wald $p$-values for the corrected MLE vs label-noise rate', pad=8)
ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5),
          frameon=True, borderaxespad=0.0)

_draw_keep_strip(strip_ax, EPS_C, _keep_fraction(corr_p))
fig.subplots_adjust(left=0.08, right=0.80, top=0.92, bottom=0.13, hspace=0.32)
save_latex_figure(str(OUT / 'fig_breast_pvalues.pdf'))
plt.close()


print("\nAll figures saved to", OUT)
for p in sorted(OUT.glob('*.pdf')):
    print(" ", p.name, p.stat().st_size, 'bytes')
