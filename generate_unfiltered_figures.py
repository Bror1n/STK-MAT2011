"""
No-filter versions of the figures for Task B (single-covariate
simulation, Section 5.2) and Task C (breast-cancer application,
Section 5.3).

The point of this script is to show the same experiments as
generate_report_figures.py but *without dropping any Monte-Carlo
replicates*.  Filtering on the bound is a valid sensitivity rule, but
it tampers with the data: the visible spread of beta_hat in the
filtered figures is the spread over the surviving subset, not over all
replicates.

Visual convention here:
    * every replicate is plotted (small dots);
    * the central tendency is the **mean** over all replicates,
      *including* those pinned at the L-BFGS-B bound;
    * the band is the **mean +/- one standard deviation** over the
      same set of replicates;
    * the y-axis is set so the entire mean +/- sd band is visible
      everywhere.  Individual scatter points falling outside that
      window are clipped (so the plot stays readable) but they are
      still in the statistics that draw the band.

This is the visualisation requested in the project notes: keep the
data, set a cutoff on the plot, but keep the standard-deviation region
fully visible.

Outputs (saved to MAT-STK2011-Project/figures/):
    fig_sim_corrected_unfiltered.pdf
    fig_sim_sd_growth_unfiltered.pdf
    fig_breast_paths_unfiltered.pdf
    fig_breast_overlay_unfiltered.pdf
    fig_breast_pvalues_unfiltered.pdf

Reproducibility: SEED = 6114 from helper_functions.corrected_mle.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.gridspec import GridSpec
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from scipy.stats import norm

from helper_functions.corrected_mle import (
    SEED, H, fit_naive, fit_corr, hessian_se, flip_labels,
)

import warnings
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
PROJ_FIG = (HERE.parent / "MAT-STK2011-Project" / "figures").resolve()
OUT = PROJ_FIG if PROJ_FIG.parent.exists() else (HERE / "figures").resolve()
OUT.mkdir(parents=True, exist_ok=True)
print(f"Saving figures to {OUT}")


# ===========================================================================
# Plot helpers: keep ALL points; mean +/- sd band; clip y-axis but keep band
# visible.
# ===========================================================================
def mean_sd(samples_2d):
    """Per-eps mean and sd over all replicates (NaNs treated as missing)."""
    mu = np.nanmean(samples_2d, axis=1)
    sd = np.nanstd(samples_2d, axis=1, ddof=1)
    return mu, sd


def yax_for_band(mu, sd, pad_frac=0.04, hard_max=None):
    """Tight y-axis: just enough margin to draw the band edge cleanly.

    Anything above mean+sd or below mean-sd (bound-hit dots, outliers) is
    clipped by matplotlib so the figure stays clean.  Caller can supply
    hard_max as a final safety cap.
    """
    finite = np.isfinite(mu) & np.isfinite(sd)
    if not np.any(finite):
        return -1.0, 1.0
    lo = np.min((mu - sd)[finite])
    hi = np.max((mu + sd)[finite])
    span = max(hi - lo, 1e-3)
    pad = pad_frac * span
    lo, hi = lo - pad, hi + pad
    if hard_max is not None:
        lo = max(lo, -hard_max)
        hi = min(hi,  hard_max)
    return lo, hi


def plot_band(ax, eps_grid, samples_2d, on_bound_2d=None, bound=None,
              color="C0", scatter_alpha=0.18, scatter_size=6, label=None):
    """Scatter all points, draw mean +/- sd band.

    Bound-hit replicates are kept in the *statistics* (so the band reflects
    them) but their scatter dots fall outside the y-window the caller sets,
    and we no longer mark them with explicit triangles -- the figure stays
    less cluttered.  ``on_bound_2d`` and ``bound`` are accepted for backward
    compatibility but currently unused.
    """
    del on_bound_2d, bound  # intentionally unused; kept for the call signature
    n_eps, n_rep = samples_2d.shape
    jitter_rng = np.random.default_rng(SEED)
    jit = jitter_rng.uniform(-0.005, 0.005, size=samples_2d.shape)

    ax.scatter(
        (eps_grid[:, None] + jit).ravel(),
        samples_2d.ravel(),
        s=scatter_size, color=color, alpha=scatter_alpha,
        edgecolor="none", zorder=2,
    )

    mu, sd = mean_sd(samples_2d)
    ax.fill_between(eps_grid, mu - sd, mu + sd, color=color, alpha=0.22,
                    zorder=4, label=label)
    ax.plot(eps_grid, mu, color=color, linewidth=1.4, zorder=5)


# ===========================================================================
# Task B: simulated single-covariate
# ===========================================================================
A_TRUE, B_TRUE = 0.5, 1.2
N_B = 1000
B_MC = 200
SIM_BOUND = 10.0

EPS_GRID = np.concatenate([
    np.linspace(0.0, 0.48, 13),
    np.linspace(0.52, 0.99, 13),
])

print(f"Task B: n={N_B}, B_MC={B_MC}, |eps_grid|={len(EPS_GRID)}, "
      f"bound=+/-{SIM_BOUND}")

naive_a = np.full((len(EPS_GRID), B_MC), np.nan)
naive_b = np.full((len(EPS_GRID), B_MC), np.nan)
corr_a  = np.full((len(EPS_GRID), B_MC), np.nan)
corr_b  = np.full((len(EPS_GRID), B_MC), np.nan)
corr_a_se = np.full((len(EPS_GRID), B_MC), np.nan)
corr_b_se = np.full((len(EPS_GRID), B_MC), np.nan)
on_bound_a = np.zeros((len(EPS_GRID), B_MC), dtype=bool)
on_bound_b = np.zeros((len(EPS_GRID), B_MC), dtype=bool)

rng_b = np.random.default_rng(SEED)
naive_fail_b = 0
scipy_fail_b = 0

for k, eps in enumerate(EPS_GRID):
    for b in range(B_MC):
        x = rng_b.standard_normal(N_B)
        y = rng_b.binomial(1, H(A_TRUE + B_TRUE * x))
        yh = flip_labels(y, eps, eps, rng=rng_b)

        # Naive GLM gives the warm start.  If statsmodels fails to
        # converge we fall back to a zero start for the corrected fit
        # (so we never drop the replicate); the naive entries are then
        # left as NaN for that replicate but the corrected fit proceeds.
        ab_n, _, _ = fit_naive(x.reshape(-1, 1), yh)
        if np.any(np.isnan(ab_n)):
            ab_warm = np.zeros(2)
            naive_fail_b += 1
        else:
            naive_a[k, b], naive_b[k, b] = ab_n
            ab_warm = ab_n

        start = ab_warm.copy() if eps < 0.5 else -ab_warm.copy()
        res = fit_corr(x.reshape(-1, 1), yh, eps, eps,
                       start=start, bound=SIM_BOUND)
        if not res.success:
            scipy_fail_b += 1
        corr_a[k, b], corr_b[k, b] = res.x
        on_bound_a[k, b] = abs(res.x[0]) >= SIM_BOUND - 0.1
        on_bound_b[k, b] = abs(res.x[1]) >= SIM_BOUND - 0.1
        # Hessian-based se via the submatrix of unbounded coordinates.
        try:
            on_bnd = np.array([on_bound_a[k, b], on_bound_b[k, b]])
            se = hessian_se(res.x, x.reshape(-1, 1), yh, eps, eps,
                            on_bound=on_bnd)
            corr_a_se[k, b], corr_b_se[k, b] = se
        except Exception:
            pass

hit_frac_a = on_bound_a.mean(axis=1)
hit_frac_b = on_bound_b.mean(axis=1)
n_total_b = corr_a.size
print(f"  bound-hit fraction (intercept): max={hit_frac_a.max():.3f},"
      f" mean={hit_frac_a.mean():.3f}")
print(f"  bound-hit fraction (slope):     max={hit_frac_b.max():.3f},"
      f" mean={hit_frac_b.mean():.3f}")
print(f"  naive GLM failures (zero-start fallback used): "
      f"{naive_fail_b} / {n_total_b}  ({100 * naive_fail_b / n_total_b:.2f}%)")
print(f"  scipy L-BFGS-B non-success flags (fit kept anyway): "
      f"{scipy_fail_b} / {n_total_b}  ({100 * scipy_fail_b / n_total_b:.2f}%)")


# --- fig_sim_corrected_unfiltered: scatter + mean +/- sd band ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
plot_band(axes[0], EPS_GRID, corr_a, on_bound_a, bound=SIM_BOUND,
          color="C0", label="MC mean +/- sd")
axes[0].axhline(A_TRUE, color="black", linestyle=":", linewidth=0.9,
                label=f"$a = {A_TRUE}$")
axes[0].axvline(0.5, color="C3", linestyle="--", linewidth=0.8, alpha=0.6)
mu_a, sd_a = mean_sd(corr_a)
ylo, yhi = yax_for_band(mu_a, sd_a, pad_frac=0.04)
axes[0].set_ylim(ylo, yhi)
axes[0].set_xlabel(r"$\varepsilon = \delta$")
axes[0].set_ylabel(r"$\widehat a$")
axes[0].set_title("Corrected MLE intercept (no filter)")
axes[0].legend(fontsize=8, loc="lower left")

plot_band(axes[1], EPS_GRID, corr_b, on_bound_b, bound=SIM_BOUND,
          color="C1", label="MC mean +/- sd")
axes[1].axhline(B_TRUE, color="black", linestyle=":", linewidth=0.9,
                label=f"$b = {B_TRUE}$")
axes[1].axhline(-B_TRUE, color="black", linestyle=":", linewidth=0.9, alpha=0.4)
axes[1].axvline(0.5, color="C3", linestyle="--", linewidth=0.8, alpha=0.6)
mu_b, sd_b = mean_sd(corr_b)
ylo, yhi = yax_for_band(mu_b, sd_b, pad_frac=0.04)
axes[1].set_ylim(ylo, yhi)
axes[1].set_xlabel(r"$\varepsilon = \delta$")
axes[1].set_ylabel(r"$\widehat b$")
axes[1].set_title("Corrected MLE slope (no filter)")
axes[1].legend(fontsize=8, loc="lower left")

fig.suptitle(
    "Corrected MLE without bound-hit filter: every replicate kept; mean +/- sd "
    "computed over all of them",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(OUT / "fig_sim_corrected_unfiltered.pdf", bbox_inches="tight")
plt.close(fig)


# --- fig_sim_sd_growth_unfiltered: sd over all reps + Hessian-se overlay + bound-hit panel ---
mc_sd_a = np.nanstd(corr_a, axis=1, ddof=1)
mc_sd_b = np.nanstd(corr_b, axis=1, ddof=1)
mean_se_a_int = np.nanmean(corr_a_se, axis=1)   # only interior fits had a Hessian-se computed
mean_se_b_int = np.nanmean(corr_b_se, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
axes[0].plot(EPS_GRID, mc_sd_a, "o-", color="C0", markersize=3.5,
             label=r"sd of $\widehat a$ (MC, all reps)")
axes[0].plot(EPS_GRID, mc_sd_b, "o-", color="C1", markersize=3.5,
             label=r"sd of $\widehat b$ (MC, all reps)")
axes[0].plot(EPS_GRID, mean_se_a_int, "s--", color="C0", alpha=0.7,
             markersize=3.0, label=r"mean Hessian se of $\widehat a$ (interior only)")
axes[0].plot(EPS_GRID, mean_se_b_int, "s--", color="C1", alpha=0.7,
             markersize=3.0, label=r"mean Hessian se of $\widehat b$ (interior only)")
axes[0].set_yscale("log")
axes[0].set_xlabel(r"$\varepsilon = \delta$")
axes[0].set_ylabel("sd / se (log scale)")
axes[0].set_title("Spread of corrected MLE (all replicates kept)")
axes[0].axvline(0.5, color="C3", linestyle="--", linewidth=0.8, alpha=0.6)
axes[0].legend(fontsize=7, loc="lower left")

axes[1].plot(EPS_GRID, hit_frac_a, "o-", color="C0", markersize=3.5,
             label=r"$\widehat a$ at bound")
axes[1].plot(EPS_GRID, hit_frac_b, "o-", color="C1", markersize=3.5,
             label=r"$\widehat b$ at bound")
axes[1].axhline(0, color="0.85", linewidth=0.5)
axes[1].set_xlabel(r"$\varepsilon = \delta$")
axes[1].set_ylabel("fraction at bound")
axes[1].set_title(r"Fraction of replicates pinned at $\pm M = 10$")
axes[1].axvline(0.5, color="C3", linestyle="--", linewidth=0.8, alpha=0.6)
axes[1].set_ylim(-0.02, 1.02)
axes[1].legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT / "fig_sim_sd_growth_unfiltered.pdf", bbox_inches="tight")
plt.close(fig)


# ===========================================================================
# Task C: breast cancer
# ===========================================================================
data = load_breast_cancer()
X_train, _, y_train, _ = train_test_split(
    data.data, data.target, test_size=171, random_state=SEED,
    stratify=data.target,
)
SEL = ["mean radius", "mean texture", "mean smoothness", "mean concave points"]
idx = np.array([list(data.feature_names).index(f) for f in SEL])
X_tr = X_train[:, idx]
mu_x = X_tr.mean(axis=0); sd_x = X_tr.std(axis=0, ddof=1)
X_tr = (X_tr - mu_x) / sd_x

ab_clean, _, _ = fit_naive(X_tr, y_train)
P = X_tr.shape[1] + 1
print(f"Task C: n={X_tr.shape[0]}, p={X_tr.shape[1]}, P={P}")

EPS_C = np.concatenate([np.linspace(0.0, 0.48, 13), np.linspace(0.52, 0.99, 13)])
B_C = 120
BOUND_C = 15.0

naive_p = np.full((len(EPS_C), B_C, P), np.nan)
corr_p  = np.full((len(EPS_C), B_C, P), np.nan)
on_bnd_c = np.zeros((len(EPS_C), B_C, P), dtype=bool)
pval_p  = np.full((len(EPS_C), B_C, P), np.nan)

rng_c = np.random.default_rng(SEED + 1)
naive_fail_c = 0
scipy_fail_c = 0

print("  starting Task C MC ...", flush=True)
for k, eps in enumerate(EPS_C):
    if k % 5 == 0:
        print(f"    eps idx {k}/{len(EPS_C)}  ({eps:.2f})", flush=True)
    for b in range(B_C):
        yn = flip_labels(y_train, eps, eps, rng=rng_c)
        # Naive GLM is used as the warm start.  If statsmodels fails to
        # converge we fall back to a zero start so we never drop the
        # replicate; the naive_p row stays NaN but corr_p proceeds.
        ab_n, _, _ = fit_naive(X_tr, yn)
        if np.any(np.isnan(ab_n)):
            ab_warm = np.zeros(P)
            naive_fail_c += 1
        else:
            naive_p[k, b] = ab_n
            ab_warm = ab_n

        start = ab_warm.copy() if eps < 0.5 else -ab_warm.copy()
        res = fit_corr(X_tr, yn, eps, eps, start=start, bound=BOUND_C)
        if not res.success:
            scipy_fail_c += 1
        corr_p[k, b] = res.x
        on_bnd_c[k, b] = np.abs(res.x) >= BOUND_C - 0.1

        # Wald p-value via the Hessian-based se, computed on the
        # submatrix of unbounded coordinates so the inference for free
        # coordinates is not coupled to phantom variability in the
        # pinned ones.  Bound-hit coordinates' p-values are NaN.
        try:
            se = hessian_se(res.x, X_tr, yn, eps, eps,
                            on_bound=on_bnd_c[k, b])
            z = res.x / se
            pval_p[k, b] = 2 * norm.sf(np.abs(z))
        except Exception:
            pass

print("  Task C MC done.", flush=True)
c_hit_frac = on_bnd_c.mean(axis=1)   # shape (n_eps, P)
n_total_c = corr_p.shape[0] * corr_p.shape[1]
print(f"  bound-hit fraction (any coord): max={c_hit_frac.max():.3f},"
      f" mean={c_hit_frac.mean():.3f}")
print(f"  naive GLM failures (zero-start fallback used): "
      f"{naive_fail_c} / {n_total_c}  ({100 * naive_fail_c / n_total_c:.2f}%)")
print(f"  scipy L-BFGS-B non-success flags (fit kept anyway): "
      f"{scipy_fail_c} / {n_total_c}  ({100 * scipy_fail_c / n_total_c:.2f}%)")


# --- fig_breast_paths_unfiltered: 2-row figure, naive top, corrected bottom, no filter ---
fig = plt.figure(figsize=(15, 6.6))
gs = GridSpec(2, P, figure=fig, hspace=0.35)
labels = ["intercept"] + SEL

for j in range(P):
    # naive: bound is not relevant
    ax_n = fig.add_subplot(gs[0, j])
    plot_band(ax_n, EPS_C, naive_p[:, :, j], on_bound_2d=None,
              color="C1", scatter_alpha=0.15, scatter_size=4)
    ax_n.axhline(ab_clean[j], color="black", linestyle=":", linewidth=0.9,
                 label=fr"clean $= {ab_clean[j]:.2f}$")
    ax_n.axvline(0.5, color="C3", linestyle="--", linewidth=0.7, alpha=0.6)
    ax_n.set_title(f"naive: {labels[j]}", fontsize=9)
    ax_n.set_xlabel(r"$\varepsilon = \delta$", fontsize=8)
    mu_j, sd_j = mean_sd(naive_p[:, :, j])
    ax_n.set_ylim(*yax_for_band(mu_j, sd_j, pad_frac=0.15))
    ax_n.legend(fontsize=7, loc="best")

    # corrected: bound-hits at +/-15 must be visible as triangles
    ax_c = fig.add_subplot(gs[1, j])
    plot_band(ax_c, EPS_C, corr_p[:, :, j],
              on_bound_2d=on_bnd_c[:, :, j], bound=BOUND_C,
              color="C2", scatter_alpha=0.15, scatter_size=4)
    ax_c.axhline(ab_clean[j], color="black", linestyle=":", linewidth=0.9)
    ax_c.axvline(0.5, color="C3", linestyle="--", linewidth=0.7, alpha=0.6)
    ax_c.set_title(f"corrected: {labels[j]}", fontsize=9)
    ax_c.set_xlabel(r"$\varepsilon = \delta$", fontsize=8)
    mu_j, sd_j = mean_sd(corr_p[:, :, j])
    # cap the y-axis at the bound so the triangles + the band are both visible
    ax_c.set_ylim(*yax_for_band(mu_j, sd_j, pad_frac=0.10,
                                ))

fig.suptitle(
    "Breast cancer -- every replicate kept; mean +/- sd computed over all of them",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "fig_breast_paths_unfiltered.pdf", bbox_inches="tight")
plt.close(fig)


# --- fig_breast_overlay_unfiltered: focal predictor mean concave points ---
j_focal = SEL.index("mean concave points") + 1   # +1 for intercept

fig, ax = plt.subplots(figsize=(8, 4.8))
plot_band(ax, EPS_C, naive_p[:, :, j_focal], on_bound_2d=None, color="C1",
          label="naive: mean +/- sd",
          scatter_alpha=0.15, scatter_size=5)
plot_band(ax, EPS_C, corr_p[:, :, j_focal],
          on_bound_2d=on_bnd_c[:, :, j_focal], bound=BOUND_C,
          color="C2", label="corrected: mean +/- sd",
          scatter_alpha=0.15, scatter_size=5)
ax.axhline(ab_clean[j_focal], color="black", linestyle="--", linewidth=1.2,
           label=fr"clean baseline $= {ab_clean[j_focal]:.2f}$")
ax.axvline(0.5, color="C3", linestyle="--", linewidth=0.8, alpha=0.6,
           label=r"pole $\varepsilon + \delta = 1$")

# y-axis: include the union of both means+/-sds, capped at +/-(BOUND_C+1)
mu_n, sd_n = mean_sd(naive_p[:, :, j_focal])
mu_c, sd_c = mean_sd(corr_p[:, :, j_focal])
all_mu = np.concatenate([mu_n, mu_c])
all_sd = np.concatenate([sd_n, sd_c])
ylo, yhi = yax_for_band(all_mu, all_sd, pad_frac=0.04)
ax.set_ylim(ylo, yhi)
ax.set_xlabel(r"$\varepsilon = \delta$")
ax.set_ylabel("coefficient on mean concave points")
ax.set_title("Naive vs corrected on mean concave points "
             "(no filter, breast-cancer data)")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "fig_breast_overlay_unfiltered.pdf", bbox_inches="tight")
plt.close(fig)


# --- fig_breast_pvalues_unfiltered ---
# Median + IQR over all replicates, per coordinate.  Median + IQR is the
# right spread summary on a log axis (mean +/- sd makes the band fall
# off a cliff to 1e-14 because sd > mean for skewed p-values).
fig, ax_p = plt.subplots(figsize=(11, 4.8))
colors_p = plt.get_cmap("tab10")

for j in range(P):
    med = np.nanmedian(pval_p[:, :, j], axis=1)
    q25 = np.nanpercentile(pval_p[:, :, j], 25, axis=1)
    q75 = np.nanpercentile(pval_p[:, :, j], 75, axis=1)
    ax_p.fill_between(EPS_C,
                      np.clip(q25, 1e-14, 1.0),
                      np.clip(q75, 1e-14, 1.0),
                      color=colors_p(j), alpha=0.18)
    ax_p.plot(EPS_C, np.clip(med, 1e-14, 1.0), "-o",
              color=colors_p(j), markersize=4, label=labels[j])
ax_p.axhline(0.05, color="black", linestyle="--", linewidth=0.9,
             label=r"$\alpha = 0.05$")
ax_p.axvline(0.5, color="C3", linestyle="--", linewidth=0.8, alpha=0.6,
             label="pole")
ax_p.set_yscale("log")
ax_p.set_ylim(1e-12, 2)
ax_p.set_xlabel(r"$\varepsilon = \delta$")
ax_p.set_ylabel(r"median Wald $p$-value (log scale)")
ax_p.set_title("Wald $p$-values on the breast-cancer corrected MLE -- "
               "median and IQR, every replicate kept")
ax_p.legend(fontsize=8, loc="lower center", ncol=3)

fig.tight_layout()
fig.savefig(OUT / "fig_breast_pvalues_unfiltered.pdf", bbox_inches="tight")
plt.close(fig)

print("\nAll unfiltered figures saved.")
for f in OUT.glob("*_unfiltered.pdf"):
    print(f"  {f.name}  ({f.stat().st_size} bytes)")
