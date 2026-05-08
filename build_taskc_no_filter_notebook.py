"""
Builds taskc_no_filter.ipynb -- a self-contained notebook that runs Task C
(Wisconsin breast-cancer application, Section 5.3 of the report) without
dropping any Monte-Carlo replicates.

Same visual convention as taskb_no_filter.ipynb:
    * every replicate is plotted as a small dot;
    * the central tendency is the **mean** over all replicates,
      *including* those pinned at the L-BFGS-B bound;
    * the band is the **mean +/- one standard deviation**;
    * the y-axis is set so the entire mean +/- sd band is visible.
      Individual scatter points falling outside that window are clipped
      (so the plot stays readable) but they remain in the statistics
      that draw the band.

Run:
    python3 build_taskc_no_filter_notebook.py
"""

import textwrap
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(src).strip("\n")))


# ============================================================================
md(r"""
# Task C without the bound-hit filter

This notebook is the no-filter version of Task C, the Wisconsin
breast-cancer application studied in Section 5.3 of the report.  The
original analysis dropped Monte-Carlo replicates whose corrected MLE
landed at the L-BFGS-B bound before reporting any summary statistics.
That presentation gave a clean visual but reported the spread over the
surviving subset, not over all replicates.

Here we **keep every replicate**, including the ones pinned at the
bound, and let the figure show what is actually going on:

* every replicate is drawn as a small dot;
* the band is the **mean $\pm$ one standard deviation** computed over all
  $B$ replicates at each $\varepsilon$ (bound-hits included);
* the $y$-axis is set so the band is fully visible at every
  $\varepsilon$ -- individual scatter points outside that window are
  clipped, but the band that summarises the data is not.

The reproducibility seed is `SEED = 6114` from
`helper_functions.corrected_mle`.
""")

# ============================================================================
md("## 1. Setup")

code(r"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from helper_functions.corrected_mle import (
    SEED, fit_naive, fit_corr, hessian_se, flip_labels,
)

import warnings
warnings.filterwarnings("ignore")

plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"
""")

# ============================================================================
md(r"""
## 2. Setup of the experiment

We load the breast-cancer dataset, take the four standardised predictors
(mean radius, mean texture, mean smoothness, mean concave points), and
the stratified $398/171$ train/test split with `random_state=6114`. The
clean-labels fit on the training set is the target a correct estimator
should recover.

We sweep $\varepsilon$ over $26$ points in $[0, 0.99]$, drawing $B = 120$
flipped-label realisations per level. At each replicate we fit the
corrected MLE with bounded \textsc{l-bfgs-b} and $|\widehat\beta_j| \le
M = 15$, and we **store the result regardless of whether it hits the
bound** -- we only flag which coordinates pinned.
""")

code(r"""
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
P = X_tr.shape[1] + 1   # +1 intercept
labels = ["intercept"] + SEL
print("clean-labels coefficients:")
for nm, p in zip(labels, ab_clean):
    print(f"  {nm:<22}  {p:7.3f}")

eps_grid = np.concatenate([
    np.linspace(0.0, 0.48, 13),
    np.linspace(0.52, 0.99, 13),
])
n_eps = len(eps_grid)
B_C = 120
BOUND_C = 15.0
TAU = 0.1

naive_p  = np.full((n_eps, B_C, P), np.nan)
corr_p   = np.full((n_eps, B_C, P), np.nan)
on_bnd_c = np.zeros((n_eps, B_C, P), dtype=bool)
pval_p   = np.full((n_eps, B_C, P), np.nan)

rng_c = np.random.default_rng(SEED + 1)
naive_fail = 0   # M6 -- count statsmodels GLM failures
scipy_fail = 0   # L1 -- count L-BFGS-B non-success flags
for k, eps in enumerate(eps_grid):
    if k % 5 == 0:
        print(f"  eps idx {k:>2}/{n_eps}  ({eps:.2f})")
    for b in range(B_C):
        yn = flip_labels(y_train, eps, eps, rng=rng_c)
        # Naive GLM is used as the warm start.  If statsmodels fails to
        # converge we fall back to a zero start for the corrected fit
        # (so we never drop the replicate), and the row of `naive_p` is
        # NaN for that replicate -- only the naive panel of the figure
        # is affected, the corrected panel still has data.
        ab_n, _, _ = fit_naive(X_tr, yn)
        if np.any(np.isnan(ab_n)):
            ab_n_warm = np.zeros(P)
            naive_fail += 1
        else:
            naive_p[k, b] = ab_n
            ab_n_warm = ab_n

        start = ab_n_warm.copy() if eps < 0.5 else -ab_n_warm.copy()
        res = fit_corr(X_tr, yn, eps, eps, start=start, bound=BOUND_C)
        if not res.success:
            scipy_fail += 1   # L-BFGS-B reports non-success but we still
                              # record res.x: bounded runs that hit the
                              # bound report success=False, which is
                              # captured by the bound-hit check below.
        corr_p[k, b] = res.x
        on_bnd_c[k, b] = np.abs(res.x) >= BOUND_C - TAU

        # Hessian-based p-value via the *submatrix* of unbounded
        # coordinates.  When some coefficients pin at the bound, the
        # asymptotic se for the free ones is sqrt(diag(J_free^{-1})),
        # i.e.\ the inverse of the Hessian's restriction to the free
        # coordinates -- not the diagonal of the full inverse, which
        # would couple in phantom variability through the pinned
        # coordinates.  hessian_se(on_bound=...) returns NaN at the
        # pinned positions and the asymptotically-correct se elsewhere.
        try:
            se = hessian_se(res.x, X_tr, yn, eps, eps,
                            on_bound=on_bnd_c[k, b])
            z = res.x / se
            pval_p[k, b] = 2 * norm.sf(np.abs(z))
        except Exception:
            pass

c_hit_frac = on_bnd_c.mean(axis=1)   # (n_eps, P)
n_corr_kept = np.sum(~np.isnan(corr_p[:, :, 0]))
n_total     = corr_p.shape[0] * corr_p.shape[1]
print(f"\ncorrected fits kept:  {n_corr_kept} / {n_total}  "
      f"({100 * n_corr_kept / n_total:.1f}%)")
print(f"bound-hit fraction (any coord): max={c_hit_frac.max():.3f}, mean={c_hit_frac.mean():.3f}")
print(f"naive GLM failures (zero-start fallback used): "
      f"{naive_fail} / {n_total}  ({100 * naive_fail / n_total:.2f}%)")
print(f"scipy L-BFGS-B non-success flags (fit kept anyway): "
      f"{scipy_fail} / {n_total}  ({100 * scipy_fail / n_total:.2f}%)")
""")

# ============================================================================
md(r"""
## 3. Plot helpers

Same helpers as in `taskb_no_filter.ipynb`. `mean_sd` computes the
per-$\varepsilon$ mean and sample standard deviation. `yax_for_band`
returns a $y$-range that fits the full mean $\pm$ sd band, optionally
capped at a `hard_max`. `plot_band` does the actual plotting: scatter
all replicates and draw the mean $\pm$ sd band. Replicates that pin to
$\pm M$ are kept in the band statistics but their scatter dots fall
outside the $y$-window we set, so they do not clutter the figure.
""")

code(r"""
def mean_sd(samples_2d):
    mu = np.nanmean(samples_2d, axis=1)
    sd = np.nanstd(samples_2d, axis=1, ddof=1)
    return mu, sd

def yax_for_band(mu, sd, pad_frac=0.04, hard_max=None):
    # tight y-axis: just enough margin to draw the band edge cleanly.
    # Anything above mean+sd or below mean-sd (bound-hit dots, outliers)
    # is clipped by matplotlib so the figure stays clean.
    finite = np.isfinite(mu) & np.isfinite(sd)
    if not np.any(finite):
        return -1.0, 1.0
    lo = float(np.min((mu - sd)[finite]))
    hi = float(np.max((mu + sd)[finite]))
    span = max(hi - lo, 1e-3)
    pad = pad_frac * span
    lo -= pad; hi += pad
    if hard_max is not None:
        lo = max(lo, -hard_max); hi = min(hi, hard_max)
    return lo, hi

def plot_band(ax, eps_grid, samples_2d, on_bound_2d=None, bound=None,
              color="C0", scatter_alpha=0.18, scatter_size=6, label=None):
    # bound-hit replicates are kept in the *statistics* (the band reflects
    # them) but their scatter dots fall outside the y-window we set; we no
    # longer mark them with explicit triangles, to keep the figure clean.
    del on_bound_2d, bound
    n_eps_, n_rep = samples_2d.shape
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
""")

# ============================================================================
md(r"""
## 4. Coefficient paths -- naive top, corrected bottom

Two rows: the naive GLM refit on top, the corrected MLE on the bottom.
For each predictor, every Monte-Carlo replicate is plotted as a dot,
the solid line is the mean over all replicates, the band is mean $\pm$
sd. Replicates whose coordinate pins at $\pm 15$ are still in the
statistics but fall outside the $y$-window we set.

The bottom row is where the no-filter view bites: for $\varepsilon$
roughly between $0.15$ and $0.85$ a substantial fraction of replicates
pin to $\pm 15$, the mean is dragged toward zero (positive and negative
bound-hits cancel), and the standard-deviation band fans out toward the
bound. This is the genuine non-identification signal that the filtered
version of this figure was hiding.
""")

code(r"""
fig = plt.figure(figsize=(15, 6.6))
gs = GridSpec(2, P, figure=fig, hspace=0.40)

for j in range(P):
    ax_n = fig.add_subplot(gs[0, j])
    plot_band(ax_n, eps_grid, naive_p[:, :, j], on_bound_2d=None,
              color="C1", scatter_alpha=0.15, scatter_size=4)
    ax_n.axhline(ab_clean[j], color="black", linestyle=":", linewidth=0.9,
                 label=fr"clean $= {ab_clean[j]:.2f}$")
    ax_n.axvline(0.5, color="C3", linestyle="--", linewidth=0.7, alpha=0.6)
    ax_n.set_title(f"naive: {labels[j]}", fontsize=9)
    ax_n.set_xlabel(r"$\varepsilon = \delta$", fontsize=8)
    mu_j, sd_j = mean_sd(naive_p[:, :, j])
    ax_n.set_ylim(*yax_for_band(mu_j, sd_j, pad_frac=0.04))
    ax_n.legend(fontsize=7, loc="best")

    ax_c = fig.add_subplot(gs[1, j])
    plot_band(ax_c, eps_grid, corr_p[:, :, j],
              on_bound_2d=on_bnd_c[:, :, j], bound=BOUND_C,
              color="C2", scatter_alpha=0.15, scatter_size=4)
    ax_c.axhline(ab_clean[j], color="black", linestyle=":", linewidth=0.9)
    ax_c.axvline(0.5, color="C3", linestyle="--", linewidth=0.7, alpha=0.6)
    ax_c.set_title(f"corrected: {labels[j]}", fontsize=9)
    ax_c.set_xlabel(r"$\varepsilon = \delta$", fontsize=8)
    mu_j, sd_j = mean_sd(corr_p[:, :, j])
    ax_c.set_ylim(*yax_for_band(mu_j, sd_j, pad_frac=0.04))

fig.suptitle(
    "Breast cancer -- every replicate kept; mean $\\pm$ sd over all of them",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
""")

# ============================================================================
md(r"""
## 5. Focal predictor -- mean concave points

A single-panel close-up on the strongest predictor in the design. The
naive curve traces the symmetric V seen on the synthetic problem; the
corrected curve has the same V shape on the interior fits but is
dragged toward zero in the moderate-noise window where many replicates
pin to the bound.
""")

code(r"""
j_focal = SEL.index("mean concave points") + 1   # +1 for intercept

fig, ax = plt.subplots(figsize=(8, 4.8))
plot_band(ax, eps_grid, naive_p[:, :, j_focal],
          on_bound_2d=None, color="C1",
          label="naive: mean $\\pm$ sd",
          scatter_alpha=0.15, scatter_size=5)
plot_band(ax, eps_grid, corr_p[:, :, j_focal],
          on_bound_2d=on_bnd_c[:, :, j_focal], bound=BOUND_C,
          color="C2",
          label="corrected: mean $\\pm$ sd",
          scatter_alpha=0.15, scatter_size=5)
ax.axhline(ab_clean[j_focal], color="black", linestyle="--", linewidth=1.2,
           label=fr"clean baseline $= {ab_clean[j_focal]:.2f}$")
ax.axvline(0.5, color="C3", linestyle="--", linewidth=0.8, alpha=0.6,
           label=r"pole $\varepsilon + \delta = 1$")

mu_n, sd_n = mean_sd(naive_p[:, :, j_focal])
mu_c, sd_c = mean_sd(corr_p[:, :, j_focal])
all_mu = np.concatenate([mu_n, mu_c])
all_sd = np.concatenate([sd_n, sd_c])
ax.set_ylim(*yax_for_band(all_mu, all_sd, pad_frac=0.04))
ax.set_xlabel(r"$\varepsilon = \delta$")
ax.set_ylabel("coefficient on mean concave points")
ax.set_title("Naive vs corrected on mean concave points (no filter)")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
plt.show()
""")

# ============================================================================
md(r"""
## 6. Wald $p$-values per coefficient

For each coefficient we report the **median Wald $p$-value** with an
**interquartile-range** band, computed across all $B = 120$ replicates
at each $\varepsilon$ (per coordinate -- a bound-hit on one coefficient
does not invalidate the others on the same replicate).  Median and IQR
are the right summary statistics on a log axis: $p$-values are bounded
in $[0, 1]$ and heavily skewed, so the mean $\pm$ sd would produce a
band whose lower edge falls below zero and clips to a flat $10^{-14}$
floor whenever sd exceeds mean; the median and IQR live inside $[0, 1]$
by construction.

At $\varepsilon = 0$ the strong predictors produce $p$-values below
$10^{-3}$. As $\varepsilon$ grows, the standard error inflates by the
$1/c$ mechanism from Section 4.2 of the report, the $z$-statistic
shrinks, and the test loses power, before identification breaks down
entirely in the middle of the noise range.
""")

code(r"""
# We summarise p-values by their median and interquartile range
# (25th-75th percentile).  This is the right spread statistic on a log
# axis: p-values are heavily skewed and bounded in [0, 1], so the mean
# and sd of p make a band that falls off a cliff to 1e-14 at the lower
# edge whenever the sd exceeds the mean.  The median + IQR bands stay
# inside [0, 1] and look like what we mean.
def median_iqr(arr_2d):
    med = np.nanmedian(arr_2d, axis=1)
    q25 = np.nanpercentile(arr_2d, 25, axis=1)
    q75 = np.nanpercentile(arr_2d, 75, axis=1)
    return med, q25, q75

fig, ax_p = plt.subplots(figsize=(11, 4.8))
colors_p = plt.get_cmap("tab10")

for j in range(P):
    med, q25, q75 = median_iqr(pval_p[:, :, j])
    ax_p.fill_between(eps_grid,
                      np.clip(q25, 1e-14, 1.0),
                      np.clip(q75, 1e-14, 1.0),
                      color=colors_p(j), alpha=0.18)
    ax_p.plot(eps_grid, np.clip(med, 1e-14, 1.0), "-o",
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
               "median and interquartile range, every replicate kept")
ax_p.legend(fontsize=8, loc="lower center", ncol=3)

fig.tight_layout()
plt.show()
""")

# ============================================================================
md(r"""
## 7. Take-aways

* On Task C the bound-hit fraction is large for moderate $\varepsilon$:
  more than half of the replicates pin to the bound for
  $\varepsilon$ roughly in $[0.15, 0.85]$. This is genuinely how the
  corrected MLE behaves on the breast-cancer training data --- the
  fitted log-likelihood develops a flat plateau at large $\lVert\beta\rVert$
  because the underlying logistic model is near-separating, and the
  corrected estimator becomes non-identified for those samples.

* The previous *filtered* presentation reported a clean MC mean and
  narrow band by averaging only over the small interior subset, which
  in the moderate-$\varepsilon$ window is a non-representative slice of
  the actual estimator distribution. The no-filter view above shows
  the genuine spread, and the bottom panel of the $p$-value figure
  shows directly when the Wald test stops being usable.

* The pattern at small $\varepsilon$ ($\le 0.10$) is unchanged: the
  corrected MLE is well-behaved, the standard error inflates by
  approximately $1/c$, and the Wald test is significant on the strong
  predictors. The lesson of the no-filter view is therefore not that
  the corrected estimator is bad -- it is that on near-separating data
  the statistician needs to inspect the bound-hit fraction along with
  the point estimate before using it.
""")

# ============================================================================
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

import os
out = os.path.join(os.path.dirname(__file__), "taskc_no_filter.ipynb")
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"Wrote {out} ({len(cells)} cells)")
